#include <cstring>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/cluster_descriptor.hpp"
#include "ck/tensor/tensor_view.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/host_utility/device_prop.hpp"

#include "tile_program.hpp"
#include "ck/tile_program/meta_data_buffer.hpp"
#include "ck/tile_program/block_tensor_distribution.hpp"
#include "ck/tile_program/block_tensor_window.hpp"
#include "ck/tile_program/static_block_distributed_tensor.hpp"
#include "ck/tile_program/load_block_distributed_tensor.hpp"
#include "ck/tile_program/store_block_distributed_tensor.hpp"
#include "ck/tile_program/block_gemm_impl_cr_as_bs.hpp"
#include "ck/tile_program/block_gemm_impl_cr_ar_bs.hpp"
#include "ck/tile_program/block_elementwise.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/fill.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"

template <typename ADataType, typename BDataType, typename CDataType, typename AccDataType>
void reference_gemm(const Tensor<ADataType>& a_m_k,
                    const Tensor<BDataType>& b_n_k,
                    Tensor<CDataType>& c_m_n)
{
    auto f_mk_kn_mn = [&](auto m, auto n) {
        const int K = a_m_k.mDesc.GetLengths()[1];

        AccDataType v_acc = 0;

        for(int k = 0; k < K; ++k)
        {
            ADataType v_a = a_m_k(m, k);
            BDataType v_b = b_n_k(n, k);

            v_acc += ck::type_convert<AccDataType>(v_a) * ck::type_convert<AccDataType>(v_b);
        }

        c_m_n(m, n) = ck::type_convert<CDataType>(v_acc);
    };

    make_ParallelTensorFunctor(f_mk_kn_mn,
                               c_m_n.mDesc.GetLengths()[0],
                               c_m_n.mDesc.GetLengths()[1])(std::thread::hardware_concurrency());
}

template <typename A0DataType,
          typename B0DataType,
          typename C0DataType,
          typename B1DataType,
          typename C1DataType,
          ck::index_t kM0PerBlock,
          ck::index_t kN0PerBlock,
          ck::index_t kK0PerBlock,
          ck::index_t kN1PerBlock>
struct GemmGemm
{
    __host__ __device__ static constexpr auto MakeA0LdsBlockDescriptor()
    {
        using namespace ck;

        constexpr auto a0_lds_block_desc_d1_d2_d3 = make_naive_tensor_descriptor_packed(
            make_tuple(kM0PerBlock / 2, 2, kK0PerBlock), Number<32>{});

        constexpr auto a0_lds_block_desc_d4_d5_d6 = transform_tensor_descriptor(
            a0_lds_block_desc_d1_d2_d3,
            make_tuple(make_xor_transform(make_tuple(kM0PerBlock / 2, kK0PerBlock), 8),
                       make_pass_through_transform(2)),
            make_tuple(Sequence<0, 2>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

        constexpr auto a0_lds_block_desc_m_k = transform_tensor_descriptor(
            a0_lds_block_desc_d4_d5_d6,
            make_tuple(make_merge_transform(make_tuple(kM0PerBlock / 2, 2)),
                       make_pass_through_transform(kK0PerBlock)),
            make_tuple(Sequence<0, 1>{}, Sequence<2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return a0_lds_block_desc_m_k;
    }

    __host__ __device__ static constexpr auto MakeB0LdsBlockDescriptor()
    {
        using namespace ck;

        constexpr auto b0_lds_block_desc_d1_d2_d3 = make_naive_tensor_descriptor_packed(
            make_tuple(kN0PerBlock / 2, 2, kK0PerBlock), Number<32>{});

        constexpr auto b0_lds_block_desc_d4_d5_d6 = transform_tensor_descriptor(
            b0_lds_block_desc_d1_d2_d3,
            make_tuple(make_xor_transform(make_tuple(kN0PerBlock / 2, kK0PerBlock), 8),
                       make_pass_through_transform(2)),
            make_tuple(Sequence<0, 2>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

        constexpr auto b0_lds_block_desc_n_k = transform_tensor_descriptor(
            b0_lds_block_desc_d4_d5_d6,
            make_tuple(make_merge_transform(make_tuple(kN0PerBlock / 2, 2)),
                       make_pass_through_transform(kK0PerBlock)),
            make_tuple(Sequence<0, 1>{}, Sequence<2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return b0_lds_block_desc_n_k;
    }

    //
    __host__ __device__ static constexpr ck::index_t GetStaticLdsSize()
    {
        return ck::math::integer_divide_ceil(
                   sizeof(A0DataType) * MakeA0LdsBlockDescriptor().GetElementSpaceSize(), 16) *
                   16 +
               sizeof(B0DataType) * MakeB0LdsBlockDescriptor().GetElementSpaceSize();
    }

    __host__ __device__ void operator()(ProgramServer& ps,
                                        const A0DataType* p_a0,
                                        const B0DataType* p_b0,
                                        const B1DataType* p_b1,
                                        C1DataType* p_c1,
                                        ck::index_t M0,
                                        ck::index_t N0,
                                        ck::index_t K0,
                                        ck::index_t N1,
                                        ck::index_t Lda0,
                                        ck::index_t Ldb0,
                                        ck::index_t Ldb1,
                                        ck::index_t Ldc1)
    {
        using namespace ck;
        using namespace ck::tile_program;
        using namespace ck::tile_program::block;

        __shared__ char p_shared_char[GetStaticLdsSize()];

        // FIXME: assume layout A0[M0, K0], B0[N0, K0], B1[N1, N0], C1[M0, N1]
        const auto a0_dram_grid = make_naive_tensor_view<AddressSpaceEnum::Global>(
            p_a0, make_tuple(M0, K0), make_tuple(Lda0, 1), Number<32>{}, Number<1>{});

        const auto b0_dram_grid = make_naive_tensor_view<AddressSpaceEnum::Global>(
            p_b0, make_tuple(N0, K0), make_tuple(Ldb0, 1), Number<32>{}, Number<1>{});

        const auto b1_dram_grid = make_naive_tensor_view<AddressSpaceEnum::Global>(
            p_b1, make_tuple(N1, N0), make_tuple(Ldb1, 1), Number<32>{}, Number<1>{});

        // divide problem
        const auto id_block = ps.get_block_1d_id();

        const auto num_tile_m0 = M0 / kM0PerBlock;
        const auto num_tile_n1 = N1 / kN1PerBlock;

        const auto block2tile = ps(make_cluster_descriptor(make_tuple(num_tile_m0, num_tile_n1)));

        const auto id_tile = block2tile.CalculateBottomIndex(make_tuple(id_block));

        const auto iM0 = ps.read_first_lane(id_tile.At<0>() * kM0PerBlock);
        const auto iN1 = ps.read_first_lane(id_tile.At<1>() * kN1PerBlock);

        // A0 tile in LDS
        A0DataType* p_a0_lds = static_cast<A0DataType*>(static_cast<void*>(p_shared_char));

        // [allow optimization] allow different LDS layouts
        constexpr auto a0_lds_block_desc = MakeA0LdsBlockDescriptor();

        auto a0_lds_block = make_tensor_view<AddressSpaceEnum::Lds>(p_a0_lds, a0_lds_block_desc);

        constexpr index_t a0_lds_block_space_size_aligned =
            math::integer_divide_ceil(sizeof(A0DataType) * a0_lds_block_desc.GetElementSpaceSize(),
                                      16) *
            16;

        // B0 tile in LDS
        B0DataType* p_b0_lds = static_cast<B0DataType*>(
            static_cast<void*>(p_shared_char + a0_lds_block_space_size_aligned));

        // [allow optimization] allow different LDS layouts
        constexpr auto b0_lds_block_desc = MakeB0LdsBlockDescriptor();

        auto b0_lds_block = make_tensor_view<AddressSpaceEnum::Lds>(p_b0_lds, b0_lds_block_desc);

        // A0 copy
        // FIXME
        constexpr auto a0_copy_dram_window_dstr = make_static_block_tensor_distribution(
            StaticTensorDistributionEncoding<Sequence<1>,
                                             Tuple<Sequence<2, 4, 16>, Sequence<4, 8>>,
                                             Tuple<Sequence<1>, Sequence<1, 2>>,
                                             Tuple<Sequence<1>, Sequence<2, 0>>,
                                             Sequence<1, 2>,
                                             Sequence<0, 1>>{});

        constexpr auto a0_copy_lds_window_dstr = a0_copy_dram_window_dstr;

        auto a0_copy_dram_window =
            make_block_window(a0_dram_grid, {iM0, 0}, a0_copy_dram_window_dstr);

        auto a0_copy_lds_window = make_block_window(a0_lds_block, {0, 0}, a0_copy_lds_window_dstr);

        // B0 copy
        // FIXME
        constexpr auto b0_copy_dram_window_dstr = make_static_block_tensor_distribution(
            StaticTensorDistributionEncoding<Sequence<1>,
                                             Tuple<Sequence<2, 4, 16>, Sequence<4, 8>>,
                                             Tuple<Sequence<1>, Sequence<1, 2>>,
                                             Tuple<Sequence<1>, Sequence<2, 0>>,
                                             Sequence<1, 2>,
                                             Sequence<0, 1>>{});

        constexpr auto b0_copy_lds_window_dstr = b0_copy_dram_window_dstr;

        auto b0_copy_dram_window =
            make_block_window(b0_dram_grid, {0, 0}, b0_copy_dram_window_dstr);

        auto b0_copy_lds_window = make_block_window(b0_lds_block, {0, 0}, b0_copy_lds_window_dstr);

        // B1 window
        // FIXME: not needed
        constexpr auto b1_copy_dram_window_dstr = make_static_block_tensor_distribution(
            StaticTensorDistributionEncoding<Sequence<1>,
                                             Tuple<Sequence<2, 4, 16>, Sequence<4, 8>>,
                                             Tuple<Sequence<1>, Sequence<1, 2>>,
                                             Tuple<Sequence<1>, Sequence<2, 0>>,
                                             Sequence<1, 2>,
                                             Sequence<0, 1>>{});

        auto b1_dram_block_window =
            make_block_window(b1_dram_grid, {iN1, 0}, b1_copy_dram_window_dstr);

        // FIXME
        auto a0_lds_gemm_window = a0_copy_lds_window;
        auto b0_lds_gemm_window = b0_copy_lds_window;

        // Acc1 tile
        auto acc1_block_tile = decltype(block_gemm_cr_ar_bs(
            block_gemm_cr_as_bs(a0_lds_gemm_window, b0_lds_gemm_window), b1_dram_block_window)){};

        // init Acc1
        block_elementwise_inout([](auto& acc1) { acc1 = 0; }, acc1_block_tile);

        index_t iN0 = 0;

        do
        {
            // Acc0 tile
            auto acc0_block_tile =
                decltype(block_gemm_cr_as_bs(a0_lds_gemm_window, b0_lds_gemm_window)){};

            // init Acc0
            block_elementwise_inout([](auto& acc0) { acc0 = 0; }, acc0_block_tile);

            index_t iK0 = 0;

            do
            {
                const auto a0_block_tile = load_block_tile(a0_copy_dram_window);
                const auto b0_block_tile = load_block_tile(b0_copy_dram_window);

                store_block_tile(a0_copy_lds_window, a0_block_tile);
                store_block_tile(b0_copy_lds_window, b0_block_tile);

                ps.block_sync_lds();

                block_gemm_cr_as_bs(acc0_block_tile, a0_lds_gemm_window, b0_lds_gemm_window);

                ps.block_sync_lds();

                move_block_window(a0_copy_dram_window, {0, kK0PerBlock});
                move_block_window(b0_copy_dram_window, {0, kK0PerBlock});

                iK0 += kK0PerBlock;

            } while(iK0 < K0);

            // convert fp32 Acc0 into fp16 c0
            const auto c0_block_tile = block_elementwise_in(
                [](const auto& acc0) { return type_convert<C0DataType>(acc0); }, acc0_block_tile);

            // tile GEMM on register + DRAM window
            // c0: register
            // b1: DRAM window
            block_gemm_cr_ar_bs(acc1_block_tile, c0_block_tile, b1_dram_block_window);

            move_block_window(a0_copy_dram_window, {0, -K0});
            move_block_window(b0_copy_dram_window, {kN0PerBlock, -K0});
            move_block_window(b1_dram_block_window, {0, kN0PerBlock});

            iN0 += kN0PerBlock;

        } while(iN0 < N0);

        // type convert
        const auto c1_block_tile = block_elementwise_in(
            [](const auto& acc1) { return ck::type_convert<C1DataType>(acc1); }, acc1_block_tile);

        // store C1
        auto c1_dram_grid = make_naive_tensor_view<AddressSpaceEnum::Global>(
            p_c1, make_tuple(M0, N1), make_tuple(Ldc1, 1), Number<32>{}, Number<1>{});

        // FIXME
        constexpr auto c1_block_distr = c1_block_tile.GetBlockDistribution();

        auto c1_dram_window = make_block_window(c1_dram_grid, {iM0, iN1}, c1_block_distr);

        store_block_tile(c1_dram_window, c1_block_tile);
    }
};

int main(int argc, char* argv[])
{
    using A0DataType = ck::half_t;
    using B0DataType = ck::half_t;
    using B1DataType = ck::half_t;
    using C0DataType = ck::half_t;
    using C1DataType = float;

    ck::index_t M0 = 13312;
    ck::index_t N0 = 4096;
    ck::index_t K0 = 128;
    ck::index_t N1 = 128;

    if(argc == 5)
    {
        M0 = std::stoi(argv[1]);
        N0 = std::stoi(argv[2]);
        K0 = std::stoi(argv[3]);
        N1 = std::stoi(argv[4]);
    }

    std::array<ck::index_t, 2> a0_lengths{M0, K0};
    std::array<ck::index_t, 2> a0_strides{K0, 1};

    std::array<ck::index_t, 2> b0_lengths{N0, K0};
    std::array<ck::index_t, 2> b0_strides{K0, 1};

    std::array<ck::index_t, 2> c0_lengths{M0, N0};
    std::array<ck::index_t, 2> c0_strides{N0, 1};

    std::array<ck::index_t, 2> b1_lengths{N1, N0};
    std::array<ck::index_t, 2> b1_strides{N0, 1};

    std::array<ck::index_t, 2> c1_lengths{M0, N1};
    std::array<ck::index_t, 2> c1_strides{N1, 1};

    // host verify
    Tensor<A0DataType> a0_host(a0_lengths, a0_strides);
    Tensor<B0DataType> b0_host(b0_lengths, b0_strides);
    Tensor<B1DataType> b1_host(b1_lengths, b1_strides);
    Tensor<C0DataType> c0_host_ref(c0_lengths, c0_strides);
    Tensor<C1DataType> c1_host_ref(c1_lengths, c1_strides);
    Tensor<C1DataType> c1_host_dev(c1_lengths, c1_strides);

    ck::utils::FillUniformDistributionIntegerValue<A0DataType>{-5.f, 5.f}(a0_host);
    ck::utils::FillUniformDistributionIntegerValue<B0DataType>{-5.f, 5.f}(b0_host);
    ck::utils::FillUniformDistributionIntegerValue<B1DataType>{-5.f, 5.f}(b1_host);

    // reference gemm
    reference_gemm<A0DataType, B0DataType, C0DataType, float>(a0_host, b0_host, c0_host_ref);
    reference_gemm<C0DataType, B1DataType, C1DataType, float>(c0_host_ref, b1_host, c1_host_ref);

    DeviceMem a0_buf(sizeof(A0DataType) * a0_host.GetElementSpaceSize());
    DeviceMem b0_buf(sizeof(B0DataType) * b0_host.GetElementSpaceSize());
    DeviceMem b1_buf(sizeof(B1DataType) * b1_host.GetElementSpaceSize());
    DeviceMem c1_buf(sizeof(C1DataType) * c1_host_ref.GetElementSpaceSize());

    a0_buf.ToDevice(a0_host.mData.data());
    b0_buf.ToDevice(b0_host.mData.data());
    b1_buf.ToDevice(b1_host.mData.data());

    constexpr ck::index_t kM0PerBlock = 128;
    constexpr ck::index_t kN0PerBlock = 128;
    constexpr ck::index_t kK0PerBlock = 32;
    constexpr ck::index_t kN1PerBlock = 128;

    constexpr ck::index_t kBlockSize = 256;
    ck::index_t kGridSize            = (M0 / kM0PerBlock) * (N1 / kN1PerBlock);

    std::cout << "grid size " << kGridSize << std::endl;

    float ave_time = launch(ProgramServer{},
                            GemmGemm<A0DataType,
                                     B0DataType,
                                     C0DataType,
                                     B1DataType,
                                     C1DataType,
                                     kM0PerBlock,
                                     kN0PerBlock,
                                     kK0PerBlock,
                                     kN1PerBlock>{},
                            kGridSize,
                            kBlockSize,
                            static_cast<A0DataType*>(a0_buf.GetDeviceBuffer()),
                            static_cast<B0DataType*>(b0_buf.GetDeviceBuffer()),
                            static_cast<B1DataType*>(b1_buf.GetDeviceBuffer()),
                            static_cast<C1DataType*>(c1_buf.GetDeviceBuffer()),
                            M0,
                            N0,
                            K0,
                            N1,
                            K0,  // Lda0
                            K0,  // Ldb0
                            N0,  // Ldb1
                            N1); // Ldc1

    c1_buf.FromDevice(c1_host_dev.mData.data());

    std::size_t flop      = std::size_t(2) * M0 * N0 * K0 + std::size_t(2) * M0 * N1 * N0;
    std::size_t num_btype = sizeof(A0DataType) * M0 * K0 + sizeof(B0DataType) * N0 * K0 +
                            sizeof(B1DataType) * N1 * N0 + sizeof(C1DataType) * M0 * N1;

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s"
              << std::endl;

#if 0
    LogRangeAsType<float>(std::cout << "ref: ", c1_host_ref.mData, ", ") << std::endl;
    LogRangeAsType<float>(std::cout << "dev: ", c1_host_dev.mData, ", ") << std::endl;
#endif

    return ck::utils::check_err(c1_host_dev, c1_host_ref);
}
