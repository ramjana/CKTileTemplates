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
#include "ck/tile_program/block_tile_elementwise_op.hpp"

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

// C = A * B
template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename AElementWiseOperation,
          typename BElementWiseOperation,
          typename CElementWiseOperation,
          ck::index_t kMPerBlock,
          ck::index_t kNPerBlock,
          ck::index_t kKPerBlock>
struct Gemm
{
    __host__ __device__ static constexpr auto MakeALdsBlockDescriptor()
    {
        using namespace ck;
#if 0
        constexpr auto a_lds_block_desc =
            make_naive_tensor_descriptor_packed(make_tuple(kMPerBlock, kKPerBlock), Number<32>{});

        return a_lds_block_desc;
#elif 0
        constexpr auto a_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(Number<kKPerBlock / 8>{}, Number<kMPerBlock>{}, Number<8>{}),
            make_tuple(Number<(kMPerBlock + 1) * 8>{}, Number<8>{}, Number<1>{}),
            Number<8>{},
            Number<1>{});

        constexpr auto a_lds_block_desc = transform_tensor_descriptor(
            a_lds_block_desc_0,
            make_tuple(make_pass_through_transform(kMPerBlock),
                       make_merge_transform(make_tuple(kKPerBlock / 8, 8))),
            make_tuple(Sequence<1>{}, Sequence<0, 2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return a_lds_block_desc;
#elif 1
        constexpr auto a_lds_block_desc_d1_d2_d3 = make_naive_tensor_descriptor_packed(
            make_tuple(kMPerBlock / 2, 2, kKPerBlock), Number<32>{});

        constexpr auto a_lds_block_desc_d4_d5_d6 = transform_tensor_descriptor(
            a_lds_block_desc_d1_d2_d3,
            make_tuple(make_xor_transform(make_tuple(kMPerBlock / 2, kKPerBlock), 8),
                       make_pass_through_transform(2)),
            make_tuple(Sequence<0, 2>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

        constexpr auto a_lds_block_desc_m_k = transform_tensor_descriptor(
            a_lds_block_desc_d4_d5_d6,
            make_tuple(make_merge_transform(make_tuple(kMPerBlock / 2, 2)),
                       make_pass_through_transform(kKPerBlock)),
            make_tuple(Sequence<0, 1>{}, Sequence<2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return a_lds_block_desc_m_k;
#endif
    }

    __host__ __device__ static constexpr auto MakeBLdsBlockDescriptor()
    {
        using namespace ck;
#if 0
        // 2D layout [N, K]
        constexpr auto b_lds_block_desc =
            make_naive_tensor_descriptor_packed(make_tuple(kNPerBlock, kKPerBlock), Number<32>{});

        return b_lds_block_desc;
#elif 0
        // [K0, M, K1] layout with padding
        constexpr auto b_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(Number<kKPerBlock / 8>{}, Number<kNPerBlock>{}, Number<8>{}),
            make_tuple(Number<(kNPerBlock + 1) * 8>{}, Number<8>{}, Number<1>{}),
            Number<8>{},
            Number<1>{});

        constexpr auto b_lds_block_desc = transform_tensor_descriptor(
            b_lds_block_desc_0,
            make_tuple(make_pass_through_transform(kNPerBlock),
                       make_merge_transform(make_tuple(kKPerBlock / 8, 8))),
            make_tuple(Sequence<1>{}, Sequence<0, 2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return b_lds_block_desc;
#elif 1
        // XOR layout
        constexpr auto b_lds_block_desc_d1_d2_d3 = make_naive_tensor_descriptor_packed(
            make_tuple(kNPerBlock / 2, 2, kKPerBlock), Number<32>{});

        constexpr auto b_lds_block_desc_d4_d5_d6 = transform_tensor_descriptor(
            b_lds_block_desc_d1_d2_d3,
            make_tuple(make_xor_transform(make_tuple(kNPerBlock / 2, kKPerBlock), 8),
                       make_pass_through_transform(2)),
            make_tuple(Sequence<0, 2>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

        constexpr auto b_lds_block_desc_n_k = transform_tensor_descriptor(
            b_lds_block_desc_d4_d5_d6,
            make_tuple(make_merge_transform(make_tuple(kNPerBlock / 2, 2)),
                       make_pass_through_transform(kKPerBlock)),
            make_tuple(Sequence<0, 1>{}, Sequence<2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return b_lds_block_desc_n_k;
#endif
    }

    //
    __host__ __device__ static constexpr ck::index_t GetStaticLdsSize()
    {
        return ck::math::integer_divide_ceil(
                   sizeof(ADataType) * MakeALdsBlockDescriptor().GetElementSpaceSize(), 16) *
                   16 +
               sizeof(BDataType) * MakeBLdsBlockDescriptor().GetElementSpaceSize();
    }

    __host__ __device__ void operator()(ProgramServer& ps,
                                        const ADataType* p_a,
                                        const BDataType* p_b,
                                        CDataType* p_c,
                                        ck::index_t M,
                                        ck::index_t N,
                                        ck::index_t K,
                                        ck::index_t Lda,
                                        ck::index_t Ldb,
                                        ck::index_t Ldc,
                                        AElementWiseOperation /* a_op */,
                                        BElementWiseOperation /* b_op */,
                                        CElementWiseOperation /* c_op */)
    {
        using namespace ck;
        using namespace ck::tile_program;
        using namespace ck::tile_program::block;

        __shared__ char p_shared_char[GetStaticLdsSize()];

        // FIXME: assume RCR layout
        const auto a_dram_grid = make_naive_tensor_view<AddressSpaceEnum::Global>(
            p_a, make_tuple(M, K), make_tuple(Lda, 1), Number<32>{}, Number<1>{});

        const auto b_dram_grid = make_naive_tensor_view<AddressSpaceEnum::Global>(
            p_b, make_tuple(N, K), make_tuple(Ldb, 1), Number<32>{}, Number<1>{});

        // divide problem
        const auto id_block = ps.get_block_1d_id();

        const auto num_tile_m = M / kMPerBlock;
        const auto num_tile_n = N / kNPerBlock;

        const auto block2tile = ps(make_cluster_descriptor(make_tuple(num_tile_m, num_tile_n)));

        const auto id_tile = block2tile.CalculateBottomIndex(make_tuple(id_block));

        const auto iM = ps.read_first_lane(id_tile.At<0>() * kMPerBlock);
        const auto iN = ps.read_first_lane(id_tile.At<1>() * kNPerBlock);

        // A tile in LDS
        ADataType* p_a_lds = static_cast<ADataType*>(static_cast<void*>(p_shared_char));

        // [allow optimization] allow different LDS layouts
        constexpr auto a_lds_block_desc = MakeALdsBlockDescriptor();

        auto a_lds_block = make_tensor_view<AddressSpaceEnum::Lds>(p_a_lds, a_lds_block_desc);

        constexpr index_t a_lds_block_space_size_aligned =
            math::integer_divide_ceil(sizeof(ADataType) * a_lds_block_desc.GetElementSpaceSize(),
                                      16) *
            16;

        // B tile in LDS
        BDataType* p_b_lds = static_cast<BDataType*>(
            static_cast<void*>(p_shared_char + a_lds_block_space_size_aligned));

        // [allow optimization] allow different LDS layouts
        constexpr auto b_lds_block_desc = MakeBLdsBlockDescriptor();

        auto b_lds_block = make_tensor_view<AddressSpaceEnum::Lds>(p_b_lds, b_lds_block_desc);

        // A copy
        // FIXME
        constexpr auto a_copy_dram_window_dstr = make_static_block_tensor_distribution(
            StaticTensorDistributionEncoding<Sequence<1>,
                                             Tuple<Sequence<2, 4, 16>, Sequence<4, 8>>,
                                             Tuple<Sequence<1>, Sequence<1, 2>>,
                                             Tuple<Sequence<1>, Sequence<2, 0>>,
                                             Sequence<1, 2>,
                                             Sequence<0, 1>>{});

        constexpr auto a_copy_lds_window_dstr = a_copy_dram_window_dstr;

        auto a_copy_dram_window = make_block_window(a_dram_grid, {iM, 0}, a_copy_dram_window_dstr);

        auto a_copy_lds_window = make_block_window(a_lds_block, {0, 0}, a_copy_lds_window_dstr);

        // B copy
        // FIXME
        constexpr auto b_copy_dram_window_dstr = make_static_block_tensor_distribution(
            StaticTensorDistributionEncoding<Sequence<1>,
                                             Tuple<Sequence<2, 4, 16>, Sequence<4, 8>>,
                                             Tuple<Sequence<1>, Sequence<1, 2>>,
                                             Tuple<Sequence<1>, Sequence<2, 0>>,
                                             Sequence<1, 2>,
                                             Sequence<0, 1>>{});

        constexpr auto b_copy_lds_window_dstr = b_copy_dram_window_dstr;

        auto b_copy_dram_window = make_block_window(b_dram_grid, {iN, 0}, b_copy_dram_window_dstr);

        auto b_copy_lds_window = make_block_window(b_lds_block, {0, 0}, b_copy_lds_window_dstr);

        // FIXME
        auto a_lds_gemm_window = a_copy_lds_window;
        auto b_lds_gemm_window = b_copy_lds_window;

        // C tile
        auto acc_block_tile = decltype(block_gemm_cr_as_bs(a_lds_gemm_window, b_lds_gemm_window)){};

        // prefetch
        // global read 0
        auto a_block_tile = load_block_tile(a_copy_dram_window);
        auto b_block_tile = load_block_tile(b_copy_dram_window);

        // move to 1
        move_block_window(a_copy_dram_window, {0, kKPerBlock});
        move_block_window(b_copy_dram_window, {0, kKPerBlock});

        // Initialize C
        block_tile_elementwise([](auto& acc) { acc = 0; }, acc_block_tile);

        // LDS write 0
        store_block_tile(a_copy_lds_window, a_block_tile);
        // global read 1
        a_block_tile = load_block_tile(a_copy_dram_window);

        // LDS write 0
        store_block_tile(b_copy_lds_window, b_block_tile);
        // global read 1
        b_block_tile = load_block_tile(b_copy_dram_window);

        index_t iK = 0;

        do
        {
            ps.block_sync_lds();

            // GEMM i
            block_gemm_cr_as_bs(acc_block_tile, a_lds_gemm_window, b_lds_gemm_window);

            ps.block_sync_lds();

            // move to i + 2
            move_block_window(a_copy_dram_window, {0, kKPerBlock});
            move_block_window(b_copy_dram_window, {0, kKPerBlock});

            // LDS write i + 1
            store_block_tile(a_copy_lds_window, a_block_tile);
            // global read i + 2
            a_block_tile = load_block_tile(a_copy_dram_window);

            // LDS write i + 1
            store_block_tile(b_copy_lds_window, b_block_tile);
            // global read i + 2
            b_block_tile = load_block_tile(b_copy_dram_window);

            iK += kKPerBlock;

        } while(iK < K - 2 * kKPerBlock);

        // tail
        {
            ps.block_sync_lds();

            // GEMM num_loop - 2
            block_gemm_cr_as_bs(acc_block_tile, a_lds_gemm_window, b_lds_gemm_window);

            ps.block_sync_lds();

            // LDS write num_loop - 1
            store_block_tile(a_copy_lds_window, a_block_tile);
            store_block_tile(b_copy_lds_window, b_block_tile);

            ps.block_sync_lds();

            // GEMM num_loop - 1
            block_gemm_cr_as_bs(acc_block_tile, a_lds_gemm_window, b_lds_gemm_window);
        }

        // FIXME
        constexpr auto c_block_distr = acc_block_tile.GetBlockDistribution();

        auto c_block_tile = make_static_block_distributed_tensor<CDataType>(c_block_distr);

        // type convert
        block_tile_elementwise(
            [](auto& c, const auto& acc) { c = ck::type_convert<CDataType>(acc); },
            c_block_tile,
            acc_block_tile);

        // store C
        auto c_dram_grid = make_naive_tensor_view<AddressSpaceEnum::Global>(
            p_c, make_tuple(M, N), make_tuple(Ldc, 1), Number<32>{}, Number<1>{});

        auto c_dram_window = make_block_window(c_dram_grid, {iM, iN}, c_block_distr);

        store_block_tile(c_dram_window, c_block_tile);
    }
};

int main(int argc, char* argv[])
{
    using ADataType = ck::half_t;
    using BDataType = ck::half_t;
    using CDataType = ck::half_t;

    ck::index_t M = 3328;
    ck::index_t N = 4096;
    ck::index_t K = 4096;

    if(argc == 4)
    {
        M = std::stoi(argv[1]);
        N = std::stoi(argv[2]);
        K = std::stoi(argv[3]);
    }

    std::array<ck::index_t, 2> a_lengths{M, K};
    std::array<ck::index_t, 2> a_strides{K, 1};

    std::array<ck::index_t, 2> b_lengths{N, K};
    std::array<ck::index_t, 2> b_strides{K, 1};

    std::array<ck::index_t, 2> c_lengths{M, N};
    std::array<ck::index_t, 2> c_strides{N, 1};

    // host verify
    Tensor<ADataType> a_host(a_lengths, a_strides);
    Tensor<BDataType> b_host(b_lengths, b_strides);
    Tensor<CDataType> c_host_ref(c_lengths, c_strides);
    Tensor<CDataType> c_host_dev(c_lengths, c_strides);

    ck::utils::FillUniformDistributionIntegerValue<ADataType>{-5.f, 5.f}(a_host);
    ck::utils::FillUniformDistributionIntegerValue<BDataType>{-5.f, 5.f}(b_host);

    // reference gemm
    reference_gemm<ADataType, ADataType, CDataType, float>(a_host, b_host, c_host_ref);

    DeviceMem a_buf(sizeof(ADataType) * a_host.GetElementSpaceSize());
    DeviceMem b_buf(sizeof(BDataType) * b_host.GetElementSpaceSize());
    DeviceMem c_buf(sizeof(CDataType) * c_host_dev.GetElementSpaceSize());

    a_buf.ToDevice(a_host.mData.data());
    b_buf.ToDevice(b_host.mData.data());

    constexpr ck::index_t kGemmMPerBlock = 128;
    constexpr ck::index_t kGemmNPerBlock = 128;
    constexpr ck::index_t kGemmKPerBlock = 32;

    constexpr ck::index_t kBlockSize = 256;
    ck::index_t kGridSize            = (M / kGemmMPerBlock) * (N / kGemmNPerBlock);

    std::cout << "grid size " << kGridSize << std::endl;

    float ave_time = launch(ProgramServer{},
                            Gemm<ADataType,
                                 BDataType,
                                 CDataType,
                                 ck::tensor_layout::gemm::RowMajor,
                                 ck::tensor_layout::gemm::ColumnMajor,
                                 ck::tensor_layout::gemm::RowMajor,
                                 ck::tensor_operation::element_wise::PassThrough,
                                 ck::tensor_operation::element_wise::PassThrough,
                                 ck::tensor_operation::element_wise::PassThrough,
                                 kGemmMPerBlock,
                                 kGemmNPerBlock,
                                 kGemmKPerBlock>{},
                            kGridSize,
                            kBlockSize,
                            static_cast<ADataType*>(a_buf.GetDeviceBuffer()),
                            static_cast<BDataType*>(b_buf.GetDeviceBuffer()),
                            static_cast<CDataType*>(c_buf.GetDeviceBuffer()),
                            M,
                            N,
                            K,
                            K,
                            K,
                            N,
                            ck::tensor_operation::element_wise::PassThrough{},
                            ck::tensor_operation::element_wise::PassThrough{},
                            ck::tensor_operation::element_wise::PassThrough{});

    c_buf.FromDevice(c_host_dev.mData.data());

    std::size_t flop = std::size_t(2) * M * N * K;
    std::size_t num_btype =
        sizeof(ADataType) * M * K + sizeof(BDataType) * K * N + sizeof(CDataType) * M * N;

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s"
              << std::endl;

    return ck::utils::check_err(c_host_dev, c_host_ref);
}
