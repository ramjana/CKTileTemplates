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
#include "ck/tile_program/block_tile_gemm.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/fill.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"

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
    // FIXME
    __host__ __device__ static constexpr ck::index_t GetDynamicLdsSize()
    {
        return ck::math::integer_divide_ceil(sizeof(ADataType) * kMPerBlock * kKPerBlock, 16) * 16 +
               sizeof(BDataType) * kNPerBlock * kKPerBlock;
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
#if 0
        (void)ps;
        (void)p_a;
        (void)p_b;
        (void)p_c;
        (void)M;
        (void)N;
        (void)K;
        (void)Lda;
        (void)Ldb;
        (void)Ldc;
        (void)a_op;
        (void)b_op;
        (void)c_op;
#else
        using namespace ck;
        using namespace ck::tile_program::block;

        __shared__ char p_shared_char[GetDynamicLdsSize()];

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

        const auto iM = id_tile.At<0>() * kMPerBlock;
        const auto iN = id_tile.At<1>() * kNPerBlock;

        // A/B tile in LDS
#if 0
        ADataType* p_a_lds = shared_memmory.get_pointer(0);
#else
        ADataType* p_a_lds = static_cast<ADataType*>(static_cast<void*>(p_shared_char));
#endif

        // [allow optimization] allow different LDS layouts
        constexpr auto a_lds_block_desc =
            make_naive_tensor_descriptor_packed(make_tuple(kMPerBlock, kKPerBlock), Number<32>{});

        auto a_lds_block = make_tensor_view<AddressSpaceEnum::Lds>(p_a_lds, a_lds_block_desc);

#if 0
        BDataType* p_b_lds = shared_memory.get_aligned_pointer(a_lds_byte);
#else
        constexpr index_t a_lds_block_space_size_aligned =
            math::integer_divide_ceil(a_lds_block_desc.GetElementSpaceSize() * sizeof(ADataType),
                                      16) *
            16;

        BDataType* p_b_lds = static_cast<BDataType*>(
            static_cast<void*>(p_shared_char + a_lds_block_space_size_aligned));
#endif

        // [allow optimization] allow different LDS layouts
        constexpr auto b_lds_block_desc =
            make_naive_tensor_descriptor_packed(make_tuple(kNPerBlock, kKPerBlock), Number<32>{});

        auto b_lds_block = make_tensor_view<AddressSpaceEnum::Lds>(p_b_lds, b_lds_block_desc);

        // A copy
        // FIXME
        constexpr auto a_dram_window_dstr = make_static_block_tensor_distribution(
            make_tuple(Sequence<2, 4, 16>{}, Sequence<4, 8>{}),
            Sequence<0>{},
            Sequence<1>{},
            Sequence<0, 1>{},
            Sequence<2, 0>{},
            Sequence<0, 1>{},
            Sequence<0, 1>{},
            Sequence<0, 1>{});

        constexpr auto a_lds_window_dstr = make_static_block_tensor_distribution(
            make_tuple(Sequence<2, 4, 16>{}, Sequence<4, 8>{}),
            Sequence<0>{},
            Sequence<1>{},
            Sequence<0, 1>{},
            Sequence<2, 0>{},
            Sequence<0, 1>{},
            Sequence<0, 1>{},
            Sequence<0, 1>{});

        auto a_dram_window = make_block_window(a_dram_grid, {iM, 0}, a_dram_window_dstr);

        auto a_lds_window = make_block_window(a_lds_block, {0, 0}, a_lds_window_dstr);

        // B copy
        // FIXME
        constexpr auto b_dram_window_dstr = make_static_block_tensor_distribution(
            make_tuple(Sequence<2, 4, 16>{}, Sequence<4, 8>{}),
            Sequence<0>{},
            Sequence<1>{},
            Sequence<0, 1>{},
            Sequence<2, 0>{},
            Sequence<0, 1>{},
            Sequence<0, 1>{},
            Sequence<0, 1>{});

        constexpr auto b_lds_window_dstr = make_static_block_tensor_distribution(
            make_tuple(Sequence<2, 4, 16>{}, Sequence<4, 8>{}),
            Sequence<0>{},
            Sequence<1>{},
            Sequence<0, 1>{},
            Sequence<2, 0>{},
            Sequence<0, 1>{},
            Sequence<0, 1>{},
            Sequence<0, 1>{});

        auto b_dram_window = make_block_window(b_dram_grid, {iN, 0}, b_dram_window_dstr);

        auto b_lds_window = make_block_window(b_lds_block, {0, 0}, b_lds_window_dstr);

        // C tile
        auto c_block_tile = decltype(block_tile_gemm<CDataType>(a_lds_block, b_lds_block)){};

#if 0
        block_tile_elementwise(c_block_tile, [](auto& c){
            c = 0;});
#endif

        index_t iK = 0;

        do
        {
            const auto a_block_tile = load_block_tile(a_dram_window);
            const auto b_block_tile = load_block_tile(b_dram_window);

            store_block_tile(a_lds_window, a_block_tile);
            store_block_tile(b_lds_window, b_block_tile);

            ps.block_sync_lds();

            block_tile_gemm(c_block_tile, a_lds_window, b_lds_window);

            ps.block_sync_lds();

            move_block_window(a_lds_window, {0, kKPerBlock});
            move_block_window(b_lds_window, {0, kKPerBlock});

            iK += kKPerBlock;
        } while(iK < K);

        // FIXME
        constexpr auto c_block_distr = c_block_tile.GetBlockDistribution();

        // store C
        auto c_dram_grid = make_naive_tensor_view<AddressSpaceEnum::Global>(
            p_c, make_tuple(M, N), make_tuple(Ldc, 1), Number<32>{}, Number<1>{});

        auto c_dram_window = make_block_window(c_dram_grid, {iM, iN}, c_block_distr);

        store_block_tile(c_dram_window, c_block_tile);
#endif
    }
};

int main()
{
    using DataType = ck::half_t;

    ck::index_t M = 4096;
    ck::index_t N = 4096;
    ck::index_t K = 4096;

    std::array<ck::index_t, 2> a_lengths{M, K};
    std::array<ck::index_t, 2> a_strides{K, 1};

    std::array<ck::index_t, 2> b_lengths{N, K};
    std::array<ck::index_t, 2> b_strides{K, 1};

    std::array<ck::index_t, 2> c_lengths{M, N};
    std::array<ck::index_t, 2> c_strides{N, 1};

    // host verify
    Tensor<DataType> a_host(a_lengths, a_strides);
    Tensor<DataType> b_host(b_lengths, b_strides);
    Tensor<DataType> c_host_ref(c_lengths, c_strides);
    Tensor<DataType> c_host_dev(c_lengths, c_strides);

    ck::utils::FillUniformDistributionIntegerValue<DataType>{-5.f, 5.f}(a_host);
    ck::utils::FillUniformDistributionIntegerValue<DataType>{-5.f, 5.f}(b_host);

#if 0
    reference_gemm(a_host,
                   b_host,
                   c_host_ref);
#endif

    DeviceMem a_buf(sizeof(DataType) * a_host.GetElementSpaceSize());
    DeviceMem b_buf(sizeof(DataType) * b_host.GetElementSpaceSize());
    DeviceMem c_buf(sizeof(DataType) * c_host_dev.GetElementSpaceSize());

    a_buf.ToDevice(a_host.mData.data());
    b_buf.ToDevice(b_host.mData.data());

    constexpr ck::index_t kGemmMPerBlock = 128;
    constexpr ck::index_t kGemmNPerBlock = 256;
    constexpr ck::index_t kGemmKPerBlock = 32;

    constexpr ck::index_t kBlockSize = 256;
    ck::index_t kGridSize            = (M / kGemmMPerBlock) * (N / kGemmNPerBlock);

    std::cout << "grid size " << kGridSize << std::endl;

    launch(ProgramServer{},
           Gemm<DataType,
                DataType,
                DataType,
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
           static_cast<DataType*>(a_buf.GetDeviceBuffer()),
           static_cast<DataType*>(b_buf.GetDeviceBuffer()),
           static_cast<DataType*>(c_buf.GetDeviceBuffer()),
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

    return ck::utils::check_err(c_host_dev, c_host_ref);
}
