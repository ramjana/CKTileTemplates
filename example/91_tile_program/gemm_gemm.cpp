#include <cstring>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/cluster_descriptor.hpp"
#include "ck/tensor/tensor_view.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/host_utility/device_prop.hpp"

#include "tile_program.hpp"
#include "ck/tile_program/tile/tile_distribution.hpp"
#include "ck/tile_program/tile/tile_elementwise.hpp"
#include "ck/tile_program/tile/tile_gemm_shape.hpp"
#include "ck/tile_program/warp_tile/warp_gemm.hpp"
#include "ck/tile_program/block_tile_pipeline/block_gemm_pipeline_agmem_bgmem_creg_v2.hpp"
#include "ck/tile_program/block_tile/block_gemm_areg_bsmem_creg_v1.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/fill.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"

#include "reference_gemm.hpp"

template <typename A0DataType,
          typename B0DataType,
          typename Acc0DataType,
          typename C0DataType,
          typename B1DataType,
          typename Acc1DataType,
          typename C1DataType,
          ck::index_t kBlockSize,
          ck::index_t kM0PerBlock,
          ck::index_t kN0PerBlock,
          ck::index_t kK0PerBlock,
          ck::index_t kN1PerBlock>
struct GemmGemm
{
    using BlockGemm0PipelineProblem =
        ck::tile_program::block::BlockGemmPipelineAGmemBGmemCRegV2Problem<
            A0DataType,
            B0DataType,
            Acc0DataType,
            kBlockSize,
            ck::tile_program::TileGemmShape<kM0PerBlock, kN0PerBlock, kK0PerBlock>>;

    using BlockGemm0PipelinePolicy =
        ck::tile_program::block::BlockGemmPipelineAGmemBGmemCRegV2DefaultPolicy;

    using BlockGemm0Pipeline =
        ck::tile_program::block::BlockGemmPipelineAGmemBGmemCRegV2<BlockGemm0PipelineProblem,
                                                                   BlockGemm0PipelinePolicy>;

    __host__ __device__ static constexpr ck::index_t GetStaticLdsSize()
    {
        return BlockGemm0Pipeline::GetStaticLdsSize();
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

        // FIXME: assume layout A0[M0, K0], B0[N0, K0], B1[N1, N0], C1[M0, N1]
        const auto a0_dram_grid = make_naive_tensor_view<AddressSpaceEnum::Global>(
            p_a0, make_tuple(M0, K0), make_tuple(Lda0, 1), Number<32>{}, Number<1>{});

        const auto b0_dram_grid = make_naive_tensor_view<AddressSpaceEnum::Global>(
            p_b0, make_tuple(N0, K0), make_tuple(Ldb0, 1), Number<32>{}, Number<1>{});

        const auto b1_dram_grid = make_naive_tensor_view<AddressSpaceEnum::Global>(
            p_b1, make_tuple(N1, N0), make_tuple(Ldb1, 1), Number<32>{}, Number<1>{});

        // divide problem
        const auto id_block = ps.get_block_id();

        const auto num_tile_m0 = M0 / kM0PerBlock;
        const auto num_tile_n1 = N1 / kN1PerBlock;

        const auto block2tile = ps(make_cluster_descriptor(make_tuple(num_tile_m0, num_tile_n1)));

        const auto id_tile = block2tile.CalculateBottomIndex(make_tuple(id_block));

        const auto iM0 = ps.read_first_lane(id_tile.At<0>() * kM0PerBlock);
        const auto iN1 = ps.read_first_lane(id_tile.At<1>() * kN1PerBlock);

        // A0 DRAM block window
        auto a0_dram_block_window = make_tile_window(
            a0_dram_grid, make_tuple(Number<kM0PerBlock>{}, Number<kK0PerBlock>{}), {iM0, 0});

        // B0 DRAM block window
        auto b0_dram_block_window = make_tile_window(
            b0_dram_grid, make_tuple(Number<kN0PerBlock>{}, Number<kK0PerBlock>{}), {0, 0});

        // B1 window
        auto b1_dram_block_window = make_tile_window(
            b1_dram_grid, make_tuple(Number<kN1PerBlock>{}, Number<kN0PerBlock>{}), {iN1, 0});

        // Block GEMM pipeline
        constexpr auto block_gemm0_pipeline = BlockGemm0Pipeline{};

        __shared__ char p_smem_char[block_gemm0_pipeline.GetStaticLdsSize()];

        // Acc1 tile
        auto acc1_block_tile = decltype(block_gemm_cr_ar_bs(
            block_gemm0_pipeline(a0_dram_block_window, b0_dram_block_window, 0, nullptr),
            b1_dram_block_window)){};

        // init Acc1
        tile_elementwise_inout([](auto& acc1) { acc1 = 0; }, acc1_block_tile);

        index_t iN0 = 0;

        do
        {
            // Acc0 tile
            const auto acc0_block_tile = block_gemm0_pipeline(
                a0_dram_block_window, b0_dram_block_window, K0 / kK0PerBlock, p_smem_char);

            // cast fp32 Acc0 into fp16 c0
            const auto c0_block_tile = tile_elementwise_in(
                [](const auto& acc0) { return type_convert<C0DataType>(acc0); }, acc0_block_tile);

            // tile GEMM on register + DRAM window
            // c0: register
            // b1: DRAM window
            block_gemm_cr_ar_bs(acc1_block_tile, c0_block_tile, b1_dram_block_window);

            // FIXME, block_gemm_pipeline should info how to reset the window
            move_tile_window(a0_dram_block_window, {0, -K0});
            move_tile_window(b0_dram_block_window, {kN0PerBlock, -K0});
            move_tile_window(b1_dram_block_window, {0, kN0PerBlock});

            iN0 += kN0PerBlock;

        } while(iN0 < N0);

        // cast Acc1DataType to C1DataType
        const auto c1_block_tile = tile_elementwise_in(
            [](const auto& acc1) { return ck::type_convert<C1DataType>(acc1); }, acc1_block_tile);

        // store C1
        auto c1_dram_grid = make_naive_tensor_view<AddressSpaceEnum::Global>(
            p_c1, make_tuple(M0, N1), make_tuple(Ldc1, 1), Number<32>{}, Number<1>{});

        auto c1_dram_window =
            make_tile_window(c1_dram_grid,
                             make_tuple(Number<kM0PerBlock>{}, Number<kN1PerBlock>{}),
                             {iM0, iN1},
                             c1_block_tile.GetTileDistribution());

        store_tile(c1_dram_window, c1_block_tile);
    }
};

int main(int argc, char* argv[])
{
    using A0DataType   = ck::half_t;
    using B0DataType   = ck::half_t;
    using Acc0DataType = float;
    using C0DataType   = ck::half_t;
    using B1DataType   = ck::half_t;
    using Acc1DataType = float;
    using C1DataType   = float;

    ck::index_t M0 = 128;
    ck::index_t N0 = 128;
    ck::index_t K0 = 32;
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

#if 0
    ck::utils::FillUniformDistributionIntegerValue<A0DataType>{-5.f, 5.f}(a0_host);
    ck::utils::FillUniformDistributionIntegerValue<B0DataType>{-5.f, 5.f}(b0_host);
    ck::utils::FillUniformDistributionIntegerValue<B1DataType>{-5.f, 5.f}(b1_host);
#else
    ck::utils::FillConstant<A0DataType>{1.f}(a0_host);
    ck::utils::FillConstant<B0DataType>{1.f}(b0_host);
    ck::utils::FillConstant<B1DataType>{1.f}(b1_host);
#endif

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
                                     Acc0DataType,
                                     C0DataType,
                                     B1DataType,
                                     Acc1DataType,
                                     C1DataType,
                                     kBlockSize,
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

    return ck::utils::check_err(c1_host_dev, c1_host_ref);
}
