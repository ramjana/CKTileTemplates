// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"

#include "ck/tile_program/tile/tile_distribution.hpp"
#include "ck/tile_program/tile/tile_elementwise.hpp"
#include "ck/tile_program/tile/tile_gemm_shape.hpp"
#include "ck/tile_program/warp_tile/warp_gemm.hpp"
#include "ck/tile_program/block_tile_pipeline/block_gemm_pipeline_agmem_bgmem_creg_v1.hpp"

// C = A * B
template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename AElementWiseOperation,
          typename BElementWiseOperation,
          typename CElementWiseOperation,
          ck::index_t kBlockSize,
          ck::index_t kMPerBlock,
          ck::index_t kNPerBlock,
          ck::index_t kKPerBlock>
struct GemmBetterPipeline
{
    __host__ __device__ static constexpr ck::index_t GetStaticLdsSize()
    {
        using namespace ck;
        using namespace ck::tile_program;
        using namespace ck::tile_program::block;

        using BlockGemmPipeline =
            BlockGemmPipelineAGmemBGmemCRegV1<ADataType,
                                              BDataType,
                                              AccDataType,
                                              kBlockSize,
                                              TileGemmShape<kMPerBlock, kNPerBlock, kKPerBlock>>;

        return BlockGemmPipeline::GetStaticLdsSize();
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

        // FIXME: assume RCR layout
        const auto a_dram_grid = make_naive_tensor_view<AddressSpaceEnum::Global>(
            p_a, make_tuple(M, K), make_tuple(Lda, 1), Number<32>{}, Number<1>{});

        const auto b_dram_grid = make_naive_tensor_view<AddressSpaceEnum::Global>(
            p_b, make_tuple(N, K), make_tuple(Ldb, 1), Number<32>{}, Number<1>{});

        // divide problem
        const auto id_block = ps.get_block_id();

        const auto num_tile_m = M / kMPerBlock;
        const auto num_tile_n = N / kNPerBlock;

        const auto block2tile = ps(make_cluster_descriptor(make_tuple(num_tile_m, num_tile_n)));

        const auto id_tile = block2tile.CalculateBottomIndex(make_tuple(id_block));

        const auto iM = ps.read_first_lane(id_tile.At<0>() * kMPerBlock);
        const auto iN = ps.read_first_lane(id_tile.At<1>() * kNPerBlock);

        // A DRAM tile window
        auto a_dram_window = make_tile_window(
            a_dram_grid, make_tuple(Number<kMPerBlock>{}, Number<kKPerBlock>{}), {iM, 0});

        // B DRAM tile window
        auto b_dram_window = make_tile_window(
            b_dram_grid, make_tuple(Number<kNPerBlock>{}, Number<kKPerBlock>{}), {iN, 0});

        auto block_gemm_pipeline =
            BlockGemmPipelineAGmemBGmemCRegV1<ADataType,
                                              BDataType,
                                              AccDataType,
                                              kBlockSize,
                                              TileGemmShape<kMPerBlock, kNPerBlock, kKPerBlock>>{};

        __shared__ char p_smem_char[block_gemm_pipeline.GetStaticLdsSize()];

        const auto acc_block_tile = block_gemm_pipeline(
            a_dram_window,
            [](auto& a) { return a; },
            b_dram_window,
            [](auto& b) { return b; },
            K / kKPerBlock,
            p_smem_char);

        // type convert
        auto c_block_tile = tile_elementwise_in(
            [](const auto& acc) { return type_convert<CDataType>(acc); }, acc_block_tile);

        // store C
        auto c_dram_grid = make_naive_tensor_view<AddressSpaceEnum::Global>(
            p_c, make_tuple(M, N), make_tuple(Ldc, 1), Number<32>{}, Number<1>{});

        auto c_dram_window =
            make_tile_window(c_dram_grid,
                             make_tuple(Number<kMPerBlock>{}, Number<kNPerBlock>{}),
                             {iM, iN},
                             c_block_tile.GetTileDistribution());

        store_tile(c_dram_window, c_block_tile);
    }
};
