// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"

#include "ck/tile_program/tile/tile_distribution.hpp"
#include "ck/tile_program/tile/tile_gemm_shape.hpp"
#include "ck/tile_program/tile/tile_elementwise.hpp"
#include "ck/tile_program/block_tile_pipeline/block_gemm_pipeline_problem.hpp"
#include "ck/tile_program/block_tile_pipeline/block_gemm_pipeline_agmem_bgmem_creg_v2.hpp"
#include "ck/tile_program/block_tile_pipeline/block_gemm_pipeline_agmem_bgmem_creg_v2_default_policy.hpp"
#include "ck/tile_program/block_tile/block_gemm_areg_bgmem_creg_problem.hpp"
#include "ck/tile_program/block_tile/block_gemm_areg_bgmem_creg_v1.hpp"
#include "ck/tile_program/block_tile/block_gemm_areg_bsmem_creg_v1_default_policy.hpp"

// C0 = A0 * B0
// C1 = C0 * B1
template <typename A0DataType,
          typename B0DataType,
          typename B1DataType,
          typename Acc0DataType,
          typename C0DataType,
          typename Acc1DataType,
          typename C1DataType,
          ck::index_t kBlockSize,
          ck::index_t kM0PerBlock,
          ck::index_t kN0PerBlock,
          ck::index_t kK0PerBlock,
          ck::index_t kN1PerBlock>
struct GemmGemm
{
    static constexpr auto I0         = ck::Number<0>{};
    static constexpr auto BlockSize  = ck::Number<kBlockSize>{};
    static constexpr auto M0PerBlock = ck::Number<kM0PerBlock>{};
    static constexpr auto N0PerBlock = ck::Number<kN0PerBlock>{};
    static constexpr auto K0PerBlock = ck::Number<kK0PerBlock>{};
    static constexpr auto N1PerBlock = ck::Number<kN1PerBlock>{};

    // block gemm0 pipeline
    using BlockGemm0Pipeline = ck::tile_program::block::BlockGemmPipelineAGmemBGmemCRegV2<
        ck::tile_program::block::BlockGemmPipelineProblem<
            A0DataType,
            B0DataType,
            Acc0DataType,
            kBlockSize,
            ck::tile_program::TileGemmShape<kM0PerBlock, kN0PerBlock, kK0PerBlock>>,
        ck::tile_program::block::BlockGemmPipelineAGmemBGmemCRegV2DefaultPolicy>;

    // block gemm1
    using BlockGemm1 = ck::tile_program::block::BlockGemmARegBGmemCRegV1<
        ck::tile_program::block::BlockGemmARegBGmemCRegProblem<
            C0DataType,
            B1DataType,
            Acc1DataType,
            kBlockSize,
            ck::tile_program::TileGemmShape<kM0PerBlock, kN1PerBlock, kN0PerBlock>>,
        ck::tile_program::block::BlockGemmARegBGmemCRegV1DefaultPolicy>;

    __device__ static constexpr ck::index_t GetStaticLdsSize()
    {
        return ck::math::max(BlockGemm0Pipeline::GetStaticLdsSize(),
                             BlockGemm1::GetStaticLdsSize());
    }

    __device__ void operator()(const A0DataType* p_a0,
                               const B0DataType* p_b0,
                               const B1DataType* p_b1,
                               C1DataType* p_c1,
                               const ck::index_t M0,
                               const ck::index_t N0,
                               const ck::index_t K0,
                               const ck::index_t N1,
                               const ck::index_t Lda0,
                               const ck::index_t Ldb0,
                               const ck::index_t Ldb1,
                               const ck::index_t Ldc1)
    {
        using namespace ck;
        using namespace ck::tile_program;
        using namespace ck::tile_program::block;

        // divide problem
        const auto id_block = get_block_id();

        const auto num_tile_m0 = M0 / kM0PerBlock;
        const auto num_tile_n1 = N1 / kN1PerBlock;

        const auto block2tile = make_cluster_descriptor(make_tuple(num_tile_m0, num_tile_n1));

        const auto id_tile = block2tile.CalculateBottomIndex(make_tuple(id_block));

        const auto iM0 = __builtin_amdgcn_readfirstlane(id_tile.At<0>() * kM0PerBlock);
        const auto iN1 = __builtin_amdgcn_readfirstlane(id_tile.At<1>() * kN1PerBlock);

        __shared__ char p_smem_char[GetStaticLdsSize()];

        // A0/B0/B1 DRAM
        // FIXME: assume layout A0[M0, K0], B0[N0, K0], B1[N1, N0], C1[M0, N1]
        const auto a0_dram_grid = make_naive_tensor_view<AddressSpaceEnum::Global>(
            p_a0, make_tuple(M0, K0), make_tuple(Lda0, 1), Number<32>{}, Number<1>{});

        const auto b0_dram_grid = make_naive_tensor_view<AddressSpaceEnum::Global>(
            p_b0, make_tuple(N0, K0), make_tuple(Ldb0, 1), Number<32>{}, Number<1>{});

        const auto b1_dram_grid = make_naive_tensor_view<AddressSpaceEnum::Global>(
            p_b1, make_tuple(N1, N0), make_tuple(Ldb1, 1), Number<32>{}, Number<1>{});

        // A0/B0/B1 DRAM window
        auto a0_dram_block_window =
            make_tile_window(a0_dram_grid, make_tuple(M0PerBlock, K0PerBlock), {iM0, 0});

        auto b0_dram_block_window =
            make_tile_window(b0_dram_grid, make_tuple(N0PerBlock, K0PerBlock), {0, 0});

        auto b1_dram_block_window =
            make_tile_window(b1_dram_grid, make_tuple(N1PerBlock, N0PerBlock), {iN1, 0});

        // Block GEMM0 pipeline
        constexpr auto block_gemm0_pipeline = BlockGemm0Pipeline{};

        // Bock GEMM1
        constexpr auto block_gemm1 = BlockGemm1{};

        // Acc1 tile
        auto acc1_block_tile = decltype(block_gemm1(
            tile_elementwise_in(
                type_convert<C0DataType, Acc0DataType>,
                block_gemm0_pipeline(a0_dram_block_window, b0_dram_block_window, 0, nullptr)),
            b1_dram_block_window,
            nullptr)){};

        // init Acc1
        tile_elementwise_inout([](auto& acc1) { acc1 = 0; }, acc1_block_tile);

        index_t iN0 = 0;

        do
        {
            // Block GEMM0 pipeline: acc0 = a0 * b0
            const auto acc0_block_tile = block_gemm0_pipeline(
                a0_dram_block_window, b0_dram_block_window, K0 / kK0PerBlock, p_smem_char);

            // type cast acc0 into c0
            const auto c0_block_tile =
                tile_elementwise_in(type_convert<C0DataType, Acc0DataType>, acc0_block_tile);

            // wait for gemm0 pipeline to finish reading Lds
            block_sync_lds();

            block_gemm1(acc1_block_tile, c0_block_tile, b1_dram_block_window, p_smem_char);

            move_tile_window(b0_dram_block_window, {kN0PerBlock, 0});
            move_tile_window(b1_dram_block_window, {0, kN0PerBlock});

            // wait for gemm1 to finish reading Lds, before next iteration
            block_sync_lds();

            iN0 += kN0PerBlock;

        } while(iN0 < N0);

        // type cast acc1 into c1
        const auto c1_block_tile =
            tile_elementwise_in(type_convert<C1DataType, Acc1DataType>, acc1_block_tile);

        // store c1
        auto c1_dram_grid = make_naive_tensor_view<AddressSpaceEnum::Global>(
            p_c1, make_tuple(M0, N1), make_tuple(Ldc1, 1), Number<32>{}, Number<1>{});

        auto c1_dram_window = make_tile_window(c1_dram_grid,
                                               make_tuple(M0PerBlock, N1PerBlock),
                                               {iM0, iN1},
                                               c1_block_tile.GetTileDistribution());

        store_tile(c1_dram_window, c1_block_tile);
    }
};
