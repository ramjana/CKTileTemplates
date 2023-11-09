// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/tile_program/block_tile_pipeline/blockgemm_pipeline_agmem_bgmem_creg_policy_impl.hpp"
#include "ck/tile_program/block_tile_pipeline/block_gemm_pipeline_agmem_bgmem_creg_v1.hpp"
#include "ck/tile_program/block_tile/block_gemm_areg_breg_creg_v1.hpp"

namespace ck {
namespace tile_program {
namespace block {

// NOTE: Assume A is K-Major
struct BlockGemmPipelineAGmemBGmemCRegV2SkipABLdsPolicy
{
    template <typename Problem>
    __host__ __device__ static constexpr auto MakeARegBlockDescriptor()
    {
        constexpr auto blockgemm = GetBlockGemm<Problem>();
        using BlockGemm          = ck::remove_cvref_t<decltype(blockgemm)>;

        return policy_impl::make_a_reg_block_descriptor<Problem, BlockGemm>();
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto MakeBRegBlockDescriptor()
    {
        constexpr auto blockgemm = GetBlockGemm<Problem>();
        using BlockGemm          = ck::remove_cvref_t<decltype(blockgemm)>;

        return policy_impl::make_b_reg_block_descriptor<Problem, BlockGemm>();
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto MakeADramTileDistribution()
    {
        constexpr auto blockgemm = GetBlockGemm<Problem>();
        using BlockGemm          = ck::remove_cvref_t<decltype(blockgemm)>;

        return policy_impl::make_a_dram_tile_distribution_skip_lds<Problem, BlockGemm>();
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto MakeBDramTileDistribution()
    {
        constexpr auto blockgemm = GetBlockGemm<Problem>();
        using BlockGemm          = ck::remove_cvref_t<decltype(blockgemm)>;

        return policy_impl::make_b_dram_tile_distribution_skip_lds<Problem, BlockGemm>();
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto GetBlockGemm()
    {
        using BlockGemmPolicy = BlockGemmARegBRegCRegV1DefaultPolicy;

        return BlockGemmARegBRegCRegV1<Problem, BlockGemmPolicy>{};
    }
};

} // namespace block
} // namespace tile_program
} // namespace ck
