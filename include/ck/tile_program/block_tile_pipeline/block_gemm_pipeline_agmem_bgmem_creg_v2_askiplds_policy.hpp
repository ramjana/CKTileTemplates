// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/tile_program/block_tile_pipeline/blockgemm_pipeline_agmem_bgmem_creg_policy_impl.hpp"
#include "ck/tile_program/block_tile_pipeline/block_gemm_pipeline_agmem_bgmem_creg_v1.hpp"

namespace ck {
namespace tile_program {
namespace block {

// NOTE: Assume A is K-Major
struct BlockGemmPipelineAGmemBGmemCRegV2SkipALdsPolicy
{
    template <typename Problem>
    __host__ __device__ static constexpr auto MakeARegBlockDescriptor()
    {
        constexpr auto blockgemm = GetBlockGemm<Problem>();
        using BlockGemm = ck::remove_cvref_t<decltype(blockgemm)>;

        return policy_impl::MakeARegBlockDescriptor<Problem, BlockGemm>();
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto MakeBLdsBlockDescriptor()
    {
        return policy_impl::MakeBLdsBlockDescriptor<Problem>();
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto MakeADramTileDistribution()
    {
        constexpr auto blockgemm = GetBlockGemm<Problem>();
        using BlockGemm = ck::remove_cvref_t<decltype(blockgemm)>;

        return policy_impl::MakeADramTileDistribution_ASkipLDS<Problem, BlockGemm>();
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto MakeBDramTileDistribution()
    {
        return policy_impl::MakeADramTileDistribution<Problem>();
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto GetBlockGemm()
    {
        using BlockGemmPolicy = BlockGemmARegBSmemCRegV1K8Policy;

        return BlockGemmARegBSmemCRegV1<Problem, BlockGemmPolicy>{};
    }
};

} // namespace block
} // namespace tile_program
} // namespace ck
