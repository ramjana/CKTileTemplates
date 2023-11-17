// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "ck/tile_program/block_tile_pipeline/block_gemm_pipeline_problem.hpp"
#include "ck/tile_program/tile/tile_gemm_shape.hpp"

namespace ck {

// Default policy for GridGemmV1
// Default policy class should not be templated, put template on member functions instead
template <ck::index_t kBlockSize_=256,
          ck::index_t kMPerBlock_=128,
          ck::index_t kNPerBlock_=128,
          ck::index_t kKPerBlock_=32>
struct GridGemmV1DefaultPolicy
{
    static constexpr index_t kBlockSize = kBlockSize_;
    static constexpr index_t kMPerBlock = kMPerBlock_;
    static constexpr index_t kNPerBlock = kNPerBlock_;
    static constexpr index_t kKPerBlock = kKPerBlock_;

    template <typename Problem>
    __host__ __device__ static constexpr auto MakeBlock2TileMap(index_t NumTilesM,
                                                                index_t NumTilesN)
    {
        const auto unmerge = make_merge_transform(make_tuple(NumTilesN, NumTilesM));

        return [unmerge](index_t block_id) {
            MultiIndex<2> unmerged;
            unmerge.CalculateLowerIndex(unmerged, make_multi_index(block_id));

            return make_multi_index(unmerged.At<1>(), unmerged.At<0>());
        };
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto GetBlockGemmPipeline()
    {
        using namespace ck::tile_program;
        using namespace ck::tile_program::block;

        using BlockGemmPipelineProblem_ =
            BlockGemmPipelineProblem<typename Problem::ADataType,
                                     typename Problem::BDataType,
                                     typename Problem::AccDataType,
                                     kBlockSize,
                                     TileGemmShape<kMPerBlock, kNPerBlock, kKPerBlock>>;

        return BlockGemmPipelineAGmemBGmemCRegV2<BlockGemmPipelineProblem_,
                                                 BlockGemmPipelineAGmemBGmemCRegV2DefaultPolicy>{};
    }
};

} // namespace ck
