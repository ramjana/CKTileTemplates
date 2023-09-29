// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"

#include "ck/tile_program/tile/tile_distribution.hpp"
#include "ck/tile_program/tile/tile_elementwise.hpp"
#include "ck/tile_program/tile/tile_gemm_shape.hpp"
#include "ck/tile_program/warp_tile/warp_gemm.hpp"
#include "ck/tile_program/block_tile_pipeline/block_gemm_pipeline_agmem_bgmem_creg_v1.hpp"
#include "ck/tile_program/block_tile_pipeline/block_gemm_pipeline_agmem_bgmem_creg_v2.hpp"
#include "ck/tile_program/grid/grid_gemm_problem.hpp"
#include "ck/tile_program/grid/grid_gemm_v1.hpp"
#include "ck/tile_program/grid/grid_gemm_v1_default_policy.hpp"

// C = A * B
template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename AElementFunction,
          typename BElementFunction,
          typename CElementFunction,
          ck::index_t kBlockSize_,
          ck::index_t kMPerBlock_,
          ck::index_t kNPerBlock_,
          ck::index_t kKPerBlock_>
struct Gemm
{
    static_assert(std::is_same_v<ALayout, ck::tensor_layout::gemm::RowMajor> &&
                  std::is_same_v<BLayout, ck::tensor_layout::gemm::ColumnMajor> &&
                  std::is_same_v<CLayout, ck::tensor_layout::gemm::RowMajor>);

    using Problem = ck::tile_program::grid::GridGemmProblem<ADataType,
                                                            BDataType,
                                                            AccDataType,
                                                            CDataType,
                                                            AElementFunction,
                                                            BElementFunction,
                                                            CElementFunction>;

    struct Policy
    {
        static constexpr ck::index_t kBlockSize = kBlockSize_;
        static constexpr ck::index_t kMPerBlock = kMPerBlock_;
        static constexpr ck::index_t kNPerBlock = kNPerBlock_;
        static constexpr ck::index_t kKPerBlock = kKPerBlock_;

        template <typename Problem>
        __host__ __device__ static constexpr auto MakeBlock2TileMap(ck::index_t NumTilesM,
                                                                    ck::index_t NumTilesN)
        {
            using namespace ck;

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
            using namespace ck;
            using namespace ck::tile_program;
            using namespace ck::tile_program::block;

            using BlockGemmPipelineProblem_ =
                BlockGemmPipelineProblem<ADataType,
                                         BDataType,
                                         AccDataType,
                                         kBlockSize,
                                         TileGemmShape<kMPerBlock, kNPerBlock, kKPerBlock>>;

            return BlockGemmPipelineAGmemBGmemCRegV2<
                BlockGemmPipelineProblem_,
                BlockGemmPipelineAGmemBGmemCRegV2DefaultPolicy>{};
        }
    };

    using GridGemm = ck::GridGemmV1<Problem, Policy>;

    __device__ void operator()(const ADataType* p_a,
                               const BDataType* p_b,
                               CDataType* p_c,
                               const ck::index_t M,
                               const ck::index_t N,
                               const ck::index_t K,
                               const ck::index_t Lda,
                               const ck::index_t Ldb,
                               const ck::index_t Ldc,
                               const AElementFunction& a_element_func,
                               const BElementFunction& b_element_func,
                               const CElementFunction& c_element_func) const
    {
        using namespace ck;
        using namespace ck::tile_program;
        using namespace ck::tile_program::block;

        // FIXME: assume RCR layout
        const auto a_dram_grid = make_naive_tensor_view<AddressSpaceEnum::Global>(
            p_a, make_tuple(M, K), make_tuple(Lda, 1), Number<32>{}, Number<1>{});

        const auto b_dram_grid = make_naive_tensor_view<AddressSpaceEnum::Global>(
            p_b, make_tuple(N, K), make_tuple(Ldb, 1), Number<32>{}, Number<1>{});

        auto c_dram_grid = make_naive_tensor_view<AddressSpaceEnum::Global>(
            p_c, make_tuple(M, N), make_tuple(Ldc, 1), Number<32>{}, Number<1>{});

        GridGemm{}(
            a_dram_grid, b_dram_grid, c_dram_grid, a_element_func, b_element_func, c_element_func);
    }
};
