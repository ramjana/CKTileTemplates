// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"

namespace ck {
namespace tile_program {

template <typename BlockTile_, // Sequence<...
          typename Gemm0BlockWarps_,
          typename Gemm0WarpTile_,
          typename Gemm1BlockWarps_,
          typename Gemm1WarpTile_,
          typename Gemm2BlockWarps_,
          typename Gemm2WarpTile_,
          typename Gemm3BlockWarps_,
          typename Gemm3WarpTile_>
struct TileFmhaBwdShape
{
    using BlockTile       = remove_cvref_t<BlockTile_>;
    using Gemm0BlockWarps = remove_cvref_t<Gemm0BlockWarps_>;
    using Gemm0WarpTile   = remove_cvref_t<Gemm0WarpTile_>;
    using Gemm1BlockWarps = remove_cvref_t<Gemm1BlockWarps_>;
    using Gemm1WarpTile   = remove_cvref_t<Gemm1WarpTile_>;
    using Gemm2BlockWarps = remove_cvref_t<Gemm2BlockWarps_>;
    using Gemm2WarpTile   = remove_cvref_t<Gemm2WarpTile_>;
    using Gemm3BlockWarps = remove_cvref_t<Gemm3BlockWarps_>;
    using Gemm3WarpTile   = remove_cvref_t<Gemm3WarpTile_>;

    static constexpr index_t kM0 = BlockTile::At(Number<0>{}); // tile size along q seqlen
    static constexpr index_t kN0 = BlockTile::At(Number<1>{}); // tile size along k seqlen
    static constexpr index_t kK0 = BlockTile::At(Number<2>{}); // tile size along gemm0 unroll
    static constexpr index_t kN1 = BlockTile::At(Number<3>{}); // tile size along v head_dim
    static constexpr index_t kK1 = BlockTile::At(Number<4>{}); // tile size along gemm1 unroll
    static constexpr index_t kK2 = BlockTile::At(Number<5>{}); // tile size along gemm2 unroll
    static constexpr index_t kQKHeaddim =
        BlockTile::At(Number<6>{}); // Q & K headdim, used for pipeline that need load Q or K at
                                    // once (or repeately load Q or K as a whole tile)
    static constexpr index_t kVHeaddim = BlockTile::At(Number<7>{}); // V headdim, used for pipeline
                                                                     // that need load V at once
};

} // namespace tile_program
} // namespace ck
