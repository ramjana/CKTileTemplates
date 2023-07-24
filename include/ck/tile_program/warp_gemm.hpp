
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tile_program/warp_gemm_impl.hpp"
#include "ck/tile_program/warp_gemm_attribute_mfma.hpp"
#include "ck/tile_program/warp_gemm_attribute_mfma_impl.hpp"

namespace ck {
namespace tile_program {
namespace warp {

using WarpGemmMfmaF16F16F32M32N32K8 =
    ck::tile_program::warp::WarpGemmImpl<ck::tile_program::warp::WarpGemmAtrributeMfma<
        ck::tile_program::warp::WarpGemmAttributeMfmaImplF16F16F32M32N32K8>>;

using WarpGemmMfmaF16F16F32M32N32K16 =
    ck::tile_program::warp::WarpGemmImpl<ck::tile_program::warp::WarpGemmAtrributeMfma<
        ck::tile_program::warp::WarpGemmAttributeMfmaImplF16F16F32M32N32K16>>;

using WarpGemmMfmaF16F16F32M16N16K16 =
    ck::tile_program::warp::WarpGemmImpl<ck::tile_program::warp::WarpGemmAtrributeMfma<
        ck::tile_program::warp::WarpGemmAttributeMfmaImplF16F16F32M16N16K16>>;

using WarpGemmMfmaF16F16F32M16N16K32 =
    ck::tile_program::warp::WarpGemmImpl<ck::tile_program::warp::WarpGemmAtrributeMfma<
        ck::tile_program::warp::WarpGemmAttributeMfmaImplF16F16F32M16N16K32>>;

} // namespace warp
} // namespace tile_program
} // namespace ck
