
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"

#include "ck/tile_program/warp_tile/warp_gemm_impl.hpp"
#include "ck/tile_program/warp_tile/warp_gemm_attribute_mfma.hpp"
#include "ck/tile_program/warp_tile/warp_gemm_attribute_mfma_impl.hpp"

namespace ck {
namespace tile_program {
namespace warp {

#if 0
template<typename DataTypes, typename WarpGemmShape>
struct WarpGemmListByDatatypeAndShape;

template<>
struct WarpGemmListByDatatypeAndShape<Tuple<half_t, half_t, float>, TileGemmShape<16, 16, 16>>
{
    using List = Tuple<
        WarpGemmImpl<WarpGemmAtrributeMfma<WarpGemmAttributeMfmaImplF16F16F32M16N16K16>>,
        WarpGemmImpl<WarpGemmAtrributeMfmaTransposedCDistribution<WarpGemmAttributeMfmaImplF16F16F32M16N16K16>>
        >;
};

template<>
struct WarpGemmListByDatatypeAndShape<Tuple<half_t, half_t, float>, TileGemmShape<32, 32, 8>>
{
    using List = Tuple<
        WarpGemmImpl<WarpGemmAtrributeMfma<WarpGemmAttributeMfmaImplF16F16F32M32N32K8>>,
        WarpGemmImpl<WarpGemmAtrributeMfmaTransposedCDistribution<WarpGemmAttributeMfmaImplF16F16F32M32N32K8>>
        >;
};

template<>
struct WarpGemmListByDatatypeAndShape<Tuple<half_t, half_t, float>, TileGemmShape<32, 32, 16>>
{
    using List = Tuple<
        WarpGemmImpl<WarpGemmAtrributeMfma<WarpGemmAttributeMfmaImplF16F16F32M32N32K16>>,
        WarpGemmImpl<WarpGemmAtrributeMfmaTransposedCDistribution<WarpGemmAttributeMfmaImplF16F16F32M32N32K16>>
        >;
};
#endif

using WarpGemmMfmaF16F16F32M32N32K8 =
    WarpGemmImpl<WarpGemmAtrributeMfma<WarpGemmAttributeMfmaImplF16F16F32M32N32K8>>;

using WarpGemmMfmaF16F16F32M32N32K16 =
    WarpGemmImpl<WarpGemmAtrributeMfma<WarpGemmAttributeMfmaImplF16F16F32M32N32K16>>;

using WarpGemmMfmaF16F16F32M32N32K16TransposedCDistribution = WarpGemmImpl<
    WarpGemmAtrributeMfmaTransposedCDistribution<WarpGemmAttributeMfmaImplF16F16F32M32N32K16>>;

using WarpGemmMfmaF16F16F32M16N16K16 =
    WarpGemmImpl<WarpGemmAtrributeMfma<WarpGemmAttributeMfmaImplF16F16F32M16N16K16>>;

using WarpGemmMfmaF16F16F32M16N16K32 =
    WarpGemmImpl<WarpGemmAtrributeMfma<WarpGemmAttributeMfmaImplF16F16F32M16N16K32>>;

} // namespace warp
} // namespace tile_program
} // namespace ck
