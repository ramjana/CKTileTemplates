// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"

#include "ck/tile_program/tile/tile_distribution.hpp"
#include "ck/tile_program/tile/tile_elementwise.hpp"
#include "ck/tile_program/tile/static_tile_gemm_shape.hpp"
#include "ck/tile_program/warp_tile/warp_gemm.hpp"

namespace ck {
namespace tile_program {
namespace block {

// Default policy for BlockGemmASmemBSmemCRegV1
// Default policy class should not be templated, put template on member functions instead
struct BlockGemmASmemBSmemCRegV1DefaultPolicy
{
    template <typename ADataType,
              typename BDataType,
              typename CDataType,
              index_t kBlockSize,
              typename BlockGemmShape>
    __host__ __device__ static constexpr auto GetConfig()
    {
        using namespace ck::tile_program::warp;

        // FIXME: use heuristic to choose parameters and WarpGEMM
#if 0
    // 128x128x32   32x32x8
    constexpr index_t MWarp = 2;
    constexpr index_t NWarp = 2;

    using WG = WarpGemmMfmaF16F16F32M32N32K8;
#elif 0
        // 128x128x32, 32x32x16, 2x2 warps
        constexpr index_t MWarp = 2;
        constexpr index_t NWarp = 2;

        using WG = WarpGemmMfmaF16F16F32M32N32K16;
#elif 0
        // 128x128x32, 32x32x16, 4x1 warps,
        constexpr index_t MWarp = 4;
        constexpr index_t NWarp = 1;

        using WG = WarpGemmMfmaF16F16F32M32N32K16;
#elif 0
        // 128x128x32, 32x32x16-Transposed C Distribution, 4x1 warps,
        constexpr index_t MWarp = 4;
        constexpr index_t NWarp = 1;

        using WG = WarpGemmMfmaF16F16F32M32N32K16TransposedCDistribution;
#elif 0
        // 128x128x32   16x16x16
        constexpr index_t MWarp = 2;
        constexpr index_t NWarp = 2;

        using WG = WarpGemmMfmaF16F16F32M16N16K16;
#elif 0
        // 128x256x32   32x32x8
        constexpr index_t MWarp = 2;
        constexpr index_t NWarp = 2;

        using WG = WarpGemmMfmaF16F16F32M32N32K8;
#elif 0
        // 128x256x32   32x32x16
        constexpr index_t MWarp = 2;
        constexpr index_t NWarp = 2;

        using WG = WarpGemmMfmaF16F16F32M32N32K16;
#elif 0
        // 128x256x32   16x16x16
        constexpr index_t MWarp = 2;
        constexpr index_t NWarp = 2;

        using WG = WarpGemmMfmaF16F16F32M16N16K16;
#elif 0
        // 256x128x32   32x32x8
        constexpr index_t MWarp = 2;
        constexpr index_t NWarp = 2;

        using WG = WarpGemmMfmaF16F16F32M32N32K8;
#elif 0
        // 256x128x32   32x32x16
        constexpr index_t MWarp = 2;
        constexpr index_t NWarp = 2;

        using WG = WarpGemmMfmaF16F16F32M32N32K16;
#elif 0
        // 256x128x32, 32x32x16, transposed C distribution
        constexpr index_t MWarp = 2;
        constexpr index_t NWarp = 2;

        using WG = WarpGemmMfmaF16F16F32M32N32K16TransposedCDistribution;
#elif 0
        // 256x128x32, 16x16x32, transposed C distribution
        constexpr index_t MWarp = 2;
        constexpr index_t NWarp = 2;

        using WG = WarpGemmMfmaF16F16F32M16N16K32TransposedCDistribution;
#endif

        static_assert(kBlockSize % get_warp_size() == 0, "wrong!");

        constexpr index_t NumWarp = kBlockSize / get_warp_size();

        constexpr index_t kMPerBlock = BlockGemmShape::kM;
        constexpr index_t kNPerBlock = BlockGemmShape::kN;
        constexpr index_t kKPerBlock = BlockGemmShape::kK;

        if constexpr(NumWarp == 4 && kMPerBlock % 128 == 0 &&
                     kNPerBlock % 128 == 0 % kKPerBlock % 16 == 0)
        {
            return make_tuple(WarpGemmMfmaF16F16F32M32N32K16{}, 2, 2);
        }
        else
        {
            return make_tuple(WarpGemmMfmaF16F16F32M32N32K16{}, 2, 2);
        }
    }
};

} // namespace block
} // namespace tile_program
} // namespace ck
