// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"

namespace ck {
namespace tile_program {
namespace warp {

struct WarpGemmAttributeMfmaImplF16F16F32M32N32K8
{
    using ADataType = half_t;
    using BDataType = half_t;
    using CDataType = float;

    using AVecType = typename vector_type<half_t, 4>::type;
    using BVecType = typename vector_type<half_t, 4>::type;
    using CVecType = typename vector_type<float, 16>::type;

    static constexpr index_t AMLane     = 32;
    static constexpr index_t BNLane     = 32;
    static constexpr index_t ABKLane    = 2;
    static constexpr index_t ABKPerLane = 4;

    static constexpr index_t CMLane     = 2;
    static constexpr index_t CNLane     = 32;
    static constexpr index_t CM0PerLane = 4;
    static constexpr index_t CM1PerLane = 4;

    // c_vec += a_vec * b_vec
    __device__ void operator()(CVecType& c_vec, const AVecType& a_vec, const BVecType& b_vec) const
    {
        c_vec = __builtin_amdgcn_mfma_f32_32x32x8f16(a_vec, b_vec, c_vec, 0, 0, 0);
    }

    // c_vec = a_vec * b_vec
    __device__ CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
        // FIXME: Is this correct?
        return __builtin_amdgcn_mfma_f32_32x32x8f16(a_vec, b_vec, CVecType{0.f}, 0, 0, 0);
    }
};

struct WarpGemmAttributeMfmaImplF16F16F32M16N16K16
{
    using ADataType = half_t;
    using BDataType = half_t;
    using CDataType = float;

    using AVecType = typename vector_type<half_t, 4>::type;
    using BVecType = typename vector_type<half_t, 4>::type;
    using CVecType = typename vector_type<float, 4>::type;

    static constexpr index_t AMLane     = 16;
    static constexpr index_t BNLane     = 16;
    static constexpr index_t ABKLane    = 4;
    static constexpr index_t ABKPerLane = 4;

    static constexpr index_t CMLane     = 4;
    static constexpr index_t CNLane     = 16;
    static constexpr index_t CM0PerLane = 1;
    static constexpr index_t CM1PerLane = 4;

    // c_vec += a_vec * b_vec
    __device__ void operator()(CVecType& c_vec, const AVecType& a_vec, const BVecType& b_vec) const
    {
        c_vec = __builtin_amdgcn_mfma_f32_16x16x16f16(a_vec, b_vec, c_vec, 0, 0, 0);
    }

    // c_vec = a_vec * b_vec
    __device__ CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
        // FIXME: Is this correct?
        return __builtin_amdgcn_mfma_f32_16x16x16f16(a_vec, b_vec, CVecType{0.f}, 0, 0, 0);
    }
};

} // namespace warp
} // namespace tile_program
} // namespace ck
