// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"
#include "ck/tile_program/block_tensor_distribution.hpp"

namespace ck {
namespace tile_program {
namespace warp {

struct WarpGemm
{
};

struct WarpGemmMfmaF16F16F32M32N32K8 : public WarpGemm
{
    using ADataType = ck::half_t;
    using BDataType = ck::half_t;
    using CDataType = float;

    static constexpr index_t AMLane     = 32;
    static constexpr index_t BNLane     = 32;
    static constexpr index_t ABKLane    = 2;
    static constexpr index_t ABKPerLane = 4;

    static constexpr index_t CMLane     = 2;
    static constexpr index_t CNLane     = 32;
    static constexpr index_t CM0PerLane = 4;
    static constexpr index_t CM1PerLane = 4;

    using AWarpDstrEncoding =
        StaticTensorDistributionEncoding<Sequence<>,
                                         Tuple<Sequence<AMLane>, Sequence<ABKLane, ABKPerLane>>,
                                         Tuple<Sequence<2, 1>>,
                                         Tuple<Sequence<0, 0>>,
                                         Sequence<2>,
                                         Sequence<1>>;

    using BWarpDstrEncoding =
        StaticTensorDistributionEncoding<Sequence<>,
                                         Tuple<Sequence<BNLane>, Sequence<ABKLane, ABKPerLane>>,
                                         Tuple<Sequence<2, 1>>,
                                         Tuple<Sequence<0, 0>>,
                                         Sequence<2>,
                                         Sequence<1>>;

    using CWarpDstrEncoding = StaticTensorDistributionEncoding<
        Sequence<>,
        Tuple<Sequence<CM0PerLane, CMLane, CM1PerLane>, Sequence<CNLane>>,
        Tuple<Sequence<1, 2>>,
        Tuple<Sequence<1, 0>>,
        Sequence<1, 1>,
        Sequence<0, 2>>;

    using AWarpDstr =
        remove_cvref_t<decltype(make_static_block_tensor_distribution(AWarpDstrEncoding{}))>;

    using BWarpDstr =
        remove_cvref_t<decltype(make_static_block_tensor_distribution(BWarpDstrEncoding{}))>;

    using CWarpDstr =
        remove_cvref_t<decltype(make_static_block_tensor_distribution(CWarpDstrEncoding{}))>;

    using AWarpTensor = StaticBlockDistributedTensor<ADataType, AWarpDstr>;
    using BWarpTensor = StaticBlockDistributedTensor<BDataType, BWarpDstr>;
    using CWarpTensor = StaticBlockDistributedTensor<CDataType, CWarpDstr>;

    __device__ void operator()(CWarpTensor& c, const AWarpTensor& a, const BWarpTensor& b) const
    {
        using AVec = typename vector_type<ADataType, AWarpTensor::GetThreadBufferSize()>::type;
        using BVec = typename vector_type<BDataType, BWarpTensor::GetThreadBufferSize()>::type;
        using CVec = typename vector_type<CDataType, CWarpTensor::GetThreadBufferSize()>::type;

        constexpr auto I0 = Number<0>{};

        const auto a_vec = a.GetThreadBuffer().template GetAsType<AVec>(I0);
        const auto b_vec = b.GetThreadBuffer().template GetAsType<BVec>(I0);
        auto c_vec       = c.GetThreadBuffer().template GetAsType<CVec>(I0);

        c_vec = __builtin_amdgcn_mfma_f32_32x32x8f16(a_vec, b_vec, c_vec, 0, 0, 0);

        c.GetThreadBuffer().template SetAsType<CVec>(I0, c_vec);
    }

    __device__ auto operator()(const AWarpTensor& a, const BWarpTensor& b) const
    {
        CWarpTensor c;

        c.Initialize(0);

        operator()(c, a, b);

        return c;
    }
};

} // namespace warp
} // namespace tile_program
} // namespace ck
