// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"

#include "ck/tile_program/tile/tile_distribution.hpp"
#include "ck/tile_program/warp_tile/warp_gemm_attribute_mfma_impl.hpp"

namespace ck {
namespace tile_program {
namespace warp {

template <typename WarpGemmAttributeMfmaImpl_>
struct WarpGemmAtrributeMfma
{
    using Impl = remove_cvref_t<WarpGemmAttributeMfmaImpl_>;

    using ADataType = typename Impl::ADataType;
    using BDataType = typename Impl::BDataType;
    using CDataType = typename Impl::CDataType;

    using AVecType = typename Impl::AVecType;
    using BVecType = typename Impl::BVecType;
    using CVecType = typename Impl::CVecType;

    static constexpr index_t M = Impl::M;
    static constexpr index_t N = Impl::N;
    static constexpr index_t K = Impl::K;

    using AWarpDstrEncoding = StaticTileDistributionEncoding<
        Sequence<>,
        Tuple<Sequence<Impl::AMLane>, Sequence<Impl::ABKLane, Impl::ABKPerLane>>,
        Tuple<Sequence<2, 1>>,
        Tuple<Sequence<0, 0>>,
        Sequence<2>,
        Sequence<1>>;

    using BWarpDstrEncoding = StaticTileDistributionEncoding<
        Sequence<>,
        Tuple<Sequence<Impl::BNLane>, Sequence<Impl::ABKLane, Impl::ABKPerLane>>,
        Tuple<Sequence<2, 1>>,
        Tuple<Sequence<0, 0>>,
        Sequence<2>,
        Sequence<1>>;

    using CWarpDstrEncoding = StaticTileDistributionEncoding<
        Sequence<>,
        Tuple<Sequence<Impl::CM0PerLane, Impl::CMLane, Impl::CM1PerLane>, Sequence<Impl::CNLane>>,
        Tuple<Sequence<1, 2>>,
        Tuple<Sequence<1, 0>>,
        Sequence<1, 1>,
        Sequence<0, 2>>;

    // c_vec += a_vec * b_vec
    __device__ void operator()(CVecType& c_vec, const AVecType& a_vec, const BVecType& b_vec) const
    {
        Impl{}(c_vec, a_vec, b_vec);
    }

    // c_vec = a_vec * b_vec
    __device__ CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
        return Impl{}(a_vec, b_vec);
    }
};

template <typename WarpGemmAttributeMfmaImpl_>
struct WarpGemmAtrributeMfmaTransposedCDistribution
{
    using Impl = remove_cvref_t<WarpGemmAttributeMfmaImpl_>;

    using ADataType = typename Impl::BDataType;
    using BDataType = typename Impl::ADataType;
    using CDataType = typename Impl::CDataType;

    using AVecType = typename Impl::BVecType;
    using BVecType = typename Impl::AVecType;
    using CVecType = typename Impl::CVecType;

    static constexpr index_t M = Impl::N;
    static constexpr index_t N = Impl::M;
    static constexpr index_t K = Impl::K;

    using AWarpDstrEncoding = StaticTileDistributionEncoding<
        Sequence<>,
        Tuple<Sequence<Impl::BNLane>, Sequence<Impl::ABKLane, Impl::ABKPerLane>>,
        Tuple<Sequence<2, 1>>,
        Tuple<Sequence<0, 0>>,
        Sequence<2>,
        Sequence<1>>;

    using BWarpDstrEncoding = StaticTileDistributionEncoding<
        Sequence<>,
        Tuple<Sequence<Impl::AMLane>, Sequence<Impl::ABKLane, Impl::ABKPerLane>>,
        Tuple<Sequence<2, 1>>,
        Tuple<Sequence<0, 0>>,
        Sequence<2>,
        Sequence<1>>;

    using CWarpDstrEncoding = StaticTileDistributionEncoding<
        Sequence<>,
        Tuple<Sequence<Impl::CNLane>, Sequence<Impl::CM0PerLane, Impl::CMLane, Impl::CM1PerLane>>,
        Tuple<Sequence<2, 1>>,
        Tuple<Sequence<1, 0>>,
        Sequence<2, 2>,
        Sequence<0, 2>>;

    // c_vec += a_vec * b_vec
    __device__ void operator()(CVecType& c_vec, const AVecType& a_vec, const BVecType& b_vec) const
    {
        Impl{}(c_vec, b_vec, a_vec);
    }

    // c_vec = a_vec * b_vec
    __device__ CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
        return Impl{}(b_vec, a_vec);
    }
};

} // namespace warp
} // namespace tile_program
} // namespace ck
