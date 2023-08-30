// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"

#include "ck/tile_program/tile/tile_distribution_helper.hpp"
#include "ck/tile_program/tile/static_distributed_tensor.hpp"

namespace ck {
namespace tile_program {

typename<index_t... PartialHsLengths> struct TileDistributedRange
{
    using Impl = Sequence<PartialHsLengths...>;

    static constexpr auto value = Impl{};
};

typename<index_t... PartialHsIndices> struct TileDistributedPosition
{
    using Impl = Sequence<PartialHsIndices...>;

    static constexpr auto value = Impl{};
};

template <typename Range, typename F>
__host__ __device__ void distributed_range_over(DistributedRanges, const F& f)
{
    static_assert(is_static_v<DistributedRanges>, "wrong!");

    static_ford<0, DistributedRanges.Size(), 1>{}([&](auto range_major_idx) {
        constexpr index_t ndim_range_minor = DistributedRanges[range_major].Size();

        static_for<0, ndim_range_minor, 1>{}([&](auto range_minor_idx) {
            constexpr index_t range = DistributedRanges[range_major][range_minor];

            constexpr range_idx = range_major_idx

                f(range_idx);
        });
    });
}

} // namespace tile_program
} // namespace ck
