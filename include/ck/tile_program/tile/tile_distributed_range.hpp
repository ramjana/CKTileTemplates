// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"

#include "ck/tile_program/tile/tile_distribution.hpp"

namespace ck {
namespace tile_program {

template <index_t... PartialHsLengths>
struct TileDistributedRange
{
    using Impl = Sequence<PartialHsLengths...>;

    static constexpr auto value = Impl{};
};

template <index_t... PartialHsIndices>
struct TileDistributedPosition
{
    using Impl = Sequence<PartialHsIndices...>;

    static constexpr auto value = Impl{};
};

template <typename PsYs2XsAdaptor_,
          typename Ys2DDescriptor_,
          typename StaticTileDistributionEncoding_>
__host__ __device__ constexpr auto get_tile_distributed_ranges(
    TileDistribution<PsYs2XsAdaptor_, Ys2DDescriptor_, StaticTileDistributionEncoding_>)
{
    using DstrEncodeDetail = remove_cvref_t<typename StaticTileDistributionEncoding_::Detail>;

    constexpr auto distributed_ranges_impl = DstrEncodeDetail::distributed_ranges_lengthss_;
    constexpr index_t ndim_range           = DstrEncodeDetail::ndim_range_major_;
    constexpr auto ndims_ranges_minor      = DstrEncodeDetail::ndims_distributed_ranges_minor_;

    constexpr auto distributed_ranges =
        TO_TUPLE_OF_SEQUENCE(distributed_ranges_impl, ndim_range, ndims_ranges_minor);

    return distributed_ranges;
}

namespace detail {

// FIXME: it's hacky to get Ys index from Range index
// return is Sequence<...>
template <typename TileDistribution_, typename TileDistributedPositions_>
__host__ __device__ constexpr auto
    get_ys_index_from_distributed_positions(TileDistribution_, TileDistributedPositions_)
{
    using Dstr             = remove_cvref_t<TileDistribution_>;
    using Positions        = remove_cvref_t<TileDistributedPositions_>;
    using DstrEncodeDetail = remove_cvref_t<typename Dstr::DstrEncode::Detail>;

    static_assert(is_static_v<Dstr>, "wrong!");
    static_assert(is_static_v<Positions>, "wrong!");

    constexpr index_t ndim_y = Dstr::NDimY;

    constexpr auto y_idx_arr = [&] {
        Array<index_t, ndim_y> y_idx;

        static_for<0, ndim_y, 1>{}([&](auto i) {
            constexpr index_t range_major = DstrEncodeDetail::ys_to_range_major_[i];
            constexpr index_t range_minor = DstrEncodeDetail::ys_to_range_minor_[i];

            y_idx(i) = Positions{}[Number<range_major>{}][Number<range_minor>{}];
        });

        return y_idx;
    }();

    return TO_SEQUENCE(y_idx_arr, ndim_y);
}

} // namespace detail

#if 0
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
#endif

} // namespace tile_program
} // namespace ck
