// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#ifndef CK_NUMBER_HPP
#define CK_NUMBER_HPP

#include "integral_constant.hpp"

namespace ck {

template <index_t N>
using Number = integral_constant<index_t, N>;

template <index_t N>
using LongNumber = integral_constant<long_index_t, N>;

template <index_t... Is>
struct Sequence;

// slice and indexing, similiar to numpy syntax, start:end:step
// TODO: step not supported yet
template <index_t start_, index_t end_, index_t step_ = 1>
struct si
{
    static constexpr index_t start = start_;
    static constexpr index_t end   = end_;
    static constexpr index_t step  = step_; // Not supported yet

    using serialized_type = Sequence<start, end, step>;

    __host__ __device__ constexpr si() {} // all length, like [:]

    // one pixel
    template <index_t x>
    __host__ __device__ constexpr si(Number<x>)
    {
    }

    // a slice [start, end)
    template <index_t x, index_t y>
    __host__ __device__ constexpr si(Number<x>, Number<y>)
    {
    }

    // a slice [start, end), with step
    template <index_t x, index_t y, index_t z>
    __host__ __device__ constexpr si(Number<x>, Number<y>, Number<z>)
    {
    }

    __host__ __device__ static constexpr auto Size() { return 2; } // TODO: no Step

    template <index_t I>
    __host__ __device__ static constexpr auto At(Number<I>)
    {
        return serialized_type::At(Number<I>{});
    }

    template <index_t I>
    __host__ __device__ constexpr auto operator[](Number<I>)
    {
        return At(Number<I>{});
    }
    template <index_t I>
    __host__ __device__ constexpr auto operator[](Number<I>) const
    {
        return At(Number<I>{});
    }
};

// deduction guide
__host__ __device__ si()->si<0, 0, 0>;
template <index_t x>
__host__ __device__ si(Number<x>)->si<x, x + 1, 1>;
template <index_t x, index_t y>
__host__ __device__ si(Number<x>, Number<y>)->si<x, y, 1>;
template <index_t x, index_t y, index_t z>
__host__ __device__ si(Number<x>, Number<y>, Number<z>)->si<x, y, z>;

template <index_t... xs>
constexpr auto make_si(Number<xs>...)
{
    static_assert(sizeof...(xs) <= 3);
    return si{Number<xs>{}...};
}

} // namespace ck
#endif
