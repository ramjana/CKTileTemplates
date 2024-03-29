// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "common_header.hpp"

namespace ck {

template <index_t N>
using MultiIndex = Array<index_t, N>;

template <typename... Xs>
__host__ __device__ constexpr auto make_multi_index(Xs&&... xs)
{
    return make_array<index_t>(index_t{xs}...);
}

template <index_t NSize>
__host__ __device__ constexpr auto make_zero_multi_index()
{
    return unpack([](auto... xs) { return make_multi_index(xs...); },
                  typename uniform_sequence_gen<NSize, 0>::type{});
}

template <typename T>
__host__ __device__ constexpr auto to_multi_index(const T& x)
{
    return unpack([](auto... ys) { return make_multi_index(ys...); }, x);
}

template <index_t NSize, typename X>
__host__ __device__ constexpr auto operator+=(MultiIndex<NSize>& y, const X& x)
{
    static_assert(X::Size() == NSize, "wrong! size not the same");
    static_for<0, NSize, 1>{}([&](auto i) { y(i) += x[i]; });
    return y;
}

template <index_t NSize, typename X>
__host__ __device__ constexpr auto operator-=(MultiIndex<NSize>& y, const X& x)
{
    static_assert(X::Size() == NSize, "wrong! size not the same");
    static_for<0, NSize, 1>{}([&](auto i) { y(i) -= x[i]; });
    return y;
}

template <index_t NSize, typename T>
__host__ __device__ constexpr auto operator+(const MultiIndex<NSize>& a, const T& b)
{
    using type = MultiIndex<NSize>;
    static_assert(T::Size() == NSize, "wrong! size not the same");
    type r;
    static_for<0, NSize, 1>{}([&](auto i) { r(i) = a[i] + b[i]; });
    return r;
}

template <index_t NSize, typename T>
__host__ __device__ constexpr auto operator-(const MultiIndex<NSize>& a, const T& b)
{
    using type = MultiIndex<NSize>;
    static_assert(T::Size() == NSize, "wrong! size not the same");
    type r;
    static_for<0, NSize, 1>{}([&](auto i) { r(i) = a[i] - b[i]; });
    return r;
}

template <index_t NSize, typename T>
__host__ __device__ constexpr auto operator*(const MultiIndex<NSize>& a, const T& b)
{
    using type = MultiIndex<NSize>;
    static_assert(T::Size() == NSize, "wrong! size not the same");
    type r;
    static_for<0, NSize, 1>{}([&](auto i) { r(i) = a[i] * b[i]; });
    return r;
}

// MultiIndex = index_t * MultiIndex
template <index_t NSize>
__host__ __device__ constexpr auto operator*(index_t a, const MultiIndex<NSize>& x)
{
    MultiIndex<NSize> r;
    static_for<0, NSize, 1>{}([&](auto i) { r(i) = a * x[i]; });
    return r;
}

// MultiIndex = MultiIndex * index_t
template <index_t NSize>
__host__ __device__ constexpr auto operator*(const MultiIndex<NSize>& x, index_t a)
{
    return a * x;
}

} // namespace ck
