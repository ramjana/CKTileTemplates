// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/tuple.hpp"

namespace ck {

template <index_t... Is>
__host__ __device__ constexpr auto make_sequence(Number<Is>...)
{
    return Sequence<Is...>{};
}

// F() returns index_t
// F use default constructor
template <typename F, index_t N>
__host__ __device__ constexpr auto generate_sequence(F, Number<N>)
{
    return typename sequence_gen<N, F>::type{};
}

// F() returns Number<>
template <typename F, index_t N>
__host__ __device__ constexpr auto generate_sequence_v2(F&& f, Number<N>)
{
    return unpack([&f](auto&&... xs) { return make_sequence(f(xs)...); },
                  typename arithmetic_sequence_gen<0, N, 1>::type{});
}

template <index_t... Is>
__host__ __device__ constexpr auto to_sequence(Tuple<Number<Is>...>)
{
    return Sequence<Is...>{};
}

} // namespace ck

// Macro function
// convert constexpr Array to Sequence
#define TO_SEQUENCE(a, n)                                                                      \
    [&a, &n] {                                                                                 \
        static_assert(a.Size() >= n, "wrong! out of bound");                                   \
                                                                                               \
        static_assert(n <= 10, "not implemented");                                             \
                                                                                               \
        if constexpr(n == 0)                                                                   \
        {                                                                                      \
            return ck::Sequence<>{};                                                           \
        }                                                                                      \
        else if constexpr(n == 1)                                                              \
        {                                                                                      \
            return ck::Sequence<a[0]>{};                                                       \
        }                                                                                      \
        else if constexpr(n == 2)                                                              \
        {                                                                                      \
            return ck::Sequence<a[0], a[1]>{};                                                 \
        }                                                                                      \
        else if constexpr(n == 3)                                                              \
        {                                                                                      \
            return ck::Sequence<a[0], a[1], a[2]>{};                                           \
        }                                                                                      \
        else if constexpr(n == 4)                                                              \
        {                                                                                      \
            return ck::Sequence<a[0], a[1], a[2], a[3]>{};                                     \
        }                                                                                      \
        else if constexpr(n == 5)                                                              \
        {                                                                                      \
            return ck::Sequence<a[0], a[1], a[2], a[3], a[4]>{};                               \
        }                                                                                      \
        else if constexpr(n == 6)                                                              \
        {                                                                                      \
            return ck::Sequence<a[0], a[1], a[2], a[3], a[4], a[5]>{};                         \
        }                                                                                      \
        else if constexpr(n == 7)                                                              \
        {                                                                                      \
            return ck::Sequence<a[0], a[1], a[2], a[3], a[4], a[5], a[6]>{};                   \
        }                                                                                      \
        else if constexpr(n == 8)                                                              \
        {                                                                                      \
            return ck::Sequence<a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]>{};             \
        }                                                                                      \
        else if constexpr(n == 9)                                                              \
        {                                                                                      \
            return ck::Sequence<a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]>{};       \
        }                                                                                      \
        else if constexpr(n == 10)                                                             \
        {                                                                                      \
            return ck::Sequence<a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9]>{}; \
        }                                                                                      \
    }()
