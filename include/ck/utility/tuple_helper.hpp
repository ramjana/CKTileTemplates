// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "functional4.hpp"
#include "tuple.hpp"

namespace ck {

template <typename F, index_t N>
__host__ __device__ constexpr auto generate_tuple(F&& f, Number<N>)
{
    return unpack([&f](auto&&... is) { return make_tuple(f(is)...); },
                  typename arithmetic_sequence_gen<0, N, 1>::type{});
}

template <typename F, index_t N>
__host__ __device__ constexpr auto generate_tie(F&& f, Number<N>)
{
    return unpack([&f](auto&&... is) { return tie(f(is)...); },
                  typename arithmetic_sequence_gen<0, N, 1>::type{});
}

// tx and ty are tuple of references, return type of will tuple of referennce (not rvalue)
template <typename... X, typename... Y>
__host__ __device__ constexpr auto concat_tuple_of_reference(const Tuple<X&...>& tx,
                                                             const Tuple<Y&...>& ty)
{
    return unpack2(
        [&](auto&&... zs) { return Tuple<decltype(zs)...>{std::forward<decltype(zs)>(zs)...}; },
        tx,
        ty);
}

namespace detail {

template <typename F, typename X, index_t... Is>
__host__ __device__ constexpr auto transform_tuples_impl(F f, const X& x, Sequence<Is...>)
{
    return make_tuple(f(x.At(Number<Is>{}))...);
}

template <typename F, typename X, typename Y, index_t... Is>
__host__ __device__ constexpr auto
transform_tuples_impl(F f, const X& x, const Y& y, Sequence<Is...>)
{
    return make_tuple(f(x.At(Number<Is>{}), y.At(Number<Is>{}))...);
}

template <typename F, typename X, typename Y, typename Z, index_t... Is>
__host__ __device__ constexpr auto
transform_tuples_impl(F f, const X& x, const Y& y, const Z& z, Sequence<Is...>)
{
    return make_tuple(f(x.At(Number<Is>{}), y.At(Number<Is>{}), z.At(Number<Is>{}))...);
}

} // namespace detail

template <typename F, typename X>
__host__ __device__ constexpr auto transform_tuples(F f, const X& x)
{
    return detail::transform_tuples_impl(
        f, x, typename arithmetic_sequence_gen<0, X::Size(), 1>::type{});
}

template <typename F, typename X, typename Y>
__host__ __device__ constexpr auto transform_tuples(F f, const X& x, const Y& y)
{
    return detail::transform_tuples_impl(
        f, x, y, typename arithmetic_sequence_gen<0, X::Size(), 1>::type{});
}

template <typename F, typename X, typename Y, typename Z>
__host__ __device__ constexpr auto transform_tuples(F f, const X& x, const Y& y, const Z& z)
{
    return detail::transform_tuples_impl(
        f, x, y, z, typename arithmetic_sequence_gen<0, X::Size(), 1>::type{});
}

namespace detail {
template <index_t J, typename F, typename... InnerType>
constexpr auto zip_tuples_impl_(F f, const InnerType&... tj)
{
    return f(tj.At(Number<J>{})...);
}

template <typename F, index_t... Is, index_t... Js, typename TupleType>
constexpr auto zip_tuples_impl(F f, const TupleType& t, Sequence<Is...>, Sequence<Js...>)
{
    return make_tuple(zip_tuples_impl_<Js>(f, t.At(Number<Is>{})...)...);
}
} // namespace detail
// zip function
//    ((0, 1, 2), (3, 4, 5)...)
// -> ((0, 3...), (1, 4...), (2, 5...))
//  the input is tuple of InnerType, which need to implement At(Number<..>), Size()
//  the output is tuple of InnerType, which is determined by F lambda/function
//  ... hence input/output InnerType can be different
//
template <typename F, typename TupleType>
constexpr auto zip_tuples(F f, const TupleType& t)
{
    // every XInnerType must have same size
    using XInnerType = remove_cvref_t<decltype(t[Number<0>{}])>;
    return detail::zip_tuples_impl(
        f,
        t,
        typename arithmetic_sequence_gen<0, TupleType::Size(), 1>::type{},
        typename arithmetic_sequence_gen<0, XInnerType::Size(), 1>::type{});
}

} // namespace ck

// Macro function
// convert constexpr Array to Tuple of Number
#define TO_TUPLE_OF_NUMBER(arr, n)                                                              \
    [&arr, &n] {                                                                                \
        static_assert(arr.Size() >= n, "wrong! out of bound");                                  \
                                                                                                \
        static_assert(n < 7, "not implemented");                                                \
                                                                                                \
        if constexpr(n == 0)                                                                    \
        {                                                                                       \
            return ck::Tuple<>{};                                                               \
        }                                                                                       \
        else if constexpr(n == 1)                                                               \
        {                                                                                       \
            return ck::Tuple<Number<arr[0]>>{};                                                 \
        }                                                                                       \
        else if constexpr(n == 2)                                                               \
        {                                                                                       \
            return ck::Tuple<Number<arr[0]>, Number<arr[1]>>{};                                 \
        }                                                                                       \
        else if constexpr(n == 3)                                                               \
        {                                                                                       \
            return ck::Tuple<Number<arr[0]>, Number<arr[1]>, Number<arr[2]>>{};                 \
        }                                                                                       \
        else if constexpr(n == 4)                                                               \
        {                                                                                       \
            return ck::Tuple<Number<arr[0]>, Number<arr[1]>, Number<arr[2]>, Number<arr[3]>>{}; \
        }                                                                                       \
        else if constexpr(n == 5)                                                               \
        {                                                                                       \
            return ck::Tuple<Number<arr[0]>,                                                    \
                             Number<arr[1]>,                                                    \
                             Number<arr[2]>,                                                    \
                             Number<arr[3]>,                                                    \
                             Number<arr[4]>>{};                                                 \
        }                                                                                       \
        else if constexpr(n == 6)                                                               \
        {                                                                                       \
            return ck::Tuple<Number<arr[0]>,                                                    \
                             Number<arr[1]>,                                                    \
                             Number<arr[2]>,                                                    \
                             Number<arr[3]>,                                                    \
                             Number<arr[4]>,                                                    \
                             Number<arr[5]>>{};                                                 \
        }                                                                                       \
    }()
