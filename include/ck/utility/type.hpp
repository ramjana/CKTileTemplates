// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "ck/utility/integral_constant.hpp"
#include "ck/utility/enable_if.hpp"

namespace ck {

// is_same
static_assert(__has_builtin(__is_same), "");

template <typename X, typename Y>
using is_same = integral_constant<bool, __is_same(X, Y)>;

template <typename X, typename Y>
inline constexpr bool is_same_v = is_same<X, Y>::value;

static_assert(__has_builtin(__type_pack_element), "");

// type_pack_element
template <index_t I, typename... Ts>
using type_pack_element = __type_pack_element<I, Ts...>;

// remove_cvref_t
template <typename T>
using remove_reference_t = typename std::remove_reference<T>::type;

template <typename T>
using remove_cv_t = typename std::remove_cv<T>::type;

template <typename T>
using remove_cvref_t = remove_cv_t<std::remove_reference_t<T>>;

template <typename T>
using remove_pointer_t = typename std::remove_pointer<T>::type;

// is_pointer
template <typename T>
inline constexpr bool is_pointer_v = std::is_pointer<T>::value;

// bit_cast
template <typename Y, typename X, typename enable_if<sizeof(X) == sizeof(Y), bool>::type = false>
__host__ __device__ constexpr Y bit_cast(const X& x)
{
    static_assert(__has_builtin(__builtin_bit_cast), "");

    return __builtin_bit_cast(Y, x);
}

} // namespace ck
