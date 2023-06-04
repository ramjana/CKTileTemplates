// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "integral_constant.hpp"
#include "sequence.hpp"
#include "tuple.hpp"

namespace ck {

template <typename T>
struct is_known_at_compile_time
{
    static constexpr bool value = false;
};

template <typename T, T X>
struct is_known_at_compile_time<integral_constant<T, X>>
{
    static constexpr bool value = true;
};

template <typename T, T X>
struct is_known_at_compile_time<const integral_constant<T, X>>
{
    static constexpr bool value = true;
};

template <typename T, T X>
struct is_known_at_compile_time<integral_constant<T, X>&>
{
    static constexpr bool value = true;
};

template <typename T, T X>
struct is_known_at_compile_time<const integral_constant<T, X>&>
{
    static constexpr bool value = true;
};

template <index_t... Is>
struct is_known_at_compile_time<Sequence<Is...>>
{
    static constexpr bool value = true;
};

template <index_t... Is>
struct is_known_at_compile_time<const Sequence<Is...>>
{
    static constexpr bool value = true;
};

template <index_t... Is>
struct is_known_at_compile_time<Sequence<Is...>&>
{
    static constexpr bool value = true;
};

template <index_t... Is>
struct is_known_at_compile_time<const Sequence<Is...>&>
{
    static constexpr bool value = true;
};

template <typename... Ts>
struct is_known_at_compile_time<Tuple<Ts...>>
{
    __host__ __device__ static constexpr bool IsKnownAtCompileTime()
    {
        bool flag = true;

        static_for<0, sizeof...(Ts), 1>{}([&flag](auto i) {
            flag &=
                is_known_at_compile_time<remove_cvref_t<type_pack_element<i.value, Ts...>>>::value;
        });

        return flag;
    }

    static constexpr bool value = IsKnownAtCompileTime();
};

template <typename... Ts>
struct is_known_at_compile_time<const Tuple<Ts...>>
{
    static constexpr bool value = is_known_at_compile_time<Tuple<Ts...>>::value;
};

template <typename... Ts>
struct is_known_at_compile_time<Tuple<Ts...>&>
{
    static constexpr bool value = is_known_at_compile_time<Tuple<Ts...>>::value;
};

template <typename... Ts>
struct is_known_at_compile_time<const Tuple<Ts...>&>
{
    static constexpr bool value = is_known_at_compile_time<Tuple<Ts...>>::value;
};

} // namespace ck
