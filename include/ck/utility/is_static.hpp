// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "ck/utility/integral_constant.hpp"
#include "ck/utility/sequence.hpp"
#include "ck/utility/tuple.hpp"

namespace ck {

template <typename T>
struct IsStatic
{
    static constexpr bool value = false;
};

//
template <typename T, T X>
struct IsStatic<integral_constant<T, X>>
{
    static constexpr bool value = true;
};

template <typename T, T X>
struct IsStatic<const integral_constant<T, X>>
{
    static constexpr bool value = true;
};

template <typename T, T X>
struct IsStatic<integral_constant<T, X>&>
{
    static constexpr bool value = true;
};

template <typename T, T X>
struct IsStatic<const integral_constant<T, X>&>
{
    static constexpr bool value = true;
};

template <index_t... Is>
struct IsStatic<Sequence<Is...>>
{
    static constexpr bool value = true;
};

template <index_t... Is>
struct IsStatic<const Sequence<Is...>>
{
    static constexpr bool value = true;
};

template <index_t... Is>
struct IsStatic<Sequence<Is...>&>
{
    static constexpr bool value = true;
};

template <index_t... Is>
struct IsStatic<const Sequence<Is...>&>
{
    static constexpr bool value = true;
};

template <typename... Ts>
struct IsStatic<Tuple<Ts...>>
{
    __host__ __device__ static constexpr bool Impl()
    {
        bool flag = true;

        static_for<0, sizeof...(Ts), 1>{}([&flag](auto i) {
            flag &= IsStatic<remove_cvref_t<type_pack_element<i.value, Ts...>>>::value;
        });

        return flag;
    }

    static constexpr bool value = Impl();
};

template <typename... Ts>
struct IsStatic<const Tuple<Ts...>>
{
    static constexpr bool value = IsStatic<Tuple<Ts...>>::value;
};

template <typename... Ts>
struct IsStatic<Tuple<Ts...>&>
{
    static constexpr bool value = IsStatic<Tuple<Ts...>>::value;
};

template <typename... Ts>
struct IsStatic<const Tuple<Ts...>&>
{
    static constexpr bool value = IsStatic<Tuple<Ts...>>::value;
};

template <typename T>
__host__ __device__ constexpr bool is_static(T)
{
    return IsStatic<T>::value;
}

// TODO: deprecate this
template <typename T>
using is_known_at_compile_time = IsStatic<T>;

} // namespace ck
