// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "ck/utility/integral_constant.hpp"
#include "ck/utility/sequence.hpp"
#include "ck/utility/tuple.hpp"

namespace ck {

template <typename T>
struct is_static
{
    static constexpr bool value = false;
};

//
template <typename T, T X>
struct is_static<integral_constant<T, X>>
{
    static constexpr bool value = true;
};

template <typename T, T X>
struct is_static<const integral_constant<T, X>>
{
    static constexpr bool value = true;
};

template <typename T, T X>
struct is_static<integral_constant<T, X>&>
{
    static constexpr bool value = true;
};

template <typename T, T X>
struct is_static<const integral_constant<T, X>&>
{
    static constexpr bool value = true;
};

template <index_t... Is>
struct is_static<Sequence<Is...>>
{
    static constexpr bool value = true;
};

template <index_t... Is>
struct is_static<const Sequence<Is...>>
{
    static constexpr bool value = true;
};

template <index_t... Is>
struct is_static<Sequence<Is...>&>
{
    static constexpr bool value = true;
};

template <index_t... Is>
struct is_static<const Sequence<Is...>&>
{
    static constexpr bool value = true;
};

template <typename... Ts>
struct is_static<Tuple<Ts...>>
{
    __host__ __device__ static constexpr bool Impl()
    {
        bool flag = true;

        static_for<0, sizeof...(Ts), 1>{}([&flag](auto i) {
            flag &= is_static<remove_cvref_t<type_pack_element<i.value, Ts...>>>::value;
        });

        return flag;
    }

    static constexpr bool value = Impl();
};

template <typename... Ts>
struct is_static<const Tuple<Ts...>>
{
    static constexpr bool value = is_static<Tuple<Ts...>>::value;
};

template <typename... Ts>
struct is_static<Tuple<Ts...>&>
{
    static constexpr bool value = is_static<Tuple<Ts...>>::value;
};

template <typename... Ts>
struct is_static<const Tuple<Ts...>&>
{
    static constexpr bool value = is_static<Tuple<Ts...>>::value;
};

template <typename T>
inline constexpr bool is_static_v = is_static<T>::value;

// TODO: deprecate this
template <typename T>
using is_known_at_compile_time = is_static<T>;

} // namespace ck
