// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <functional>
#include <optional>
#include <type_traits>
#include <utility>

namespace ck {

template <typename FirstPredicate,
          typename FirstTypeCarrier,
          typename SecondPredicate,
          typename SecondTypeCarrier,
          typename... Args,
          typename TypeReceiver>
__host__ auto type_select(FirstPredicate first_predicate,
                          FirstTypeCarrier first_type_carrier,
                          SecondPredicate second_predicate,
                          SecondTypeCarrier second_type_carrier,
                          Args... args,
                          TypeReceiver&& type_receiver,
                          std::optional<std::function<void()>> error_handler = std::nullopt)
    -> std::enable_if_t<sizeof...(Args) % 2 == 0 && std::is_invocable_r_v<bool, FirstPredicate> &&
                            std::is_invocable_v<TypeReceiver&&, FirstTypeCarrier>,
                        bool>
{
    if(first_predicate())
    {
        type_receiver(first_type_carrier);
        return true;
    }

    return type_select(second_predicate,
                       second_type_carrier,
                       args...,
                       std::forward<TypeReceiver>(type_receiver),
                       std::move(error_handler));
}

template <typename Predicate, typename TypeCarrier, typename TypeReceiver>
__host__ auto type_select(Predicate predicate,
                          TypeCarrier type_carrier,
                          TypeReceiver&& type_receiver,
                          std::optional<std::function<void()>> error_handler = std::nullopt)
    -> std::enable_if_t<std::is_invocable_r_v<bool, Predicate> &&
                            std::is_invocable_v<TypeReceiver&&, TypeCarrier>,
                        bool>
{
    if(predicate())
    {
        type_receiver(type_carrier);
        return true;
    }

    if(error_handler.has_value())
    {
        std::invoke(*error_handler);
    }

    return false;
}

} // namespace ck
