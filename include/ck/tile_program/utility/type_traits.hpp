// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <type_traits>

#define PP_DEFINE_INDIRECT_MEMBER_TYPE_GETTER(traits, path, member) \
template <typename T, typename = void> \
struct traits##_impl { \
  using type = typename traits##_impl<typename T::path>::type; \
}; \
template <typename T> \
struct traits##_impl<T, std::void_t<typename T::member>> \
{ \
  using type = typename T::member; \
}; \
template <typename T> \
using traits = typename traits##_impl<T>::type

#define PP_DEFINE_INDIRECT_MEMBER_GETTER(traits, path, member) \
template <typename T, typename = void> \
struct traits##_impl { \
  static constexpr auto value = traits##_impl<typename T::path>::value; \
}; \
template <typename T> \
struct traits##_impl<T, std::void_t<decltype(T::member)>> \
{ \
  static constexpr auto value = T::member; \
}; \
template <typename T> \
static constexpr auto traits = traits##_impl<T>::value
