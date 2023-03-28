// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "buffer_view.hpp"

// FIXME: deprecate BynamicBuffer

namespace ck {

// FIXME: remove
template <AddressSpaceEnum BufferAddressSpace,
          typename T,
          typename ElementSpaceSize,
          bool InvalidElementUseNumericalZeroValue>
using DynamicBuffer =
    BufferView<BufferAddressSpace, T, ElementSpaceSize, InvalidElementUseNumericalZeroValue>;

// FIXME: remove
template <AddressSpaceEnum BufferAddressSpace, typename T, typename ElementSpaceSize>
__host__ __device__ constexpr auto make_dynamic_buffer(T* p, ElementSpaceSize element_space_size)
{
    return DynamicBuffer<BufferAddressSpace, T, ElementSpaceSize, true>{p, element_space_size};
}

// FIXME: remove
template <
    AddressSpaceEnum BufferAddressSpace,
    typename T,
    typename ElementSpaceSize,
    typename X,
    typename enable_if<is_same<remove_cvref_t<T>, remove_cvref_t<X>>::value, bool>::type = false>
__host__ __device__ constexpr auto
make_dynamic_buffer(T* p, ElementSpaceSize element_space_size, X invalid_element_value)
{
    return DynamicBuffer<BufferAddressSpace, T, ElementSpaceSize, false>{
        p, element_space_size, invalid_element_value};
}

} // namespace ck
