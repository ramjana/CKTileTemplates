// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "buffer_view.hpp"

// FIXME: deprecate DynamicBuffer, use BufferView instead

namespace ck {

// FIXME: deprecate DynamicBuffer, use BufferView instead
template <AddressSpaceEnum BufferAddressSpace,
          typename T,
          typename ElementSpaceSize,
          bool InvalidElementUseNumericalZeroValue>
using DynamicBuffer =
    BufferView<BufferAddressSpace, T, ElementSpaceSize, InvalidElementUseNumericalZeroValue>;

// FIXME: deprecate make_dynamic_buffer, use make_buffer_view instead
template <AddressSpaceEnum BufferAddressSpace, typename T, typename ElementSpaceSize>
__host__ __device__ constexpr auto make_dynamic_buffer(T* p, ElementSpaceSize element_space_size)
{
    return make_buffer_view<BufferAddressSpace, T, ElementSpaceSize>(p, element_space_size);
}

// FIXME: deprecate make_dynamic_buffer, use make_buffer_view instead
template <
    AddressSpaceEnum BufferAddressSpace,
    typename T,
    typename ElementSpaceSize,
    typename X,
    typename enable_if<is_same<remove_cvref_t<T>, remove_cvref_t<X>>::value, bool>::type = false>
__host__ __device__ constexpr auto
make_dynamic_buffer(T* p, ElementSpaceSize element_space_size, X invalid_element_value)
{
    return make_buffer_view<BufferAddressSpace, T, ElementSpaceSize>(
        p, element_space_size, invalid_element_value);
}

} // namespace ck
