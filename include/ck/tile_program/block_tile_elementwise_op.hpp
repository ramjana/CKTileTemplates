// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"
#include "ck/tile_program/block_tensor_distribution.hpp"

namespace ck {
namespace tile_program {
namespace block {

// TODO: support tensors with different distribution
template <typename ElementOp, typename... BlockTensors>
__host__ __device__ void block_tile_elementwise(const ElementOp& element_op,
                                                BlockTensors&... block_tensors)
{
    // TODO: make sure all tensors have same lengths and distribution
    // static_assert(xxx);

    constexpr index_t thread_buffer_size =
        type_pack_element<0, BlockTensors...>::GetThreadBufferSize();

    static_for<0, thread_buffer_size, 1>{}(
        [&](auto i) { element_op(block_tensors.GetThreadBuffer()(i)...); });
}

} // namespace block
} // namespace tile_program
} // namespace ck
