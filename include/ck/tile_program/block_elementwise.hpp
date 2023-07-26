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
template <typename InOutElementOp, typename... InOutBlockTensors>
__host__ __device__ void block_elementwise_inout(const InOutElementOp& inout_element_op,
                                                 InOutBlockTensors&... inout_block_tensors)
{
    // TODO: make sure all tensors have same lengths and distribution
    // static_assert(xxx);

    constexpr index_t thread_buffer_size =
        type_pack_element<0, InOutBlockTensors...>::GetThreadBufferSize();

    static_for<0, thread_buffer_size, 1>{}(
        [&](auto i) { inout_element_op(inout_block_tensors.GetThreadBuffer()(i)...); });
}

template <typename InElementOp, typename... InBlockTensors>
__host__ __device__ auto block_elementwise_in(const InElementOp& in_element_op,
                                              const InBlockTensors&... in_block_tensors)
{
    using OutDataType = decltype(in_element_op(typename InBlockTensors::DataType{}...));

    // TODO: make sure all tensors have same lengths and distribution
    // static_assert(xxx);
    constexpr auto in_block_distr = type_pack_element<0, InBlockTensors...>::GetBlockDistribution();

    constexpr index_t thread_buffer_size =
        type_pack_element<0, InBlockTensors...>::GetThreadBufferSize();

    auto out_block_tensor = make_static_block_distributed_tensor<OutDataType>(in_block_distr);

    static_for<0, thread_buffer_size, 1>{}([&](auto i) {
        out_block_tensor.GetThreadBuffer()(i) =
            in_element_op(in_block_tensors.GetThreadBuffer()[i]...);
    });

    return out_block_tensor;
}

} // namespace block
} // namespace tile_program
} // namespace ck
