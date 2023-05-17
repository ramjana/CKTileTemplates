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

template <typename T_, typename StaticBlockDistribution_>
struct StaticBlockDistributedTensor
{
    using T                       = remove_cvref_t<T_>;
    using StaticBlockDistribution = remove_cvref_t<StaticBlockDistribution_>;

    static_assert(StaticBlockDistribution::IsKnownAtCompileTime(),
                  "wrong! StaticBlockDistribution should be known at compile tile");

    using ThreadTensorDesc =
        remove_cvref_t<decltype(StaticBlockDistribution{}.GetYs2DidDescriptor())>;

    static constexpr index_t kThreadElementSpaceSize = ThreadTensorDesc{}.GetElementSpaceSize();

    __host__ __device__ static constexpr auto GetBlockDistribution()
    {
        return StaticBlockDistribution{};
    }

    __host__ __device__ void Initialize(const T& x) { thread_buf_.Initialize(x); }

    __host__ __device__ constexpr const auto& GetThreadBuffer() const { return thread_buf_; }

    __host__ __device__ constexpr auto& GetThreadBuffer() { return thread_buf_; }

    //
    StaticBuffer<AddressSpaceEnum::Vgpr, T, kThreadElementSpaceSize, true> thread_buf_;
};

template <typename T, typename StaticBlockDistribution>
__host__ __device__ constexpr auto
make_static_block_distributed_tensor(const StaticBlockDistribution&)
{
    return StaticBlockDistributedTensor<remove_cvref_t<T>,
                                        remove_cvref_t<StaticBlockDistribution>>{};
}

} // namespace block
} // namespace tile_program
} // namespace ck
