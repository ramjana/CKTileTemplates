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

template <typename DataType_, typename StaticBlockDistribution_>
struct StaticBlockDistributedTensor
{
    using DataType                = remove_cvref_t<DataType_>;
    using StaticBlockDistribution = remove_cvref_t<StaticBlockDistribution_>;

    static_assert(StaticBlockDistribution::IsStatic(),
                  "wrong! StaticBlockDistribution should be known at compile tile");

    using ThreadTensorDesc =
        remove_cvref_t<decltype(StaticBlockDistribution{}.GetYs2DDescriptor())>;

    static constexpr index_t kThreadElementSpaceSize = ThreadTensorDesc{}.GetElementSpaceSize();

    __host__ __device__ static constexpr auto GetBlockDistribution()
    {
        return StaticBlockDistribution{};
    }

    __host__ __device__ void Initialize(const DataType& x) { thread_buf_.Initialize(x); }

    __host__ __device__ constexpr const auto& GetThreadBuffer() const { return thread_buf_; }

    __host__ __device__ constexpr auto& GetThreadBuffer() { return thread_buf_; }

    __host__ __device__ static constexpr index_t GetThreadBufferSize()
    {
        return kThreadElementSpaceSize;
    }

    template <index_t... YSliceOrigins, index_t... YSliceLengths>
    __host__ __device__ auto GetSlicedThreadData(Sequence<YSliceOrigins...>,
                                                 Sequence<YSliceLengths...>) const
    {
        constexpr auto sliced_thread_tensor_desc =
            make_naive_tensor_descriptor_packed(make_tuple(YSliceLengths...));

        StaticBuffer<AddressSpaceEnum::Vgpr,
                     DataType,
                     sliced_thread_tensor_desc.GetElementSpaceSize(),
                     true>
            sliced_thread_data;

        static_ford<Sequence<YSliceLengths...>>{}([&](auto idx) {
            constexpr auto idx_ys = idx + Sequence<YSliceOrigins...>{};

            sliced_thread_data(Number<sliced_thread_tensor_desc.CalculateOffset(idx)>{}) =
                thread_buf_[Number<ThreadTensorDesc{}.CalculateOffset(idx_ys)>{}];
        });

        return sliced_thread_data;
    }

    template <index_t... YSliceOrigins, index_t... YSliceLengths, index_t NSlicedData>
    __host__ __device__ void SetSlicedThreadData(
        Sequence<YSliceOrigins...>,
        Sequence<YSliceLengths...>,
        const StaticBuffer<AddressSpaceEnum::Vgpr, DataType, NSlicedData, true>& sliced_thread_data)
    {
        constexpr auto sliced_thread_tensor_desc =
            make_naive_tensor_descriptor_packed(make_tuple(YSliceLengths...));

        static_ford<Sequence<YSliceLengths...>>{}([&](auto idx) {
            constexpr auto idx_ys = idx + Sequence<YSliceOrigins...>{};

            thread_buf_(Number<ThreadTensorDesc{}.CalculateOffset(idx_ys)>{}) =
                sliced_thread_data[Number<sliced_thread_tensor_desc.CalculateOffset(idx)>{}];
        });
    }

    //
    StaticBuffer<AddressSpaceEnum::Vgpr, DataType, kThreadElementSpaceSize, true> thread_buf_;
};

template <typename DataType, typename StaticBlockDistribution>
__host__ __device__ constexpr auto
make_static_block_distributed_tensor(const StaticBlockDistribution&)
{
    return StaticBlockDistributedTensor<remove_cvref_t<DataType>,
                                        remove_cvref_t<StaticBlockDistribution>>{};
}

} // namespace block
} // namespace tile_program
} // namespace ck
