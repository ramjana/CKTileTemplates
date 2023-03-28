// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/thread_private_buffer.hpp"
#include "ck/utility/buffer_view.hpp"

namespace ck {

template <typename DataType, typename TensorDescriptor>
struct ThreadPrivateTensor
{
    using DataType_         = DataType;
    using T                 = DataType_;
    using TensorDescriptor_ = remove_cvref_t<TensorDescriptor>;

    __host__ __device__ constexpr ThreadPrivateTensor() = default;

    __host__ __device__ constexpr ThreadPrivateTensor(const TensorDescriptor_& desc)
        : buf_view_{buf_, desc_.GetElementSpaceSize()}, desc_{desc}
    {
    }

    // member
    TensorDescriptor_ desc_;

    ThreadPrivateBuffer<DataType_> buf_;

    BufferView<AddressSpaceEnum::Vgpr, DataType_, index_t, true> buf_view_;

    // function
#if 0
    template <typename Idx>
    __device__ constexpr const T& operator[](const Idx& idx) const
    {
        const auto coord = make_tensor_coordinate(desc_, idx);

        const index_t offset = coord.GetOffset();

        const bool is_valid = coordinate_has_valid_offset(desc_, coord);

        return buf_view_.template Get<T>(i, is_valid_element);
    }

    template <typename Idx>
    __device__ constexpr T& operator()(const Idx& idx)
    {
        const auto coord = make_tensor_coordinate(desc_, idx);

        const index_t offset = coord.GetOffset();

        const bool is_valid = coordinate_has_valid_offset(desc_, coord);

        return buf_view_.template Get<T>(i, is_valid_element);
    }
#endif

    // idx is the index of T, not X. idx should be aligned to X
    template <typename X, typename Idx>
    __device__ constexpr X Get(const Idx& idx, bool is_valid_element) const
    {
        const auto coord = make_tensor_coordinate(desc_, idx);

        const index_t offset = coord.GetOffset();

        const bool is_valid = coordinate_has_valid_offset(desc_, coord);

        return buf_view_.template Get<X>(offset, is_valid_element);
    }

    // idx is the index of T, not X. idx should be aligned to X
    template <typename X, typename Idx>
    __device__ void Set(const Idx& idx, bool is_valid_element, const X& x)
    {
        const auto coord = make_tensor_coordinate(desc_, idx);

        const index_t offset = coord.GetOffset();

        const bool is_valid = coordinate_has_valid_offset(desc_, coord);

        buf_view_.template Set<X>(offset, is_valid_element, x);
    }
};

} // namespace ck
