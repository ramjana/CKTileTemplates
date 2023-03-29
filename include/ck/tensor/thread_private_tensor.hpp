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

    __host__ __device__ constexpr ThreadPrivateTensor()
        // FIXME: remove static_cast
        : desc_{}, buf_view_{buf_.p_data_, static_cast<index_t>(desc_.GetElementSpaceSize())}
    {
    }

    __host__ __device__ constexpr ThreadPrivateTensor(const TensorDescriptor_& desc)
        // FIXME: remove static_cast
        : desc_{desc}, buf_view_{buf_.p_data_, static_cast<index_t>(desc_.GetElementSpaceSize())}
    {
    }

    // member
    TensorDescriptor_ desc_;

    ThreadPrivateBuffer<DataType_> buf_;

    // FIXME: remove assumption that type of BufferSize should be index_t
    BufferView<AddressSpaceEnum::Vgpr, DataType_, index_t, true> buf_view_;

    // function
#if 1
    // FIXME: doesn't do is_valid check
    template <typename Idx>
    __device__ constexpr const T& operator[](const Idx& idx) const
    {
        // FIXME: remove to_multi_index
        const auto coord = make_tensor_coordinate(desc_, to_multi_index(idx));

        const index_t offset = coord.GetOffset();

        return buf_view_[offset];
    }

    // FIXME: doesn't do is_valid check
    template <typename Idx>
    __device__ constexpr T& operator()(const Idx& idx)
    {
        // FIXME: remove to_multi_index
        const auto coord = make_tensor_coordinate(desc_, to_multi_index(idx));

        const index_t offset = coord.GetOffset();

        return buf_view_(offset);
    }
#endif

    // idx is the index of T, not X. idx should be aligned to X
    template <typename X, typename Idx>
    __device__ constexpr X Get(const Idx& idx) const
    {
        const auto coord = make_tensor_coordinate(desc_, idx);

        const index_t offset = coord.GetOffset();

        const bool is_valid = coordinate_has_valid_offset(desc_, coord);

        return buf_view_.template Get<X>(offset, is_valid);
    }

    // idx is the index of T, not X. idx should be aligned to X
    template <typename X, typename Idx>
    __device__ void Set(const Idx& idx, const X& x)
    {
        const auto coord = make_tensor_coordinate(desc_, idx);

        const index_t offset = coord.GetOffset();

        const bool is_valid = coordinate_has_valid_offset(desc_, coord);

        buf_view_.template Set<X>(offset, is_valid, x);
    }
};

} // namespace ck
