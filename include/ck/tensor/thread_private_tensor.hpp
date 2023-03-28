// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace ck {

template <typename DataType, typename TensorDescriptor>
struct ThreadPrivateTensor
{
    using DataType_         = DataType;
    using TensorDescriptor_ = remove_cvref_t<TensorDescriptor>;

    __host__ __device__ constexpr ThreadPrivateTensor() = delete;

    __host__ __device__ constexpr ThreadPrivateTensor(const TensorDescriptor_& desc)
        : buf_view_{buf_, desc_.GetElementSpaceSize()}, desc_{desc}
    {
    }

    // member
    TensorDescriptor_ desc_;

    ThreadBuffer<DataType_> buf_;

    BufferView<AddressSpaceEnum::Vgpr, DataType_, index_t, true> buf_view_;
};

} // namespace ck
