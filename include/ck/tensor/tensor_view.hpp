// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace ck {

template <typename Buffer_, typename TensorDesc_>
struct TensorView
{
    using Buffer     = remove_reference_t<Buffer_>;
    using DataType   = typename Buffer::type;
    using TensorDesc = remove_cvref_t<TensorDesc_>;

    __host__ __device__ constexpr TensorView() = delete;

    __host__ __device__ constexpr TensorView(Buffer& buffer, TensorDesc desc)
        : buf_{buffer}, desc_{desc}
    {
    }

    __host__ __device__ constexpr auto& GetTensorDescriptor() const { return desc_; }

    __host__ __device__ static constexpr index_t GetNumOfDimension()
    {
        return TensorDesc::GetNumOfTopDimension();
    }

    // member
    Buffer& buf_;

    TensorDesc desc_;
};

template <typename Buffer, typename TensorDesc>
__host__ __device__ constexpr auto make_tensor_view(Buffer& buffer, const TensorDesc& desc)
{
    return TensorView<Buffer, remove_cvref_t<TensorDesc>>{buffer, desc};
}

template <typename Buffer,
          typename... Lengths,
          typename... Strides,
          typename enable_if<sizeof...(Lengths) == sizeof...(Strides), bool>::type = false>
__host__ __device__ constexpr auto make_naive_tensor_view(Buffer& buffer,
                                                          const Tuple<Lengths...>& lengths,
                                                          const Tuple<Strides...>& strides)
{
    auto desc = make_naive_tensor_descriptor(lengths, strides);

    return TensorView<Buffer, decltype(desc)>{buffer, desc};
}

template <typename Buffer, typename... Lengths>
__host__ __device__ constexpr auto make_naive_tensor_view_packed(Buffer& buffer,
                                                                 const Tuple<Lengths...>& lengths)
{
    auto desc = make_naive_tensor_descriptor_packed(lengths);

    return TensorView<Buffer, decltype(desc)>{buffer, desc};
}

template <typename OldTensorView,
          typename NewTransforms,
          typename NewLowerDimensionOldVisibleIdss,
          typename NewUpperDimensionNewVisibleIdss>
__host__ __device__ constexpr auto transform_tensor_view(const OldTensorView& old_tensor_view,
                                                         const NewTransforms& new_transforms,
                                                         NewLowerDimensionOldVisibleIdss,
                                                         NewUpperDimensionNewVisibleIdss)
{
    const auto new_desc = transform_tensor_descriptor(old_tensor_view.desc_,
                                                      new_transforms,
                                                      NewLowerDimensionOldVisibleIdss{},
                                                      NewUpperDimensionNewVisibleIdss{});

    return TensorView<typename OldTensorView::Buffer, remove_cvref_t<decltype(new_desc)>>{
        old_tensor_view.buf_, new_desc};
}

} // namespace ck
