// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"

namespace ck {
namespace block {

template <typename Distribution, typename T, index_t MaxThreadTensorElementSize>
struct BlockDistributedTensor
{
    __device__ BlockDistributedTensor(const Distribution& distribution)
        : distribution_{distribution},
          thread_tensor_desc_{xxx},
          thread_tensor_{data_, thread_tensor_desc_}
    {
    }

    __device__ constexpr auto GetDistribution() const { return distribution_; }

    //
    // e.g. [wid, lid, m3, k2, k3] to [m, k]
    BlockTensorDistribution<XXX> distribution_;

    // thread Tensor
    // e.g. [m3, k2, k3]
    using ThreadTensorDesc = TensorDescriptor<xxx>;
    TensorTensorDesc thread_tensor_desc_;

    TensorView<AddressSpaceEnum::Vgpr, true, T, ThreadTensorDesc> thread_tensor_;

    //
    T data_[MaxThreadTensorElementSize];
};

template <index_t NDim, typename XXX>
__device__ constexpr auto make_distributed_tensor(const MultiIndex<NDim>& lengths,
                                                  const BlockTensorDistribution<XXX>& distribution)
{
    return BlockDistributedTensor<Distribution, T, kThreadTensorElementSize>{distribution};
}

} // namespace block
} // namespace ck
