// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"

namespace ck {
namespace block {

template <xxx....>
struct BlockTensorDistribution
{
    // tensor adaptor from per block tensor to per thread tensor
    // e.g. [wid, lid, m3, k2, k3] to [m, k]
    TensorAdaptor<xxx> adaptor_;
};

} // namespace block
} // namespace ck
