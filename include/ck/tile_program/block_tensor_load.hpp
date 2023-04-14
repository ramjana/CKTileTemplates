// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"

namespace ck {
namespace block {

template <typename BlockTensorWindow, typename Strategy>
__device__ auto load(BlockTensorWindow& src_window, const Strategy& strategy)
{
    // make empty distributed tensor
    auto dst_dist_tensor =
        make_distributed_tensor(src_window.GetLengths(), src_window.GetDistribution());

    // copy data from src window into distributed tensor
    // get adaptor top lengths
    const auto =

        // return
        return dst_dist_tensor;
}

// Ideally, Distribtion must be the same between BlockwiseTensorWindow dst and
// BlockDistributedTensor src When they are not the same, the fall back could be:
//    1. blockwiwse shuffle of distributed tensor
//    2. create a new BlockTensorWindow for dst, with the same block distribition as src
template <typename xxx, typename Strategy>
__device__ void
store(BlockTensorWindow<xxx>& dst, const BlockDistributedTensor<xxx>& src, const Strategy& strategy)
{
}

} // namespace block
} // namespace ck
