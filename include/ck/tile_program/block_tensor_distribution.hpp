// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"

namespace ck {
namespace tile_program {
namespace block {

template <xxx....>
struct BlockTensorDistribution
{
    // Unmerge lengths [m0, m1, m2...] from [m']
    // Unmerge lengths [k0, k1, k2...] from [k']
    Tuple<UnmergeLengthss, ...> unmerge_lengthss;

    // merge [wid] from [m0, ..., k0, ...]
    Sequence<...> ids_wid_to_ms_ks;

    // merge [lid] from [m0, ..., k0, ...]
    Sequence<...> ids_lid_to_ms_ks;

    // merge [did] from [m0, ..., k0, ...]
    Sequence<...> ids_did_to_ms_ks;
};

template <typename BlockDistribution>
__device__ constexpr auto make_block_distribution_adaptor(const BlockDistrition& block_distribution)
{
    constexpr auto transforms_lowDimIdss_upDimIdss = [&]() {}();
}

} // namespace block
} // namespace tile_program
} // namespace ck
