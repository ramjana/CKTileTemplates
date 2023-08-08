// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"

#include "tile_program.hpp"
#include "ck/tile_program/tile/tile_distribution.hpp"
#include "ck/tile_program/tile/tile_elementwise.hpp"

template <typename ADataType,
          typename AccDataType,
          typename BDataType,
          ck::index_t kBlockSize,
          ck::index_t kMPerBlock,
          ck::index_t kNPerBlock>
struct Reduce
{
    __host__ __device__ void operator()(
        ProgramServer& ps, const ADataType* p_a, BDataType* p_b, ck::index_t M, ck::index_t N) const
    {
        (void)ps;
        (void)p_a;
        (void)p_b;
        (void)M;
        (void)N;
    }
};
