// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"

#include "ck/tile_program/tile/tile_distribution.hpp"

namespace ck {
namespace tile_program {

// sweep over a distributed span of a distribted tile
template <typename TileDistributedSpan_, typename F>
__host__ __device__ void sweep_tile_span(TileDistributedSpan_, const F& f)
{
    using DstrSpan = remove_cvref_t<TileDistributedSpan_>;

    static_ford<typename DstrSpan::Impl>{}([&](auto dstr_idx_impl) {
        constexpr auto dstr_idx = detail::make_tile_distributed_index(dstr_idx_impl);

        f(dstr_idx);
    });
}

} // namespace tile_program
} // namespace ck
