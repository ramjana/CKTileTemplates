// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"

#include "ck/tile_program/tile/tile_distribution_helper.hpp"
#include "ck/tile_program/tile/static_distributed_tensor.hpp"

namespace ck {
namespace tile_program {
namespace warp {

// 2D to 1D reduce
template <typename OutDataType_,
          typename StaticDistributedTensor_,
          index_t... InReduceDims,
          typename ReduceFuncIn,
          typename InDataType_>
__device__ auto warp_tile_reduce_in(const StaticDistributedTensor_& in_tensor,
                                    Sequence<InReduceDims...> in_reduce_dims,
                                    const ReduceFuncIn& /* reduce_func_in */,
                                    const InDataType_& reduce_init)
{
    using namespace ck::tile_program;

    using InDataType  = typename StaticDistributedTensor_::DataType;
    using OutDataType = remove_cvref_t<OutDataType_>;

    static_assert(is_same_v<InDataType, remove_cvref_t<InDataType_>>, "wrong!");

    // declare out_warp_tensor
    constexpr auto out_dstr = detail::make_reduce_tile_distribution_encoding(
        in_tensor.GetTileDistribution(), in_reduce_dims);

    auto out_tensor = make_static_distributed_tensor<OutDataType>(out_dstr);

    // initialize out_warp_tensor
    tile_elementwise_inout([&](auto& out) { out = type_convert<OutDataType>(reduce_init); },
                           out_tensor);

    // in-thread reduction
#if 0
    // FIXME
    range_over(get_distributed_tensor_range(in_tensor), [&](auto in_range_idx) {
        // FIXME
        const auto out_range_idx = get_range_subset(in_range_idx, Sequence<0>{});

        // FIXME
        out_tensor(out_range_idx) =
            reduce_func_inout(out_tensor[out_range_idx], in_tensor[in_range_idx]);
    });
#endif

    // cross-thread but in-warp reduction
#if 0
    sync_reduce_warp_distributed_tensor(reduce_func, out_tensor);
#endif

    return out_tensor;
}

} // namespace warp
} // namespace tile_program
} // namespace ck
