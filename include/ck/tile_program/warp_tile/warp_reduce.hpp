// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"

#include "ck/tile_program/tile/tile_distribution_helper.hpp"
#include "ck/tile_program/tile/static_distributed_tensor.hpp"

namespace ck {
namespace tile_program {
namespace warp {

// FIXME: this is for 2D to 1D reduce only, need to support n-D
template <typename AccDistributedTensor_,
          typename InDistributedTensor_,
          index_t... InReduceDims,
          typename ReduceFuncAccIn>
__host__ __device__ void warp_tile_reduce_acc_in(AccDistributedTensor_& acc_tensor,
                                                 const InDistributedTensor_& in_tensor,
                                                 Sequence<InReduceDims...>,
                                                 const ReduceFuncAccIn& reduce_func_acc_in)
{
    using namespace ck::tile_program;

#if 1
    (void)acc_tensor;
    (void)reduce_func_acc_in;
    (void)in_tensor;
#endif

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
}

template <typename AccDataType_,
          typename InDistributedTensor_,
          index_t... InReduceDims,
          typename ReduceFuncAccIn,
          typename InDataType_>
__host__ __device__ auto warp_tile_reduce_in(const InDistributedTensor_& in_tensor,
                                             Sequence<InReduceDims...> in_reduce_dims,
                                             const ReduceFuncAccIn& reduce_func_acc_in,
                                             const InDataType_& reduce_init)
{
    using namespace ck::tile_program;

    using InDataType  = typename InDistributedTensor_::DataType;
    using AccDataType = remove_cvref_t<AccDataType_>;

    static_assert(is_same_v<InDataType, remove_cvref_t<InDataType_>>, "wrong!");

    // declare acc_tensor
    constexpr auto acc_dstr =
        make_static_tile_distribution(detail::make_reduce_tile_distribution_encoding(
            InDistributedTensor_::GetTileDistribution().GetStaticTileDistributionEncoding(),
            Sequence<InReduceDims...>{}));

    auto acc_tensor = make_static_distributed_tensor<AccDataType>(acc_dstr);

    // init acc_tensor
    tile_elementwise_inout([&](auto& acc) { acc = type_convert<AccDataType>(reduce_init); },
                           acc_tensor);

    // reduce
    warp_tile_reduce_acc_in(acc_tensor, in_tensor, in_reduce_dims, reduce_func_acc_in);

    return acc_tensor;
}

} // namespace warp
} // namespace tile_program
} // namespace ck
