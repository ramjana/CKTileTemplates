// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"

#include "ck/tile_program/tile/static_distributed_tensor.hpp"
#include "ck/tile_program/tile/static_tile_distribution_encoding_helper.hpp"

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
#if 1
    (void)acc_tensor;
    (void)reduce_func_acc_in;
    (void)in_tensor;
#endif

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};

    // in-thread reduction
#if 0
    constexpr auto in_reduce_dims = Sequence<InReduceDims...>{};

    constexpr index_t ndim_in        = InDistributedTensor_::GetNumOfDimension();
    constexpr index_t ndim_in_reduce = in_reduce_dims.Size();
    constexpr index_t ndim_in_free   = ndim_in - ndim_in_reduce;

    constexpr auto in_free_dims_arr = [&] {
        Array<bool, ndim_free> is_free_dims{true};

        for(index_t i = 0; i < ndim_reduce; i++)
        {
            is_free_dims(in_reduce_dims[i]) = false;
        }

        Array<index_t, ndim_free> in_free_dims{-1};

        index_t cnt = 0;

        for(index_t i = 0; i < ndim_in; i++)
        {
            if(is_free_dims[i])
            {
                in_free_dims(cnt) = i;

                cnt++
            }
        }

        return is_free_dims;
    }();

    constexpr auto in_free_dims = TO_SEQUENCE(is_free_dims_arr, ndim_in_free);
#endif

#if 1
    constexpr auto spans = InDistributedTensor_::GetDistributedSpans();

    using Span0 = remove_cvref_t<decltype(spans[I0])>;
    using Span1 = remove_cvref_t<decltype(spans[I1])>;

    // FIXME
    static_ford<typename Span0::Impl>{}([&](auto dstr_idx_impl_i0) {
        constexpr auto dstr_idx_i0  = detail::make_tile_distributed_index(dstr_idx_impl_i0);
        constexpr auto acc_dstr_idx = make_tuple(dstr_idx_i0);

        auto acc = acc_tensor.GetElementFromTileDistributedIndex(acc_dstr_idx);

        //  FIXME
        static_ford<typename Span1::Impl>{}([&](auto dstr_idx_impl_i1) {
            constexpr auto dstr_idx_i1 = detail::make_tile_distributed_index(dstr_idx_impl_i1);
            constexpr auto in_dstr_idx = make_tuple(dstr_idx_i0, dstr_idx_i1);

            const auto in = in_tensor.GetElementFromTileDistributedIndex(in_dstr_idx);

            reduce_func_acc_in(acc, in);
        });

        acc_tensor.SetElementFromTileDistributedIndex(acc_dstr_idx, acc);
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
    using InDataType  = typename InDistributedTensor_::DataType;
    using AccDataType = remove_cvref_t<AccDataType_>;

    static_assert(is_same_v<InDataType, remove_cvref_t<InDataType_>>, "wrong!");

    // declare acc_tensor
    constexpr auto acc_dstr = make_static_tile_distribution(
        ck::tile_program::detail::make_reduce_tile_distribution_encoding(
            InDistributedTensor_::GetTileDistribution().GetStaticTileDistributionEncoding(),
            Sequence<InReduceDims...>{}));

    auto acc_tensor = make_static_distributed_tensor<AccDataType>(acc_dstr);

    // init acc_tensor
    tile_elementwise_inout([&](auto& acc) { acc = type_convert<AccDataType>(reduce_init); },
                           acc_tensor);

    // warp reduce
    warp_tile_reduce_acc_in(acc_tensor, in_tensor, in_reduce_dims, reduce_func_acc_in);

    return acc_tensor;
}

} // namespace warp
} // namespace tile_program
} // namespace ck
