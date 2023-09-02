// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"

#include "ck/tile_program/tile/static_distributed_tensor.hpp"
#include "ck/tile_program/tile/static_tile_distribution_encoding_helper.hpp"
#include "ck/tile_program/tile/distributed_tile_sweep.hpp"

namespace ck {
namespace tile_program {
namespace warp {

namespace detail {

template <typename AccDistributedTensor_, typename ReduceFuncAccIn>
__host__ __device__ void
reduce_and_broadcast_replication_of_distributed_tensor(AccDistributedTensor_& acc_tensor,
                                                       const ReduceFuncAccIn& reduce_func_acc_in)
{
    using Dstr             = typename AccDistributedTensor_::StaticTileDistribution;
    using DstrEncode       = typename Dstr::DstrEncode;
    using DstrEncodeDetail = typename DstrEncode::Detail;

    constexpr index_t NDimP = Dstr::GetNumOfDimensionP();
    constexpr index_t NDimR = Dstr::GetNumOfDimensionR();

    // FIXME: this is for block reduce
    const auto ps_idx = make_array<index_t>(get_warp_id(), get_lane_id());
    const auto rs_idx = acc_tensor.GetTileDistribution().CalculateRsIndexFromPsIndex(ps_idx);

    constexpr index_t thread_buf_size = AccDistributedTensor_::GetThreadBufferSize();

    // loop over thread data
    static_for<0, thread_buf_size, 1>{}([&](auto i) {
        auto v = acc_tensor.GetThreadBuffer()[i];

        // reduce replication
        static_for<0, NDimR, 1>{}([&](auto idim_r) {
            constexpr index_t r_length = DstrEncode::rs_lengths_[idim_r];

            constexpr index_t lid_over_rid_derivative =
                DstrEncodeDetail::ps_over_rs_derivative_[NDimP - 1][idim_r];

            static_assert(math::is_power_of_two_integer(r_length),
                          "wrong! only support power of 2 reduction");

            constexpr index_t nstage = math::integer_log2_floor(r_length);

            // sweep forward to get data from other lane for reduction
            static_for<0, nstage, 1>{}([&](auto istage) {
                constexpr index_t lid_delta =
                    lid_over_rid_derivative * (1 << (nstage - istage - 1));

                const auto v_remote = warp_shuffle_down(v, lid_delta);

                // reduce
                reduce_func_acc_in(v, v_remote);
            });
        });

        // broadcast replication
        static_for<0, NDimR, 1>{}([&](auto idim_r) {
            const index_t r_id = rs_idx[idim_r];

            constexpr index_t r_length = DstrEncode::rs_lengths_[idim_r];

            constexpr index_t lid_over_rid_derivative =
                DstrEncodeDetail::ps_over_rs_derivative_[NDimP - 1][idim_r];

            static_assert(math::is_power_of_two_integer(r_length),
                          "wrong! only support power of 2 reduction");

            constexpr index_t nstage = math::integer_log2_floor(r_length);

            // sweep backward to get data from other lane for broadcast
            static_for<0, nstage, 1>{}([&](auto istage) {
                constexpr index_t lid_delta_tmp = lid_over_rid_derivative * (1 << istage);

                // read from other or from self
                const index_t lid_delta = r_id < (1 << istage) ? 0 : lid_delta_tmp;

                v = warp_shuffle_up(v, lid_delta);
            });

            acc_tensor.GetThreadBuffer()(i) = v;
        });
    });
}

} // namespace detail

// FIXME: this is for 2D to 1D reduce only, need to support n-D
template <typename AccDistributedTensor_,
          typename InDistributedTensor_,
          index_t... InReduceDims,
          typename ReduceFuncAccIn>
__device__ void warp_tile_reduce_acc_in(AccDistributedTensor_& acc_tensor,
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

    constexpr auto spans = InDistributedTensor_::GetDistributedSpans();

    // in-thread reduction
    // FIXME: hard coded to be 2D to 1D reduction
    sweep_tile_span(spans[I0], [&](auto dstr_idx_i0) {
        constexpr auto acc_dstr_idx = make_tuple(dstr_idx_i0);

        auto acc = acc_tensor.GetElementFromTileDistributedIndices(acc_dstr_idx);

        // FIXME
        sweep_tile_span(spans[I1], [&](auto dstr_idx_i1) {
            constexpr auto in_dstr_idx = make_tuple(dstr_idx_i0, dstr_idx_i1);

            const auto in = in_tensor.GetElementFromTileDistributedIndices(in_dstr_idx);

            reduce_func_acc_in(acc, in);
        });

        acc_tensor.SetElementFromTileDistributedIndices(acc_dstr_idx, acc);
    });

    // cross-thread but in-warp reduction
#if 1
    detail::reduce_and_broadcast_replication_of_distributed_tensor(acc_tensor, reduce_func_acc_in);
#endif
}

// FIXME: dummy function for tile program
template <typename AccDistributedTensor_,
          typename InDistributedTensor_,
          index_t... InReduceDims,
          typename ReduceFuncAccIn>
__host__ void warp_tile_reduce_acc_in(AccDistributedTensor_&,
                                      const InDistributedTensor_&,
                                      Sequence<InReduceDims...>,
                                      const ReduceFuncAccIn&)
{
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
