// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"
#include "ck/tensor_description/tensor_adaptor_coordinate.hpp"

namespace ck {
namespace tile_program {
namespace block {

template <typename BottomTensorView_, typename BlockTensorDistribution_>
struct BlockTensorWindow
{
    using BottomTensorView = remove_reference_t<BottomTensorView_>;
    using BlockTensorDstr  = remove_cvref_t<BlockTensorDistribution_>;

    using WindowAdaptor    = typename BlockTensorDstr::WidLidYs2XsAdaptor;
    using BottomTensorDesc = typename BottomTensorView::TensorDesc;

    static constexpr index_t NDimWindowAdaptorTop = WindowAdaptor::GetNumOfTopDimension();
    static constexpr index_t NDimBottomTensor     = BottomTensorDesc::GetNumOfDimension();

    using WindowAdaptorCoord = decltype(
        make_tensor_adaptor_coordinate(WindowAdaptor{}, MultiIndex<NDimWindowAdaptorTop>{}));

    using BottomTensorCoord =
        decltype(make_tensor_coordinate(BottomTensorDesc{}, MultiIndex<NDimBottomTensor>{}));

    __device__ constexpr BlockTensorWindow(const BottomTensorView& bottom_tensor_view,
                                           const MultiIndex<NDimBottomTensor>& block_window_origin,
                                           const BlockTensorDstr& block_tensor_distribution)
        : window_adaptor_{block_tensor_distribution.GetWidLidYs2XsAdaptor()},
          bottom_tensor_view_{bottom_tensor_view},
          window_adaptor_thread_coord_{make_tensor_adaptor_coordinate(
              window_adaptor_, block_tensor_distribution.CalculateThreadWidLidYsOrigin())},
          bottom_tensor_thread_coord_{make_tensor_coordinate(
              bottom_tensor_view_.GetTensorDescriptor(),
              block_window_origin + window_adaptor_thread_coord_.GetBottomIndex())}
    {
    }

    // move thread's window adaptor coordiante
    // e.g. [wid, lid, y0, y1, ...] ==> [x0, x1, ...]
    __device__ void
    MoveWindowAdaptorThreadCoordinate(const MultiIndex<NDimWindowAdaptorTop>& idx_diff_adaptor)
    {
        move_tensor_adaptor_coordinate(
            window_adaptor_, window_adaptor_thread_coord_, idx_diff_adaptor);
    }

    // move thread's botom tensor coordiante
    // [x0', x1', ... ] ==> [offset]
    __device__ void
    MoveBottomTensorThreadCoordinate(const MultiIndex<NDimBottomTensor>& idx_diff_tensor)
    {
        move_tensor_coordinate(bottom_tensor_view_.GetTensorDescriptor(),
                               bottom_tensor_thread_coord_,
                               idx_diff_tensor);
    }

    // move thread's window adaptor coordinate and bottom tensor coordinate
    // e.g. [wid, lid, y0, y1, ...] ==> [x0, x1, ...] ==> [x0', x1', ...] ==> [offset]
    __device__ void MoveWindowAdaptorAndBottomTensorThreadCoordinate(
        const MultiIndex<NDimWindowAdaptorTop>& idx_diff_adaptor_top)
    {
        Array<index_t, NDimBottomTensor> idx_diff_adaptor_bottom;

        move_tensor_adaptor_coordinate(window_adaptor_,
                                       window_adaptor_thread_coord_,
                                       idx_diff_adaptor_top,
                                       idx_diff_adaptor_bottom);

        move_tensor_coordinate(bottom_tensor_view_.GetTensorDescriptor(),
                               bottom_tensor_thread_coord_,
                               idx_diff_adaptor_bottom);
    }

    // this is the adaptor for window
    // [wid, lid, y0, y1, ...] ==> [x0, x1, ...]
    WindowAdaptor window_adaptor_;

    // this is the bottom tensor
    // [x0', x1', ...] ==> [offset]
    BottomTensorView bottom_tensor_view_;

    // [wid, lid, y0, y1, ...] ==> [x0, x1, ...]
    WindowAdaptorCoord window_adaptor_thread_coord_;

    // [x0', x1', ...] ==> [offset]
    BottomTensorCoord bottom_tensor_thread_coord_;
};

template <typename Tensor_, typename BlockTensorDistribution_>
__device__ constexpr auto
make_block_tensor_window(const Tensor_& tensor,
                         const MultiIndex<Tensor_::GetNumOfDimension()>& origin,
                         const BlockTensorDistribution_& block_tensor_distribution)
{
    return BlockTensorWindow<remove_cvref_t<Tensor_>, remove_cvref_t<BlockTensorDistribution_>>{
        tensor, origin, block_tensor_distribution};
}

template <typename BlockTensorWindow_, typename Index>
__device__ void move_block_tensor_window(BlockTensorWindow_& window, const Index& step)
{
    STATIC_ASSERT(BlockTensorWindow_::GetNumOfDimension() == Index::Size(), "");

    window.MoveBottomTensorThreadCoordinate(step);
}

#if 0
template <index_t BlockSize, index_t... BlockWindowLengths, template Strategy>
__host__ __device__ constexpr auto
make_block_tensor_window_from_strategy(Sequence<BlockWindowLengths...>, const Stragety& strategy)
{

    constexpr auto xs_unmerge_up_lengthss = xxx;

    constexpr auto dims_wid_2_xs_major = xxx;
    constexpr auto dims_wid_2_xs_minor = xxx;

    constexpr auto dims_lid_2_xs_major = xxx;
    constexpr auto dims_lid_2_xs_minor = xxx;

    constexpr auto dims_ys_2_xs_major = xxx;
    constexpr auto dims_ys_2_xs_minor = xxx;

    constexpr auto ys_order = xxx;

    return make_block_tensor_distribution(xs_unmerge_up_lengthss,
                                          dims_wid_2_xs_major,
                                          dims_wid_2_xs_minor,
                                          dims_lid_2_xs_major,
                                          dims_lid_2_xs_minor,
                                          dims_ys_2_xs_major,
                                          dims_ys_2_xs_minor,
                                          ys_order);
}
#endif

} // namespace block
} // namespace tile_program
} // namespace ck
