// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"
#include "ck/tensor_description/tensor_adaptor_coordinate.hpp"
#include "ck/tile_program/block_tensor_distribution.hpp"

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

    using AdaptorTopIndex   = Array<index_t, NDimWindowAdaptorTop>;
    using BottomTensorIndex = Array<index_t, NDimBottomTensor>;

    using WindowAdaptorCoord =
        decltype(make_tensor_adaptor_coordinate(WindowAdaptor{}, AdaptorTopIndex{}));

    using BottomTensorCoord =
        decltype(make_tensor_coordinate(BottomTensorDesc{}, BottomTensorIndex{}));

    // FIXME: host dummy constructor for tile program
    __host__ constexpr BlockTensorWindow(const BottomTensorView& bottom_tensor_view,
                                         const BottomTensorIndex&,
                                         const BlockTensorDstr&)
        : bottom_tensor_view_{bottom_tensor_view},
          bottom_tensor_thread_coord_{},
          block_tensor_dstr_{},
          window_adaptor_thread_coord_{}
    {
    }

    __device__ constexpr BlockTensorWindow(const BottomTensorView& bottom_tensor_view,
                                           const BottomTensorIndex& block_window_origin,
                                           const BlockTensorDstr& block_tensor_distribution)
        : bottom_tensor_view_{bottom_tensor_view},
          bottom_tensor_thread_coord_{},
          block_tensor_dstr_{block_tensor_distribution},
          window_adaptor_thread_coord_{make_tensor_adaptor_coordinate(
              block_tensor_distribution.GetWidLidYs2XsAdaptor(),
              block_tensor_distribution.CalculateThreadWidLidYsOrigin())}
    {
        BottomTensorIndex bottom_tensor_thread_origin_idx;

        for(index_t i = 0; i < NDimBottomTensor; ++i)
        {
            bottom_tensor_thread_origin_idx(i) =
                block_window_origin[i] + window_adaptor_thread_coord_.GetBottomIndex()[i];
        }

        bottom_tensor_thread_coord_ = make_tensor_coordinate(
            bottom_tensor_view_.GetTensorDescriptor(), bottom_tensor_thread_origin_idx);
    }

    __host__ __device__ static constexpr index_t GetNumOfDimension() { return NDimBottomTensor; }

    __host__ __device__ static constexpr bool HasStaticBlockTensorDistribution()
    {
        return BlockTensorDstr::IsKnownAtCompileTime();
    }

    __host__ __device__ constexpr auto GetBlockTensorDistribution() const
    {
        return block_tensor_dstr_;
    }

    __host__ __device__ constexpr auto GetBottomTensorView() const { return bottom_tensor_view_; }

    __host__ __device__ constexpr auto GetBottomTensorThreadCoordinate() const
    {
        return bottom_tensor_thread_coord_;
    }

    // move thread's window adaptor coordiante
    // [wid, lid, y0, y1, ...] ==> [x0, x1, ...]
    __device__ void MoveWindowAdaptorThreadCoordinate(const AdaptorTopIndex& idx_diff_adaptor)
    {
        move_tensor_adaptor_coordinate(block_tensor_dstr_.GetWidLidYs2XsAdaptor(),
                                       window_adaptor_thread_coord_,
                                       idx_diff_adaptor);
    }

    // move thread's botom tensor coordiante
    // [x0', x1', ... ] ==> [offset]
    __device__ void MoveBottomTensorThreadCoordinate(const BottomTensorIndex& idx_diff_tensor)
    {
        move_tensor_coordinate(bottom_tensor_view_.GetTensorDescriptor(),
                               bottom_tensor_thread_coord_,
                               idx_diff_tensor);
    }

    // move thread's window adaptor coordinate and bottom tensor coordinate
    // [wid, lid, y0, y1, ...] ==> [x0, x1, ...] ==> [x0', x1', ...] ==> [offset]
    __device__ void
    MoveWindowAdaptorAndBottomTensorThreadCoordinate(const AdaptorTopIndex& idx_diff_adaptor_top)
    {
        Array<index_t, NDimBottomTensor> idx_diff_adaptor_bottom;

        move_tensor_adaptor_coordinate(block_tensor_dstr_.GetWidLidYs2XsAdaptor(),
                                       window_adaptor_thread_coord_,
                                       idx_diff_adaptor_top,
                                       idx_diff_adaptor_bottom);

        move_tensor_coordinate(bottom_tensor_view_.GetTensorDescriptor(),
                               bottom_tensor_thread_coord_,
                               idx_diff_adaptor_bottom);
    }

    // this is the bottom tensor
    // [x0', x1', ...] ==> [offset]
    // tensor view and per-thread coordinate for bottom tensor
    BottomTensorView bottom_tensor_view_;
    BottomTensorCoord bottom_tensor_thread_coord_;

    // Block tensor distribution, which contains:
    //   1. adaptor for window: [wid, lid, y0, y1, ...] ==> [x0, x1, ...]
    //   2. thread descriptor for thread tensor in register
    BlockTensorDstr block_tensor_dstr_;

    //    thread window coordinate
    WindowAdaptorCoord window_adaptor_thread_coord_;
};

template <typename TensorView_, typename BlockTensorDistribution_>
__host__ __device__ constexpr auto
make_block_tensor_window(const TensorView_& tensor_view,
                         const Array<index_t, TensorView_::GetNumOfDimension()>& origin,
                         const BlockTensorDistribution_& block_tensor_distribution)
{
    return BlockTensorWindow<remove_cvref_t<TensorView_>, remove_cvref_t<BlockTensorDistribution_>>{
        tensor_view, origin, block_tensor_distribution};
}

// FIXME: dummy host function for tile program
template <typename BlockTensorWindow_>
__host__ void move_block_tensor_window(BlockTensorWindow_&,
                                       const MultiIndex<BlockTensorWindow_::GetNumOfDimension()>&)
{
}

template <typename BlockTensorWindow_>
__device__ void
move_block_tensor_window(BlockTensorWindow_& window,
                         const MultiIndex<BlockTensorWindow_::GetNumOfDimension()>& step)
{
    window.MoveBottomTensorThreadCoordinate(step);
}

} // namespace block
} // namespace tile_program
} // namespace ck
