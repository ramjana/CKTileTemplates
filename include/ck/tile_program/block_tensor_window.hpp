// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"
#include "ck/tensor_description/tensor_adaptor_coordinate.hpp"
#include "ck/tile_program/tile_distribution.hpp"

namespace ck {
namespace tile_program {

template <typename BottomTensorView_, typename BlockTensorDistribution_>
struct BlockTensorWindow
{
    using BottomTensorView = remove_reference_t<BottomTensorView_>;
    using BlockTensorDstr  = remove_cvref_t<BlockTensorDistribution_>;

    using WindowAdaptor    = typename BlockTensorDstr::PsYs2XsAdaptor;
    using BottomTensorDesc = typename BottomTensorView::TensorDesc;

    using DataType = typename BottomTensorView::DataType;

    static constexpr index_t NDimWindowAdaptorTop = WindowAdaptor::GetNumOfTopDimension();
    static constexpr index_t NDimBottomTensor     = BottomTensorDesc::GetNumOfDimension();

    static_assert(WindowAdaptor::IsStatic(), "wrong!");

    static_assert(NDimBottomTensor == WindowAdaptor::GetNumOfBottomDimension(),
                  "wrong! inconsistent # of diemsnions");

    using AdaptorTopIndex   = Array<index_t, NDimWindowAdaptorTop>;
    using BottomTensorIndex = Array<index_t, NDimBottomTensor>;

    using WindowAdaptorCoord =
        decltype(make_tensor_adaptor_coordinate(WindowAdaptor{}, AdaptorTopIndex{}));

    using BottomTensorCoord =
        decltype(make_tensor_coordinate(BottomTensorDesc{}, BottomTensorIndex{}));

    __host__ __device__ constexpr BlockTensorWindow() = default;

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
          block_window_origin_{block_window_origin},
          bottom_tensor_thread_coord_{},
          block_tensor_dstr_{block_tensor_distribution},
          window_adaptor_thread_coord_{
              make_tensor_adaptor_coordinate(block_tensor_distribution.GetPsYs2XsAdaptor(),
                                             AdaptorTopIndex{get_warp_id(), get_lane_id(), 0})}
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
        return BlockTensorDstr::IsStatic();
    }

    __host__ __device__ constexpr auto GetWindowLengths() const
    {
        return WindowAdaptor{}.GetBottomDimensionLengths();
    }

    __host__ __device__ constexpr auto GetBlockTensorDistribution() const
    {
        return block_tensor_dstr_;
    }

    __host__ __device__ constexpr auto GetBottomTensorView() const { return bottom_tensor_view_; }

    __host__ __device__ constexpr auto GetBlockWindowOrigin() const { return block_window_origin_; }

    __host__ __device__ constexpr auto GetBottomTensorThreadCoordinate() const
    {
        return bottom_tensor_thread_coord_;
    }

    // move thread's window adaptor coordiante
    // [p0, p1, ..., y0, y1, ...] ==> [x0, x1, ...]
    __device__ void MoveWindowAdaptorThreadCoordinate(const AdaptorTopIndex& idx_diff_adaptor)
    {
        move_tensor_adaptor_coordinate(
            block_tensor_dstr_.GetPsYs2XsAdaptor(), window_adaptor_thread_coord_, idx_diff_adaptor);
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
    // [p0, p1, ..., y0, y1, ...] ==> [x0, x1, ...] ==> [x0', x1', ...] ==> [offset]
    __device__ void
    MoveWindowAdaptorAndBottomTensorThreadCoordinate(const AdaptorTopIndex& idx_diff_adaptor_top)
    {
        Array<index_t, NDimBottomTensor> idx_diff_adaptor_bottom;

        move_tensor_adaptor_coordinate(block_tensor_dstr_.GetPsYs2XsAdaptor(),
                                       window_adaptor_thread_coord_,
                                       idx_diff_adaptor_top,
                                       idx_diff_adaptor_bottom);

        move_tensor_coordinate(bottom_tensor_view_.GetTensorDescriptor(),
                               bottom_tensor_thread_coord_,
                               idx_diff_adaptor_bottom);
    }

    // return vector dimension among [y0, y1, ...]
    __host__ __device__ static constexpr auto GetWindowAdaptorYsSafeVectorLengthStrides()
    {
        // bottom tensor top dimension vector lengths and strides
        const auto [bottom_tensor_top_dim_vector_lengths, bottom_tensor_top_dim_vector_strides] =
            BottomTensorDesc::GetTopDimensionSafeVectorLengthStrides();

        // window vector lengths/strides
        const auto window_adaptor_bottom_dim_vector_lengths = bottom_tensor_top_dim_vector_lengths;
        const auto window_adaptor_bottom_dim_vector_strides = bottom_tensor_top_dim_vector_strides;

        // window adaptor [p0, p1, ..., y0, y1, ...]
        Array<index_t, WindowAdaptor::GetNumOfHiddenDimension()> window_adaptor_vector_lengths{-1};
        Array<index_t, WindowAdaptor::GetNumOfHiddenDimension()> window_adaptor_vector_strides{-1};

        constexpr auto window_adaptor_bottom_dims = WindowAdaptor::GetBottomDimensionHiddenIds();

        set_container_subset(window_adaptor_vector_lengths,
                             window_adaptor_bottom_dims,
                             window_adaptor_bottom_dim_vector_lengths);
        set_container_subset(window_adaptor_vector_strides,
                             window_adaptor_bottom_dims,
                             window_adaptor_bottom_dim_vector_strides);

        const auto [window_adaptor_ps_ys_vector_lengths, window_adaptor_ps_ys_vector_strides] =
            WindowAdaptor{}.GetTopDimensionSafeVectorLengthStrides(window_adaptor_vector_lengths,
                                                                   window_adaptor_vector_strides);

        // [y0, y1, ...]
        constexpr auto y_dims =
            typename arithmetic_sequence_gen<BlockTensorDstr::GetNumOfDimensionP(),
                                             NDimWindowAdaptorTop,
                                             1>::type{};

        return make_tuple(get_container_subset(window_adaptor_ps_ys_vector_lengths, y_dims),
                          get_container_subset(window_adaptor_ps_ys_vector_strides, y_dims));
    }

    // this is the bottom tensor view
    // [x0', x1', ...] ==> [offset]
    BottomTensorView bottom_tensor_view_;

    // the blockwise origin ([x0', x1', ...]) of block window on bottom tensor
    BottomTensorIndex block_window_origin_;

    // per-thread coordinate for bottom tensor
    BottomTensorCoord bottom_tensor_thread_coord_;

    // Block tensor distribution, which contains:
    //   1. adaptor for window: [p0, p1, ..., y0, y1, ...] ==> [x0, x1, ...]
    //   2. thread descriptor for thread tensor in register: [y0, y1, ...] ==> [d]
    BlockTensorDstr block_tensor_dstr_;

    //    thread window coordinate
    WindowAdaptorCoord window_adaptor_thread_coord_;
};

// TODO: use strategy
template <typename TensorView_, typename BlockTensorDistribution_>
__host__ __device__ constexpr auto
make_block_window(const TensorView_& tensor_view,
                  const Array<index_t, TensorView_::GetNumOfDimension()>& origin,
                  const BlockTensorDistribution_& block_tensor_distribution)
{
    return BlockTensorWindow<remove_cvref_t<TensorView_>, remove_cvref_t<BlockTensorDistribution_>>{
        tensor_view, origin, block_tensor_distribution};
}

// FIXME: dummy host function for tile program
template <typename BlockTensorWindow_>
__host__ void move_block_window(BlockTensorWindow_&,
                                const MultiIndex<BlockTensorWindow_::GetNumOfDimension()>&)
{
}

template <typename BlockTensorWindow_>
__device__ void move_block_window(BlockTensorWindow_& window,
                                  const MultiIndex<BlockTensorWindow_::GetNumOfDimension()>& step)
{
    window.block_window_origin_ += step;

    window.MoveBottomTensorThreadCoordinate(step);
}

} // namespace tile_program
} // namespace ck
