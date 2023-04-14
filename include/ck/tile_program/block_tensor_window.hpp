// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"

namespace ck {
namespace tile_program {
namespace block {

template <typename TensorView>
struct TensorBlockWindow
{
    using WindowAdaptor = TensorAdaptor<xxx>;

    using TensorDesc = typename Tensor::TensorDescriptor;

    using WindowAdaptorCoord = TensorAdaptorCoordinate<xxx>;

    using TensorCoord = TensorCoordinate<Tensor::TensorDescriptor::GetNumOfHiddenDimension(),
                                         Tensor::TensorDescriptor::GetVisibleDimensionIds()>;

    __device__ constexpr TensorBlockWindow(xxx)
        : window_adaptor_{xxx},
          tensor_{xxx},
          window_adaptor_thread_coord_{make_tensor_adaptor_coordinate(xxx)},
          tensor_thread_coord_{make_tensor_coordiante(xxx)}
    {
    }

    // this move thread's window adaptor coordiante
    // e.g. [wid, lid, m3, k2, k3] to [m', k']
    __device__ void
    MoveWindowAdaptorThreadCoordinate(const MultiIndex<NumDimWindowAdaptorTop>& adaptor_step)
    {
        move_tensor_adaptor_coordinate(window_adaptor_, window_adaptor_thread_coord_, adaptor_step);
    }

    // this move thread's tensor coordiante
    // e.g. [m, k] to [offset]
    __device__ void MoveTensorThreadCoordinate(const MultiIndex<NumDimTensor>& tensor_step)
    {
        move_tensor_coordinate(tensor_.GetTensorDescriptor(), tensor_thread_coord_, tensor_step);
    }

    // this move thread's window adapto coordinate and tensor coordinate
    // e.g. [wid, lid, m3, k2, k3] to [m', k'] to [offset]
    __device__ void MoveWindowAdaptorAndTensorThreadCoordinate(
        const MultiIndex<NumDimWindowAdaptorTop>& adaptor_step)
    {
        move_tensor_adaptor_coordinate(window_adaptor_, window_adaptor_thread_coord_, adaptor_step);

        const auto tensor_step = window_adaptor_thread_coord_.GetBottomIndex();

        move_tensor_coordinate(tensor_.GetTensorDescriptor(), tensor_thread_coord_, tensor_step);
    }

    // this is the adaptor for window
    // e.g. [wid, lid, m3, k2, k3] to [m', k']
    WindowAdaptor window_adaptor_;

    // this is the tensor
    // e.g. [m, k] to [offset]
    BottomTensorView tensor_;

    // e.g. [wid, lid, m3, k2, k3] to [m, k]
    WindowAdaptorCoord window_adaptor_thread_coord_;

    // e.g. [m, k] to [offset]
    BottomTensorCoord tensor_thread_coord_;
};

template <index_t BlockSize, typename xxx, index_t Lengths..., typename Strategy>
__device__ constexpr auto make_tensor_block_window(const BlockTileProgram& btp,
                                                   const TensorView<xxx>& tensor,
                                                   const Sequence<Lengths...>& window_lengths,
                                                   const MultiIndex<xxx>& origin,
                                                   const Strategy& block_distribution_strategy)
{
    // FIXME: not implemented
    // block distribution
    const auto block_distribution =
        get_block_tensor_distribution(btp, window_lengths, block_distribution_strategy);

    // adaptor
    const auto adptor_ms_ks_to_m_k = make_single_stage_tensor_adaptor(transforms, up

    constexpr auto window_adaptor_wid_lid_xs = xxxx;

    const auto widnow_adaptor_xs =
        transform_tensor_adaptor(window_aadptor_wid_lid_xs, xxx, xxx, xxx);

    // window adaptor coordinate
    auto window_adaptor_coord = make_tensor_adaptor_coordinate(window_adaptor_xs, {0, 0, 0, ...});

    // tensor visible index
    auto tensor_visible_idx = window_adaptor_coord.GetBottomIndex() + origin;

    auto tensor_coord = make_tensor_coordinate(tensor.GetTensorDescriptor(), tensor_visible_idx);

    //
    return TensorBlockWindow<xxx>{xxx};
}

template <typename xxx>
__device__ void move_tensor_block_window(TensorBlockWindow<xxx>& window,
                                         const MultiIndex<xxx>& step)
{
    window.MoveTensorThreadCoordinate(step);
}

} // namespace block
} // namespace tile_program
} // namespace ck
