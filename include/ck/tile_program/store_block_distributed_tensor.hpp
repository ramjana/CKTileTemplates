// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"
#include "ck/tensor_description/tensor_space_filling_curve.hpp"
#include "ck/tile_program/block_tensor_distribution.hpp"

namespace ck {
namespace tile_program {
namespace block {

// FIXME: host dummy function for tile program
template <typename BottomTensorView_, typename BlockTensorDistribution_, typename DataType_>
__host__ void
store_block_tile(BlockTensorWindow<BottomTensorView_, BlockTensorDistribution_>&,
                 const StaticBlockDistributedTensor<DataType_, BlockTensorDistribution_>&)
{
}

template <typename BottomTensorView_, typename BlockTensorDistribution_, typename DataType_>
__device__ void store_block_tile(
    BlockTensorWindow<BottomTensorView_, BlockTensorDistribution_>& block_tensor_window,
    const StaticBlockDistributedTensor<DataType_, BlockTensorDistribution_>& block_dstr_tensor)
{
    using DataType         = remove_cvref_t<typename BottomTensorView_::DataType>;
    using BottomTensorView = remove_cvref_t<BottomTensorView_>;
    using BlockTensorDstr  = remove_cvref_t<BlockTensorDistribution_>;
    using BlockWindow      = BlockTensorWindow<BottomTensorView, BlockTensorDstr>;

    static_assert(is_same_v<remove_cvref_t<DataType_>, DataType>, "wrong!");
    static_assert(BlockWindow::HasStaticBlockTensorDistribution(), "wrong!");

    constexpr auto block_dstr = BlockTensorDstr{};

    constexpr auto thread_tensor_lengths_ys =
        to_sequence(block_dstr.GetYs2DidDescriptor().GetLengths());

    constexpr index_t ndim_ys = thread_tensor_lengths_ys.Size();

    // FIXME:
    constexpr index_t ScalarPerVector = 1;
    constexpr index_t VectorDimY      = 1;

    // FIXME:
    using DimAccessOrder = typename arithmetic_sequence_gen<0, ndim_ys, 1>::type;

    constexpr auto scalars_per_access_arr = generate_array(
        [&](auto i) { return (i == VectorDimY) ? ScalarPerVector : 1; }, Number<ndim_ys>{});

    constexpr auto scalars_per_access = TO_SEQUENCE(scalars_per_access_arr, ndim_ys);

    using vector_type_t = vector_type_maker_t<DataType, ScalarPerVector>;
    using vector_t      = typename vector_type_t::type;

    using SFC_Ys = SpaceFillingCurve<decltype(thread_tensor_lengths_ys),
                                     DimAccessOrder,
                                     decltype(scalars_per_access)>;

    constexpr index_t num_access = SFC_Ys::GetNumOfAccess();

    static_assert(num_access > 0, "wrong! num_access should be larger than 0");

    // loop over thread tensor space [y0, y1, ...]
    static_for<0, num_access, 1>{}([&](auto iAccess) {
        // data index [y0, y1, ...]
        constexpr auto idx_ys_start = SFC_Ys::GetIndex(iAccess);

        // read from block distributed tensor
        vector_type_t vec;

        static_for<0, ScalarPerVector, 1>{}([&](auto j) {
            constexpr auto idx_ys = generate_array(
                [&](auto jj) {
                    return jj == VectorDimY ? (idx_ys_start[jj] + j) : idx_ys_start[jj];
                },
                Number<ndim_ys>{});

            constexpr index_t did = block_dstr.GetYs2DidDescriptor().CalculateOffset(idx_ys);

            vec.template AsType<DataType>()(j) =
                block_dstr_tensor.GetThreadBuffer().template At<did>();
        });

        const vector_t vec_value = vec.template AsType<vector_t>().template At<0>();

        // write into bottom tensor
        block_tensor_window.GetBottomTensorView().template SetVectorizedElements<vector_t>(
            block_tensor_window.GetBottomTensorThreadCoordinate(), vec_value);

        // move thread coordinate
        if constexpr(iAccess.value != num_access - 1)
        {
            constexpr auto idx_diff_ys = SFC_Ys::GetForwardStep(iAccess);

            constexpr auto idx_diff_wid_lid_ys =
                container_concat(Array<index_t, 2>{0, 0}, idx_diff_ys);

            block_tensor_window.MoveWindowAdaptorAndBottomTensorThreadCoordinate(
                idx_diff_wid_lid_ys);
        }
    });

    // move thread coordinate back to origin
    {
        constexpr auto idx_diff_ys = SFC_Ys::GetStepBetween(Number<num_access - 1>{}, Number<0>{});

        constexpr auto idx_diff_wid_lid_ys = container_concat(Array<index_t, 2>{0, 0}, idx_diff_ys);

        block_tensor_window.MoveWindowAdaptorAndBottomTensorThreadCoordinate(idx_diff_wid_lid_ys);
    }
}

} // namespace block
} // namespace tile_program
} // namespace ck
