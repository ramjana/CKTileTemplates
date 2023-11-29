// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tile_program/tile/store_tile.hpp"
#include "ck/tile_program/tile/tile_elementwise.hpp"

template <typename KGradaccDataType_,
          typename KGradDataType_>
struct FmhaBwdKGradEpilogueProblem
{
    using KGradaccDataType   = ck::remove_cvref_t<KGradaccDataType_>;
    using KGradDataType = ck::remove_cvref_t<KGradDataType_>;
};

template <typename Problem_, typename Policy_ = void>
struct FmhaBwdKGradEpilogue
{
    using Problem       = ck::remove_cvref_t<Problem_>;
    using KGradaccDataType   = ck::remove_cvref_t<typename Problem::KGradaccDataType>;
    using KGradDataType = ck::remove_cvref_t<typename Problem::KGradDataType>;

    __host__ __device__ static constexpr ck::index_t GetSmemSize() { return 0; }

    template <typename KGradDramWindowTmp, typename KGradAccTile>
    __device__ auto operator()(KGradDramWindowTmp& dk_dram_window_tmp, const KGradAccTile& dk_acc_tile)
    {
        using namespace ck;
        using namespace ck::tile_program;

        const auto dk = tile_elementwise_in(type_convert<KGradDataType, KGradaccDataType>, dk_acc_tile);
        store_tile(dk_dram_window_tmp, dk);
    }
};

template <typename VGradaccDataType_,
          typename VGradDataType_>
struct FmhaBwdVGradEpilogueProblem
{
    using VGradaccDataType   = ck::remove_cvref_t<VGradaccDataType_>;
    using VGradDataType = ck::remove_cvref_t<VGradDataType_>;
};

template <typename Problem_, typename Policy_ = void>
struct FmhaBwdVGradEpilogue
{
    using Problem       = ck::remove_cvref_t<Problem_>;
    using VGradaccDataType   = ck::remove_cvref_t<typename Problem::VGradaccDataType>;
    using VGradDataType = ck::remove_cvref_t<typename Problem::VGradDataType>;

    __host__ __device__ static constexpr ck::index_t GetSmemSize() { return 0; }

    template <typename VGradDramWindowTmp, typename VGradAccTile>
    __device__ auto operator()(VGradDramWindowTmp& dv_dram_window_tmp, const VGradAccTile& dv_acc_tile)
    {
        using namespace ck;
        using namespace ck::tile_program;

        const auto dv = tile_elementwise_in(type_convert<VGradDataType, VGradaccDataType>, dv_acc_tile);
        store_tile(dv_dram_window_tmp, dv);
    }
};

