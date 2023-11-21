// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "ck/utility/type.hpp"

namespace ck {
namespace tile_program {
namespace block {

template <typename QDataType_,
          typename KDataType_,
          typename VDataType_,
          typename GemmDataType_,
          typename LSEDataType_,
          typename AccDataType_,
          typename SMPLComputeDataType_,
          typename DDataType_,
          typename ZDataType_,
          typename ODataType_,
          typename OGradDataType_,
          typename QGradDataType_,
          typename KGradDataType_,
          typename VGradDataType_,
          index_t kBlockSize_,
          typename BlockFmhaBwdShape_>
struct BlockFmhaBwdPipelineProblem
{
    using QDataType           = remove_cvref_t<QDataType_>;
    using KDataType           = remove_cvref_t<KDataType_>;
    using VDataType           = remove_cvref_t<VDataType_>;
    using GemmDataType        = remove_cvref_t<GemmDataType_>;
    using LSEDataType         = remove_cvref_t<LSEDataType_>;
    using AccDataType         = remove_cvref_t<AccDataType_>;
    using SMPLComputeDataType = remove_cvref_t<SMPLComputeDataType_>;
    using DDataType           = remove_cvref_t<DDataType_>;
    using ZDataType           = remove_cvref_t<ZDataType_>;
    using ODataType           = remove_cvref_t<ODataType_>;
    using OGradDataType       = remove_cvref_t<OGradDataType_>;
    using QGradDataType       = remove_cvref_t<QGradDataType_>;
    using KGradDataType       = remove_cvref_t<KGradDataType_>;
    using VGradDataType       = remove_cvref_t<VGradDataType_>;
    using BlockFmhaBwdShape   = remove_cvref_t<BlockFmhaBwdShape_>;

    static constexpr index_t kBlockSize = kBlockSize_;
};

} // namespace block
} // namespace tile_program
} // namespace ck
