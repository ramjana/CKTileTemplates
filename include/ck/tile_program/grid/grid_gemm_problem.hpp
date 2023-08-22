// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "ck/utility/type.hpp"

#include "ck/tile_program/block_tile_pipeline/block_gemm_pipeline_problem.hpp"
#include "ck/tile_program/utility/type_traits.hpp"

namespace ck {
namespace tile_program {
namespace grid {

template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CDataType_,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          index_t kBlockSize_,
          typename BlockGemmShape>
struct GridGemmProblem
{
    using Sub = block::
        BlockGemmPipelineProblem<ADataType, BDataType, AccDataType, kBlockSize_, BlockGemmShape>;

    using CDataType = CDataType_;
};

PP_DEFINE_INDIRECT_MEMBER_TYPE_GETTER(GetADataType, Sub, ADataType);
PP_DEFINE_INDIRECT_MEMBER_TYPE_GETTER(GetBDataType, Sub, BDataType);
PP_DEFINE_INDIRECT_MEMBER_TYPE_GETTER(GetCDataType, Sub, CDataType);

PP_DEFINE_INDIRECT_MEMBER_GETTER(GetMPerBlock, Sub, kM);
PP_DEFINE_INDIRECT_MEMBER_GETTER(GetNPerBlock, Sub, kN);
PP_DEFINE_INDIRECT_MEMBER_GETTER(GetKPerBlock, Sub, kK);

} // namespace grid
} // namespace tile_program
} // namespace ck
