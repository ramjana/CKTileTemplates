// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tile_program/block_tile_pipeline/block_fmha_bwd_pipeline.hpp"

namespace ck {
namespace tile_program {
namespace block {

namespace impl {
template <bool QLoadOnce,
          bool QTLoadOnce,
          bool KLoadOnce,
          bool KTLoadOnce,
          bool VLoadOnce,
          bool OGradLoadOnce,
          bool OGradTLoadOnce>
struct BlockFmhaBwdPipelineDispatcher;

// clang-format off
// #############################################| QLoadOnce| QTLoadOnce| KLoadOnce| KTLoadOnce| VLoadOnce| OGradLoadOnce| OGradTLoadOnce|
template<> struct BlockFmhaBwdPipelineDispatcher<      true,       true,      true,       true,      true,          true,           true> { using Type = BlockFmhaBwdPipelineV1; };
template<> struct BlockFmhaBwdPipelineDispatcher<      true,       true,      true,       true,      true,          true,          false> { using Type = BlockFmhaBwdPipelineV2; };
template<> struct BlockFmhaBwdPipelineDispatcher<      true,      false,      true,       true,      true,          true,           true> { using Type = BlockFmhaBwdPipelineV3; };
template<> struct BlockFmhaBwdPipelineDispatcher<      true,      false,      true,       true,      true,          true,          false> { using Type = BlockFmhaBwdPipelineV4; };
template<> struct BlockFmhaBwdPipelineDispatcher<      true,       true,      true,       true,      true,         false,          false> { using Type = BlockFmhaBwdPipelineV5; };
template<> struct BlockFmhaBwdPipelineDispatcher<     false,      false,      true,       true,      true,          true,           true> { using Type = BlockFmhaBwdPipelineV6; };
template<> struct BlockFmhaBwdPipelineDispatcher<      true,      false,      true,       true,      true,         false,          false> { using Type = BlockFmhaBwdPipelineV7; };
template<> struct BlockFmhaBwdPipelineDispatcher<     false,      false,      true,       true,      true,          true,          false> { using Type = BlockFmhaBwdPipelineV8; };
template<> struct BlockFmhaBwdPipelineDispatcher<     false,      false,      true,       true,      true,         false,          false> { using Type = BlockFmhaBwdPipelineV9; };
template<> struct BlockFmhaBwdPipelineDispatcher<      true,      false,      true,      false,      true,          true,          false> { using Type = BlockFmhaBwdPipelineV10; };
template<> struct BlockFmhaBwdPipelineDispatcher<      true,      false,      true,      false,      true,         false,          false> { using Type = BlockFmhaBwdPipelineV11; };
template<> struct BlockFmhaBwdPipelineDispatcher<     false,      false,      true,      false,      true,          true,          false> { using Type = BlockFmhaBwdPipelineV12; };
template<> struct BlockFmhaBwdPipelineDispatcher<     false,      false,      true,      false,      true,         false,          false> { using Type = BlockFmhaBwdPipelineV13; };
template<> struct BlockFmhaBwdPipelineDispatcher<     false,      false,     false,      false,      true,         false,          false> { using Type = BlockFmhaBwdPipelineV14; };
template<> struct BlockFmhaBwdPipelineDispatcher<     false,      false,     false,      false,     false,         false,          false> { using Type = BlockFmhaBwdPipelineV15; };
// clang-format o
} // namespace impl

template <bool QLoadOnce,
          bool QTLoadOnce,
          bool KLoadOnce,
          bool KTLoadOnce,
          bool VLoadOnce,
          bool OGradLoadOnce,
          bool OGradTLoadOnce>
using BlockFmhaBwdPipelineDispatcher = typename impl::
    BlockFmhaBwdPipelineDispatcher<QLoadOnce, QTLoadOnce, KLoadOnce, KTLoadOnce, VLoadOnce, OGradLoadOnce, OGradTLoadOnce>::Type;

} // namespace block
} // namespace tile_program
} // namespace ck
