// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

//FileName -> block_gemm_pipeline_agmem_bgmem_creg_v2_policy.hpp
#pragma once

#include "ck/tile_program/block_tile_pipeline/block_gemm_pipeline_agmem_bgmem_creg_v1.hpp"

namespace ck {
namespace tile_program {
namespace block {

// Default policy for BlockGemmPipelineAGmemBGmemCRegV2
// Default policy class should not be templated, put template on member functions instead
// NOTE: policy should be binded to its corresponding operation. It's just a coincidence that
//   BlockGemmPipelineAGmemBGmemCRegV2DefaultPolicy is the same as
//   BlockGemmPipelineAGmemBGmemCRegV1DefaultPolicy
using BlockGemmPipelineAGmemBGmemCRegV2DefaultPolicy =
    BlockGemmPipelineAGmemBGmemCRegV1DefaultPolicy;

// NOTE: Assume A is K-Major
struct BlockGemmPipelineAGmemBGmemCRegV2SkipALdsPolicy : BlockGemmPipelineAGmemBGmemCRegV2DefaultPolicy
{
    // TODO: More general API name called MakeABlockDescriptor, It could be Reg/Lds.
    // With general API name we can implement various policy with inherit and overload.
    template <typename Problem>
    __host__ __device__ static constexpr auto MakeARegBlockDescriptor()
    {
        using namespace ck;

        constexpr index_t kMPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t kKPerBlock = Problem::BlockGemmShape::kK;

        constexpr auto a_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(Number<kKPerBlock / 8>{}, Number<kMPerBlock>{}, Number<8>{}),
            make_tuple(Number<(kMPerBlock + 1) * 8>{}, Number<8>{}, Number<1>{}),
            Number<8>{},
            Number<1>{});

        constexpr auto a_lds_block_desc = transform_tensor_descriptor(
            a_lds_block_desc_0,
            make_tuple(make_pass_through_transform(kMPerBlock),
                       make_merge_transform(make_tuple(kKPerBlock / 8, 8))),
            make_tuple(Sequence<1>{}, Sequence<0, 2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return a_lds_block_desc;
    }

    // kK = KRepeat * kABKLane * KPerRead
    // kM = MRepeat * MWarp * Mma.kAMLane

    // e.g. Mma = V_MFMA_F32_32x32x8F16, kM = 128, kK = 32, blocksize = 256, 
    // MRepeat = 1, MWarp = 4,  Mma.kAMLane = 32;
    // KRepeat = 2, kABKLane = 2, KPerRead = 8
    template <typename Problem>
    __host__ __device__ static constexpr auto MakeADramTileDistribution()
    {
        using ADataType = remove_cvref_t<typename Problem::ADataType>;
        
        constexpr auto BlockGemm = GetBlockGemm<Problem>();
        constexpr auto config = decltype(BlockGemm)::BlockGemmPolicy::template GetWarpGemmMWarpNWarp<Problem>();
        
        using WG = remove_cvref_t<decltype(config.template At<0>())>;
        
        constexpr index_t MWarp = config.template At<1>();

        constexpr index_t kBlockSize = Problem::kBlockSize;

        constexpr index_t kMPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t kKPerBlock = Problem::BlockGemmShape::kK;

        // Try replace with
        // detail::make_embed_tile_distribution_encoding(
        // a_block_outer_dstr_encoding, typename WG::AWarpDstrEncoding{});
        constexpr index_t K2 = 16 / sizeof(ADataType);
        constexpr index_t K1 = WG::WarpGemmAttribute::Impl::kABKLane;
        constexpr index_t K0 = kKPerBlock / (K1 * K2);

        constexpr index_t M2 = WG::WarpGemmAttribute::Impl::kAMLane;
        constexpr index_t M1 = MWarp;
        constexpr index_t M0 = kMPerBlock / (M2 * M1);

        return make_static_tile_distribution(
            StaticTileDistributionEncoding<Sequence<1>,
                                           Tuple<Sequence<M0, M1, M2>, Sequence<K0, K1, K2>>,
                                           Tuple<Sequence<1>, Sequence<1, 2>>,
                                           Tuple<Sequence<1>, Sequence<2, 1>>,
                                           Sequence<2, 1, 2>,
                                           Sequence<0, 0, 2>>{});
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto GetBlockGemm()
    {
        using BlockGemmPolicy = BlockGemmARegBSmemCRegV1DefaultPolicy;

        return BlockGemmARegBSmemCRegV1<Problem, BlockGemmPolicy>{};
    }
};


} // namespace block
} // namespace tile_program
} // namespace ck
