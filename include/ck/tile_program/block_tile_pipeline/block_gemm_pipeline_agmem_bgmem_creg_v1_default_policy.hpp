// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"

#include "ck/tile_program/tile/tile_distribution.hpp"
#include "ck/tile_program/tile/tile_elementwise.hpp"
#include "ck/tile_program/tile/tile_gemm_shape.hpp"
#include "ck/tile_program/warp_tile/warp_gemm.hpp"
#include "ck/tile_program/block_tile/block_gemm_asmem_bsmem_creg_v1.hpp"

namespace ck {
namespace tile_program {
namespace block {

// Default policy for BlockGemmPipelineAGmemBGmemCRegV1
// Default policy class should not be templated, put template on member functions instead
struct BlockGemmPipelineAGmemBGmemCRegV1DefaultPolicy
{
    using BlockGemmPolicy = BlockGemmASmemBSmemCRegV1DefaultPolicy;

#if 0
    // 2d
    template <typename Problem>
    __host__ __device__ static constexpr auto MakeALdsBlockDescriptor()
    {
        using namespace ck;

        constexpr index_t kMPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t kKPerBlock = Problem::BlockGemmShape::kK;

        constexpr auto a_lds_block_desc =
            make_naive_tensor_descriptor_packed(make_tuple(kMPerBlock, kKPerBlock), Number<32>{});

        return a_lds_block_desc;
    }

    // 2d
    template <typename Problem>
    __host__ __device__ static constexpr auto MakeBLdsBlockDescriptor()
    {
        using namespace ck;

        constexpr index_t kNPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t kKPerBlock = Problem::BlockGemmShape::kK;

        constexpr auto b_lds_block_desc =
            make_naive_tensor_descriptor_packed(make_tuple(kNPerBlock, kKPerBlock), Number<32>{});

        return b_lds_block_desc;
    }
#elif 1
    // 3d + padding
    template <typename Problem>
    __host__ __device__ static constexpr auto MakeALdsBlockDescriptor()
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

    // 3d + padding
    template <typename Problem>
    __host__ __device__ static constexpr auto MakeBLdsBlockDescriptor()
    {
        using namespace ck;

        constexpr index_t kNPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t kKPerBlock = Problem::BlockGemmShape::kK;

        constexpr auto b_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(Number<kKPerBlock / 8>{}, Number<kNPerBlock>{}, Number<8>{}),
            make_tuple(Number<(kNPerBlock + 1) * 8>{}, Number<8>{}, Number<1>{}),
            Number<8>{},
            Number<1>{});

        constexpr auto b_lds_block_desc = transform_tensor_descriptor(
            b_lds_block_desc_0,
            make_tuple(make_pass_through_transform(kNPerBlock),
                       make_merge_transform(make_tuple(kKPerBlock / 8, 8))),
            make_tuple(Sequence<1>{}, Sequence<0, 2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return b_lds_block_desc;
    }
#elif 1
    // fake XOR
    template <typename Problem>
    __host__ __device__ static constexpr auto MakeALdsBlockDescriptor()
    {
        using namespace ck;

        using ADataType = remove_cvref_t<typename Problem::ADataType>;

        constexpr index_t kMPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t kKPerBlock = Problem::BlockGemmShape::kK;

        constexpr auto a_lds_block_desc_d1_d2_d3 = make_naive_tensor_descriptor_packed(
            make_tuple(Number<kMPerBlock / 2>{}, Number<2>{}, Number<kKPerBlock>{}),
            Number<kKPerBlock>{});

        constexpr index_t kK1 = 16 / sizeof(ADataType);

        constexpr auto a_lds_block_desc_d4_d5_d6 = transform_tensor_descriptor(
            a_lds_block_desc_d1_d2_d3,
            make_tuple(
                make_xor_transform(make_tuple(Number<kMPerBlock / 2>{}, Number<kKPerBlock>{}), kK1),
                make_pass_through_transform(2)),
            make_tuple(Sequence<0, 2>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

        constexpr auto a_lds_block_desc_m_k = transform_tensor_descriptor(
            a_lds_block_desc_d4_d5_d6,
            make_tuple(make_merge_transform(make_tuple(Number<kMPerBlock / 2>{}, Number<2>{})),
                       make_pass_through_transform(kKPerBlock)),
            make_tuple(Sequence<0, 1>{}, Sequence<2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return a_lds_block_desc_m_k;
    }

    // fake XOR
    template <typename Problem>
    __host__ __device__ static constexpr auto MakeBLdsBlockDescriptor()
    {
        using namespace ck;

        using BDataType = remove_cvref_t<typename Problem::BDataType>;

        constexpr index_t kNPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t kKPerBlock = Problem::BlockGemmShape::kK;

        constexpr auto b_lds_block_desc_d1_d2_d3 = make_naive_tensor_descriptor_packed(
            make_tuple(Number<kNPerBlock / 2>{}, Number<2>{}, Number<kKPerBlock>{}),
            Number<kKPerBlock>{});

        constexpr index_t kK1 = 16 / sizeof(BDataType);

        constexpr auto b_lds_block_desc_d4_d5_d6 = transform_tensor_descriptor(
            b_lds_block_desc_d1_d2_d3,
            make_tuple(
                make_xor_transform(make_tuple(Number<kNPerBlock / 2>{}, Number<kKPerBlock>{}), kK1),
                make_pass_through_transform(2)),
            make_tuple(Sequence<0, 2>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

        constexpr auto b_lds_block_desc_n_k = transform_tensor_descriptor(
            b_lds_block_desc_d4_d5_d6,
            make_tuple(make_merge_transform(make_tuple(Number<kNPerBlock / 2>{}, Number<2>{})),
                       make_pass_through_transform(kKPerBlock)),
            make_tuple(Sequence<0, 1>{}, Sequence<2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return b_lds_block_desc_n_k;
    }
#endif

    template <typename Problem>
    __host__ __device__ static constexpr auto MakeADramTileDistribution()
    {
        using ADataType = remove_cvref_t<typename Problem::ADataType>;

        constexpr index_t kBlockSize = Problem::kBlockSize;

        constexpr index_t kMPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t kKPerBlock = Problem::BlockGemmShape::kK;

        constexpr index_t K1 = 16 / sizeof(ADataType);
        constexpr index_t K0 = kKPerBlock / K1;
        constexpr index_t M2 = get_warp_size() / K0;
#if 1 // coalesce reading for each blocks
        constexpr index_t M1 = kBlockSize / get_warp_size();
        constexpr index_t M0 = kMPerBlock / (M2 * M1);

        return make_static_tile_distribution(
            StaticTileDistributionEncoding<Sequence<1>,
                                           Tuple<Sequence<M0, M1, M2>, Sequence<K0, K1>>,
                                           Tuple<Sequence<1>, Sequence<1, 2>>,
                                           Tuple<Sequence<1>, Sequence<2, 0>>,
                                           Sequence<1, 2>,
                                           Sequence<0, 1>>{});
#else // coalesce reading for each warps
        constexpr index_t M0 = kBlockSize / get_warp_size();
        constexpr index_t M1 = kMPerBlock / (M2 * M0);

        return make_static_tile_distribution(
            StaticTileDistributionEncoding<Sequence<1>,
                                           Tuple<Sequence<M0, M1, M2>, Sequence<K0, K1>>,
                                           Tuple<Sequence<1>, Sequence<1, 2>>,
                                           Tuple<Sequence<0>, Sequence<2, 0>>,
                                           Sequence<1, 2>,
                                           Sequence<1, 1>>{});
#endif
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto MakeBDramTileDistribution()
    {
        using BDataType = remove_cvref_t<typename Problem::BDataType>;

        constexpr index_t kBlockSize = Problem::kBlockSize;

        constexpr index_t kNPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t kKPerBlock = Problem::BlockGemmShape::kK;

        constexpr index_t K1 = 16 / sizeof(BDataType);
        constexpr index_t K0 = kKPerBlock / K1;
        constexpr index_t N2 = get_warp_size() / K0;
#if 1 // coalesce reading for each blocks
        constexpr index_t N1 = kBlockSize / get_warp_size();
        constexpr index_t N0 = kNPerBlock / (N2 * N1);

        return make_static_tile_distribution(
            StaticTileDistributionEncoding<Sequence<1>,
                                           Tuple<Sequence<N0, N1, N2>, Sequence<K0, K1>>,
                                           Tuple<Sequence<1>, Sequence<1, 2>>,
                                           Tuple<Sequence<1>, Sequence<2, 0>>,
                                           Sequence<1, 2>,
                                           Sequence<0, 1>>{});
#else // coalesce reading for each warps
        constexpr index_t N0 = kBlockSize / get_warp_size();
        constexpr index_t N1 = kNPerBlock / (N2 * N0);

        return make_static_tile_distribution(
            StaticTileDistributionEncoding<Sequence<1>,
                                           Tuple<Sequence<N0, N1, N2>, Sequence<K0, K1>>,
                                           Tuple<Sequence<1>, Sequence<1, 2>>,
                                           Tuple<Sequence<0>, Sequence<2, 0>>,
                                           Sequence<1, 2>,
                                           Sequence<1, 1>>{});
#endif
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto GetBlockGemm()
    {
        return BlockGemmASmemBSmemCRegV1<Problem, BlockGemmPolicy>{};
    }

    template <typename Problem>
    __host__ static constexpr auto ScheduleReductionLoop()
    {
    }

    template <typename Problem>
    __device__ static constexpr auto ScheduleReductionLoop()
    {
        constexpr index_t BlockSize = Problem::kBlockSize;

        constexpr auto MPerBlock = Problem::BlockGemmShape::kM;
        constexpr auto NPerBlock = Problem::BlockGemmShape::kN;
        constexpr auto KPerBlock = Problem::BlockGemmShape::kK;

        constexpr auto config = BlockGemmPolicy::template GetWarpGemmMWarpNWarp<Problem>();

        using WarpGemm = remove_cvref_t<decltype(config.template At<0>())>;

        constexpr auto MPerWarp = WarpGemm::kM;
        constexpr auto NPerWarp = WarpGemm::kN;
        constexpr auto KPerWarp = WarpGemm::kK;

        constexpr index_t MWarp = config.template At<1>();
        constexpr index_t NWarp = config.template At<2>();

        constexpr index_t MIterPerWarp = MPerBlock / (MWarp * MPerWarp);
        constexpr index_t NIterPerWarp = NPerBlock / (NWarp * NPerWarp);
        constexpr index_t KIterPerWarp = KPerBlock / KPerWarp;

        constexpr index_t NumInstPerWarpGemm = 2;

        constexpr index_t TileSize = (MPerBlock + NPerBlock) * KPerBlock;

        constexpr index_t LdsStoreScalarPerVector   = 8;
        constexpr index_t GlobalLoadScalarPerVector = 8;

        constexpr index_t NumLdsLoad    = KIterPerWarp * (MIterPerWarp + NIterPerWarp);
        constexpr index_t NumLdsStore   = TileSize / BlockSize / LdsStoreScalarPerVector;
        constexpr index_t NumGlobalLoad = TileSize / BlockSize / GlobalLoadScalarPerVector;
        constexpr index_t NumMfma = KIterPerWarp * MIterPerWarp * NIterPerWarp * NumInstPerWarpGemm;

        static_assert(NumLdsStore <= NumGlobalLoad);

        enum SchedGroupMask
        {
            DS_READ   = 1u << 8,
            MFMA      = 1u << 3,
            DS_WRITE  = 1u << 9,
            VMEM_READ = 1u << 5,
        };

        // pipeline #1
        static_for<0, KIterPerWarp * 2, 1>{}([](auto) {
            __builtin_amdgcn_sched_group_barrier(DS_READ, NumLdsLoad / KIterPerWarp / 2, 1);
            __builtin_amdgcn_sched_group_barrier(MFMA, NumMfma / KIterPerWarp / 2, 1);
        });
    }
};

} // namespace block
} // namespace tile_program
} // namespace ck
