// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"

#include "ck/tile_program/tile/tile_distribution.hpp"
#include "ck/tile_program/tile/tile_elementwise.hpp"
#include "ck/tile_program/tile/static_tile_gemm_shape.hpp"
#include "ck/tile_program/warp_tile/warp_gemm.hpp"
#include "ck/tile_program/block_tile/block_gemm_impl_cr_as_bs.hpp"

namespace ck {
namespace tile_program {
namespace block {

// Default policy for BlockGemmPipelineV1
// Default policy class should not be templated, put template on member functions instead
struct BlockGemmPipelineV1DefaultPolicy
{
    template <typename ADataType, typename BlockGemmShape>
    __host__ __device__ static constexpr auto MakeALdsBlockDescriptor()
    {
        using namespace ck;

        static constexpr index_t kMPerBlock = BlockGemmShape::kM;
        static constexpr index_t kKPerBlock = BlockGemmShape::kK;

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

    template <typename BDataType, typename BlockGemmShape>
    __host__ __device__ static constexpr auto MakeBLdsBlockDescriptor()
    {
        using namespace ck;

        static constexpr index_t kNPerBlock = BlockGemmShape::kN;
        static constexpr index_t kKPerBlock = BlockGemmShape::kK;

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

    template <typename ADataType, typename BlockGemmShape>
    __host__ __device__ static constexpr auto MakeADramTileDistribution()
    {
        using namespace ck::tile_program;

        static constexpr index_t kMPerBlock = BlockGemmShape::kM;
        static constexpr index_t kKPerBlock = BlockGemmShape::kK;

        constexpr index_t K1 = 16 / sizeof(ADataType);
        constexpr index_t K0 = kKPerBlock / K1;
        constexpr index_t M2 = get_warp_size() / K0;
        constexpr index_t M1 = kBlockSize / get_warp_size();
        constexpr index_t M0 = kMPerBlock / (M2 * M1);

        return make_static_tile_distribution(
            StaticTileDistributionEncoding<Sequence<1>,
                                           Tuple<Sequence<M0, M1, M2>, Sequence<K0, K1>>,
                                           Tuple<Sequence<1>, Sequence<1, 2>>,
                                           Tuple<Sequence<1>, Sequence<2, 0>>,
                                           Sequence<1, 2>,
                                           Sequence<0, 1>>{});
    }

    template <typename BDataType, typename BlockGemmShape>
    __host__ __device__ static constexpr auto MakeBDramTileDistribution()
    {
        using namespace ck::tile_program;

        static constexpr index_t kNPerBlock = BlockGemmShape::kN;
        static constexpr index_t kKPerBlock = BlockGemmShape::kK;

        constexpr index_t K1 = 16 / sizeof(BDataType);
        constexpr index_t K0 = kKPerBlock / K1;
        constexpr index_t N2 = get_warp_size() / K0;
        constexpr index_t N1 = kBlockSize / get_warp_size();
        constexpr index_t N0 = kNPerBlock / (N2 * N1);

        return make_static_tile_distribution(
            StaticTileDistributionEncoding<Sequence<1>,
                                           Tuple<Sequence<N0, N1, N2>, Sequence<K0, K1>>,
                                           Tuple<Sequence<1>, Sequence<1, 2>>,
                                           Tuple<Sequence<1>, Sequence<2, 0>>,
                                           Sequence<1, 2>,
                                           Sequence<0, 1>>{});
    }

    template <typename ADataType_, typename BDataType_, typename CDataType_, index_t kBlockSize>
    __host__ __device__ static constexpr auto GetBlockGemm()
    {
        return BlockGemmV1<ADataType, BDataType, CDataType, kBlockSize, BlockGemmV1DefaultPolicy>{};
    }

}; // struct BlockGemmPipelineV1DefaultPolicy

} // namespace block
} // namespace tile_program
} // namespace ck
