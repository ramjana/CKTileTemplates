// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"

#include "tile_program.hpp"
#include "ck/tile_program/tile/tile_distribution.hpp"
#include "ck/tile_program/tile/tile_elementwise.hpp"

template <typename ADataType,
          typename AccDataType,
          typename BDataType,
          ck::index_t kBlockSize,
          ck::index_t kMPerBlock,
          ck::index_t kNPerBlock>
struct Reduce
{
    __host__ __device__ void operator()(
        ProgramServer& ps, const ADataType* p_a, BDataType* p_b, ck::index_t M, ck::index_t N) const
    {
#if 0
        (void)ps;
        (void)p_a;
        (void)p_b;
        (void)M;
        (void)N;
#else
        using namespace ck;

        const auto a_m_n = make_naive_tensor_view<AddressSpaceEnum::Global>(
            p_a, make_tuple(M, N), make_tuple(N, 1), Number<32>{}, Number<1>{});

        const auto id_block = ps.get_block_id();

        const auto num_tile_m = ps.read_first_lane(M / kMPerBlock);

        const auto block2tile = ps(make_cluster_descriptor(make_tuple(num_tile_m)));

        const auto i_m = block2tile.CalculateBottomIndex(make_multi_index(id_block));

        const auto iM = ps.read_first_lane(i_m[0]) * kMPerBlock;

        // A window
        auto a_block_window =
            make_tile_window(a_m_n,
                             make_tuple(Number<kMPerBlock>{}, Number<kNPerBlock>{}),
                             {iM, 0},
                             MakeABlockTileDistribution());

        // B tile
        auto b_block_tile =
            decltype(block_tile_reduce_in(a_block_tile, math::max<AccDataType, ADataType>)){};

        // init B tile
        tile_elementwise_inout([](auto& b) { b = NumericLimits<ADataType>::Min(); }, b_block_tile);

        // loop
        index_t iN = 0;

        do
        {
            const auto a_block_tile = load_tile(a_block_window);

            block_tile_reduce_inout(b_block_tile, a_block_tile, math::max<AccDataType, ADataType>);

            move_tile_window(a_block_window, {0, kNPerBlock});

            iN += kNPerBlock;

        } while(iN < N);

        // B window
        auto b_block_window = make_tile_window(b_m, make_tuple(Number<kMPerBlock>{}), {iM});

        // store B
        store_tile(b_block_window, b_block_tile);
#endif
    }
};
