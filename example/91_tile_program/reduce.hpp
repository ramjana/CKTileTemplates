// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"

#include "tile_program.hpp"
#include "ck/tile_program/tile/tile_distribution.hpp"
#include "ck/tile_program/tile/tile_window.hpp"
#include "ck/tile_program/tile/load_tile.hpp"
#include "ck/tile_program/tile/store_tile.hpp"
#include "ck/tile_program/tile/tile_elementwise.hpp"
#include "ck/tile_program/warp_tile/warp_reduce.hpp"

template <typename ADataType,
          typename AccDataType,
          typename BDataType,
          ck::index_t kBlockSize,
          ck::index_t kMPerBlock,
          ck::index_t kNPerBlock>
struct Reduce
{
    __host__ __device__ static constexpr auto MakeABlockTileDistribution()
    {
        using namespace ck;
        using namespace ck::tile_program;

        return make_static_tile_distribution(
            StaticTileDistributionEncoding<Sequence<>,
                                           Tuple<Sequence<2, 2, 4, 2, 4>, Sequence<2, 2, 32>>,
                                           Tuple<Sequence<1, 2>, Sequence<1, 2>>,
                                           Tuple<Sequence<1, 1>, Sequence<3, 2>>,
                                           Sequence<1, 2, 1, 1>,
                                           Sequence<0, 0, 2, 4>>{});
    }

    __host__ __device__ void operator()(
        ProgramServer& ps, const ADataType* p_a, BDataType* p_b, ck::index_t M, ck::index_t N) const
    {
        using namespace ck;
        using namespace ck::tile_program;
        using namespace ck::tile_program::warp;

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

        auto f_max = [](AccDataType acc, ADataType a) { return acc > a ? acc : a; };

#if 0
        // B tile
        // FIXME: block_tile_reduce_in
        auto b_block_tile = decltype(warp_tile_reduce_in<AccDataType>(
            load_tile(a_block_window), Sequence<1>{}, f_max, NumericLimits<ADataType>::Min())){};

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

        // B
        const auto b_m = make_naive_tensor_view_packed<AddressSpaceEnum::Global>(
            p_b, make_tuple(M), Number<32>{});

        // B window
        auto b_block_window =
            make_tile_window(b_block_tile, make_tuple(Number<kMPerBlock>{}), {iM});

        // store B
        store_tile(b_block_window, b_block_tile);
#else
        (void)p_b;
        (void)a_block_window;
        (void)f_max;

        constexpr auto in_dstr = ck::tile_program::detail::make_reduce_tile_distribution_encoding(
            a_block_window.GetTileDistribution().GetStaticTileDistributionEncoding(),
            Sequence<0>{});

        if(ProgramServer::get_block_id() == 0 && ProgramServer::get_thread_id() == 0)
        {
            print(in_dstr);
            printf("\n");
        }
#endif
    }
};
