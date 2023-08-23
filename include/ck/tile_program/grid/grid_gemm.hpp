// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace ck {
namespace tile_program {
namespace grid {

template <typename Problem, typename Policy>
struct GridGemm
{
    using ADataType        = typename Problem::ADataType;
    using BDataType        = typename Problem::BDataType;
    using CDataType        = typename Problem::CDataType;
    using AElementFunction = typename Problem::AElementFunction;
    using BElementFunction = typename Problem::BElementFunction;
    using CElementFunction = typename Problem::CElementFunction;

    static constexpr auto kMPerBlock = Policy::kMPerBlock;
    static constexpr auto kNPerBlock = Policy::kNPerBlock;
    static constexpr auto kKPerBlock = Policy::kKPerBlock;

    using BlockGemmPipeline = typename Policy::template BlockGemmPipeline<Problem>;

    template <typename ADramGrid, typename BDramGrid, typename CDramGrid>
    __host__ __device__ void operator()(ProgramServer& ps,
                                        const ADramGrid& a_dram_grid,
                                        const BDramGrid& b_dram_grid,
                                        CDramGrid& c_dram_grid,
                                        ck::index_t M,
                                        ck::index_t N,
                                        ck::index_t K,
                                        const AElementFunction& a_element_func,
                                        const BElementFunction& b_element_func,
                                        const CElementFunction& c_element_func) const
    {
        using namespace ck;
        using namespace ck::tile_program;
        using namespace ck::tile_program::block;

        // divide problem
        const auto id_block = ps.get_block_id();

        const auto num_tile_m = M / kMPerBlock;
        const auto num_tile_n = N / kNPerBlock;

        const auto block2tile = ps(Policy::MakeBlock2TileMap(num_tile_m, num_tile_n));

        const auto id_tile = block2tile(id_block);

        const auto iM = ps.read_first_lane(id_tile.template At<0>() * kMPerBlock);
        const auto iN = ps.read_first_lane(id_tile.template At<1>() * kNPerBlock);

        // A DRAM block window
        auto a_dram_block_window = make_tile_window(
            a_dram_grid, make_tuple(Number<kMPerBlock>{}, Number<kKPerBlock>{}), {iM, 0});

        // B DRAM block window
        auto b_dram_block_window = make_tile_window(
            b_dram_grid, make_tuple(Number<kNPerBlock>{}, Number<kKPerBlock>{}), {iN, 0});

        // Block GEMM pipeline
        constexpr auto block_gemm_pipeline = BlockGemmPipeline{};

        __shared__ char p_smem_char[block_gemm_pipeline.GetStaticLdsSize()];

        const auto acc_block_tile = block_gemm_pipeline(a_dram_block_window,
                                                        a_element_func,
                                                        b_dram_block_window,
                                                        b_element_func,
                                                        K / kKPerBlock,
                                                        p_smem_char);

        // cast to CDataType and apply CElementFunction
        const auto c_block_tile = tile_elementwise_in(
            [&](const auto& acc) { return c_element_func(type_convert<CDataType>(acc)); },
            acc_block_tile);

        // store C
        auto c_dram_window = make_tile_window(
            c_dram_grid, make_tuple(Number<kMPerBlock>{}, Number<kNPerBlock>{}), {iM, iN});

        store_tile(c_dram_window, c_block_tile);
    }
};

} // namespace grid
} // namespace tile_program
} // namespace ck
