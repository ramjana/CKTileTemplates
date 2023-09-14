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
#include "ck/tile_program/block_tile_pipeline/block_gemm_pipeline_agmem_bgmem_creg_v2.hpp"
#include "ck/tile_program/block_tile_pipeline/block_gemm_pipeline_problem.hpp"
#include "ck/tile_program/block_tile/block_gemm_areg_bsmem_creg_v1.hpp"
#include "ck/tile_program/block_tile/block_reduce.hpp"

// S[M0, N0] = Q[M0, K0] * K[N0, K0]
// P[M0, N0] = Softmax(S[M0, N0])
// O[M0, N1] = P[M0, N0] * V[N1, N0]
template <typename QDataType,
          typename KDataType,
          typename SaccDataType,
          typename SMPLComputeDataType,
          typename PDataType,
          typename VDataType,
          typename OaccDataType,
          typename ODataType,
          ck::index_t kBlockSize,
          ck::index_t kM0PerBlock,
          ck::index_t kN0PerBlock,
          ck::index_t kK0PerBlock,
          ck::index_t kN1PerBlock>
struct GemmSoftmaxGemm
{
    // block gemm0 pipeline
    using BlockGemm0Pipeline = ck::tile_program::block::BlockGemmPipelineAGmemBGmemCRegV2<
        ck::tile_program::block::BlockGemmPipelineProblem<
            QDataType,
            KDataType,
            SaccDataType,
            kBlockSize,
            ck::tile_program::TileGemmShape<kM0PerBlock, kN0PerBlock, kK0PerBlock>>,
        ck::tile_program::block::BlockGemmPipelineAGmemBGmemCRegV2DefaultPolicy>;

    // block gemm1
    using BlockGemm1 = ck::tile_program::block::BlockGemmARegBSmemCRegV1<
        ck::tile_program::block::BlockGemmARegBSmemCRegV1Problem<
            PDataType,
            VDataType,
            OaccDataType,
            kBlockSize,
            ck::tile_program::TileGemmShape<kM0PerBlock, kN1PerBlock, kN0PerBlock>>,
        ck::tile_program::block::BlockGemmARegBSmemCRegV1DefaultPolicy>;

#if 0
    // 2d
    __device__ static constexpr auto MakeVLdsBlockDescriptor()
    {
        using namespace ck;

        constexpr index_t kNPerBlock = kN1PerBlock;
        constexpr index_t kKPerBlock = kN0PerBlock;

        constexpr auto b_lds_block_desc =
            make_naive_tensor_descriptor_packed(make_tuple(kNPerBlock, kKPerBlock), Number<32>{});

        return b_lds_block_desc;
    }
#else
    // fake XOR
    __device__ static constexpr auto MakeVLdsBlockDescriptor()
    {
        using namespace ck;

        using BDataType = VDataType;

        constexpr index_t kNPerBlock = kN1PerBlock;
        constexpr index_t kKPerBlock = kN0PerBlock;

        constexpr auto b_lds_block_desc_d1_d2_d3 = make_naive_tensor_descriptor_packed(
            make_tuple(kNPerBlock / 2, 2, kKPerBlock), Number<kKPerBlock>{});

        constexpr index_t kK1 = 16 / sizeof(BDataType);

        constexpr auto b_lds_block_desc_d4_d5_d6 = transform_tensor_descriptor(
            b_lds_block_desc_d1_d2_d3,
            make_tuple(make_xor_transform(make_tuple(kNPerBlock / 2, kKPerBlock), kK1),
                       make_pass_through_transform(2)),
            make_tuple(Sequence<0, 2>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

        constexpr auto b_lds_block_desc_n_k = transform_tensor_descriptor(
            b_lds_block_desc_d4_d5_d6,
            make_tuple(make_merge_transform(make_tuple(kNPerBlock / 2, 2)),
                       make_pass_through_transform(kKPerBlock)),
            make_tuple(Sequence<0, 1>{}, Sequence<2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return b_lds_block_desc_n_k;
    }
#endif

    __device__ static constexpr auto MakeVDramTileDistribution()
    {
        using namespace ck;
        using namespace ck::tile_program;

        using BDataType = VDataType;

        constexpr index_t kNPerBlock = kN1PerBlock;
        constexpr index_t kKPerBlock = kN0PerBlock;

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

    __device__ static constexpr ck::index_t GetStaticLdsSize()
    {
        using namespace ck;

        return math::max(BlockGemm0Pipeline::GetStaticLdsSize(),
                         static_cast<index_t>(MakeVLdsBlockDescriptor().GetElementSpaceSize() *
                                              sizeof(VDataType)));
    }

    __device__ void operator()(const QDataType* q_ptr,
                               const KDataType* k_ptr,
                               const VDataType* v_ptr,
                               ODataType* o_ptr,
                               ck::index_t M0,
                               ck::index_t N0,
                               ck::index_t K0,
                               ck::index_t N1,
                               ck::index_t StrideQ,
                               ck::index_t StrideK,
                               ck::index_t StrideV,
                               ck::index_t StrideO)
    {
        using namespace ck;
        using namespace ck::tile_program;
        using namespace ck::tile_program::block;

        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};

        // FIXME: assume layout Q[M0, K0], K[N0, K0], V[N1, N0], O[M0, N1]
        const auto q_dram_grid = make_naive_tensor_view<AddressSpaceEnum::Global>(
            q_ptr, make_tuple(M0, K0), make_tuple(StrideQ, 1), Number<32>{}, Number<1>{});

        const auto k_dram_grid = make_naive_tensor_view<AddressSpaceEnum::Global>(
            k_ptr, make_tuple(N0, K0), make_tuple(StrideK, 1), Number<32>{}, Number<1>{});

        const auto v_dram_grid = make_naive_tensor_view<AddressSpaceEnum::Global>(
            v_ptr, make_tuple(N1, N0), make_tuple(StrideV, 1), Number<32>{}, Number<1>{});

        // divide problem
        const auto num_tile_n1 = N1 / kN1PerBlock;

        const auto id_block = get_block_id();

        const auto id_tile_m = id_block / num_tile_n1;
        const auto id_tile_n = id_block - id_tile_m * num_tile_n1;

        const auto iM0 = __builtin_amdgcn_readfirstlane(id_tile_m * kM0PerBlock);
        const auto iN1 = __builtin_amdgcn_readfirstlane(id_tile_n * kN1PerBlock);

        // allocate LDS
        __shared__ char smem_ptr[GetStaticLdsSize()];

        // Q DRAM block window
        auto q_dram_block_window = make_tile_window(
            q_dram_grid, make_tuple(Number<kM0PerBlock>{}, Number<kK0PerBlock>{}), {iM0, 0});

        // K DRAM block window
        auto k_dram_block_window = make_tile_window(
            k_dram_grid, make_tuple(Number<kN0PerBlock>{}, Number<kK0PerBlock>{}), {0, 0});

        // Block GEMM0 pipeline
        constexpr auto block_gemm0_pipeline = BlockGemm0Pipeline{};

        // V DRAM window
        auto v_dram_block_window =
            make_tile_window(v_dram_grid,
                             make_tuple(Number<kN1PerBlock>{}, Number<kN0PerBlock>{}),
                             {iN1, 0},
                             MakeVDramTileDistribution());

        // V LDS tensor view: occupies the same LDS allocation as block_gemm0_pipeline
        auto v_lds_block = make_tensor_view<AddressSpaceEnum::Lds>(
            reinterpret_cast<VDataType*>(smem_ptr), MakeVLdsBlockDescriptor());

        auto v_lds_block_window = make_tile_window(
            v_lds_block, make_tuple(Number<kN1PerBlock>{}, Number<kN0PerBlock>{}), {0, 0});

        // Block GEMM1
        constexpr auto block_gemm1 = BlockGemm1{};

        //
        const auto f_max = [](auto e0, auto e1) { return max(e0, e1); };
        const auto f_sum = [](auto e0, auto e1) { return e0 + e1; };

        // infer Sacc, S, P, M, L, Oacc type
        using SaccBlockTileType =
            decltype(block_gemm0_pipeline(q_dram_block_window, k_dram_block_window, 0, nullptr));

        using SBlockTileType = decltype(tile_elementwise_in(
            type_convert<SMPLComputeDataType, SaccDataType>, SaccBlockTileType{}));

        using PBlockTileType = decltype(tile_elementwise_in(type_convert<PDataType, SaccDataType>,
                                                            SaccBlockTileType{}));

        using MLBlockTileType = decltype(block_tile_reduce<SMPLComputeDataType>(
            SBlockTileType{}, Sequence<1>{}, f_max, SMPLComputeDataType{0}));

        using OaccBlockTileType = decltype(block_gemm1(PBlockTileType{}, v_dram_block_window));

        // init Oacc, M, L
        auto o_acc_block_tile = OaccBlockTileType{};
        auto m                = MLBlockTileType{};
        auto l                = MLBlockTileType{};

        tile_elementwise_inout([](auto& e) { e = 0; }, o_acc_block_tile);
        tile_elementwise_inout([](auto& e) { e = NumericLimits<SMPLComputeDataType>::Lowest(); },
                               m);
        tile_elementwise_inout([](auto& e) { e = 0; }, l);

        // loop over Column of S (J loop)
        index_t iN0 = 0;

        do
        {
            // Sacc{j} = Q * K{j}
            const auto s_acc_block_tile = block_gemm0_pipeline(
                q_dram_block_window, k_dram_block_window, K0 / kK0PerBlock, smem_ptr);

            // S{j}
            const auto s_block_tile = tile_elementwise_in(
                type_convert<SMPLComputeDataType, SaccDataType>, s_acc_block_tile);

            // m_local = rowmax(S{j})
            auto m_local = block_tile_reduce<SMPLComputeDataType>(
                s_block_tile, Sequence<1>{}, f_max, NumericLimits<SMPLComputeDataType>::Lowest());

            block_tile_reduce_sync(m_local, f_max);

            // m{j-1}
            const auto m_old = m;

            // m{j}
            tile_elementwise_inout(
                [](auto& m_e, auto m_old_e, auto m_local_e) { m_e = max(m_old_e, m_local_e); },
                m,
                m_old,
                m_local);

            // P{j}
            auto p = make_static_distributed_tensor<SMPLComputeDataType>(
                s_block_tile.GetTileDistribution());

            constexpr auto p_spans = decltype(p)::GetDistributedSpans();

            sweep_tile_span(p_spans[I0], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);

                const auto m_e = m.GetElementFromTileDistributedIndices(i_idx);

                sweep_tile_span(p_spans[I1], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);

                    const auto s_e = s_block_tile.GetElementFromTileDistributedIndices(i_j_idx);

                    const auto p_e = math::exp(s_e - m_e);

                    p.SetElementFromTileDistributedIndices(i_j_idx, p_e);
                });
            });

            // rowsum(P{j})
            auto rowsum_p = block_tile_reduce<SMPLComputeDataType>(
                p, Sequence<1>{}, f_sum, SMPLComputeDataType{0});

            block_tile_reduce_sync(rowsum_p, f_sum);

            // l{j}, O{j}
            sweep_tile_span(p_spans[I0], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);

                const auto m_old_e = m_old.GetElementFromTileDistributedIndices(i_idx);
                const auto m_e     = m.GetElementFromTileDistributedIndices(i_idx);
                const auto l_old_v = l.GetElementFromTileDistributedIndices(i_idx);

                const auto tmp  = math::exp(m_old_e - m_e);
                const auto tmp2 = 1 / tmp;

                auto l_e = tmp * l_old_v + rowsum_p.GetElementFromTileDistributedIndices(i_idx);

                l.SetElementFromTileDistributedIndices(i_idx, l_e);

                sweep_tile_span(p_spans[I1], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);

                    // O{j}
                    const auto o_acc_old_v =
                        o_acc_block_tile.GetElementFromTileDistributedIndices(i_j_idx);

#if 0 // debug
      // this use the same equation from FA v2 paper, but produce -nan
                    const auto o_e = o_old_v * tmp2;
#elif 1
                    // this use different equation from FA v2 paper, but produce correct result
                    (void) tmp2;
                    const auto o_acc_e = o_acc_old_v * tmp;
#endif

                    o_acc_block_tile.SetElementFromTileDistributedIndices(i_j_idx, o_acc_e);
                });
            });

            // type cast P{j}
            const auto p_block_tile =
                tile_elementwise_in(type_convert<PDataType, SMPLComputeDataType>, p);

            // Block GEMM1: Oacc{j} += P{j} * V{j}
            {
                // load V{j}
                const auto v_block_tile = load_tile(v_dram_block_window);

                // wait for block gemm0 pipeline to finish
                block_sync_lds();

                store_tile(v_lds_block_window, v_block_tile);

                // wait for store_tile to finish
                block_sync_lds();

                // Oacc{j} += P{j} * V{j}
                block_gemm1(o_acc_block_tile, p_block_tile, v_lds_block_window);

                // wait for block gemm1 to finish
                block_sync_lds();
            }

            // move tile windows
            move_tile_window(k_dram_block_window, {kN0PerBlock, 0});
            move_tile_window(v_dram_block_window, {0, kN0PerBlock});

            iN0 += kN0PerBlock;

        } while(iN0 < N0);

        // O
        constexpr auto o_spans = decltype(o_acc_block_tile)::GetDistributedSpans();

        sweep_tile_span(o_spans[I0], [&](auto idx0) {
            constexpr auto i_idx = make_tuple(idx0);

            const auto l_e = l.GetElementFromTileDistributedIndices(i_idx);

            const auto tmp = 1 / l_e;

            sweep_tile_span(o_spans[I1], [&](auto idx1) {
                constexpr auto i_j_idx = make_tuple(idx0, idx1);

                const auto o_acc_e = o_acc_block_tile.GetElementFromTileDistributedIndices(i_j_idx);

                const auto o_acc_new_e = o_acc_e * tmp;

                o_acc_block_tile.SetElementFromTileDistributedIndices(i_j_idx, o_acc_new_e);
            });
        });

        // type cast Oacc into O
        const auto o_block_tile =
            tile_elementwise_in(type_convert<ODataType, OaccDataType>, o_acc_block_tile);

        // store O
        auto o_dram_grid = make_naive_tensor_view<AddressSpaceEnum::Global>(
            o_ptr, make_tuple(M0, N1), make_tuple(StrideO, 1), Number<32>{}, Number<1>{});

        auto o_dram_window =
            make_tile_window(o_dram_grid,
                             make_tuple(Number<kM0PerBlock>{}, Number<kN1PerBlock>{}),
                             {iM0, iN1},
                             o_block_tile.GetTileDistribution());

        store_tile(o_dram_window, o_block_tile);
    }
};
