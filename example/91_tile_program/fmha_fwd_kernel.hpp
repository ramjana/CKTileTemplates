// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor/tensor_view.hpp"
#include "ck/tile_program/tile/tile_window.hpp"

// S[seqlen_q, seqlen_k] = Q[seqlen_q, hdim_q] * K[seqlen_k, hdim_q]
// P[seqlen_q, seqlen_k] = Softmax(S[seqlen_q, seqlen_k])
// O[seqlen_q, hdim_v] = P[seqlen_q, seqlen_k] * V[hdim_v, seqlen_k]

#define C_LOG2E 1.44269504088896340736 // log2(e)

template <typename TilePartitioner_, typename FmhaPipeline_, typename EpiloguePipeline_>
struct FmhaFwdKernel
{
    using TilePartitioner                   = ck::remove_cvref_t<TilePartitioner_>;
    using FmhaPipeline                      = ck::remove_cvref_t<FmhaPipeline_>;
    using EpiloguePipeline                  = ck::remove_cvref_t<EpiloguePipeline_>;
    static constexpr ck::index_t kBlockSize = FmhaPipeline::kBlockSize;

    using QDataType = ck::remove_cvref_t<typename FmhaPipeline::QDataType>;
    using KDataType = ck::remove_cvref_t<typename FmhaPipeline::KDataType>;
    using VDataType = ck::remove_cvref_t<typename FmhaPipeline::VDataType>;
    using ODataType = ck::remove_cvref_t<typename FmhaPipeline::ODataType>;

    struct Kargs
    {
        const QDataType* q_ptr;
        const KDataType* k_ptr;
        const VDataType* v_ptr;
        ODataType* o_ptr;

        ck::index_t seqlen_q;
        ck::index_t seqlen_k;
        ck::index_t hdim_q;
        ck::index_t hdim_v;

        float scale;

        ck::index_t stride_q;
        ck::index_t stride_k;
        ck::index_t stride_v;
        ck::index_t stride_o;

        ck::index_t nhead_stride_q;
        ck::index_t nhead_stride_k;
        ck::index_t nhead_stride_v;
        ck::index_t nhead_stride_o;

        // attributes for batch mode
        ck::index_t batch_stride_q = 0;
        ck::index_t batch_stride_k = 0;
        ck::index_t batch_stride_v = 0;
        ck::index_t batch_stride_o = 0;

        // attributes for group mode. only support shape=[1, seqlen, nhead, hdim] in group
        // mode
        const ck::index_t* seqstart_q_ptr = nullptr;
        const ck::index_t* seqstart_k_ptr = nullptr;
        const ck::index_t* seqlen_k_ptr   = nullptr;
    };

    // initialize kernel arguments for batch mode
    __host__ static constexpr Kargs MakeKargs(const void* q_ptr,
                                              const void* k_ptr,
                                              const void* v_ptr,
                                              void* o_ptr,
                                              ck::index_t seqlen_q,
                                              ck::index_t seqlen_k,
                                              ck::index_t hdim_q,
                                              ck::index_t hdim_v,
                                              float scale,
                                              ck::index_t stride_q,
                                              ck::index_t stride_k,
                                              ck::index_t stride_v,
                                              ck::index_t stride_o,
                                              ck::index_t nhead_stride_q,
                                              ck::index_t nhead_stride_k,
                                              ck::index_t nhead_stride_v,
                                              ck::index_t nhead_stride_o,
                                              ck::index_t batch_stride_q,
                                              ck::index_t batch_stride_k,
                                              ck::index_t batch_stride_v,
                                              ck::index_t batch_stride_o)
    {
        Kargs kargs;

        kargs.q_ptr = reinterpret_cast<const QDataType*>(q_ptr);
        kargs.k_ptr = reinterpret_cast<const KDataType*>(k_ptr);
        kargs.v_ptr = reinterpret_cast<const VDataType*>(v_ptr);
        kargs.o_ptr = reinterpret_cast<ODataType*>(o_ptr);

        kargs.seqlen_q = seqlen_q;
        kargs.seqlen_k = seqlen_k;
        kargs.hdim_q   = hdim_q;
        kargs.hdim_v   = hdim_v;

        kargs.scale = scale;

        kargs.stride_q = stride_q;
        kargs.stride_k = stride_k;
        kargs.stride_v = stride_v;
        kargs.stride_o = stride_o;

        kargs.nhead_stride_q = nhead_stride_q;
        kargs.nhead_stride_k = nhead_stride_k;
        kargs.nhead_stride_v = nhead_stride_v;
        kargs.nhead_stride_o = nhead_stride_o;

        kargs.batch_stride_q = batch_stride_q;
        kargs.batch_stride_k = batch_stride_k;
        kargs.batch_stride_v = batch_stride_v;
        kargs.batch_stride_o = batch_stride_o;

        return kargs;
    }

    // initialize kernel arguments for group mode
    __host__ static constexpr Kargs MakeKargs(const void* q_ptr,
                                              const void* k_ptr,
                                              const void* v_ptr,
                                              void* o_ptr,
                                              const void* seqstart_q_ptr,
                                              const void* seqstart_k_ptr,
                                              const void* seqlen_k_ptr,
                                              ck::index_t max_seqlen_q,
                                              ck::index_t hdim_q,
                                              ck::index_t hdim_v,
                                              float scale,
                                              ck::index_t stride_q,
                                              ck::index_t stride_k,
                                              ck::index_t stride_v,
                                              ck::index_t stride_o,
                                              ck::index_t nhead_stride_q,
                                              ck::index_t nhead_stride_k,
                                              ck::index_t nhead_stride_v,
                                              ck::index_t nhead_stride_o)
    {
        Kargs kargs;

        kargs.q_ptr = reinterpret_cast<const QDataType*>(q_ptr);
        kargs.k_ptr = reinterpret_cast<const KDataType*>(k_ptr);
        kargs.v_ptr = reinterpret_cast<const VDataType*>(v_ptr);
        kargs.o_ptr = reinterpret_cast<ODataType*>(o_ptr);

        kargs.seqlen_q = max_seqlen_q;
        kargs.seqlen_k = 0; // will be set inside the kernel
        kargs.hdim_q   = hdim_q;
        kargs.hdim_v   = hdim_v;

        kargs.scale = scale;

        kargs.stride_q = stride_q;
        kargs.stride_k = stride_k;
        kargs.stride_v = stride_v;
        kargs.stride_o = stride_o;

        kargs.nhead_stride_q = nhead_stride_q;
        kargs.nhead_stride_k = nhead_stride_k;
        kargs.nhead_stride_v = nhead_stride_v;
        kargs.nhead_stride_o = nhead_stride_o;

        kargs.seqstart_q_ptr = reinterpret_cast<const ck::index_t*>(seqstart_q_ptr);
        kargs.seqstart_k_ptr = reinterpret_cast<const ck::index_t*>(seqstart_k_ptr);
        kargs.seqlen_k_ptr   = reinterpret_cast<const ck::index_t*>(seqlen_k_ptr);

        return kargs;
    }

    __host__ static constexpr auto GridSize(ck::index_t batch_size_,
                                            ck::index_t nhead_,
                                            ck::index_t seqlen_q_,
                                            ck::index_t hdim_v_)
    {
        return TilePartitioner::GridSize(batch_size_, nhead_, seqlen_q_, hdim_v_);
    }

    __host__ static constexpr auto BlockSize() { return dim3(kBlockSize); }

    __host__ __device__ static constexpr ck::index_t GetSmemSize()
    {
        return ck::math::max(FmhaPipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
    }

    __device__ void operator()(Kargs kargs) const
    {
        using namespace ck;
        using namespace ck::tile_program;
        using namespace ck::tile_program::block;

        // allocate LDS
        __shared__ char smem_ptr[GetSmemSize()];

        // divide problem
        const auto [i_tile_m, i_tile_n, i_nhead, i_batch] =
            TilePartitioner{}(kargs.seqlen_q, kargs.hdim_v);

        const index_t i_m0 = __builtin_amdgcn_readfirstlane(i_tile_m * FmhaPipeline::kM0);
        const index_t i_n1 = __builtin_amdgcn_readfirstlane(i_tile_n * FmhaPipeline::kN1);

        const bool in_batch_mode =
            (kargs.seqstart_q_ptr == nullptr || kargs.seqstart_k_ptr == nullptr);

        const index_t query_start = (in_batch_mode ? 0 : kargs.seqstart_q_ptr[i_batch]);
        const index_t key_start   = (in_batch_mode ? 0 : kargs.seqstart_k_ptr[i_batch]);

        const index_t batch_offset_q =
            (in_batch_mode ? i_batch * kargs.batch_stride_q : query_start * kargs.stride_q);
        const index_t batch_offset_k =
            (in_batch_mode ? i_batch * kargs.batch_stride_k : key_start * kargs.stride_k);
        const index_t batch_offset_v = (in_batch_mode ? i_batch * kargs.batch_stride_v : 0);
        const index_t batch_offset_o =
            (in_batch_mode ? i_batch * kargs.batch_stride_o : query_start * kargs.stride_o);

        // get real # queries & # keys under group mode
        if(!in_batch_mode)
        {
            const auto adjusted_seqstart_q_ptr = kargs.seqstart_q_ptr + i_batch;
            kargs.seqlen_q = adjusted_seqstart_q_ptr[1] - adjusted_seqstart_q_ptr[0];

            if(kargs.seqlen_k_ptr != nullptr)
            {
                kargs.seqlen_k = kargs.seqlen_k_ptr[i_batch];
            }
            else
            {
                const auto adjusted_seqstart_k_ptr = kargs.seqstart_k_ptr + i_batch;
                kargs.seqlen_k = adjusted_seqstart_k_ptr[1] - adjusted_seqstart_k_ptr[0];
            }
        }

        // for simplicity, batch stride we just modify the pointer
        const QDataType* q_ptr = kargs.q_ptr + i_nhead * kargs.nhead_stride_q + batch_offset_q;
        const KDataType* k_ptr = kargs.k_ptr + i_nhead * kargs.nhead_stride_k + batch_offset_k;
        const VDataType* v_ptr =
            kargs.v_ptr + i_nhead * kargs.nhead_stride_v + key_start + batch_offset_v;
        ODataType* o_ptr = kargs.o_ptr + i_nhead * kargs.nhead_stride_o + batch_offset_o;

        // Q/K/V DRAM and DRAM window
        // FIXME: assume layout Q[seqlen_q, hdim_q], K[seqlen_k, hdim_q], V[hdim_v, seqlen_k],
        const auto q_dram = make_naive_tensor_view<AddressSpaceEnum::Global>(
            q_ptr,
            make_tuple(kargs.seqlen_q, kargs.hdim_q),
            make_tuple(kargs.stride_q, 1),
            Number<32>{},
            Number<1>{});

        const auto k_dram = make_naive_tensor_view<AddressSpaceEnum::Global>(
            k_ptr,
            make_tuple(kargs.seqlen_k, kargs.hdim_q),
            make_tuple(kargs.stride_k, 1),
            Number<32>{},
            Number<1>{});

        const auto v_dram = make_naive_tensor_view<AddressSpaceEnum::Global>(
            v_ptr,
            make_tuple(kargs.hdim_v, kargs.seqlen_k),
            make_tuple(kargs.stride_v, 1),
            Number<32>{},
            Number<1>{});

        auto q_dram_window = make_tile_window(
            q_dram,
            [&]() {
                if constexpr(FmhaPipeline::kQLoadOnce)
                    return make_tuple(Number<FmhaPipeline::kM0>{},
                                      Number<FmhaPipeline::kK0BlockLength>{});
                else
                    return make_tuple(Number<FmhaPipeline::kM0>{}, Number<FmhaPipeline::kK0>{});
            }(),
            {i_m0, 0});

        auto k_dram_window = make_tile_window(
            k_dram, make_tuple(Number<FmhaPipeline::kN0>{}, Number<FmhaPipeline::kK0>{}), {0, 0});

        auto v_dram_window =
            make_tile_window(v_dram,
                             make_tuple(Number<FmhaPipeline::kN1>{}, Number<FmhaPipeline::kK1>{}),
                             {i_n1, 0});

        auto o_acc_tile = FmhaPipeline{}(q_dram_window,
                                         k_dram_window,
                                         v_dram_window,
                                         kargs.scale,
                                         kargs.seqlen_k / FmhaPipeline::kN0,
                                         kargs.hdim_q / FmhaPipeline::kK0,
                                         smem_ptr);

        // O DRAM and O DRAM window
        auto o_dram = make_naive_tensor_view<AddressSpaceEnum::Global>(
            o_ptr,
            make_tuple(kargs.seqlen_q, kargs.hdim_v),
            make_tuple(kargs.stride_o, 1),
            Number<32>{},
            Number<1>{});

        auto o_dram_window =
            make_tile_window(o_dram,
                             make_tuple(Number<FmhaPipeline::kM0>{}, Number<FmhaPipeline::kN1>{}),
                             {i_m0, i_n1});

        EpiloguePipeline{}(o_dram_window, o_acc_tile);
    }
};
