// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"
#include "ck/tile_program/block_tensor_distribution.hpp"

namespace ck {
namespace tile_program {
namespace block {

struct WarpGemm
{
};

struct WarpGemmXdlFp16M32N32K8 : public WarpGemm
{
    using ADataType = ck::half_t;
    using BDataType = ck::half_t;
    using CDataType = float;

    static constexpr index_t AMLane     = 32;
    static constexpr index_t BNLane     = 32;
    static constexpr index_t ABKLane    = 2;
    static constexpr index_t ABKPerLane = 4;

    static constexpr index_t CMLane     = 2;
    static constexpr index_t CNLane     = 32;
    static constexpr index_t CM0PerLane = 4;
    static constexpr index_t CM1PerLane = 4;

    // FIXME: implement hierarical distribution and then reimplement this
    using AWarpDstr = decltype(make_static_block_tensor_distribution(
        Sequence<1>{},
        make_tuple(Sequence<AMLane>{}, Sequence<ABKLane, ABKPerLane>{}),
        Sequence<0>{},
        Sequence<0>{},
        Sequence<2, 1>{},
        Sequence<0, 0>{},
        Sequence<2>{},
        Sequence<1>{}));

    // FIXME: implement hierarical distribution and then reimplement this
    using BWarpDstr = decltype(make_static_block_tensor_distribution(
        Sequence<1>{},
        make_tuple(Sequence<BNLane>{}, Sequence<ABKLane, ABKPerLane>{}),
        Sequence<0>{},
        Sequence<0>{},
        Sequence<2, 1>{},
        Sequence<0, 0>{},
        Sequence<2>{},
        Sequence<1>{}));

    // FIXME: implement hierarical distribution and then reimplement this
    using CWarpDstr = decltype(make_static_block_tensor_distribution(
        Sequence<1>{},
        make_tuple(Sequence<CM0PerLane, CMLane, CM1PerLane>{}, Sequence<CNLane>{}),
        Sequence<0>{},
        Sequence<0>{},
        Sequence<1, 2>{},
        Sequence<1, 0>{},
        Sequence<1, 1>{},
        Sequence<0, 2>{}));

    using AWarpTensor = StaticBlockDistributedTensor<ADataType, AWarpDstr>;
    using BWarpTensor = StaticBlockDistributedTensor<BDataType, BWarpDstr>;
    using CWarpTensor = StaticBlockDistributedTensor<CDataType, CWarpDstr>;

    __device__ void operator()(CWarpTensor& c, const AWarpTensor& a, const BWarpTensor& b) const
    {
        using AVec = typename vector_type<ADataType, AWarpTensor::GetThreadBufferSize()>::type;
        using BVec = typename vector_type<BDataType, BWarpTensor::GetThreadBufferSize()>::type;
        using CVec = typename vector_type<CDataType, CWarpTensor::GetThreadBufferSize()>::type;

        constexpr auto I0 = Number<0>{};

        const auto a_vec = a.GetThreadBuffer().template GetAsType<AVec>(I0);
        const auto b_vec = b.GetThreadBuffer().template GetAsType<BVec>(I0);
        auto c_vec       = c.GetThreadBuffer().template GetAsType<CVec>(I0);

        c_vec = __builtin_amdgcn_mfma_f32_32x32x8f16(a_vec, b_vec, c_vec, 0, 0, 0);

        c.GetThreadBuffer().template SetAsType<CVec>(I0, c_vec);
    }

    __device__ auto operator()(const AWarpTensor& a, const BWarpTensor& b) const
    {
        CWarpTensor c;

        c.Initialize(0);

        operator()(c, a, b);

        return c;
    }
};

template <typename CBlockTensor, typename ABlockWindowTmp, typename BBlockWindowTmp>
__device__ void block_tile_gemm(CBlockTensor& c_block_tensor,
                                const ABlockWindowTmp& a_block_window_tmp,
                                const BBlockWindowTmp& b_block_window_tmp)
{
    // FIXME: use heuristic to choose paramters and WarpGEMM
    constexpr index_t MWarp = 2;
    constexpr index_t NWarp = 2;

    constexpr index_t MXdlPerWarp = 2;
    constexpr index_t NXdlPerWarp = 2;
    constexpr index_t KXdlPerWarp = 4;

    using WG = WarpGemmXdlFp16M32N32K8;

    // FIXME: create block dstr from existing wave dstr
    constexpr auto a_block_dstr = make_static_block_tensor_distribution(
        Sequence<NWarp, 1>{},
        make_tuple(Sequence<MXdlPerWarp, MWarp, WG::AMLane>{},
                   Sequence<KXdlPerWarp, WG::ABKLane, WG::ABKPerLane>{}),
        Sequence<1, 0, 0>{},
        Sequence<1, 0, 1>{},
        Sequence<2, 1>{},
        Sequence<1, 2>{},
        Sequence<1, 2, 2>{},
        Sequence<0, 0, 2>{});

    // FIXME: create block dstr from existing wave dstr
    constexpr auto b_block_dstr = make_static_block_tensor_distribution(
        Sequence<MWarp, 1>{},
        make_tuple(Sequence<NXdlPerWarp, NWarp, WG::BNLane>{},
                   Sequence<KXdlPerWarp, WG::ABKLane, WG::ABKPerLane>{}),
        Sequence<0, 1, 0>{},
        Sequence<0, 1, 1>{},
        Sequence<2, 1>{},
        Sequence<1, 2>{},
        Sequence<1, 2, 2>{},
        Sequence<0, 0, 2>{});

    // FIXME: create block dstr from existing wave dstr
    constexpr auto c_block_dstr = make_static_block_tensor_distribution(
        Sequence<1>{},
        make_tuple(Sequence<MXdlPerWarp, MWarp, WG::CM0PerLane, WG::CMLane, WG::CM1PerLane>{},
                   Sequence<NXdlPerWarp, NWarp, WG::CNLane>{}),
        Sequence<1, 2, 0>{},
        Sequence<1, 1, 0>{},
        Sequence<1, 2>{},
        Sequence<3, 2>{},
        Sequence<1, 2, 1, 1>{},
        Sequence<0, 0, 2, 4>{});

    static_assert(is_same_v<remove_cvref_t<decltype(c_block_dstr)>,
                            remove_cvref_t<decltype(CBlockTensor::GetBlockDistribution())>>,
                  "wrong!");

    // construct A/B-block-window from A/B-block-distribution
    auto a_block_window = make_block_window(a_block_window_tmp.GetBottomTensorView(),
                                            a_block_window_tmp.GetBlockWindowOrigin(),
                                            a_block_dstr);

    auto b_block_window = make_block_window(b_block_window_tmp.GetBottomTensorView(),
                                            b_block_window_tmp.GetBlockWindowOrigin(),
                                            b_block_dstr);

    // hot loop:
    static_for<0, KXdlPerWarp, 1>{}([&](auto kIter) {
        static_for<0, MXdlPerWarp, 1>{}([&](auto mIter) {
            // read A warp tensor
            WG::AWarpTensor a_warp_tensor;

            a_warp_tensor.GetThreadBuffer() =
                detail::load_sliced_thread_data_from_block_tensor_window(
                    a_block_window,
                    MultiIndex<3>{mIter, kIter, 0},
                    Sequence<1, 1, WG::ABKPerLane>{});

            static_for<0, NXdlPerWarp, 1>{}([&](auto nIter) {
                // read B warp tensor
                WG::BWarpTensor b_warp_tensor;

                b_warp_tensor.GetThreadBuffer() =
                    detail::load_sliced_thread_data_from_block_tensor_window(
                        b_block_window,
                        MultiIndex<3>{nIter, kIter, 0},
                        Sequence<1, 1, WG::ABKPerLane>{});

                // read C warp tensor from C block tensor
                WG::CWarpTensor c_warp_tensor;

                c_warp_tensor.SetSlicedThreadData(
                    Sequence<0, 0>{},
                    Sequence<WG::CM0PerLane, WG::CM1PerLane>{},
                    c_block_tensor.GetSlicedThreadData(
                        Sequence<mIter, nIter, 0, 0>{},
                        Sequence<1, 1, WG::CM0PerLane, WG::CM1PerLane>{}));

                WG{}(c_warp_tensor, a_warp_tensor, b_warp_tensor);

                // write C warp tensor into C block tensor
                c_block_tensor.SetSlicedThreadData(
                    Sequence<mIter, nIter, 0, 0>{},
                    Sequence<1, 1, WG::CM0PerLane, WG::CM1PerLane>{},
                    c_warp_tensor.GetSlicedThreadData(Sequence<0, 0>{},
                                                      Sequence<WG::CM0PerLane, WG::CM1PerLane>{}));
            });
        });
    });
}

template <typename ABlockWindow, typename BBlockWindow>
__host__ __device__ auto block_tile_gemm(const ABlockWindow& a_block_window,
                                         const BBlockWindow& b_block_window)
{
    // FIXME: use heuristic to choose paramters and WarpGEMM
    constexpr index_t MWarp = 2;
    constexpr index_t NWarp = 2;

    constexpr index_t MXdlPerWarp = 2;
    constexpr index_t NXdlPerWarp = 2;

    using WG = WarpGemmXdlFp16M32N32K8;

    using CDataType = typename WG::CDataType;

    // FIXME: create block dstr from existing wave dstr
    constexpr auto c_block_dstr = make_static_block_tensor_distribution(
        Sequence<1>{},
        make_tuple(Sequence<MXdlPerWarp, MWarp, WG::CM0PerLane, WG::CMLane, WG::CM1PerLane>{},
                   Sequence<NXdlPerWarp, NWarp, WG::CNLane>{}),
        Sequence<1, 2, 0>{},
        Sequence<1, 1, 0>{},
        Sequence<1, 2>{},
        Sequence<3, 2>{},
        Sequence<1, 2, 1, 1>{},
        Sequence<0, 0, 2, 4>{});

    auto c_block_tensor = make_static_block_distributed_tensor<CDataType>(c_block_dstr);

    block_tile_gemm(c_block_tensor, a_block_window, b_block_window);

    return c_block_tensor;
}

#if 1
// FIXME: remove: dummy host function for tile programming
template <typename CBlockTensor, typename ABlockWindowTmp, typename BBlockWindowTmp>
__host__ void block_tile_gemm(CBlockTensor&, const ABlockWindowTmp&, const BBlockWindowTmp&)
{
}
#endif

} // namespace block
} // namespace tile_program
} // namespace ck
