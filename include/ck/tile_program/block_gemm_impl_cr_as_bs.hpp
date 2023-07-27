// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"

#include "ck/tile_program/tile_distribution.hpp"
#include "ck/tile_program/tile_elementwise.hpp"
#include "ck/tile_program/warp_gemm.hpp"

namespace ck {
namespace tile_program {
namespace block {

// A is block window on shared memory
// B is block window on shared memory
// C is block distributed tensor
template <typename CBlockTensor, typename ABlockWindowTmp, typename BBlockWindowTmp>
__device__ void block_gemm_cr_as_bs(CBlockTensor& c_block_tensor,
                                    const ABlockWindowTmp& a_block_window_tmp,
                                    const BBlockWindowTmp& b_block_window_tmp)
{
    using namespace ck::tile_program::warp;

    // FIXME: use heuristic to choose parameters and WarpGEMM
#if 0
    // 128x128x32   32x32x8
    constexpr index_t MWarp = 2;
    constexpr index_t NWarp = 2;

    constexpr index_t MIterPerWarp = 2;
    constexpr index_t NIterPerWarp = 2;
    constexpr index_t KIterPerWarp = 4;

    using WG = WarpGemmMfmaF16F16F32M32N32K8;
#elif 0
    // 128x128x32, 32x32x16, 2x2 warps
    constexpr index_t MWarp = 2;
    constexpr index_t NWarp = 2;

    constexpr index_t MIterPerWarp = 2;
    constexpr index_t NIterPerWarp = 2;
    constexpr index_t KIterPerWarp = 2;

    using WG = WarpGemmMfmaF16F16F32M32N32K16;
#elif 0
    // 128x128x32, 32x32x16, 4x1 warps,
    constexpr index_t MWarp = 4;
    constexpr index_t NWarp = 1;

    constexpr index_t MIterPerWarp = 1;
    constexpr index_t NIterPerWarp = 4;
    constexpr index_t KIterPerWarp = 2;

    using WG = WarpGemmMfmaF16F16F32M32N32K16;
#elif 0
    // 128x128x32, 32x32x16-Transposed C Distribution, 4x1 warps,
    constexpr index_t MWarp = 4;
    constexpr index_t NWarp = 1;

    constexpr index_t MIterPerWarp = 1;
    constexpr index_t NIterPerWarp = 4;
    constexpr index_t KIterPerWarp = 2;

    using WG = WarpGemmMfmaF16F16F32M32N32K16TransposedCDistribution;
#elif 0
    // 128x128x32   16x16x16
    constexpr index_t MWarp = 2;
    constexpr index_t NWarp = 2;

    constexpr index_t MIterPerWarp = 4;
    constexpr index_t NIterPerWarp = 4;
    constexpr index_t KIterPerWarp = 2;

    using WG = WarpGemmMfmaF16F16F32M16N16K16;
#elif 0
    // 128x256x32   32x32x8
    constexpr index_t MWarp = 2;
    constexpr index_t NWarp = 2;

    constexpr index_t MIterPerWarp = 2;
    constexpr index_t NIterPerWarp = 4;
    constexpr index_t KIterPerWarp = 4;

    using WG = WarpGemmMfmaF16F16F32M32N32K8;
#elif 0
    // 128x256x32   32x32x16
    constexpr index_t MWarp = 2;
    constexpr index_t NWarp = 2;

    constexpr index_t MIterPerWarp = 2;
    constexpr index_t NIterPerWarp = 4;
    constexpr index_t KIterPerWarp = 2;

    using WG = WarpGemmMfmaF16F16F32M32N32K16;
#elif 0
    // 128x256x32   16x16x16
    constexpr index_t MWarp = 2;
    constexpr index_t NWarp = 2;

    constexpr index_t MIterPerWarp = 4;
    constexpr index_t NIterPerWarp = 8;
    constexpr index_t KIterPerWarp = 2;

    using WG = WarpGemmMfmaF16F16F32M16N16K16;
#elif 1
    // 256x128x32   32x32x16
    constexpr index_t MWarp = 2;
    constexpr index_t NWarp = 2;

    constexpr index_t MIterPerWarp = 4;
    constexpr index_t NIterPerWarp = 2;
    constexpr index_t KIterPerWarp = 2;

    using WG = WarpGemmMfmaF16F16F32M32N32K16;
#endif

    // FIXME A/BlockWindow lengths need to be static;
    // static_assert

#if 0 // debug
    constexpr index_t MPerBlock = ABlockWindowTmp{}.GetWindowLengths()[Number<0>{}];
    constexpr index_t NPerBlock = BBlockWindowTmp{}.GetWindowLengths()[Number<0>{}];
    constexpr index_t KPerBlock = ABlockWindowTmp{}.GetWindowLengths()[Number<1>{}];

    constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WG::M);
    constexpr index_t NIterPerWarp = NPerBlock / (NWarp * WG::N);
    constexpr index_t KIterPerWarp = KPerBlock / WG::K;

    static_assert(MPerBlock == 128, "wrong!");
    static_assert(NPerBlock == 256, "wrong!");
    static_assert(KPerBlock ==  32, "wrong!");
    static_assert(WG::M == 16, "wrong!");
    static_assert(WG::N == 16, "wrong!");
    static_assert(WG::K == 16, "wrong!");
#endif

    constexpr auto a_block_outer_dstr_encoding =
        StaticTileDistributionEncoding<Sequence<NWarp>,
                                       Tuple<Sequence<MIterPerWarp, MWarp>, Sequence<KIterPerWarp>>,
                                       Tuple<Sequence<1, 0>>,
                                       Tuple<Sequence<1, 0>>,
                                       Sequence<1, 2>,
                                       Sequence<0, 0>>{};

    constexpr auto b_block_outer_dstr_encoding =
        StaticTileDistributionEncoding<Sequence<MWarp>,
                                       Tuple<Sequence<NIterPerWarp, NWarp>, Sequence<KIterPerWarp>>,
                                       Tuple<Sequence<0, 1>>,
                                       Tuple<Sequence<0, 1>>,
                                       Sequence<1, 2>,
                                       Sequence<0, 0>>{};

    constexpr auto c_block_outer_dstr_encoding = StaticTileDistributionEncoding<
        Sequence<>,
        Tuple<Sequence<MIterPerWarp, MWarp>, Sequence<NIterPerWarp, NWarp>>,
        Tuple<Sequence<1, 2>>,
        Tuple<Sequence<1, 1>>,
        Sequence<1, 2>,
        Sequence<0, 0>>{};

    //  WG::AWarpDstrEncoding{}.foo();

    constexpr auto a_block_dstr_encode =
        embed_tile_distribution_encoding(a_block_outer_dstr_encoding, WG::AWarpDstrEncoding{});

    //  a_block_dstr_encode.foo();

    //  WG::BWarpDstrEncoding{}.foo();

    constexpr auto b_block_dstr_encode =
        embed_tile_distribution_encoding(b_block_outer_dstr_encoding, WG::BWarpDstrEncoding{});

    //  b_block_dstr_encode.foo();

    //  WG::CWarpDstrEncoding{}.foo();

    constexpr auto c_block_dstr_encode =
        embed_tile_distribution_encoding(c_block_outer_dstr_encoding, WG::CWarpDstrEncoding{});

    //  c_block_dstr_encode.foo();

    constexpr auto a_block_dstr = make_static_tile_distribution(a_block_dstr_encode);
    constexpr auto b_block_dstr = make_static_tile_distribution(b_block_dstr_encode);

    static_assert(
        is_same_v<remove_cvref_t<decltype(c_block_dstr_encode)>,
                  remove_cvref_t<decltype(
                      CBlockTensor::GetTileDistribution().GetStaticTileDistributionEncoding())>>,
        "wrong!");

    // construct A/B-block-window from A/B-block-distribution
    auto a_block_window = make_tile_window(a_block_window_tmp.GetBottomTensorView(),
                                           a_block_window_tmp.GetWindowOrigin(),
                                           a_block_dstr);

    auto b_block_window = make_tile_window(b_block_window_tmp.GetBottomTensorView(),
                                           b_block_window_tmp.GetWindowOrigin(),
                                           b_block_dstr);

    constexpr auto a_warp_y_lengths = to_sequence(WG::AWarpDstr{}.GetYs2DDescriptor().GetLengths());
    constexpr auto b_warp_y_lengths = to_sequence(WG::BWarpDstr{}.GetYs2DDescriptor().GetLengths());
    constexpr auto c_warp_y_lengths = to_sequence(WG::CWarpDstr{}.GetYs2DDescriptor().GetLengths());

    constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<WG::CWarpDstr::NDimY, 0>{};

    // hot loop:
    static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
        static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
            // read A warp tensor from A block window
            WG::AWarpTensor a_warp_tensor;

            a_warp_tensor.GetThreadBuffer() = detail::load_sliced_thread_data_from_tile_window(
                a_block_window,
                MultiIndex<2 + WG::AWarpDstr::NDimY>{mIter, kIter, 0},
                merge_sequences(Sequence<1, 1>{}, a_warp_y_lengths));

            static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                // read B warp tensor from B Block window
                WG::BWarpTensor b_warp_tensor;

                b_warp_tensor.GetThreadBuffer() = detail::load_sliced_thread_data_from_tile_window(
                    b_block_window,
                    MultiIndex<2 + WG::BWarpDstr::NDimY>{nIter, kIter, 0},
                    merge_sequences(Sequence<1, 1>{}, b_warp_y_lengths));

                // read C warp tensor from C block tensor
                WG::CWarpTensor c_warp_tensor;

                c_warp_tensor.GetThreadBuffer() = c_block_tensor.GetSlicedThreadData(
                    merge_sequences(Sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                    merge_sequences(Sequence<1, 1>{}, c_warp_y_lengths));

                // warp GEMM
                WG{}(c_warp_tensor, a_warp_tensor, b_warp_tensor);

                // write C warp tensor into C block tensor
                c_block_tensor.SetSlicedThreadData(
                    merge_sequences(Sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                    merge_sequences(Sequence<1, 1>{}, c_warp_y_lengths),
                    c_warp_tensor.GetThreadBuffer());
            });
        });
    });
}

template <typename ABlockWindow, typename BBlockWindow>
__host__ __device__ auto block_gemm_cr_as_bs(const ABlockWindow& a_block_window,
                                             const BBlockWindow& b_block_window)
{
    using namespace ck::tile_program::warp;

    // FIXME: use heuristic to choose paramters and WarpGEMM
#if 0
    // 128x128x32   32x32x8
    constexpr index_t MWarp = 2;
    constexpr index_t NWarp = 2;

    constexpr index_t MIterPerWarp = 2;
    constexpr index_t NIterPerWarp = 2;

    using WG = WarpGemmMfmaF16F16F32M32N32K8;
#elif 0
    // 128x128x32   32x32x16
    constexpr index_t MWarp = 2;
    constexpr index_t NWarp = 2;

    constexpr index_t MIterPerWarp = 2;
    constexpr index_t NIterPerWarp = 2;

    using WG = WarpGemmMfmaF16F16F32M32N32K16;
#elif 0
    // 128x128x32, 32x32x16, 4x1 warps,
    constexpr index_t MWarp = 4;
    constexpr index_t NWarp = 1;

    constexpr index_t MIterPerWarp = 1;
    constexpr index_t NIterPerWarp = 4;

    using WG = WarpGemmMfmaF16F16F32M32N32K16;
#elif 0
    // 128x128x32, 32x32x16-Transposed C Distribution, 4x1 warps,
    constexpr index_t MWarp = 4;
    constexpr index_t NWarp = 1;

    constexpr index_t MIterPerWarp = 1;
    constexpr index_t NIterPerWarp = 4;

    using WG = WarpGemmMfmaF16F16F32M32N32K16TransposedCDistribution;
#elif 0
    // 128x128x32   16x16x16
    constexpr index_t MWarp = 2;
    constexpr index_t NWarp = 2;

    constexpr index_t MIterPerWarp = 4;
    constexpr index_t NIterPerWarp = 4;

    using WG = WarpGemmMfmaF16F16F32M16N16K16;
#elif 0
    // 128x256x32   32x32x8
    constexpr index_t MWarp = 2;
    constexpr index_t NWarp = 2;

    constexpr index_t MIterPerWarp = 2;
    constexpr index_t NIterPerWarp = 4;

    using WG = WarpGemmMfmaF16F16F32M32N32K8;
#elif 0
    // 128x256x32   32x32x16
    constexpr index_t MWarp = 2;
    constexpr index_t NWarp = 2;

    constexpr index_t MIterPerWarp = 2;
    constexpr index_t NIterPerWarp = 4;

    using WG = WarpGemmMfmaF16F16F32M32N32K16;
#elif 0
    // 128x256x32   16x16x16
    constexpr index_t MWarp = 2;
    constexpr index_t NWarp = 2;

    constexpr index_t MIterPerWarp = 4;
    constexpr index_t NIterPerWarp = 8;

    using WG = WarpGemmMfmaF16F16F32M16N16K16;
#elif 1
    // 256x128x32   32x32x16
    constexpr index_t MWarp = 2;
    constexpr index_t NWarp = 2;

    constexpr index_t MIterPerWarp = 4;
    constexpr index_t NIterPerWarp = 2;

    using WG = WarpGemmMfmaF16F16F32M32N32K16;
#endif

    constexpr auto c_block_outer_dstr_encoding = StaticTileDistributionEncoding<
        Sequence<>,
        Tuple<Sequence<MIterPerWarp, MWarp>, Sequence<NIterPerWarp, NWarp>>,
        Tuple<Sequence<1, 2>>,
        Tuple<Sequence<1, 1>>,
        Sequence<1, 2>,
        Sequence<0, 0>>{};

    constexpr auto c_block_dstr_encode =
        embed_tile_distribution_encoding(c_block_outer_dstr_encoding, WG::CWarpDstrEncoding{});

    constexpr auto c_block_dstr = make_static_tile_distribution(c_block_dstr_encode);

    using CDataType = typename WG::CDataType;

    auto c_block_tensor = make_static_distributed_tensor<CDataType>(c_block_dstr);

    tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tensor);

    block_gemm_cr_as_bs(c_block_tensor, a_block_window, b_block_window);

    return c_block_tensor;
}

// FIXME: remove: dummy host function for tile programming
template <typename CBlockTensor, typename ABlockWindowTmp, typename BBlockWindowTmp>
__host__ void block_gemm_cr_as_bs(CBlockTensor&, const ABlockWindowTmp&, const BBlockWindowTmp&)
{
}

} // namespace block
} // namespace tile_program
} // namespace ck
