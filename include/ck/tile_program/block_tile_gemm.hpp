// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"
#include "ck/tile_program/block_tensor_distribution.hpp"
#include "ck/tile_program/warp_gemm.hpp"
#include "ck/tile_program/warp_gemm_attribute_mfma.hpp"
#include "ck/tile_program/warp_gemm_attribute_mfma_impl.hpp"

namespace ck {
namespace tile_program {
namespace block {

#if 1
using WarpGemmMfmaF16F16F32M32N32K8 =
    ck::tile_program::warp::WarpGemm<ck::tile_program::warp::WarpGemmAtrributeMfma<
        ck::tile_program::warp::WarpGemmAttributeMfmaImplF16F16F32M32N32K8>>;

using WarpGemmMfmaF16F16F32M32N32K16 =
    ck::tile_program::warp::WarpGemm<ck::tile_program::warp::WarpGemmAtrributeMfma<
        ck::tile_program::warp::WarpGemmAttributeMfmaImplF16F16F32M32N32K16>>;

using WarpGemmMfmaF16F16F32M16N16K16 =
    ck::tile_program::warp::WarpGemm<ck::tile_program::warp::WarpGemmAtrributeMfma<
        ck::tile_program::warp::WarpGemmAttributeMfmaImplF16F16F32M16N16K16>>;
#endif

template <typename CBlockTensor, typename ABlockWindowTmp, typename BBlockWindowTmp>
__device__ void block_tile_gemm(CBlockTensor& c_block_tensor,
                                const ABlockWindowTmp& a_block_window_tmp,
                                const BBlockWindowTmp& b_block_window_tmp)
{
    // FIXME: use heuristic to choose paramters and WarpGEMM
#if 0
    // 128x128x32   32x32x8
    constexpr index_t MWarp = 2;
    constexpr index_t NWarp = 2;

    constexpr index_t MIterPerWarp = 2;
    constexpr index_t NIterPerWarp = 2;
    constexpr index_t KIterPerWarp = 4;

    using WG = WarpGemmMfmaF16F16F32M32N32K8;
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
#elif 1
    // 128x256x32   32x32x16
    constexpr index_t MWarp = 2;
    constexpr index_t NWarp = 2;

    constexpr index_t MIterPerWarp = 2;
    constexpr index_t NIterPerWarp = 4;
    constexpr index_t KIterPerWarp = 2;

    using WG = WarpGemmMfmaF16F16F32M32N32K16;
#elif 1
    // 128x256x32   16x16x16
    constexpr index_t MWarp = 2;
    constexpr index_t NWarp = 2;

    constexpr index_t MIterPerWarp = 4;
    constexpr index_t NIterPerWarp = 8;
    constexpr index_t KIterPerWarp = 2;

    using WG = WarpGemmMfmaF16F16F32M16N16K16;
#endif

    constexpr auto a_block_outer_dstr_encoding = StaticTensorDistributionEncoding<
        Sequence<NWarp>,
        Tuple<Sequence<MIterPerWarp, MWarp>, Sequence<KIterPerWarp>>,
        Tuple<Sequence<1, 0>>,
        Tuple<Sequence<1, 0>>,
        Sequence<1, 2>,
        Sequence<0, 0>>{};

    constexpr auto b_block_outer_dstr_encoding = StaticTensorDistributionEncoding<
        Sequence<MWarp>,
        Tuple<Sequence<NIterPerWarp, NWarp>, Sequence<KIterPerWarp>>,
        Tuple<Sequence<0, 1>>,
        Tuple<Sequence<0, 1>>,
        Sequence<1, 2>,
        Sequence<0, 0>>{};

    constexpr auto c_block_outer_dstr_encoding = StaticTensorDistributionEncoding<
        Sequence<>,
        Tuple<Sequence<MIterPerWarp, MWarp>, Sequence<NIterPerWarp, NWarp>>,
        Tuple<Sequence<1, 2>>,
        Tuple<Sequence<1, 1>>,
        Sequence<1, 2>,
        Sequence<0, 0>>{};

    constexpr auto a_block_dstr_encode =
        embed_tensor_distribution_encoding(a_block_outer_dstr_encoding, WG::AWarpDstrEncoding{});

    constexpr auto b_block_dstr_encode =
        embed_tensor_distribution_encoding(b_block_outer_dstr_encoding, WG::BWarpDstrEncoding{});

    constexpr auto c_block_dstr_encode =
        embed_tensor_distribution_encoding(c_block_outer_dstr_encoding, WG::CWarpDstrEncoding{});

    constexpr auto a_block_dstr = make_static_block_tensor_distribution(a_block_dstr_encode);
    constexpr auto b_block_dstr = make_static_block_tensor_distribution(b_block_dstr_encode);
    constexpr auto c_block_dstr = make_static_block_tensor_distribution(c_block_dstr_encode);

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

    constexpr auto a_warp_y_lengths = to_sequence(WG::AWarpDstr{}.GetYs2DDescriptor().GetLengths());
    constexpr auto b_warp_y_lengths = to_sequence(WG::BWarpDstr{}.GetYs2DDescriptor().GetLengths());
    constexpr auto c_warp_y_lengths = to_sequence(WG::CWarpDstr{}.GetYs2DDescriptor().GetLengths());

    // hot loop:
    static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
        static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
            // read A warp tensor from A block window
            WG::AWarpTensor a_warp_tensor;

            a_warp_tensor.GetThreadBuffer() =
                detail::load_sliced_thread_data_from_block_tensor_window(
                    a_block_window,
                    MultiIndex<2 + WG::AWarpDstr::NDimY>{mIter, kIter, 0},
                    merge_sequences(Sequence<1, 1>{}, a_warp_y_lengths));

            static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                // read B warp tensor from B Block window
                WG::BWarpTensor b_warp_tensor;

                b_warp_tensor.GetThreadBuffer() =
                    detail::load_sliced_thread_data_from_block_tensor_window(
                        b_block_window,
                        MultiIndex<2 + WG::BWarpDstr::NDimY>{nIter, kIter, 0},
                        merge_sequences(Sequence<1, 1>{}, b_warp_y_lengths));

                // read C warp tensor from C block tensor
                WG::CWarpTensor c_warp_tensor;

                constexpr auto c_warp_y_index_zeros =
                    uniform_sequence_gen_t<WG::CWarpDstr::NDimY, 0>{};

                c_warp_tensor.SetSlicedThreadData(
                    c_warp_y_index_zeros,
                    c_warp_y_lengths,
                    c_block_tensor.GetSlicedThreadData(
                        merge_sequences(Sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                        merge_sequences(Sequence<1, 1>{}, c_warp_y_lengths)));

                // warp GEMM
                WG{}(c_warp_tensor, a_warp_tensor, b_warp_tensor);

                // write C warp tensor into C block tensor
                c_block_tensor.SetSlicedThreadData(
                    merge_sequences(Sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                    merge_sequences(Sequence<1, 1>{}, c_warp_y_lengths),
                    c_warp_tensor.GetSlicedThreadData(Sequence<0, 0>{}, c_warp_y_lengths));
            });
        });
    });
}

template <typename ABlockWindow, typename BBlockWindow>
__host__ __device__ auto block_tile_gemm(const ABlockWindow& a_block_window,
                                         const BBlockWindow& b_block_window)
{
    // FIXME: use heuristic to choose paramters and WarpGEMM
#if 0
    // 128x128x32   32x32x8
    constexpr index_t MWarp = 2;
    constexpr index_t NWarp = 2;

    constexpr index_t MIterPerWarp = 2;
    constexpr index_t NIterPerWarp = 2;

    using WG = WarpGemmMfmaF16F16F32M32N32K8;
#elif 0
    // 128x128x32   16x16x16
    constexpr index_t MWarp = 2;
    constexpr index_t NWarp = 2;

    constexpr index_t MIterPerWarp = 4;
    constexpr index_t NIterPerWarp = 4;

    using WG = WarpGemmMfmaF16F16F32M16N16K16;
#elif 1
    // 128x256x32   32x32x8
    constexpr index_t MWarp = 2;
    constexpr index_t NWarp = 2;

    constexpr index_t MIterPerWarp = 2;
    constexpr index_t NIterPerWarp = 4;

    using WG = WarpGemmMfmaF16F16F32M32N32K8;
#elif 1
    // 128x256x32   16x16x16
    constexpr index_t MWarp = 2;
    constexpr index_t NWarp = 2;

    constexpr index_t MIterPerWarp = 4;
    constexpr index_t NIterPerWarp = 8;

    using WG = WarpGemmMfmaF16F16F32M16N16K16;
#endif

    constexpr auto c_block_outer_dstr_encoding = StaticTensorDistributionEncoding<
        Sequence<>,
        Tuple<Sequence<MIterPerWarp, MWarp>, Sequence<NIterPerWarp, NWarp>>,
        Tuple<Sequence<1, 2>>,
        Tuple<Sequence<1, 1>>,
        Sequence<1, 2>,
        Sequence<0, 0>>{};

    constexpr auto c_block_dstr_encode =
        embed_tensor_distribution_encoding(c_block_outer_dstr_encoding, WG::CWarpDstrEncoding{});

    constexpr auto c_block_dstr = make_static_block_tensor_distribution(c_block_dstr_encode);

    using CDataType = typename WG::CDataType;

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
