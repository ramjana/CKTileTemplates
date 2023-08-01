// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"
#include "ck/tile_program/block_tensor_distribution.hpp"
#include "ck/tile_program/warp_gemm.hpp"

namespace ck {
namespace tile_program {
namespace block {

// A is block window on shared memory
// B is block distributed tensor
// C is block distributed tensor
template <typename CBlockTensor, typename ABlockTensorTmp, typename BBlockWindowTmp>
__device__ void block_gemm_cr_ar_bs(CBlockTensor& c_block_tensor,
                                    const ABlockTensorTmp& a_block_tensor_tmp,
                                    const BBlockWindowTmp& b_block_window_tmp)
{
    using namespace ck::tile_program::warp;

    // FIXME: use heuristic to choose parameters and WarpGEMM
#if 1
    // 128x128x128   32x32x8
    constexpr index_t MWarp = 4;
    constexpr index_t NWarp = 1;

    constexpr index_t MIterPerWarp = 1;
    constexpr index_t NIterPerWarp = 4;
    constexpr index_t KIterPerWarp = 16;

    using WG = WarpGemmMfmaF16F16F32M32N32K8;
#endif

    // FIXME A/BlockWindow lengths need to be static;
    // static_assert

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
        Sequence<1>,
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

    static_assert(
        is_same_v<remove_cvref_t<decltype(c_block_dstr_encode)>,
                  remove_cvref_t<decltype(
                      CBlockTensor::GetBlockDistribution().GetStaticTensorDistributionEncoding())>>,
        "wrong!");

#if 0
    // FIXME: need method to check a_block_tensor and a_block_tensor_tmp have equivalent distribution
    static_assert(
        is_same_v<remove_cvref_t<decltype(a_block_dstr_encode)>,
                  remove_cvref_t<decltype(
                      ABlockTensorTmp::GetBlockDistribution().GetStaticTensorDistributionEncoding())>>,
        "wrong!");
#endif

    // FIXME: need method to check a_block_tensor and a_block_tensor_tmp have equivalent
    // distribution
    auto a_block_tensor =
        make_static_block_distributed_tensor<typename ABlockTensorTmp::DataType>(a_block_dstr);

    a_block_tensor.GetThreadBuffer() = a_block_tensor_tmp.GetThreadBuffer();

    // construct B-block-window from B-block-distribution
    auto b_block_window = make_block_window(b_block_window_tmp.GetBottomTensorView(),
                                            b_block_window_tmp.GetBlockWindowOrigin(),
                                            b_block_dstr);

    constexpr auto a_warp_y_lengths = to_sequence(WG::AWarpDstr{}.GetYs2DDescriptor().GetLengths());
    constexpr auto b_warp_y_lengths = to_sequence(WG::BWarpDstr{}.GetYs2DDescriptor().GetLengths());
    constexpr auto c_warp_y_lengths = to_sequence(WG::CWarpDstr{}.GetYs2DDescriptor().GetLengths());

    constexpr auto a_warp_y_index_zeros = uniform_sequence_gen_t<WG::AWarpDstr::NDimY, 0>{};
    constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<WG::CWarpDstr::NDimY, 0>{};

    // hot loop:
    static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
        static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
            // read A warp tensor from A block tensor
            WG::AWarpTensor a_warp_tensor;

            a_warp_tensor.GetThreadBuffer() = a_block_tensor.GetSlicedThreadData(
                merge_sequences(Sequence<mIter, kIter>{}, a_warp_y_index_zeros),
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

template <typename ABlockTensor, typename BBlockWindow>
__host__ __device__ auto block_gemm_cr_ar_bs(const ABlockTensor& a_block_tensor,
                                             const BBlockWindow& b_block_window)
{
    using namespace ck::tile_program::warp;

    // FIXME: use heuristic to choose paramters and WarpGEMM
#if 1
    // 128x128x128   32x32x8
    constexpr index_t MWarp = 4;
    constexpr index_t NWarp = 1;

    constexpr index_t MIterPerWarp = 1;
    constexpr index_t NIterPerWarp = 4;

    using WG = WarpGemmMfmaF16F16F32M32N32K8;
#endif

    constexpr auto c_block_outer_dstr_encoding = StaticTensorDistributionEncoding<
        Sequence<1>,
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

    block_elementwise_inout([](auto& c) { c = 0; }, c_block_tensor);

    block_gemm_cr_ar_bs(c_block_tensor, a_block_tensor, b_block_window);

    return c_block_tensor;
}

// FIXME: remove: dummy host function for tile programming
template <typename CBlockTensor, typename ABlockWindowTmp, typename BBlockWindowTmp>
__host__ void block_gemm_cr_ar_bs(CBlockTensor&, const ABlockWindowTmp&, const BBlockWindowTmp&)
{
}

} // namespace block
} // namespace tile_program
} // namespace ck
