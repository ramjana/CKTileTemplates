// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"

#include "ck/tile_program/tile/static_tile_distribution_helper.hpp"
#include "ck/tile_program/tile/tile_distribution.hpp"
#include "ck/tile_program/tile/load_tile.hpp"
#include "ck/tile_program/tile/tile_elementwise.hpp"
#include "ck/tile_program/tile/tile_gemm_shape.hpp"
#include "ck/tile_program/warp_tile/warp_gemm.hpp"
#include "ck/tile_program/block_tile/block_gemm_asmem_bsmem_creg_v1_default_policy.hpp"

namespace ck {
namespace tile_program {
namespace block {

// Problem Description for BlockGemmASmemBSmemCRegV1
template <typename ADataType_,
          typename BDataType_,
          typename CDataType_,
          index_t kBlockSize_,
          typename BlockGemmShape_>
struct BlockGemmASmemBSmemCRegV1Problem
{
    using ADataType      = remove_cvref_t<ADataType_>;
    using BDataType      = remove_cvref_t<BDataType_>;
    using CDataType      = remove_cvref_t<CDataType_>;
    using BlockGemmShape = remove_cvref_t<BlockGemmShape_>;

    static constexpr index_t kBlockSize = kBlockSize_;
};

// A is block window on shared memory
// B is block window on shared memory
// C is block distributed tensor
template <typename Problem, typename Policy = BlockGemmASmemBSmemCRegV1DefaultPolicy>
struct BlockGemmASmemBSmemCRegV1
{
    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType      = remove_cvref_t<typename Problem::BDataType>;
    using CDataType      = remove_cvref_t<typename Problem::CDataType>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

    static constexpr index_t kBlockSize = Problem::kBlockSize;

#if 0
    // C += A * B
    template <typename CBlockTensor, typename ABlockWindowTmp, typename BBlockWindowTmp>
    __device__ void operator()(CBlockTensor& c_block_tensor,
                               const ABlockWindowTmp& a_block_window_tmp,
                               const BBlockWindowTmp& b_block_window_tmp) const
    {
        static_assert(is_same_v<ADataType, typename ABlockWindowTmp::DataType> &&
                          is_same_v<BDataType, typename BBlockWindowTmp::DataType> &&
                          is_same_v<CDataType, typename CBlockTensor::DataType>,
                      "wrong!");

        constexpr index_t MPerBlock = ABlockWindowTmp{}.GetWindowLengths()[Number<0>{}];
        constexpr index_t NPerBlock = BBlockWindowTmp{}.GetWindowLengths()[Number<0>{}];
        constexpr index_t KPerBlock = ABlockWindowTmp{}.GetWindowLengths()[Number<1>{}];

        static_assert(MPerBlock == BlockGemmShape::kM && NPerBlock == BlockGemmShape::kN &&
                          KPerBlock == BlockGemmShape::kK,
                      "wrong!");

        constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();

        using WG = remove_cvref_t<decltype(config.template At<0>())>;

        constexpr index_t MWarp = config.template At<1>();
        constexpr index_t NWarp = config.template At<2>();

        constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WG::kM);
        constexpr index_t NIterPerWarp = NPerBlock / (NWarp * WG::kN);
        constexpr index_t KIterPerWarp = KPerBlock / WG::kK;

        constexpr auto a_block_outer_dstr_encoding = StaticTileDistributionEncoding<
            Sequence<NWarp>,
            Tuple<Sequence<MIterPerWarp, MWarp>, Sequence<KIterPerWarp>>,
            Tuple<Sequence<1, 0>>,
            Tuple<Sequence<1, 0>>,
            Sequence<1, 2>,
            Sequence<0, 0>>{};

        constexpr auto b_block_outer_dstr_encoding = StaticTileDistributionEncoding<
            Sequence<MWarp>,
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

        constexpr auto a_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            a_block_outer_dstr_encoding, typename WG::AWarpDstrEncoding{});

        constexpr auto b_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            b_block_outer_dstr_encoding, typename WG::BWarpDstrEncoding{});

        constexpr auto c_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            c_block_outer_dstr_encoding, typename WG::CWarpDstrEncoding{});

        constexpr auto a_block_dstr = make_static_tile_distribution(a_block_dstr_encode);
        constexpr auto b_block_dstr = make_static_tile_distribution(b_block_dstr_encode);

        static_assert(is_same_v<remove_cvref_t<decltype(c_block_dstr_encode)>,
                                remove_cvref_t<decltype(CBlockTensor::GetTileDistribution()
                                                            .GetStaticTileDistributionEncoding())>>,
                      "wrong!");

        // construct A/B-block-window from A/B-block-distribution
        auto a_block_window = make_tile_window(a_block_window_tmp.GetBottomTensorView(),
                                               a_block_window_tmp.GetWindowLengths(),
                                               a_block_window_tmp.GetWindowOrigin(),
                                               a_block_dstr);

        auto b_block_window = make_tile_window(b_block_window_tmp.GetBottomTensorView(),
                                               b_block_window_tmp.GetWindowLengths(),
                                               b_block_window_tmp.GetWindowOrigin(),
                                               b_block_dstr);

        using AWarpDstr = typename WG::AWarpDstr;
        using BWarpDstr = typename WG::BWarpDstr;
        using CWarpDstr = typename WG::CWarpDstr;

        using AWarpTensor = typename WG::AWarpTensor;
        using BWarpTensor = typename WG::BWarpTensor;
        using CWarpTensor = typename WG::CWarpTensor;

        constexpr auto a_warp_y_lengths = to_sequence(AWarpDstr{}.GetYs2DDescriptor().GetLengths());
        constexpr auto b_warp_y_lengths = to_sequence(BWarpDstr{}.GetYs2DDescriptor().GetLengths());
        constexpr auto c_warp_y_lengths = to_sequence(CWarpDstr{}.GetYs2DDescriptor().GetLengths());
        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

        // hot loop:
        static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
            static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                // read A warp tensor from A block window
                AWarpTensor a_warp_tensor;

                a_warp_tensor.GetThreadBuffer() = detail::load_sliced_thread_data_from_tile_window(
                    a_block_window,
                    MultiIndex<2 + AWarpDstr::NDimY>{mIter, kIter, 0},
                    merge_sequences(Sequence<1, 1>{}, a_warp_y_lengths));

                static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                    // read B warp tensor from B Block window
                    BWarpTensor b_warp_tensor;

                    b_warp_tensor.GetThreadBuffer() =
                        detail::load_sliced_thread_data_from_tile_window(
                            b_block_window,
                            MultiIndex<2 + BWarpDstr::NDimY>{nIter, kIter, 0},
                            merge_sequences(Sequence<1, 1>{}, b_warp_y_lengths));

                    // read C warp tensor from C block tensor
                    CWarpTensor c_warp_tensor;

                    c_warp_tensor.GetThreadBuffer() = c_block_tensor.GetYSlicedThreadData(
                        merge_sequences(Sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                        merge_sequences(Sequence<1, 1>{}, c_warp_y_lengths));

                    // warp GEMM
                    WG{}(c_warp_tensor, a_warp_tensor, b_warp_tensor);

                    // write C warp tensor into C block tensor
                    c_block_tensor.SetYSlicedThreadData(
                        merge_sequences(Sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                        merge_sequences(Sequence<1, 1>{}, c_warp_y_lengths),
                        c_warp_tensor.GetThreadBuffer());
                });
            });
        });
    }
#else
    // C += A * B
    template <typename CBlockTensor, typename ABlockWindowTmp, typename BBlockWindowTmp>
    __device__ void operator()(CBlockTensor& c_block_tensor,
                               const ABlockWindowTmp& a_block_window_tmp,
                               const BBlockWindowTmp& b_block_window_tmp) const
    {
        static_assert(is_same_v<ADataType, typename ABlockWindowTmp::DataType> &&
                          is_same_v<BDataType, typename BBlockWindowTmp::DataType> &&
                          is_same_v<CDataType, typename CBlockTensor::DataType>,
                      "wrong!");

        constexpr index_t MPerBlock = ABlockWindowTmp{}.GetWindowLengths()[Number<0>{}];
        constexpr index_t NPerBlock = BBlockWindowTmp{}.GetWindowLengths()[Number<0>{}];
        constexpr index_t KPerBlock = ABlockWindowTmp{}.GetWindowLengths()[Number<1>{}];

        static_assert(MPerBlock == BlockGemmShape::kM && NPerBlock == BlockGemmShape::kN &&
                          KPerBlock == BlockGemmShape::kK,
                      "wrong!");

        constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();

        using WG = remove_cvref_t<decltype(config.template At<0>())>;

        constexpr index_t MWarp = config.template At<1>();
        constexpr index_t NWarp = config.template At<2>();

        constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WG::kM);
        constexpr index_t NIterPerWarp = NPerBlock / (NWarp * WG::kN);
        constexpr index_t KIterPerWarp = KPerBlock / WG::kK;

        constexpr index_t MPerBlockPerIter = MPerBlock / MIterPerWarp;
        constexpr index_t NPerBlockPerIter = NPerBlock / NIterPerWarp;
        constexpr index_t KPerBlockPerIter = KPerBlock / KIterPerWarp;

        constexpr auto a_block_outer_dstr_encoding =
            StaticTileDistributionEncoding<Sequence<NWarp>,
                                           Tuple<Sequence<1, MWarp>, Sequence<1>>,
                                           Tuple<Sequence<1, 0>>,
                                           Tuple<Sequence<1, 0>>,
                                           Sequence<1, 2>,
                                           Sequence<0, 0>>{};

        constexpr auto b_block_outer_dstr_encoding =
            StaticTileDistributionEncoding<Sequence<MWarp>,
                                           Tuple<Sequence<1, NWarp>, Sequence<1>>,
                                           Tuple<Sequence<0, 1>>,
                                           Tuple<Sequence<0, 1>>,
                                           Sequence<1, 2>,
                                           Sequence<0, 0>>{};

        constexpr auto a_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            a_block_outer_dstr_encoding, typename WG::AWarpDstrEncoding{});

        constexpr auto b_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            b_block_outer_dstr_encoding, typename WG::BWarpDstrEncoding{});

        constexpr auto a_block_dstr = make_static_tile_distribution(a_block_dstr_encode);
        constexpr auto b_block_dstr = make_static_tile_distribution(b_block_dstr_encode);

        // construct A-block-window
        using ABlockWindow = decltype(
            make_tile_window(a_block_window_tmp.GetBottomTensorView(),
                             make_tuple(Number<MPerBlockPerIter>{}, Number<KPerBlockPerIter>{}),
                             a_block_window_tmp.GetWindowOrigin(),
                             a_block_dstr));

        Array<Array<ABlockWindow, KIterPerWarp>, MIterPerWarp> a_block_windows{{
            make_tile_window(a_block_window_tmp.GetBottomTensorView(),
                             make_tuple(Number<MPerBlockPerIter>{}, Number<KPerBlockPerIter>{}),
                             a_block_window_tmp.GetWindowOrigin(),
                             a_block_dstr)}};

        for(index_t mIter = 0; mIter < MIterPerWarp; mIter++)
        {
            for(index_t kIter = 0; kIter < KIterPerWarp; kIter++)
            {
                move_tile_window(a_block_windows(mIter)(kIter),
                                 {mIter * MPerBlockPerIter, kIter * KPerBlockPerIter});
            }
        }

        // construct B-block-window
        using BBlockWindow = decltype(
            make_tile_window(b_block_window_tmp.GetBottomTensorView(),
                             make_tuple(Number<NPerBlockPerIter>{}, Number<KPerBlockPerIter>{}),
                             b_block_window_tmp.GetWindowOrigin(),
                             b_block_dstr));

        Array<Array<BBlockWindow, KIterPerWarp>, NIterPerWarp> b_block_windows{{
            make_tile_window(b_block_window_tmp.GetBottomTensorView(),
                             make_tuple(Number<NPerBlockPerIter>{}, Number<KPerBlockPerIter>{}),
                             b_block_window_tmp.GetWindowOrigin(),
                             b_block_dstr)}};

        for(index_t nIter = 0; nIter < NIterPerWarp; nIter++)
        {
            for(index_t kIter = 0; kIter < KIterPerWarp; kIter++)
            {
                move_tile_window(b_block_windows(nIter)(kIter),
                                 {nIter * NPerBlockPerIter, kIter * KPerBlockPerIter});
            }
        }

        using CWarpDstr = typename WG::CWarpDstr;

        using AWarpTensor = typename WG::AWarpTensor;
        using BWarpTensor = typename WG::BWarpTensor;
        using CWarpTensor = typename WG::CWarpTensor;

        constexpr auto c_warp_y_lengths = to_sequence(CWarpDstr{}.GetYs2DDescriptor().GetLengths());
        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

        // hot loop:
        static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
            static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                // read A warp tensor from A block window
                AWarpTensor a_warp_tensor;

                a_warp_tensor.GetThreadBuffer() = detail::load_thread_data_from_tile_window(
                    a_block_windows(mIter)(kIter));

                static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                    // read B warp tensor from B Block window
                    BWarpTensor b_warp_tensor;

                    b_warp_tensor.GetThreadBuffer() =
                        detail::load_thread_data_from_tile_window(
                            b_block_windows(nIter)(kIter));

                    // read C warp tensor from C block tensor
                    CWarpTensor c_warp_tensor;

                    c_warp_tensor.GetThreadBuffer() = c_block_tensor.GetYSlicedThreadData(
                        merge_sequences(Sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                        merge_sequences(Sequence<1, 1>{}, c_warp_y_lengths));

                    // warp GEMM
                    WG{}(c_warp_tensor, a_warp_tensor, b_warp_tensor);

                    // write C warp tensor into C block tensor
                    c_block_tensor.SetYSlicedThreadData(
                        merge_sequences(Sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                        merge_sequences(Sequence<1, 1>{}, c_warp_y_lengths),
                        c_warp_tensor.GetThreadBuffer());
                });
            });
        });
    }
#endif

#if 0
    // C = A * B
    template <typename ABlockWindowTmp, typename BBlockWindowTmp>
    __device__ auto operator()(const ABlockWindowTmp& a_block_window_tmp,
                               const BBlockWindowTmp& b_block_window_tmp) const
    {
        static_assert(is_same_v<ADataType, typename ABlockWindowTmp::DataType> &&
                          is_same_v<BDataType, typename BBlockWindowTmp::DataType>,
                      "wrong!");

        constexpr index_t MPerBlock = ABlockWindowTmp{}.GetWindowLengths()[Number<0>{}];
        constexpr index_t NPerBlock = BBlockWindowTmp{}.GetWindowLengths()[Number<0>{}];
        constexpr index_t KPerBlock = ABlockWindowTmp{}.GetWindowLengths()[Number<1>{}];

        static_assert(MPerBlock == BlockGemmShape::kM && NPerBlock == BlockGemmShape::kN &&
                          KPerBlock == BlockGemmShape::kK,
                      "wrong!");

        constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();

        using WG = remove_cvref_t<decltype(config.template At<0>())>;

        constexpr index_t MWarp = config.template At<1>();
        constexpr index_t NWarp = config.template At<2>();

        constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WG::kM);
        constexpr index_t NIterPerWarp = NPerBlock / (NWarp * WG::kN);
        constexpr index_t KIterPerWarp = KPerBlock / WG::kK;

        constexpr auto a_block_outer_dstr_encoding = StaticTileDistributionEncoding<
            Sequence<NWarp>,
            Tuple<Sequence<MIterPerWarp, MWarp>, Sequence<KIterPerWarp>>,
            Tuple<Sequence<1, 0>>,
            Tuple<Sequence<1, 0>>,
            Sequence<1, 2>,
            Sequence<0, 0>>{};

        constexpr auto b_block_outer_dstr_encoding = StaticTileDistributionEncoding<
            Sequence<MWarp>,
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

        constexpr auto a_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            a_block_outer_dstr_encoding, typename WG::AWarpDstrEncoding{});

        constexpr auto b_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            b_block_outer_dstr_encoding, typename WG::BWarpDstrEncoding{});

        constexpr auto c_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            c_block_outer_dstr_encoding, typename WG::CWarpDstrEncoding{});

        constexpr auto a_block_dstr = make_static_tile_distribution(a_block_dstr_encode);
        constexpr auto b_block_dstr = make_static_tile_distribution(b_block_dstr_encode);
        constexpr auto c_block_dstr = make_static_tile_distribution(c_block_dstr_encode);

        // construct A/B-block-window from A/B-block-distribution
        auto a_block_window = make_tile_window(a_block_window_tmp.GetBottomTensorView(),
                                               a_block_window_tmp.GetWindowLengths(),
                                               a_block_window_tmp.GetWindowOrigin(),
                                               a_block_dstr);

        auto b_block_window = make_tile_window(b_block_window_tmp.GetBottomTensorView(),
                                               b_block_window_tmp.GetWindowLengths(),
                                               b_block_window_tmp.GetWindowOrigin(),
                                               b_block_dstr);

        static_assert(is_same_v<CDataType, typename WG::CDataType>, "wrong!");

        // Construct C-Block-Tensor
        auto c_block_tensor = make_static_distributed_tensor<CDataType>(c_block_dstr);

        using AWarpDstr = typename WG::AWarpDstr;
        using BWarpDstr = typename WG::BWarpDstr;
        using CWarpDstr = typename WG::CWarpDstr;

        using AWarpTensor = typename WG::AWarpTensor;
        using BWarpTensor = typename WG::BWarpTensor;
        using CWarpTensor = typename WG::CWarpTensor;

        constexpr auto a_warp_y_lengths = to_sequence(AWarpDstr{}.GetYs2DDescriptor().GetLengths());
        constexpr auto b_warp_y_lengths = to_sequence(BWarpDstr{}.GetYs2DDescriptor().GetLengths());
        constexpr auto c_warp_y_lengths = to_sequence(CWarpDstr{}.GetYs2DDescriptor().GetLengths());

        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

        // hot loop:
        static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
            static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                // read A warp tensor from A block window
                AWarpTensor a_warp_tensor;

                a_warp_tensor.GetThreadBuffer() = detail::load_sliced_thread_data_from_tile_window(
                    a_block_window,
                    MultiIndex<2 + AWarpDstr::NDimY>{mIter, kIter, 0},
                    merge_sequences(Sequence<1, 1>{}, a_warp_y_lengths));

                static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                    // read B warp tensor from B Block window
                    BWarpTensor b_warp_tensor;

                    b_warp_tensor.GetThreadBuffer() =
                        detail::load_sliced_thread_data_from_tile_window(
                            b_block_window,
                            MultiIndex<2 + BWarpDstr::NDimY>{nIter, kIter, 0},
                            merge_sequences(Sequence<1, 1>{}, b_warp_y_lengths));

                    // read C warp tensor from C block tensor
                    CWarpTensor c_warp_tensor;

                    // warp GEMM
                    if constexpr(KIterPerWarp == 0)
                    {
                        // c = a * b
                        c_warp_tensor = WG{}(a_warp_tensor, b_warp_tensor);
                    }
                    else
                    {
                        // c += a * b
                        c_warp_tensor.GetThreadBuffer() = c_block_tensor.GetYSlicedThreadData(
                            merge_sequences(Sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(Sequence<1, 1>{}, c_warp_y_lengths));

                        WG{}(c_warp_tensor, a_warp_tensor, b_warp_tensor);
                    }

                    // write C warp tensor into C block tensor
                    c_block_tensor.SetYSlicedThreadData(
                        merge_sequences(Sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                        merge_sequences(Sequence<1, 1>{}, c_warp_y_lengths),
                        c_warp_tensor.GetThreadBuffer());
                });
            });
        });

        return c_block_tensor;
    }
#else
    // C = A * B
    template <typename ABlockWindowTmp, typename BBlockWindowTmp>
    __device__ auto operator()(const ABlockWindowTmp& a_block_window_tmp,
                               const BBlockWindowTmp& b_block_window_tmp) const
    {
        static_assert(is_same_v<ADataType, typename ABlockWindowTmp::DataType> &&
                          is_same_v<BDataType, typename BBlockWindowTmp::DataType>,
                      "wrong!");

        constexpr index_t MPerBlock = ABlockWindowTmp{}.GetWindowLengths()[Number<0>{}];
        constexpr index_t NPerBlock = BBlockWindowTmp{}.GetWindowLengths()[Number<0>{}];
        constexpr index_t KPerBlock = ABlockWindowTmp{}.GetWindowLengths()[Number<1>{}];

        static_assert(MPerBlock == BlockGemmShape::kM && NPerBlock == BlockGemmShape::kN &&
                          KPerBlock == BlockGemmShape::kK,
                      "wrong!");

        constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();

        using WG = remove_cvref_t<decltype(config.template At<0>())>;

        constexpr index_t MWarp = config.template At<1>();
        constexpr index_t NWarp = config.template At<2>();

        constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WG::kM);
        constexpr index_t NIterPerWarp = NPerBlock / (NWarp * WG::kN);
        constexpr index_t KIterPerWarp = KPerBlock / WG::kK;

        constexpr index_t MPerBlockPerIter = MPerBlock / MIterPerWarp;
        constexpr index_t NPerBlockPerIter = NPerBlock / NIterPerWarp;
        constexpr index_t KPerBlockPerIter = KPerBlock / KIterPerWarp;

        constexpr auto a_block_outer_dstr_encoding = StaticTileDistributionEncoding<
            Sequence<NWarp>,
            Tuple<Sequence<1, MWarp>, Sequence<1>>,
            Tuple<Sequence<1, 0>>,
            Tuple<Sequence<1, 0>>,
            Sequence<1, 2>,
            Sequence<0, 0>>{};

        constexpr auto b_block_outer_dstr_encoding = StaticTileDistributionEncoding<
            Sequence<MWarp>,
            Tuple<Sequence<1, NWarp>, Sequence<1>>,
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

        constexpr auto a_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            a_block_outer_dstr_encoding, typename WG::AWarpDstrEncoding{});

        constexpr auto b_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            b_block_outer_dstr_encoding, typename WG::BWarpDstrEncoding{});

        constexpr auto c_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            c_block_outer_dstr_encoding, typename WG::CWarpDstrEncoding{});

        constexpr auto a_block_dstr = make_static_tile_distribution(a_block_dstr_encode);
        constexpr auto b_block_dstr = make_static_tile_distribution(b_block_dstr_encode);
        constexpr auto c_block_dstr = make_static_tile_distribution(c_block_dstr_encode);

        // construct A-block-window
        using ABlockWindow = decltype(
            make_tile_window(a_block_window_tmp.GetBottomTensorView(),
                             make_tuple(Number<MPerBlockPerIter>{}, Number<KPerBlockPerIter>{}),
                             a_block_window_tmp.GetWindowOrigin(),
                             a_block_dstr));

        Array<Array<ABlockWindow, KIterPerWarp>, MIterPerWarp> a_block_windows{{
            make_tile_window(a_block_window_tmp.GetBottomTensorView(),
                             make_tuple(Number<MPerBlockPerIter>{}, Number<KPerBlockPerIter>{}),
                             a_block_window_tmp.GetWindowOrigin(),
                             a_block_dstr)}};

        for(index_t mIter = 0; mIter < MIterPerWarp; mIter++)
        {
            for(index_t kIter = 0; kIter < KIterPerWarp; kIter++)
            {
                move_tile_window(a_block_windows(mIter)(kIter),
                                 {mIter * MPerBlockPerIter, kIter * KPerBlockPerIter});
            }
        }

        // construct B-block-window
        using BBlockWindow = decltype(
            make_tile_window(b_block_window_tmp.GetBottomTensorView(),
                             make_tuple(Number<NPerBlockPerIter>{}, Number<KPerBlockPerIter>{}),
                             b_block_window_tmp.GetWindowOrigin(),
                             b_block_dstr));

        Array<Array<BBlockWindow, KIterPerWarp>, NIterPerWarp> b_block_windows{{
            make_tile_window(b_block_window_tmp.GetBottomTensorView(),
                             make_tuple(Number<NPerBlockPerIter>{}, Number<KPerBlockPerIter>{}),
                             b_block_window_tmp.GetWindowOrigin(),
                             b_block_dstr)}};

        for(index_t nIter = 0; nIter < NIterPerWarp; nIter++)
        {
            for(index_t kIter = 0; kIter < KIterPerWarp; kIter++)
            {
                move_tile_window(b_block_windows(nIter)(kIter),
                                 {nIter * NPerBlockPerIter, kIter * KPerBlockPerIter});
            }
        }

        static_assert(is_same_v<CDataType, typename WG::CDataType>, "wrong!");

        // Construct C-Block-Tensor
        auto c_block_tensor = make_static_distributed_tensor<CDataType>(c_block_dstr);

        using CWarpDstr = typename WG::CWarpDstr;

        using AWarpTensor = typename WG::AWarpTensor;
        using BWarpTensor = typename WG::BWarpTensor;
        using CWarpTensor = typename WG::CWarpTensor;

        constexpr auto c_warp_y_lengths = to_sequence(CWarpDstr{}.GetYs2DDescriptor().GetLengths());
        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

        // hot loop:
        static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
            static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                // read A warp tensor from A block window
                AWarpTensor a_warp_tensor;

                a_warp_tensor.GetThreadBuffer() = detail::load_thread_data_from_tile_window(
                    a_block_windows(mIter)(kIter));

                static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                    // read B warp tensor from B Block window
                    BWarpTensor b_warp_tensor;

                    b_warp_tensor.GetThreadBuffer() =
                        detail::load_thread_data_from_tile_window(
                            b_block_windows(nIter)(kIter));

                    // read C warp tensor from C block tensor
                    CWarpTensor c_warp_tensor;

                    // warp GEMM
                    if constexpr(KIterPerWarp == 0)
                    {
                        // c = a * b
                        c_warp_tensor = WG{}(a_warp_tensor, b_warp_tensor);
                    }
                    else
                    {
                        // c += a * b
                        c_warp_tensor.GetThreadBuffer() = c_block_tensor.GetYSlicedThreadData(
                            merge_sequences(Sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(Sequence<1, 1>{}, c_warp_y_lengths));

                        WG{}(c_warp_tensor, a_warp_tensor, b_warp_tensor);
                    }

                    // write C warp tensor into C block tensor
                    c_block_tensor.SetYSlicedThreadData(
                        merge_sequences(Sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                        merge_sequences(Sequence<1, 1>{}, c_warp_y_lengths),
                        c_warp_tensor.GetThreadBuffer());
                });
            });
        });

        return c_block_tensor;
    }
#endif
};

} // namespace block
} // namespace tile_program
} // namespace ck
