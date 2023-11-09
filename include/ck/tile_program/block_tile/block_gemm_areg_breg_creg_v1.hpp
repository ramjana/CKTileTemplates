// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"

#include "ck/tile_program/tile/static_tile_distribution_helper.hpp"
#include "ck/tile_program/tile/tile_distribution.hpp"
#include "ck/tile_program/tile/tile_elementwise.hpp"
#include "ck/tile_program/tile/tile_gemm_shape.hpp"
#include "ck/tile_program/warp_tile/warp_gemm.hpp"
#include "ck/tile_program/block_tile/block_gemm_areg_breg_creg_problem.hpp"
#include "ck/tile_program/block_tile/block_gemm_areg_breg_creg_v1_default_policy.hpp"
// #include "ck/tile_program/block_tile/block_gemm_areg_breg_creg_v1_iteratek_policy.hpp"

namespace ck {
namespace tile_program {
namespace block {

// A is block distributed tensor
// B is block distributed tensor
// C is block distributed tensor
template <typename Problem_, typename Policy_ = BlockGemmARegBRegCRegV1DefaultPolicy>
struct BlockGemmARegBRegCRegV1
{
    using ADataType       = remove_cvref_t<typename Problem_::ADataType>;
    using BDataType       = remove_cvref_t<typename Problem_::BDataType>;
    using CDataType       = remove_cvref_t<typename Problem_::CDataType>;
    using BlockGemmShape  = remove_cvref_t<typename Problem_::BlockGemmShape>;
    using BlockGemmPolicy = Policy_;

    static constexpr index_t kBlockSize = Problem_::kBlockSize;

    // C += A * B
    template <typename CBlockTensor, typename ABlockTensorTmp, typename BBlockTensorTmp>
    __device__ void operator()(CBlockTensor& c_block_tensor,
                               const ABlockTensorTmp& a_block_tensor_tmp,
                               const BBlockTensorTmp& b_block_tensor_tmp) const
    {
        static_assert(is_same_v<ADataType, remove_cv_t<typename ABlockTensorTmp::DataType>> &&
                          is_same_v<BDataType, remove_cv_t<typename BBlockTensorTmp::DataType>> &&
                          is_same_v<CDataType, remove_cv_t<typename CBlockTensor::DataType>>,
                      "wrong!");

        constexpr index_t MPerBlock = ABlockTensorTmp{}.GetLengths()[Number<0>{}];
        constexpr index_t NPerBlock = BBlockTensorTmp{}.GetLengths()[Number<0>{}];
        constexpr index_t KPerBlock = ABlockTensorTmp{}.GetLengths()[Number<1>{}];

        static_assert(MPerBlock == BlockGemmShape::kM && NPerBlock == BlockGemmShape::kN &&
                          KPerBlock == BlockGemmShape::kK,
                      "wrong!");

        constexpr auto config = BlockGemmPolicy::template GetWarpGemmMWarpNWarp<Problem_>();

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
            Tuple<Sequence<1, 0>>,
            Tuple<Sequence<1, 0>>,
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

        // constrcut from A-block-tensor from A-Block-tensor-tmp
        // FIXME: need method to check a_block_tensor and a_block_tensor_tmp have equivalent
        // distribution
        auto a_block_tensor =
            make_static_distributed_tensor<typename ABlockTensorTmp::DataType>(a_block_dstr);
        
        auto b_block_tensor =
            make_static_distributed_tensor<typename BBlockTensorTmp::DataType>(b_block_dstr);

        a_block_tensor.GetThreadBuffer() = a_block_tensor_tmp.GetThreadBuffer();
        b_block_tensor.GetThreadBuffer() = b_block_tensor_tmp.GetThreadBuffer();


        // check C-block-distribution
        static_assert(is_same_v<remove_cvref_t<decltype(c_block_dstr_encode)>,
                                remove_cvref_t<decltype(CBlockTensor::GetTileDistribution()
                                                            .GetStaticTileDistributionEncoding())>>,
                      "wrong!");

        using AWarpDstr = typename WG::AWarpDstr;
        using BWarpDstr = typename WG::BWarpDstr;
        using CWarpDstr = typename WG::CWarpDstr;

        using AWarpTensor = typename WG::AWarpTensor;
        using BWarpTensor = typename WG::BWarpTensor;
        using CWarpTensor = typename WG::CWarpTensor;

        constexpr auto a_warp_y_lengths = to_sequence(AWarpDstr{}.GetYs2DDescriptor().GetLengths());
        constexpr auto b_warp_y_lengths = to_sequence(BWarpDstr{}.GetYs2DDescriptor().GetLengths());
        constexpr auto c_warp_y_lengths = to_sequence(CWarpDstr{}.GetYs2DDescriptor().GetLengths());

        constexpr auto a_warp_y_index_zeros = uniform_sequence_gen_t<AWarpDstr::NDimY, 0>{};
        constexpr auto b_warp_y_index_zeros = uniform_sequence_gen_t<BWarpDstr::NDimY, 0>{};
        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

        // hot loop:
        static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
            static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                // read A warp tensor from A block tensor
                AWarpTensor a_warp_tensor;

                a_warp_tensor.GetThreadBuffer() = a_block_tensor.GetYSlicedThreadData(
                    merge_sequences(Sequence<mIter, kIter>{}, a_warp_y_index_zeros),
                    merge_sequences(Sequence<1, 1>{}, a_warp_y_lengths));

                static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                    BWarpTensor b_warp_tensor;

                    b_warp_tensor.GetThreadBuffer() = b_block_tensor.GetYSlicedThreadData(
                    merge_sequences(Sequence<nIter, kIter>{}, b_warp_y_index_zeros),
                    merge_sequences(Sequence<1, 1>{}, b_warp_y_lengths));

                    // read C warp tensor from C block tensor
                    CWarpTensor c_warp_tensor;

                    c_warp_tensor.GetThreadBuffer() = c_block_tensor.GetYSlicedThreadData(
                        merge_sequences(Sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                        merge_sequences(Sequence<1, 1>{}, c_warp_y_lengths));
#if 0
        printf("(m: %02d, n: %02d, k:%02d) Blockid: %02d, Tid: %03d, a_thread_buf: %04x %04x %04x %04x|\n",
            mIter.value, nIter.value, kIter.value,
            get_block_1d_id(), get_thread_local_1d_id(),
            *(reinterpret_cast<const uint16_t*>(&(a_warp_tensor.GetThreadBuffer()[Number<0>{}]))),
            *(reinterpret_cast<const uint16_t*>(&(a_warp_tensor.GetThreadBuffer()[Number<1>{}]))),
            *(reinterpret_cast<const uint16_t*>(&(a_warp_tensor.GetThreadBuffer()[Number<2>{}]))),
            *(reinterpret_cast<const uint16_t*>(&(a_warp_tensor.GetThreadBuffer()[Number<3>{}])))
            );
#endif
#if 0
        printf("(m: %02d, n: %02d, k:%02d) Blockid: %02d, Tid: %03d, b_thread_buf: %04x %04x %04x %04x %04x %04x %04x %04x|\n",
            mIter.value, nIter.value, kIter.value, get_block_1d_id(), get_thread_local_1d_id(),
            *(reinterpret_cast<const uint16_t*>(&(b_warp_tensor.GetThreadBuffer()[Number<0>{}]))),
            *(reinterpret_cast<const uint16_t*>(&(b_warp_tensor.GetThreadBuffer()[Number<1>{}]))),
            *(reinterpret_cast<const uint16_t*>(&(b_warp_tensor.GetThreadBuffer()[Number<2>{}]))),
            *(reinterpret_cast<const uint16_t*>(&(b_warp_tensor.GetThreadBuffer()[Number<3>{}]))),
            *(reinterpret_cast<const uint16_t*>(&(b_warp_tensor.GetThreadBuffer()[Number<4>{}]))),
            *(reinterpret_cast<const uint16_t*>(&(b_warp_tensor.GetThreadBuffer()[Number<5>{}]))),
            *(reinterpret_cast<const uint16_t*>(&(b_warp_tensor.GetThreadBuffer()[Number<6>{}]))),
            *(reinterpret_cast<const uint16_t*>(&(b_warp_tensor.GetThreadBuffer()[Number<7>{}])))
            );
#endif
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

    // C = A * B
    template <typename ABlockTensorTmp, typename BBlockTensorTmp>
    __device__ auto operator()(const ABlockTensorTmp& a_block_tensor_tmp,
                               const BBlockTensorTmp& b_block_tensor_tmp) const
    {
        static_assert(is_same_v<ADataType, remove_cv_t<typename ABlockTensorTmp::DataType>> &&
                          is_same_v<BDataType, remove_cv_t<typename BBlockTensorTmp::DataType>>,
                      "wrong!");

        constexpr index_t MPerBlock = ABlockTensorTmp{}.GetLengths()[Number<0>{}];
        constexpr index_t NPerBlock = BBlockTensorTmp{}.GetLengths()[Number<0>{}];
        constexpr index_t KPerBlock = ABlockTensorTmp{}.GetLengths()[Number<1>{}];

        static_assert(MPerBlock == BlockGemmShape::kM && NPerBlock == BlockGemmShape::kN &&
                          KPerBlock == BlockGemmShape::kK,
                      "wrong!");

        constexpr auto config = BlockGemmPolicy::template GetWarpGemmMWarpNWarp<Problem_>();

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
        
        // Be careful about Warp Mapping, in M->N order
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

        // constrcut from A-block-tensor from A-Block-tensor-tmp
        // FIXME: need method to check a_block_tensor and a_block_tensor_tmp have equivalent
        // distribution
        auto a_block_tensor =
            make_static_distributed_tensor<typename ABlockTensorTmp::DataType>(a_block_dstr);
        
        auto b_block_tensor =
            make_static_distributed_tensor<typename BBlockTensorTmp::DataType>(b_block_dstr);

        a_block_tensor.GetThreadBuffer() = a_block_tensor_tmp.GetThreadBuffer();
        b_block_tensor.GetThreadBuffer() = b_block_tensor_tmp.GetThreadBuffer();

        using AWarpDstr = typename WG::AWarpDstr;
        using BWarpDstr = typename WG::BWarpDstr;
        using CWarpDstr = typename WG::CWarpDstr;

        using AWarpTensor = typename WG::AWarpTensor;
        using BWarpTensor = typename WG::BWarpTensor;
        using CWarpTensor = typename WG::CWarpTensor;

        constexpr auto a_warp_y_lengths = to_sequence(AWarpDstr{}.GetYs2DDescriptor().GetLengths());
        constexpr auto b_warp_y_lengths = to_sequence(BWarpDstr{}.GetYs2DDescriptor().GetLengths());
        constexpr auto c_warp_y_lengths = to_sequence(CWarpDstr{}.GetYs2DDescriptor().GetLengths());

        constexpr auto a_warp_y_index_zeros = uniform_sequence_gen_t<AWarpDstr::NDimY, 0>{};
        constexpr auto b_warp_y_index_zeros = uniform_sequence_gen_t<BWarpDstr::NDimY, 0>{};
        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

        // Construct C-Block-Tensor
        auto c_block_tensor = make_static_distributed_tensor<CDataType>(c_block_dstr);

        // hot loop:
        static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
            static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                // read A warp tensor from A block tensor
                AWarpTensor a_warp_tensor;

                a_warp_tensor.GetThreadBuffer() = a_block_tensor.GetYSlicedThreadData(
                    merge_sequences(Sequence<mIter, kIter>{}, a_warp_y_index_zeros),
                    merge_sequences(Sequence<1, 1>{}, a_warp_y_lengths));

                static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                    BWarpTensor b_warp_tensor;

                    b_warp_tensor.GetThreadBuffer() = b_block_tensor.GetYSlicedThreadData(
                    merge_sequences(Sequence<nIter, kIter>{}, b_warp_y_index_zeros),
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

        return c_block_tensor;
    }
};

} // namespace block
} // namespace tile_program
} // namespace ck
