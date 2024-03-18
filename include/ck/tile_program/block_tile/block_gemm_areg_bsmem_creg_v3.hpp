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
#include "ck/tile_program/block_tile/block_gemm_areg_bsmem_creg_problem.hpp"
#include "ck/tile_program/block_tile/block_gemm_areg_bsmem_creg_v3_default_policy.hpp"

namespace ck {
namespace tile_program {
namespace block {

// A is block distributed tensor
// B is block window on shared memory
// C is block distributed tensor
template <typename Problem_, typename Policy_ = BlockGemmARegBSmemCRegV3DefaultPolicy>
struct BlockGemmARegBSmemCRegV3
{
    using Problem        = remove_cvref_t<Problem_>;
    using Policy         = remove_cvref_t<Policy_>;
    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType      = remove_cvref_t<typename Problem::BDataType>;
    using CDataType      = remove_cvref_t<typename Problem::CDataType>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

    static constexpr index_t kBlockSize = Problem::kBlockSize;

    // C += A * B
    template <typename CBlockTensor, typename ABlockTensorTmp, typename BBlockWindowTmp>
    __device__ void operator()(CBlockTensor& c_block_tensor,
                               const ABlockTensorTmp& a_block_tensor_tmp,
                               const BBlockWindowTmp& b_block_window_tmp) const
    {
        static_assert(is_same_v<ADataType, remove_cv_t<typename ABlockTensorTmp::DataType>> &&
                          is_same_v<BDataType, remove_cv_t<typename BBlockWindowTmp::DataType>> &&
                          is_same_v<CDataType, remove_cv_t<typename CBlockTensor::DataType>>,
                      "wrong!");

        constexpr index_t MPerBlock = ABlockTensorTmp{}.GetLengths()[Number<0>{}];
        constexpr index_t NPerBlock = BBlockWindowTmp{}.GetWindowLengths()[Number<0>{}];
        constexpr index_t KPerBlock = ABlockTensorTmp{}.GetLengths()[Number<1>{}];

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
	constexpr index_t prefetch_buf = 2;

        constexpr index_t NPerBlockPerIter = NPerBlock / NIterPerWarp;
        constexpr index_t KPerBlockPerIter = KPerBlock / KIterPerWarp;

        const index_t iNWarp = get_warp_id() % NWarp;

        constexpr auto a_block_outer_dstr_encoding = StaticTileDistributionEncoding<
            Sequence<NWarp>,
            Tuple<Sequence<MIterPerWarp, MWarp>, Sequence<KIterPerWarp>>,
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

        constexpr auto c_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            c_block_outer_dstr_encoding, typename WG::CWarpDstrEncoding{});

        constexpr auto a_block_dstr = make_static_tile_distribution(a_block_dstr_encode);

        // constrcut from A-block-tensor from A-Block-tensor-tmp
        // FIXME: need method to check a_block_tensor and a_block_tensor_tmp have equivalent
        // distribution
        auto a_block_tensor =
            make_static_distributed_tensor<typename ABlockTensorTmp::DataType>(a_block_dstr);

        a_block_tensor.GetThreadBuffer() = a_block_tensor_tmp.GetThreadBuffer();

        // construct B-warp-window
        auto b_warp_window_tmp = make_tile_window(
            b_block_window_tmp.GetBottomTensorView(),
            make_tuple(Number<WG::kN>{}, Number<WG::kK>{}),
            b_block_window_tmp.GetWindowOrigin() + MultiIndex<2>{iNWarp * WG::kN, 0},
            make_static_tile_distribution(typename WG::BWarpDstrEncoding{}));

#if 0 // FIXME: using Array will cause register spill
        Array<Array<decltype(b_warp_window_tmp), KIterPerWarp>, NIterPerWarp> b_warp_windows{
            {b_warp_window_tmp}};

        for(index_t nIter = 0; nIter < NIterPerWarp; nIter++)
        {
            for(index_t kIter = 0; kIter < KIterPerWarp; kIter++)
            {
                move_tile_window(b_warp_windows(nIter)(kIter),
                                 {nIter * NPerBlockPerIter, kIter * KPerBlockPerIter});
            }
        }
#else
        StaticallyIndexedArray<StaticallyIndexedArray<decltype(b_warp_window_tmp), KIterPerWarp>,
                               NIterPerWarp>
            b_warp_windows;

        static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                b_warp_windows(nIter)(kIter) = b_warp_window_tmp;

                move_tile_window(b_warp_windows(nIter)(kIter),
                                 {nIter * NPerBlockPerIter, kIter * KPerBlockPerIter});
            });
        });
#endif

        // check C-block-distribution
        static_assert(is_same_v<remove_cvref_t<decltype(c_block_dstr_encode)>,
                                remove_cvref_t<decltype(CBlockTensor::GetTileDistribution()
                                                            .GetStaticTileDistributionEncoding())>>,
                      "wrong!");

        using AWarpDstr = typename WG::AWarpDstr;
        using CWarpDstr = typename WG::CWarpDstr;

        using AWarpTensor = typename WG::AWarpTensor;
        using BWarpTensor = typename WG::BWarpTensor;
        using CWarpTensor = typename WG::CWarpTensor;

        constexpr auto a_warp_y_lengths = to_sequence(AWarpDstr{}.GetYs2DDescriptor().GetLengths());
        constexpr auto c_warp_y_lengths = to_sequence(CWarpDstr{}.GetYs2DDescriptor().GetLengths());

        constexpr auto a_warp_y_index_zeros = uniform_sequence_gen_t<AWarpDstr::NDimY, 0>{};
        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

        StaticallyIndexedArray<BWarpTensor, NIterPerWarp> b_warp_tensor_0;
        StaticallyIndexedArray<BWarpTensor, NIterPerWarp> b_warp_tensor_1;

	//Prefetch[0] for B_warp_tenso
        static_for<0, 1, 1>{}([&](auto kiter) {
            static_for<0, NIterPerWarp, 1>{}([&](auto niter) {
                 // read B warp tensor from B Block window -- buffer[0]
                 b_warp_tensor_0(niter) = load_tile(b_warp_windows(niter)(Number<kiter+0>{}));
            });
        });

	//prefetch B for K=i iteration
	__builtin_amdgcn_sched_barrier(0);

        // hot loop:
        static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
            static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                // read A warp tensor from A block tensor
                AWarpTensor a_warp_tensor;

                a_warp_tensor.GetThreadBuffer() = a_block_tensor.GetYSlicedThreadData(
                     merge_sequences(Sequence<mIter, kIter>{}, a_warp_y_index_zeros),
                     merge_sequences(Sequence<1, 1>{}, a_warp_y_lengths));


	        auto prefetch_idx = Number<kIter%prefetch_buf>{};
                static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                    // read C warp tensor from C block tensor
                    CWarpTensor c_warp_tensor;

                    c_warp_tensor.GetThreadBuffer() = c_block_tensor.GetYSlicedThreadData(
                        merge_sequences(Sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                        merge_sequences(Sequence<1, 1>{}, c_warp_y_lengths));
		    //constexpr index_t prefetchCnt = 2; 
                    // warp GEMM
	 	    if constexpr (KIterPerWarp > (kIter+1))
		    {
		        if (prefetch_idx == 0)
                        {
			    if constexpr (nIter == 0 && NIterPerWarp <= 2)
			    {
                                WG{}(c_warp_tensor, a_warp_tensor, b_warp_tensor_0(nIter));
                                static_for<0, NIterPerWarp, 1>{}([&](auto niter) {
                                    b_warp_tensor_1(niter) = load_tile(b_warp_windows(niter)(Number<kIter+1>{}));
			        });
	                        __builtin_amdgcn_sched_barrier(0x0);
			    }
			    else if constexpr (nIter == 0 && NIterPerWarp > 2 )
			    {
                                WG{}(c_warp_tensor, a_warp_tensor, b_warp_tensor_0(nIter));
                                static_for<0, 2, 1>{}([&](auto niter) {
                                    b_warp_tensor_1(niter) = load_tile(b_warp_windows(niter)(Number<kIter+1>{}));
			        });
	                        __builtin_amdgcn_sched_barrier(0x0);
		            }
			    else if constexpr (nIter == 1 && NIterPerWarp > 2 )
			    {
                                WG{}(c_warp_tensor, a_warp_tensor, b_warp_tensor_0(nIter));
                                static_for<0, 2, 1>{}([&](auto niter) {
                                    b_warp_tensor_1(Number<niter+2>{}) = load_tile(b_warp_windows(Number<niter+2>{})(Number<kIter+1>{}));
			        });
	                        __builtin_amdgcn_sched_barrier(0x0);
		            }
			    else if constexpr (nIter > 1) 
	                    {
                                WG{}(c_warp_tensor, a_warp_tensor, b_warp_tensor_0(nIter));
		            }
                        }
			else
                        {
			    //prefetch next K buffer
			    if constexpr (nIter == 0 && NIterPerWarp <= 2 )
			    {
                                WG{}(c_warp_tensor, a_warp_tensor, b_warp_tensor_1(nIter));
                                static_for<0, NIterPerWarp, 1>{}([&](auto niter) {
                                    b_warp_tensor_0(niter) = load_tile(b_warp_windows(niter)(Number<kIter+1>{}));
			        });    
	                        __builtin_amdgcn_sched_barrier(0x0);
			    }
			    else if constexpr (nIter == 0 && NIterPerWarp > 2 )
			    {
                                WG{}(c_warp_tensor, a_warp_tensor, b_warp_tensor_1(nIter));
                                static_for<0, 2, 1>{}([&](auto niter) {
                                    b_warp_tensor_0(niter) = load_tile(b_warp_windows(niter)(Number<kIter+1>{}));
			        });    
	                        __builtin_amdgcn_sched_barrier(0x0);
			    }
			    if constexpr (nIter == 1 && NIterPerWarp > 2 )
			    {
                                WG{}(c_warp_tensor, a_warp_tensor, b_warp_tensor_1(nIter));
                                static_for<0, 2, 1>{}([&](auto niter) {
                                    b_warp_tensor_0(Number<niter+2>{}) = load_tile(b_warp_windows(Number<niter+2>{})(Number<kIter+1>{}));
			        });    
	                        __builtin_amdgcn_sched_barrier(0x0);
			    }
			    else if constexpr (nIter > 1) 
		            {
                                WG{}(c_warp_tensor, a_warp_tensor, b_warp_tensor_1(nIter));
		            }

		        }
		    }
	            else
		    {
		        if (prefetch_idx)
                        {
                            WG{}(c_warp_tensor, a_warp_tensor, b_warp_tensor_1(nIter));
                        }
                        else
                        {
                            WG{}(c_warp_tensor, a_warp_tensor, b_warp_tensor_0(nIter));
                        }
		    }
                    // WG{}(c_warp_tensor, a_warp_tensor, b_warp_tensor_array[nIter]);

                    // write C warp tensor into C block tensor
                    c_block_tensor.SetYSlicedThreadData(
                        merge_sequences(Sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                        merge_sequences(Sequence<1, 1>{}, c_warp_y_lengths),
                        c_warp_tensor.GetThreadBuffer());
                });
            });
	    __builtin_amdgcn_sched_barrier(0);
        });
    }

    __device__ constexpr auto MakeCBlockTile() const
    {
        constexpr index_t MPerBlock = BlockGemmShape::kM;
        constexpr index_t NPerBlock = BlockGemmShape::kN;

        constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();

        using WG = remove_cvref_t<decltype(config.template At<0>())>;

        constexpr index_t MWarp = config.template At<1>();
        constexpr index_t NWarp = config.template At<2>();

        constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WG::kM);
        constexpr index_t NIterPerWarp = NPerBlock / (NWarp * WG::kN);
        // constexpr index_t KIterPerWarp = KPerBlock / WG::kK;

        constexpr auto c_block_outer_dstr_encoding = StaticTileDistributionEncoding<
            Sequence<>,
            Tuple<Sequence<MIterPerWarp, MWarp>, Sequence<NIterPerWarp, NWarp>>,
            Tuple<Sequence<1, 2>>,
            Tuple<Sequence<1, 1>>,
            Sequence<1, 2>,
            Sequence<0, 0>>{};

        constexpr auto c_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            c_block_outer_dstr_encoding, typename WG::CWarpDstrEncoding{});
        constexpr auto c_block_dstr = make_static_tile_distribution(c_block_dstr_encode);
        auto c_block_tensor         = make_static_distributed_tensor<CDataType>(c_block_dstr);
        return c_block_tensor;
    }

    // C = A * B
    template <typename ABlockTensorTmp, typename BBlockWindowTmp>
    __device__ auto operator()(const ABlockTensorTmp& a_block_tensor_tmp,
                               const BBlockWindowTmp& b_block_window_tmp) const
    {
        auto c_block_tensor = MakeCBlockTile();
        operator()(c_block_tensor, a_block_tensor_tmp, b_block_window_tmp);
        return c_block_tensor;
    }
};

} // namespace block
} // namespace tile_program
} // namespace ck
