// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"

#include "ck/tile_program/tile/tile_distribution.hpp"
#include "ck/tile_program/tile/tile_elementwise.hpp"
#include "ck/tile_program/tile/tile_gemm_shape.hpp"
#include "ck/tile_program/warp_tile/warp_gemm.hpp"
#include "ck/tile_program/block_tile_pipeline/block_gemm_pipeline_agmem_bgmem_creg_v1.hpp"
#include "ck/tile_program/block_tile_pipeline/block_gemm_pipeline_agmem_bgmem_creg_v2.hpp"
#include "ck/tile_program/grid/grid_gemm_problem.hpp"
#include "ck/tile_program/grid/grid_gemm_v1.hpp"
#include "ck/tile_program/grid/grid_gemm_v1_default_policy.hpp"

// C = A * B
template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename AElementFunction,
          typename BElementFunction,
          typename CElementFunction,
          ck::index_t kAAlignment,
          ck::index_t kBAlignment,
          ck::index_t kCAlignment,
          ck::index_t kBlockSize_,
          ck::index_t kMPerBlock_,
          ck::index_t kNPerBlock_,
          ck::index_t kKPerBlock_>
struct Gemm
{
    using GridGemmProblem = ck::tile_program::grid::GridGemmProblem<ADataType,
                                                                    BDataType,
                                                                    AccDataType,
                                                                    CDataType,
                                                                    AElementFunction,
                                                                    BElementFunction,
                                                                    CElementFunction>;
    using GridGemmPolicy = ck::GridGemmV1DefaultPolicy<kBlockSize_, kMPerBlock_, kNPerBlock_, kKPerBlock_>;

    using GridGemm = ck::GridGemmV1<GridGemmProblem, GridGemmPolicy>;

    __device__ void operator()(const ADataType* p_a,
                               const BDataType* p_b,
                               CDataType* p_c,
                               const ck::index_t M,
                               const ck::index_t N,
                               const ck::index_t K,
                               const ck::index_t Lda,
                               const ck::index_t Ldb,
                               const ck::index_t Ldc,
                               const AElementFunction& a_element_func,
                               const BElementFunction& b_element_func,
                               const CElementFunction& c_element_func) const
    {
        using namespace ck;
        using namespace ck::tile_program;
        using namespace ck::tile_program::block;

        const auto a_dram = [&] {
            if constexpr(is_same_v<ALayout, ck::tensor_layout::gemm::RowMajor>)
            {
                return make_naive_tensor_view<AddressSpaceEnum::Global>(
                    p_a, make_tuple(M, K), make_tuple(Lda, 1), Number<kAAlignment>{}, Number<1>{});
            }
            else
            {
                const auto a_k_m_desc = make_naive_tensor_view<AddressSpaceEnum::Global>(
                    p_a, make_tuple(K, M), make_tuple(Lda, 1), Number<kAAlignment>{}, Number<1>{});

                return transform_tensor_view(
                    a_k_m_desc,
                    make_tuple(make_pass_through_transform(M), make_pass_through_transform(K)),
                    make_tuple(Sequence<1>{}, Sequence<0>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));
            }
        }();

        const auto b_dram = [&] {
            if constexpr(is_same_v<BLayout, ck::tensor_layout::gemm::ColumnMajor>)
            {
                return make_naive_tensor_view<AddressSpaceEnum::Global>(
                    p_b, make_tuple(N, K), make_tuple(Ldb, 1), Number<kBAlignment>{}, Number<1>{});
            }
            else
            {
                const auto b_k_n_desc = make_naive_tensor_view<AddressSpaceEnum::Global>(
                    p_b, make_tuple(K, N), make_tuple(Ldb, 1), Number<kBAlignment>{}, Number<1>{});

                return transform_tensor_view(
                    b_k_n_desc,
                    make_tuple(make_pass_through_transform(N), make_pass_through_transform(K)),
                    make_tuple(Sequence<1>{}, Sequence<0>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));
            }
        }();

        const auto c_dram = [&] {
            if constexpr(is_same_v<CLayout, ck::tensor_layout::gemm::RowMajor>)
            {
                return make_naive_tensor_view<AddressSpaceEnum::Global>(
                    p_c, make_tuple(M, N), make_tuple(Ldc, 1), Number<kCAlignment>{}, Number<1>{});
            }
            else
            {
                const auto c_n_m_desc = make_naive_tensor_view<AddressSpaceEnum::Global>(
                    p_c, make_tuple(N, M), make_tuple(Ldc, 1), Number<kCAlignment>{}, Number<1>{});

                return transform_tensor_view(
                    c_n_m_desc,
                    make_tuple(make_pass_through_transform(M), make_pass_through_transform(N)),
                    make_tuple(Sequence<1>{}, Sequence<0>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));
            }
        }();

        GridGemm{}(a_dram, b_dram, c_dram, a_element_func, b_element_func, c_element_func);
    }
};
