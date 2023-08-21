// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"

#include "ck/tile_program/tile/tile_distribution.hpp"

namespace ck {
namespace tile_program {
namespace detail {

template <typename OuterDstr, typename InnerDstr>
__host__ __device__ constexpr auto make_embed_tile_distribution_encoding(OuterDstr, InnerDstr)
{
    static_assert(OuterDstr::NDimX == InnerDstr::NDimX, "wrong!");

    constexpr index_t NDimHMajor = OuterDstr::NDimX;

    using RsLengths =
        sequence_merge_t<typename OuterDstr::RsLengths, typename InnerDstr::RsLengths>;

    constexpr auto hs_lengthss = generate_tuple(
        [&](auto i) {
            return merge_sequences(typename OuterDstr::HsLengthss{}[i],
                                   typename InnerDstr::HsLengthss{}[i]);
        },
        Number<NDimHMajor>{});

    //
    constexpr auto rhs_major_2_ndim_outer_rhs_minor = [&]() {
        Array<index_t, NDimHMajor + 1> rhs_major_2_ndim_outer_rhs_minor_;

        // R dimension
        rhs_major_2_ndim_outer_rhs_minor_(0) = OuterDstr::RsLengths::Size();

        // Hs dimensions
        static_for<0, NDimHMajor, 1>{}([&](auto i) {
            rhs_major_2_ndim_outer_rhs_minor_(i + 1) = typename OuterDstr::HsLengthss{}[i].Size();
        });

        return rhs_major_2_ndim_outer_rhs_minor_;
    }();

    // Ps2RHssMinor
    constexpr auto updated_inner_ps_2_rhss_minor = generate_tuple(
        [&](auto p) {
            constexpr auto inner_p_2_rhss_major = typename InnerDstr::Ps2RHssMajor{}[p];
            constexpr auto inner_p_2_rhss_minor = typename InnerDstr::Ps2RHssMinor{}[p];

            constexpr index_t ndim_tmp = inner_p_2_rhss_minor.Size();

            constexpr auto updated_inner_p_2_rhss_minor = [&]() {
                Array<index_t, ndim_tmp> updated_inner_p_2_rhss_minor_;

                for(index_t i = 0; i < ndim_tmp; i++)
                {
                    index_t rh_major = inner_p_2_rhss_major[i];

                    index_t ndim_outer_h_minor = rhs_major_2_ndim_outer_rhs_minor[rh_major];

                    updated_inner_p_2_rhss_minor_(i) = inner_p_2_rhss_minor[i] + ndim_outer_h_minor;
                }

                return updated_inner_p_2_rhss_minor_;
            }();

            return TO_SEQUENCE(updated_inner_p_2_rhss_minor, ndim_tmp);
        },
        Number<InnerDstr::NDimP>{});

    // Ys2RHsMinor
    constexpr auto updated_inner_ys_2_rhs_minor = [&]() {
        constexpr auto inner_ys_2_rhs_major = typename InnerDstr::Ys2RHsMajor{};
        constexpr auto inner_ys_2_rhs_minor = typename InnerDstr::Ys2RHsMinor{};

        constexpr index_t ndim_tmp = inner_ys_2_rhs_minor.Size();

        constexpr auto updated_inner_ys_2_rhs_minor_ = [&]() {
            Array<index_t, ndim_tmp> updated_inner_ys_2_rhs_minor__;

            for(index_t i = 0; i < ndim_tmp; i++)
            {
                index_t rh_major = inner_ys_2_rhs_major[i];

                index_t ndim_outer_h_minor = rhs_major_2_ndim_outer_rhs_minor[rh_major];

                updated_inner_ys_2_rhs_minor__(i) = inner_ys_2_rhs_minor[i] + ndim_outer_h_minor;
            }

            return updated_inner_ys_2_rhs_minor__;
        }();

        return TO_SEQUENCE(updated_inner_ys_2_rhs_minor_, ndim_tmp);
    }();

    //
    constexpr auto ps_2_rhss_major =
        container_concat(typename OuterDstr::Ps2RHssMajor{}, typename InnerDstr::Ps2RHssMajor{});

    constexpr auto ps_2_rhss_minor =
        container_concat(typename OuterDstr::Ps2RHssMinor{}, updated_inner_ps_2_rhss_minor);

    //
    constexpr auto ys_2_rhs_major =
        merge_sequences(typename OuterDstr::Ys2RHsMajor{}, typename InnerDstr::Ys2RHsMajor{});

    constexpr auto ys_2_rhs_minor =
        merge_sequences(typename OuterDstr::Ys2RHsMinor{}, updated_inner_ys_2_rhs_minor);

    return StaticTileDistributionEncoding<RsLengths,
                                          remove_cvref_t<decltype(hs_lengthss)>,
                                          remove_cvref_t<decltype(ps_2_rhss_major)>,
                                          remove_cvref_t<decltype(ps_2_rhss_minor)>,
                                          remove_cvref_t<decltype(ys_2_rhs_major)>,
                                          remove_cvref_t<decltype(ys_2_rhs_minor)>>{};
}

#if 0
template <typename InDstr, index_t... ReduceDims>
__host__ __device__ constexpr auto make_reduce_tile_distribution_encoding(InDstr,
                                                                          Sequence<ReduceDims...>)
{


    return OutDstr;
}
#endif

} // namespace detail
} // namespace tile_program
} // namespace ck
