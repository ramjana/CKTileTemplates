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

template <typename InDstr, index_t... InReduceDimXs>
__host__ __device__ constexpr auto
make_reduce_tile_distribution_encoding(InDstr, Sequence<InReduceDimXs...> reduce_dim_xs_in)
{
#if 1
    constexpr auto I1 = Number<1>{};

    // FIXME
    constexpr index_t max_ndim_rh_minor_in = 20;
    constexpr index_t max_ndim_r_out       = 20;
    constexpr index_t max_ndim_y_out       = 20;

    //
    constexpr index_t ndim_p           = InDstr::NDimP;
    constexpr index_t ndim_x_in        = InDstr::NDimX;
    constexpr index_t ndim_y_in        = InDstr::NDimY;
    constexpr index_t ndim_rh_major_in = InDstr::NDimX + 1;
    constexpr index_t ndim_x_out       = ndim_x_in - sizeof...(InReduceDimXs);

    // is_rh_major_in_for_reduce
    Array<bool, ndim_rh_major_in> is_rh_major_in_for_reduce{false};

    for(index_t i = 0; i < reduce_dim_xs_in.Size(); i++)
    {
        index_t rh_major = reduce_dim_xs_in[i] + 1;

        is_rh_major_in_for_reduce(rh_major) = true;
    }

    // is_y_in_for_reduce
    Array<bool, ndim_y_in> is_y_in_for_reduce{false};

    for(index_t i = 0; i < ndim_y_in; i++)
    {
        index_t rh_major = InDstr::ys_to_rhs_major_[i];

        if(is_rh_major_in_for_reduce[rh_major])
        {
            is_y_in_for_reduce(i) = true;
        }
    }

    // is_rh_minor_in_for_y_reduce
    Array<Array<bool, max_ndim_rh_minor_in>, ndim_rh_major_in> is_rh_minor_in_for_y_reduce{{false}};

    static_for<0, ndim_y_in, 1>{}([&](auto i) {
        index_t rh_major = InDstr::ys_to_rhs_major_[i];
        index_t rh_minor = InDstr::ys_to_rhs_minor_[i];

        if(is_y_in_for_reduce[i])
        {
            is_rh_minor_in_for_y_reduce(rh_major)(rh_minor) = true;
        }
    });

    // in2out_rh_major
    Array<index_t, ndim_rh_major_in> in2out_rh_major{-1};
    index_t cnt_ndim_rh_major_out = 0;

    for(index_t i = 0; i < ndim_rh_major_in; i++)
    {
        if(is_rh_major_in_for_reduce[i])
        {
            in2out_rh_major(i) = 0;
        }
        else
        {
            in2out_rh_major(i) = cnt_ndim_rh_major_out;

            cnt_ndim_rh_major_out++;
        }
    }

    // ndim_rh_major_out
    const index_t ndim_rh_major_out = cnt_ndim_rh_major_out;

    // rs_lengths_out, in2out_rh_minor
    Array<index_t, max_ndim_r_out> rs_lengths_out{-1};
    Array<Array<index_t, max_ndim_rh_minor_in>, ndim_rh_major_in> in2out_rh_minor{{-1}};

    // loop over input R dim
    for(index_t i = 0; i < InDstr::rs_lengths_.Size(); i++)
    {
        // rs_lengths_out
        rs_lengths_out(i) = InDstr::rs_lengths_[i];

        // in2out_rh_minor
        in2out_rh_minor(0)(i) = i;
    }

    // loop over input H Dim
    index_t cnt_ndim_r_out = InDstr::rs_lengths_.Size();

    static_for<1, ndim_rh_major_in, 1>{}([&](auto rh_major_in) {
        constexpr index_t ndim_rh_minor_in = InDstr::hs_lengthss_[rh_major_in - I1].Size();

        if(is_rh_major_in_for_reduce[rh_major_in])
        {
            for(index_t rh_minor_in = 0; rh_minor_in < ndim_rh_minor_in; rh_minor_in++)
            {
                if(not is_rh_minor_in_for_y_reduce[rh_major_in][rh_minor_in])
                {
                    // in2out_rh_minor
                    in2out_rh_minor(rh_major_in)(rh_minor_in) = cnt_ndim_r_out;

                    cnt_ndim_r_out++;
                }
            }
        }
        else
        {
            for(index_t rh_minor_in = 0; rh_minor_in < ndim_rh_minor_in; rh_minor_in++)
            {
                // in2out_rh_minor
                in2out_rh_minor(rh_major_in)(rh_minor_in) = rh_minor_in;
            }
        }
    });

    // ndim_r_out
    const index_t ndim_r_out = cnt_ndim_r_out;

    // hs_lengthss_out
    Array<Array<index_t, max_ndim_rh_minor_in>, ndim_x_out> hs_lengthss_out{{-1}};

    index_t cnt_ndim_x_out = 0;

    static_for<0, ndim_x_in, 1>{}([&](auto i) {
        if(not is_rh_major_in_for_reduce[i + I1])
        {
            static_for<0, InDstr::hs_lengthss_[i].Size(), 1>{}(
                [&](auto j) { hs_lengthss_out(cnt_ndim_x_out)(j) = InDstr::hs_lengthss_[i][j]; });

            cnt_ndim_x_out++;
        }
    });

    // ps_to_rhss_major_out, ps_to_rhss_minor_out
    Array<Array<index_t, max_ndim_rh_minor_in>, ndim_p> ps_to_rhss_major_out{{-1}};
    Array<Array<index_t, max_ndim_rh_minor_in>, ndim_p> ps_to_rhss_minor_out{{-1}};

    static_for<0, ndim_p, 1>{}([&](auto idim_p) {
        static_for<0, InDstr::ps_to_rhss_major_[idim_p].Size(), 1>{}([&](auto idim_low) {
            index_t rh_major_in = InDstr::ps_to_rhss_major_[idim_p][idim_low];
            index_t rh_minor_in = InDstr::ps_to_rhss_minor_[idim_p][idim_low];

            ps_to_rhss_major_out(idim_p)(idim_low) = in2out_rh_major[rh_major_in];
            ps_to_rhss_minor_out(idim_p)(idim_low) = in2out_rh_minor[rh_major_in][rh_minor_in];
        });
    });

    // ys_to_rhs_major_out, ys_to_rhs_minor_out
    Array<index_t, max_ndim_y_out> ys_to_rhs_major_out{-1};
    Array<index_t, max_ndim_y_out> ys_to_rhs_minor_out{-1};

    index_t cnt_ndim_y_out = 0;

    static_for<0, ndim_y_in, 1>{}([&](auto i) {
        if(not is_y_in_for_reduce[i])
        {
            index_t rh_major_in = InDstr::ys_to_rhs_major_[i];
            index_t rh_minor_in = InDstr::ys_to_rhs_minor_[i];

            ys_to_rhs_major_out(cnt_ndim_y_out) = in2out_rh_major[rh_major_in];
            ys_to_rhs_minor_out(cnt_ndim_y_out) = in2out_rh_minor[rh_major_in][rh_minor_in];

            cnt_ndim_y_out++;
        }
    });

    // ndim_y_out
    const index_t ndim_y_out = cnt_ndim_y_out;

    if(ProgramServer::get_block_id() == 0 && ProgramServer::get_thread_id() == 0)
    {
        printf("ndim_rh_major_out: ");
        print(ndim_rh_major_out);
        printf("\n");

        printf("ndim_r_out: ");
        print(ndim_r_out);
        printf("\n");

        printf("ndim_y_out: ");
        print(ndim_y_out);
        printf("\n");

        printf("is_y_in_for_reduce: ");
        print(is_y_in_for_reduce);
        printf("\n");

        printf("is_rh_major_in_for_reduce: ");
        print(is_rh_major_in_for_reduce);
        printf("\n");

        printf("is_rh_minor_in_for_y_reduce: ");
        print(is_rh_minor_in_for_y_reduce);
        printf("\n");

        printf("in2out_rh_major: ");
        print(in2out_rh_major);
        printf("\n");

        printf("in2out_rh_minor: ");
        print(in2out_rh_minor);
        printf("\n");

        printf("hs_lengthss_out: ");
        print(hs_lengthss_out);
        printf("\n");

        printf("ps_to_rhss_major_out: ");
        print(ps_to_rhss_major_out);
        printf("\n");

        printf("ps_to_rhss_minor_out: ");
        print(ps_to_rhss_minor_out);
        printf("\n");

        printf("ys_to_rhs_major_out: ");
        print(ys_to_rhs_major_out);
        printf("\n");

        printf("ys_to_rhs_minor_out: ");
        print(ys_to_rhs_minor_out);
        printf("\n");
    }
#else
    (void)reduce_dim_xs_in;
#endif
}

} // namespace detail
} // namespace tile_program
} // namespace ck
