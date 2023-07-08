// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"

namespace ck {
namespace tile_program {
namespace block {

namespace detail {

template <index_t NDimMax>
__host__ __device__ constexpr auto make_sequential_index(index_t ibegin, index_t iend)
{
    Array<index_t, NDimMax> arr{0};

    for(index_t i = 0; i < iend - ibegin; ++i)
    {
        arr(i) = ibegin + i;
    }

    return arr;
}

// this returns a constexpr encoding of BlockTensorDistribution
// TODO: reimplement as Hierachical-Distribution
template <typename StaticTensorDistributionEncoding_>
__host__ __device__ constexpr auto
    make_adaptor_encoding_for_tensor_distribution(StaticTensorDistributionEncoding_)
{
    using RsLengths    = typename StaticTensorDistributionEncoding_::RsLengths;
    using HsLengthss   = typename StaticTensorDistributionEncoding_::HsLengthss;
    using Ps2RHssMajor = typename StaticTensorDistributionEncoding_::Ps2RHssMajor;
    using Ps2RHssMinor = typename StaticTensorDistributionEncoding_::Ps2RHssMinor;
    using Ys2RHsMajor  = typename StaticTensorDistributionEncoding_::Ys2RHsMajor;
    using Ys2RHsMinor  = typename StaticTensorDistributionEncoding_::Ys2RHsMinor;

    constexpr index_t kMaxNumTransforms = 20;
    constexpr index_t kMaxMetaDataSize  = 128;
    constexpr index_t kMaxNumDim        = 10;

    using Name     = IndexTransformEnum;
    using MetaData = MetaDataBuffer<kMaxMetaDataSize>;
    using NumDim   = index_t;
    using Dims     = Array<index_t, kMaxNumDim>;
    using Lengths  = Array<index_t, kMaxNumDim>;

    // window Adaptor
    //   bottom dims [x0, x1, x2, ...]
    //   top dims [p0, p1, ..., y0, y1, ...]
    constexpr index_t ndim_x = HsLengthss::Size();

    // Dim Ids: [idim_x_major, idim_x_minor] to [idim_hidden]
    Array<Array<index_t, kMaxNumDim>, ndim_x + 1> rh_major_rh_minor_to_hidden_ids;
    Array<Array<index_t, kMaxNumDim>, ndim_x + 1> rh_major_rh_minor_to_hidden_lengths;

    auto trans = Array<Tuple<Name, MetaData, NumDim, Dims, NumDim, Dims>, kMaxNumTransforms>{};

    index_t num_tran       = 0;
    index_t hidden_dim_cnt = ndim_x;

    // this is Replicate transform
    {
        constexpr index_t ndim_r_minor = RsLengths::Size();

        constexpr auto r_minor_lengths = RsLengths{};

        trans(num_tran++) = {
            IndexTransformEnum::Replicate,
            MetaData{to_array<index_t, ndim_r_minor>(r_minor_lengths)},
            NumDim{0},
            Dims{},
            NumDim{ndim_r_minor},
            make_sequential_index<kMaxNumDim>(hidden_dim_cnt, hidden_dim_cnt + ndim_r_minor)};

        for(index_t i = 0; i < ndim_r_minor; ++i)
        {
            rh_major_rh_minor_to_hidden_ids(0)(i)     = hidden_dim_cnt;
            rh_major_rh_minor_to_hidden_lengths(0)(i) = r_minor_lengths[i];

            hidden_dim_cnt++;
        }
    };

    // these are Unmerge transforms for X dimesions
    static_for<0, ndim_x, 1>{}([&trans,
                                &num_tran,
                                &hidden_dim_cnt,
                                &rh_major_rh_minor_to_hidden_ids,
                                &rh_major_rh_minor_to_hidden_lengths](auto idim_x) {
        constexpr auto h_minor_lengths = tuple_element_t<idim_x, HsLengthss>{};

        constexpr index_t ndim_h_minor = h_minor_lengths.Size();

        trans(num_tran++) = {
            IndexTransformEnum::UnMerge,
            MetaData{to_array<index_t, ndim_h_minor>(h_minor_lengths)},
            NumDim{1},
            Dims{idim_x},
            NumDim{ndim_h_minor},
            make_sequential_index<kMaxNumDim>(hidden_dim_cnt, hidden_dim_cnt + ndim_h_minor)};

        for(index_t i = 0; i < ndim_h_minor; ++i)
        {
            rh_major_rh_minor_to_hidden_ids(idim_x + 1)(i)     = hidden_dim_cnt;
            rh_major_rh_minor_to_hidden_lengths(idim_x + 1)(i) = h_minor_lengths[i];

            hidden_dim_cnt++;
        }
    });

    // transform: P dimensions
    constexpr index_t ndim_p = Ps2RHssMajor::Size();

    Dims hidden_dim_id_ps;

    static_for<0, ndim_p, 1>{}([&](auto iDimP) {
        //
        index_t hidden_dim_id_p = hidden_dim_cnt++;

        hidden_dim_id_ps(iDimP) = hidden_dim_id_p;

        constexpr auto p2RHsMajor = Ps2RHssMajor{}[iDimP];
        constexpr auto p2RHsMinor = Ps2RHssMinor{}[iDimP];

        static_assert(p2RHsMajor.Size() == p2RHsMinor.Size(), "wrong!");

        constexpr index_t ndim_low = p2RHsMajor.Size();

        Dims low_dims;
        Lengths low_lengths;

        for(index_t i = 0; i < ndim_low; ++i)
        {
            index_t rh_major = p2RHsMajor[i];
            index_t rh_minor = p2RHsMinor[i];
            low_dims(i)      = rh_major_rh_minor_to_hidden_ids[rh_major][rh_minor];
            low_lengths(i)   = rh_major_rh_minor_to_hidden_lengths[rh_major][rh_minor];
        }

        trans(num_tran++) = {IndexTransformEnum::Merge,
                             MetaData{to_array<index_t, ndim_low>(low_lengths)},
                             NumDim{ndim_low},
                             low_dims,
                             NumDim{1},
                             Dims{hidden_dim_id_p}};
    });

    constexpr index_t ndim_bottom = ndim_x;

    constexpr auto bottom_dim_ids = make_sequential_index<kMaxNumDim>(0, ndim_bottom);

    constexpr auto ys_to_rhs_major = Ys2RHsMajor{};
    constexpr auto ys_to_rhs_minor = Ys2RHsMinor{};

    constexpr index_t ndim_y   = Ys2RHsMajor::Size();
    constexpr index_t ndim_top = ndim_p + ndim_y;

    auto top_dim_ids = hidden_dim_id_ps;

    {
        for(index_t i = 0; i < ndim_y; ++i)
        {
            index_t rh_major        = ys_to_rhs_major[i];
            index_t rh_minor        = ys_to_rhs_minor[i];
            top_dim_ids(ndim_p + i) = rh_major_rh_minor_to_hidden_ids[rh_major][rh_minor];
        }
    }

    //
    const auto ps_ys_to_xs_adaptor_encoding =
        make_tuple(trans, num_tran, bottom_dim_ids, ndim_bottom, top_dim_ids, ndim_top);

    // descriptor: [y0, y1, ...] to [d]
    Lengths y_lengths;
    index_t d_length = 1;

    for(index_t i = 0; i < ndim_y; ++i)
    {
        index_t rh_major = ys_to_rhs_major[i];
        index_t rh_minor = ys_to_rhs_minor[i];
        index_t y_length = rh_major_rh_minor_to_hidden_lengths[rh_major][rh_minor];
        y_lengths(i)     = y_length;
        d_length *= y_length;
    }

    auto tran = make_tuple(IndexTransformEnum::UnMerge,
                           MetaData{to_array<index_t, ndim_y>(y_lengths)},
                           NumDim{1},
                           Dims{0},
                           NumDim{ndim_y},
                           make_sequential_index<kMaxNumDim>(1, ndim_y + 1));

    const auto ys_to_d_adaptor_encoding = make_tuple(
        make_tuple(tran), 1, Dims{0}, 1, make_sequential_index<kMaxNumDim>(1, ndim_y + 1), ndim_y);

    return make_tuple(ps_ys_to_xs_adaptor_encoding, ys_to_d_adaptor_encoding, d_length);
}

} // namespace detail

template <typename PsYs2XsAdaptor_, typename Ys2DDescriptor_>
struct BlockTensorDistribution
{
    using PsYs2XsAdaptor = remove_cvref_t<PsYs2XsAdaptor_>;
    using Ys2DDescriptor = remove_cvref_t<Ys2DDescriptor_>;

    static constexpr index_t NDimX = PsYs2XsAdaptor::GetNumOfBottomDimension();
    static constexpr index_t NDimY = Ys2DDescriptor::GetNumOfTopDimension();
    static constexpr index_t NDimP = PsYs2XsAdaptor::GetNumOfTopDimension() - NDimY;

    PsYs2XsAdaptor ps_ys_to_xs_;
    Ys2DDescriptor ys_to_d_;

    __host__ __device__ static constexpr index_t GetNumOfDimensionX() { return NDimX; }
    __host__ __device__ static constexpr index_t GetNumOfDimensionY() { return NDimY; }
    __host__ __device__ static constexpr index_t GetNumOfDimensionP() { return NDimP; }

    __host__ __device__ constexpr auto GetLengths() const
    {
        ps_ys_to_xs_.GetBottomDimensionLengths();
    }

    __host__ __device__ constexpr const auto& GetPsYs2XsAdaptor() const { return ps_ys_to_xs_; }

    __host__ __device__ constexpr const auto& GetYs2DDescriptor() const { return ys_to_d_; }

    __host__ __device__ static constexpr bool IsStatic()
    {
        return PsYs2XsAdaptor::IsStatic() && Ys2DDescriptor::IsStatic();
    }

    __host__ __device__ void Print() const
    {
        printf("{");
        printf("BlockTensorDistribution, ");
        ps_ys_to_xs_.Print();
        ys_to_d_.Print();
        printf("}");
    }
};

template <typename RsLengths_,    // Sequence<...>
          typename HsLengthss_,   // Tuple<Sequence<...>, ...>
          typename Ps2RHssMajor_, // Tuple<Sequence<...>, ...>
          typename Ps2RHssMinor_, // Tuple<Sequence<...>, ...>
          typename Ys2RHsMajor_,  // Sequence<...>
          typename Ys2RHsMinor_   // Sequence<...>
          >
struct StaticTensorDistributionEncoding
{
    using RsLengths    = remove_cvref_t<RsLengths_>;
    using HsLengthss   = remove_cvref_t<HsLengthss_>;
    using Ps2RHssMajor = remove_cvref_t<Ps2RHssMajor_>;
    using Ps2RHssMinor = remove_cvref_t<Ps2RHssMinor_>;
    using Ys2RHsMajor  = remove_cvref_t<Ys2RHsMajor_>;
    using Ys2RHsMinor  = remove_cvref_t<Ys2RHsMinor_>;

    static_assert(Ps2RHssMajor::Size() == Ps2RHssMinor::Size(), "wrong!");
    static_assert(Ys2RHsMajor::Size() == Ys2RHsMinor::Size(), "wrong!");

    static constexpr index_t NDimX = HsLengthss::Size();
    static constexpr index_t NDimP = Ps2RHssMajor::Size();
    static constexpr index_t NDimY = Ys2RHsMajor::Size();
};

template <typename OuterDstr, typename InnerDstr>
__host__ __device__ constexpr auto embed_tensor_distribution_encoding(OuterDstr, InnerDstr)
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

    return StaticTensorDistributionEncoding<RsLengths,
                                            remove_cvref_t<decltype(hs_lengthss)>,
                                            remove_cvref_t<decltype(ps_2_rhss_major)>,
                                            remove_cvref_t<decltype(ps_2_rhss_minor)>,
                                            remove_cvref_t<decltype(ys_2_rhs_major)>,
                                            remove_cvref_t<decltype(ys_2_rhs_minor)>>{};
}

// this returns a constexpr BlockTensorDistribution
template <typename StaticTensorDistributionEncoding_>
__host__ __device__ constexpr auto make_block_tensor_distribution(StaticTensorDistributionEncoding_)
{
    constexpr auto encode =
        detail::make_adaptor_encoding_for_tensor_distribution(StaticTensorDistributionEncoding_{});

    constexpr auto encoded_ps_ys_to_xs_adaptor = encode.template At<0>();
    constexpr auto encoded_ys_to_d_adaptor     = encode.template At<1>();
    constexpr index_t d_length                 = encode.template At<2>();

    constexpr auto ps_ys_to_xs_adaptor =
        CONSTRUCT_TENSOR_ADAPTOR_FROM_ENCODING(encoded_ps_ys_to_xs_adaptor);

    constexpr auto ys_to_d_adaptor =
        CONSTRUCT_TENSOR_ADAPTOR_FROM_ENCODING(encoded_ys_to_d_adaptor);

    constexpr auto ys_to_d_descriptor =
        make_tensor_descriptor_from_adaptor(ys_to_d_adaptor, d_length);

    return BlockTensorDistribution<remove_cvref_t<decltype(ps_ys_to_xs_adaptor)>,
                                   remove_cvref_t<decltype(ys_to_d_descriptor)>>{
        ps_ys_to_xs_adaptor, ys_to_d_descriptor};
}

// this returns a static BlockTensorDistribution
template <typename StaticTensorDistributionEncoding_>
__host__ __device__ constexpr auto
    make_static_block_tensor_distribution(StaticTensorDistributionEncoding_)
{
    constexpr auto encode =
        detail::make_adaptor_encoding_for_tensor_distribution(StaticTensorDistributionEncoding_{});

    constexpr auto encoded_ps_ys_to_xs_adaptor = encode.template At<0>();
    constexpr auto encoded_ys_to_d_adaptor     = encode.template At<1>();
    constexpr index_t d_length                 = encode.template At<2>();

    constexpr auto ps_ys_to_xs_adaptor =
        CONSTRUCT_STATIC_TENSOR_ADAPTOR_FROM_ENCODING(encoded_ps_ys_to_xs_adaptor);

    constexpr auto ys_to_d_adaptor =
        CONSTRUCT_STATIC_TENSOR_ADAPTOR_FROM_ENCODING(encoded_ys_to_d_adaptor);

    constexpr auto ys_to_d_descriptor =
        make_tensor_descriptor_from_adaptor(ys_to_d_adaptor, Number<d_length>{});

    return BlockTensorDistribution<remove_cvref_t<decltype(ps_ys_to_xs_adaptor)>,
                                   remove_cvref_t<decltype(ys_to_d_descriptor)>>{
        ps_ys_to_xs_adaptor, ys_to_d_descriptor};
}

} // namespace block
} // namespace tile_program
} // namespace ck
