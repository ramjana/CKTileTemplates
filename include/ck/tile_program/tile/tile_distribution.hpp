// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"
#include "ck/tensor_description/macro_func_tensor_adaptor_from_encoding.hpp"

namespace ck {
namespace tile_program {

template <typename RsLengths_,    // Sequence<...>
          typename HsLengthss_,   // Tuple<Sequence<...>, ...>
          typename Ps2RHssMajor_, // Tuple<Sequence<...>, ...>
          typename Ps2RHssMinor_, // Tuple<Sequence<...>, ...>
          typename Ys2RHsMajor_,  // Sequence<...>
          typename Ys2RHsMinor_   // Sequence<...>
          >
struct StaticTileDistributionEncoding
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

    static constexpr auto rs_lengths_       = RsLengths{};
    static constexpr auto hs_lengthss_      = HsLengthss{};
    static constexpr auto ps_to_rhss_major_ = Ps2RHssMajor{};
    static constexpr auto ps_to_rhss_minor_ = Ps2RHssMinor{};
    static constexpr auto ys_to_rhs_major_  = Ys2RHsMajor{};
    static constexpr auto ys_to_rhs_minor_  = Ys2RHsMinor{};

    // redundant but useful info
    struct Detail
    {
        static constexpr index_t ndim_rh_major_    = NDimX + 1;
        static constexpr index_t ndim_range_major_ = NDimX;

        // Array
        static constexpr auto ndims_rhs_minor_ = generate_array(
            [](auto i) {
                if constexpr(i.value == 0)
                {
                    return rs_lengths_.Size();
                }
                else
                {
                    return hs_lengthss_[i - Number<1>{}].Size();
                }
            },
            Number<NDimX + 1>{});

        //
        static constexpr index_t max_ndim_rh_minor_ =
            container_reduce(ndims_rhs_minor_, math::maximize<index_t>{}, 0);

        // Array of Array
        static constexpr auto rhs_major_minor_to_ys_ = [] {
            Array<Array<index_t, max_ndim_rh_minor_>, NDimX + 1> rhs_major_minor_to_ys_tmp{{-1}};

            static_for<0, NDimY, 1>{}([&](auto i) {
                constexpr index_t rh_major = ys_to_rhs_major_[i];
                constexpr index_t rh_minor = ys_to_rhs_minor_[i];

                rhs_major_minor_to_ys_tmp(rh_major)(rh_minor) = i;
            });

            return rhs_major_minor_to_ys_tmp;
        }();

        // Array
        static constexpr auto ndims_range_minor_ = [] {
            Array<index_t, NDimX> ndims_range_minor{0};

            for(index_t i = 0; i < NDimY; i++)
            {
                const index_t range_major = ys_to_rhs_major_[i] - 1;

                ndims_range_minor(range_major)++;
            }

            return ndims_range_minor;
        }();

        //
        static constexpr index_t max_ndim_range_minor_ =
            container_reduce(ndims_range_minor_, math::maximize<index_t>{}, 0);

        // Array
        static constexpr auto rhs_major_minor_to_range_minor_ = [] {
            Array<Array<index_t, max_ndim_rh_minor_>, ndim_rh_major_>
                rhs_major_minor_to_range_minor{{-1}};

            static_for<0, ndim_rh_major_, 1>{}([&](auto rh_major) {
                constexpr index_t ndim_rh_minor = ndims_rhs_minor_[rh_major];

                index_t cnt_ndim_range_minor = 0;

                static_for<0, ndim_rh_minor, 1>{}([&](auto rh_minor) {
                    constexpr index_t idim_y = rhs_major_minor_to_ys_[rh_major][rh_minor];

                    if(idim_y >= 0)
                    {
                        rhs_major_minor_to_range_minor(rh_major)(rh_minor) = cnt_ndim_range_minor;

                        cnt_ndim_range_minor++;
                    }
                });
            });

            return rhs_major_minor_to_range_minor;
        }();

        // Array
        static constexpr auto ys_to_range_major_ =
            generate_array([](auto i) { return ys_to_rhs_major_[i] - 1; }, Number<NDimY>{});

        //
        static constexpr auto ys_to_range_minor_ = generate_array(
            [](auto i) {
                return rhs_major_minor_to_range_minor_[ys_to_rhs_major_[i]][ys_to_rhs_minor_[i]];
            },
            Number<NDimY>{});

        //
        static constexpr auto distributed_ranges_lengthss_ = [] {
            Array<Array<index_t, max_ndim_range_minor_>, ndim_range_major_>
                distributed_ranges_lengthss{{-1}};

            static_for<0, NDimY, 1>{}([&](auto i) {
                const index_t rh_major = ys_to_rhs_major_[i];
                const index_t rh_minor = ys_to_rhs_minor_[i];

                const index_t h_length = hs_lengthss_[Number<rh_major - 1>{}][rh_minor];

                const index_t range_major = rh_major - 1;
                const index_t range_minor = rhs_major_minor_to_range_minor_[rh_major][rh_minor];

                distributed_ranges_lengthss(range_major)(range_minor) = h_length;
            });

            return distributed_ranges_lengthss;
        }();

        static constexpr auto ndims_distributed_ranges_minor_ = [] {
            Array<index_t, ndim_range_major_> ndims_distributed_ranges_minor{0};

            static_for<0, NDimY, 1>{}([&](auto i) {
                const index_t range_major = ys_to_rhs_major_[i] - 1;

                ndims_distributed_ranges_minor(range_major)++;
            });

            return ndims_distributed_ranges_minor;
        }();

        __host__ __device__ void Print() const
        {
            printf("StaticTileDistributionEncoding::Detail{");
            //
            printf("ndim_rh_major_: ");
            print(ndim_rh_major_);
            printf(", ");
            //
            printf("ndim_range_major_: ");
            print(ndim_range_major_);
            printf(", ");
            //
            printf("ndims_rhs_minor_: ");
            print(ndims_rhs_minor_);
            printf(", ");
            //
            printf("ndim_rh_major_: ");
            print(ndim_rh_major_);
            printf(", ");
            //
            printf("max_ndim_rh_minor_: ");
            print(max_ndim_rh_minor_);
            printf(", ");
            //
            printf("rhs_major_minor_to_ys_: ");
            print(rhs_major_minor_to_ys_);
            printf(", ");
            //
            printf("ndims_range_minor_: ");
            print(ndims_range_minor_);
            printf(", ");
            //
            printf("max_ndim_range_minor_: ");
            print(max_ndim_range_minor_);
            printf(", ");
            //
            printf("ys_to_range_major_: ");
            print(ys_to_range_major_);
            printf(", ");
            //
            printf("ys_to_range_minor_: ");
            print(ys_to_range_minor_);
            printf(", ");
            //
            printf("distributed_ranges_lengthss_: ");
            print(distributed_ranges_lengthss_);
            printf(", ");
            //
            printf("ndims_distributed_ranges_minor_: ");
            print(ndims_distributed_ranges_minor_);
            //
            printf("}");
        }
    };

    __host__ __device__ void Print() const
    {
        printf("StaticTileDistributionEncoding{");
        //
        printf("NDimX: %d, NDimP: %d, NDimY: %d, ", NDimX, NDimP, NDimY);
        //
        printf("rs_lengths_: ");
        print(rs_lengths_);
        printf(", ");
        //
        printf("hs_lengthss_: ");
        print(hs_lengthss_);
        printf(", ");
        //
        printf("ps_to_rhss_major_: ");
        print(ps_to_rhss_major_);
        printf(", ");
        //
        printf("ps_to_rhss_minor_: ");
        print(ps_to_rhss_minor_);
        printf(", ");
        //
        printf("ys_to_rhs_major_: ");
        print(ys_to_rhs_major_);
        printf(", ");
        //
        printf("ys_to_rhs_minor_: ");
        print(ys_to_rhs_minor_);
        printf(", ");
        //
        printf("Detail: ");
        print(Detail{});
        //
        printf("}");
    }
};

template <typename PsYs2XsAdaptor_,
          typename Ys2DDescriptor_,
          typename StaticTileDistributionEncoding_>
struct TileDistribution
{
    using PsYs2XsAdaptor = remove_cvref_t<PsYs2XsAdaptor_>;
    using Ys2DDescriptor = remove_cvref_t<Ys2DDescriptor_>;
    using DstrEncode     = remove_cvref_t<StaticTileDistributionEncoding_>;

    static_assert(PsYs2XsAdaptor::IsStatic() && Ys2DDescriptor::IsStatic(),
                  "wrong! should be static");

    static constexpr index_t NDimX = PsYs2XsAdaptor::GetNumOfBottomDimension();
    static constexpr index_t NDimY = Ys2DDescriptor::GetNumOfTopDimension();
    static constexpr index_t NDimP = PsYs2XsAdaptor::GetNumOfTopDimension() - NDimY;

    PsYs2XsAdaptor ps_ys_to_xs_;
    Ys2DDescriptor ys_to_d_;

    __host__ __device__ static constexpr index_t GetNumOfDimensionX() { return NDimX; }
    __host__ __device__ static constexpr index_t GetNumOfDimensionY() { return NDimY; }
    __host__ __device__ static constexpr index_t GetNumOfDimensionP() { return NDimP; }

    __host__ __device__ static constexpr auto GetLengths()
    {
#if 0
        // FIXME: TensorAdaptor::GetBottomDimensionLengths is wrong. re-enable this after it's fixed
        ps_ys_to_xs_.GetBottomDimensionLengths();
#else
        return generate_tuple(
            [&](auto i) {
                constexpr index_t x_length =
                    container_reduce(typename DstrEncode::HsLengthss{}[i], math::multiplies{}, 1);

                return Number<x_length>{};
            },
            Number<NDimX>{});
#endif
    }

    __host__ __device__ constexpr const auto& GetPsYs2XsAdaptor() const { return ps_ys_to_xs_; }

    __host__ __device__ constexpr const auto& GetYs2DDescriptor() const { return ys_to_d_; }

    __host__ __device__ static constexpr auto GetStaticTileDistributionEncoding()
    {
        return DstrEncode{};
    }

    __host__ __device__ static constexpr auto GetDistributedRanges()
    {
        constexpr auto distributed_ranges_impl = DstrEncode::Detail::distributed_ranges_lengthss_;
        constexpr index_t ndim_range           = DstrEncode::Detail::ndim_range_major_;
        constexpr auto ndims_ranges_minor = DstrEncode::Detail::ndims_distributed_ranges_minor_;

        constexpr auto distributed_ranges =
            TO_TUPLE_OF_SEQUENCE(distributed_ranges_impl, ndim_range, ndims_ranges_minor);

        return distributed_ranges;
    }

    // FIXME: it's hacky to get Ys index from Range index
    template <typename DistributedRangeIndex>
    __host__ __device__ static constexpr auto
    GetYsIndexFromDistributedRangeIndex(DistributedRangeIndex dstr_range_idx)
    {
        static_assert(is_static_v<DistributedRangeIndex>, "wrong!");

        Array<index_t, NDimY> y_idx;

        static_for<0, NDimY, 1>{}([&](auto i) {
            constexpr index_t range_major = DstrEncode::Detail::ys_to_range_major_[i];
            constexpr index_t range_minor = DstrEncode::Detail::ys_to_range_minor_[i];

            y_idx(i) = dstr_range_idx[Number<range_major>{}][Number<range_minor>{}];
        });

        return y_idx;
    }

    __host__ __device__ static constexpr bool IsStatic()
    {
        return PsYs2XsAdaptor::IsStatic() && Ys2DDescriptor::IsStatic();
    }

    __host__ __device__ void Print() const
    {
        printf("TileDistribution{");
        //
        printf("StaticTileDistributionEncoding: ");
        print(DstrEncode{});
        printf(", ");
        //
        printf("ps_ys_to_xs_: ");
        print(ps_ys_to_xs_);
        printf(", ");
        //
        printf("ys_to_d_: ");
        print(ys_to_d_);
        //
        printf("}");
    }
};

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

// this returns a constexpr encoding of TileDistribution
template <typename StaticTileDistributionEncoding_>
__host__ __device__ constexpr auto
    make_adaptor_encoding_for_tile_distribution(StaticTileDistributionEncoding_)
{
    using RsLengths    = typename StaticTileDistributionEncoding_::RsLengths;
    using HsLengthss   = typename StaticTileDistributionEncoding_::HsLengthss;
    using Ps2RHssMajor = typename StaticTileDistributionEncoding_::Ps2RHssMajor;
    using Ps2RHssMinor = typename StaticTileDistributionEncoding_::Ps2RHssMinor;
    using Ys2RHsMajor  = typename StaticTileDistributionEncoding_::Ys2RHsMajor;
    using Ys2RHsMinor  = typename StaticTileDistributionEncoding_::Ys2RHsMinor;

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

// this returns a constexpr TileDistribution
template <typename StaticTileDistributionEncoding_>
__host__ __device__ constexpr auto make_tile_distribution(StaticTileDistributionEncoding_)
{
    constexpr auto encode =
        detail::make_adaptor_encoding_for_tile_distribution(StaticTileDistributionEncoding_{});

    constexpr auto encoded_ps_ys_to_xs_adaptor = encode.template At<0>();
    constexpr auto encoded_ys_to_d_adaptor     = encode.template At<1>();
    constexpr index_t d_length                 = encode.template At<2>();

    constexpr auto ps_ys_to_xs_adaptor =
        CONSTRUCT_TENSOR_ADAPTOR_FROM_ENCODING(encoded_ps_ys_to_xs_adaptor);

    constexpr auto ys_to_d_adaptor =
        CONSTRUCT_TENSOR_ADAPTOR_FROM_ENCODING(encoded_ys_to_d_adaptor);

    constexpr auto ys_to_d_descriptor =
        make_tensor_descriptor_from_adaptor(ys_to_d_adaptor, d_length);

    return TileDistribution<remove_cvref_t<decltype(ps_ys_to_xs_adaptor)>,
                            remove_cvref_t<decltype(ys_to_d_descriptor)>,
                            remove_cvref_t<StaticTileDistributionEncoding_>>{ps_ys_to_xs_adaptor,
                                                                             ys_to_d_descriptor};
}

// this returns a static TileDistribution
template <typename StaticTileDistributionEncoding_>
__host__ __device__ constexpr auto make_static_tile_distribution(StaticTileDistributionEncoding_)
{
    constexpr auto encode =
        detail::make_adaptor_encoding_for_tile_distribution(StaticTileDistributionEncoding_{});

    constexpr auto encoded_ps_ys_to_xs_adaptor = encode.template At<0>();
    constexpr auto encoded_ys_to_d_adaptor     = encode.template At<1>();
    constexpr index_t d_length                 = encode.template At<2>();

    constexpr auto ps_ys_to_xs_adaptor =
        CONSTRUCT_STATIC_TENSOR_ADAPTOR_FROM_ENCODING(encoded_ps_ys_to_xs_adaptor);

    constexpr auto ys_to_d_adaptor =
        CONSTRUCT_STATIC_TENSOR_ADAPTOR_FROM_ENCODING(encoded_ys_to_d_adaptor);

    constexpr auto ys_to_d_descriptor =
        make_tensor_descriptor_from_adaptor(ys_to_d_adaptor, Number<d_length>{});

    return TileDistribution<remove_cvref_t<decltype(ps_ys_to_xs_adaptor)>,
                            remove_cvref_t<decltype(ys_to_d_descriptor)>,
                            remove_cvref_t<StaticTileDistributionEncoding_>>{ps_ys_to_xs_adaptor,
                                                                             ys_to_d_descriptor};
}

} // namespace tile_program
} // namespace ck
