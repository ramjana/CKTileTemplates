// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"

#include "ck/tile_program/tile/tile_distribution.hpp"

namespace ck {
namespace tile_program {

template <typename DataType_, typename StaticTileDistribution_>
struct StaticDistributedTensor
{
    using DataType               = remove_cvref_t<DataType_>;
    using StaticTileDistribution = remove_cvref_t<StaticTileDistribution_>;

    static_assert(StaticTileDistribution::IsStatic(),
                  "wrong! StaticTileDistribution should be known at compile tile");

    using ThreadTensorDesc = remove_cvref_t<decltype(StaticTileDistribution{}.GetYs2DDescriptor())>;

    static constexpr index_t kThreadElementSpaceSize = ThreadTensorDesc{}.GetElementSpaceSize();

    __host__ __device__ static constexpr auto GetLengths()
    {
        return StaticTileDistribution::GetLengths();
    }

    __host__ __device__ static constexpr auto GetTileDistribution()
    {
        return StaticTileDistribution{};
    }

    __host__ __device__ void Initialize(const DataType& x) { thread_buf_.Initialize(x); }

    __host__ __device__ constexpr const auto& GetThreadBuffer() const { return thread_buf_; }

    __host__ __device__ constexpr auto& GetThreadBuffer() { return thread_buf_; }

    __host__ __device__ static constexpr index_t GetThreadBufferSize()
    {
        return kThreadElementSpaceSize;
    }

    template <typename... SIs>
    __host__ __device__ constexpr auto operator()(SIs...) const
    {
        static_assert(sizeof...(SIs) == StaticTileDistribution::NDimX);
        constexpr auto origins_lengths =
            zip_tuples([&](auto... ts) { return make_sequence(ts...); }, make_tuple(SIs{}...));
        return slice_static_distributed_tensor_from_x(
            *this, origins_lengths[Number<0>{}], origins_lengths[Number<1>{}]);
    }

    template <index_t... YSliceOrigins, index_t... YSliceLengths>
    __host__ __device__ auto GetSlicedThreadData(Sequence<YSliceOrigins...>,
                                                 Sequence<YSliceLengths...>) const
    {
        static_assert(sizeof...(YSliceOrigins) == StaticTileDistribution::NDimY &&
                          sizeof...(YSliceLengths) == StaticTileDistribution::NDimY,
                      "wrong!");

        constexpr auto sliced_thread_tensor_desc =
            make_naive_tensor_descriptor_packed(make_tuple(YSliceLengths...));

        StaticBuffer<AddressSpaceEnum::Vgpr,
                     DataType,
                     sliced_thread_tensor_desc.GetElementSpaceSize(),
                     true>
            sliced_thread_data;

        static_ford<Sequence<YSliceLengths...>>{}([&](auto idx) {
            constexpr auto idx_ys = idx + Sequence<YSliceOrigins...>{};

            sliced_thread_data(Number<sliced_thread_tensor_desc.CalculateOffset(idx)>{}) =
                thread_buf_[Number<ThreadTensorDesc{}.CalculateOffset(idx_ys)>{}];
        });

        return sliced_thread_data;
    }

    template <index_t... YSliceOrigins, index_t... YSliceLengths, index_t NSlicedData>
    __host__ __device__ void SetSlicedThreadData(
        Sequence<YSliceOrigins...>,
        Sequence<YSliceLengths...>,
        const StaticBuffer<AddressSpaceEnum::Vgpr, DataType, NSlicedData, true>& sliced_thread_data)
    {
        static_assert(sizeof...(YSliceOrigins) == StaticTileDistribution::NDimY &&
                          sizeof...(YSliceLengths) == StaticTileDistribution::NDimY,
                      "wrong!");

        constexpr auto sliced_thread_tensor_desc =
            make_naive_tensor_descriptor_packed(make_tuple(YSliceLengths...));

        static_ford<Sequence<YSliceLengths...>>{}([&](auto idx) {
            constexpr auto idx_ys = idx + Sequence<YSliceOrigins...>{};

            thread_buf_(Number<ThreadTensorDesc{}.CalculateOffset(idx_ys)>{}) =
                sliced_thread_data[Number<sliced_thread_tensor_desc.CalculateOffset(idx)>{}];
        });
    }

    //
    StaticBuffer<AddressSpaceEnum::Vgpr, DataType, kThreadElementSpaceSize, true> thread_buf_;
};

template <typename DataType, typename StaticTileDistribution>
__host__ __device__ constexpr auto make_static_distributed_tensor(const StaticTileDistribution&)
{
    return StaticDistributedTensor<remove_cvref_t<DataType>,
                                   remove_cvref_t<StaticTileDistribution>>{};
}

//
// slice tensor from x_dim, result in split in y_dim, not p_dim.
// We don't support slice cross p_dim (aka, slice different threads)
// also, sliced along y_dim need be the first dim of current dim.
// Multiply Y dim before sliced dim does not make sense
//
// e.g
//       X0           X1
//       <1, 4, 32> - <4, 1, 4, 2, 4>  | slice origin:<0, 0>, len:<0, 32>, (0 means all length)
//        Y  P  P      Y  P  Y  P  Y
//   =>  <1, 4, 32> - <1, 1, 4, 2, 4> -> OK
//                     |--> slice along this Y dim, is the first dim of X1, totally 4 slices
//
//       X0           X1
//       <1, 4, 32> - <4, 1, 4, 2, 4>  | slice origin:<0, 0>, len:<0, 8>, (0 means all length)
//        Y  P  P      Y  P  Y  P  Y
//   =>  <1, 4, 32> - <1, 1, 1, 2, 4> -> OK
//                           |--> slice along this Y dim, the P dim is 1 in the left, so is OK
//                                 totally 16 slices
//
//       X0           X1
//       <1, 4, 32> - <4, 1, 4, 2, 4>  | slice origin:<0, 0>, len:<0, 4>, (0 means all length)
//        Y  P  P      Y  P  Y  P  Y
//   =>  <1, 4, 32> - <1, 1, 1, 1, 4> -> Fail
//                              |--> slice along this P dim, will split threads, not supported
//
//       X0           X1
//       <1, 4, 32> - <4, 1, 4, 2, 4>  | slice origin:<0, 0>, len:<0, 16>, (0 means all length)
//        Y  P  P      Y  P  Y  P  Y
//   =>  <1, 4, 32> - <1, 1, 2, 2, 4> -> Fail
//                           |--> slice along this Y dim, but this Y sim need to split into 2
//                           subdime
//                                we do not support this strategy now, but you can change the
//                                original x dim like this:
//
//       X0           X1
//       <1, 4, 32> - <4, 1, 2, 2, 2, 4>  | slice origin:<0, 0>, len:<0, 16>, (0 means all length)
//        Y  P  P      Y  P  Y  Y  P  Y
//   =>  <1, 4, 32> - <1, 1, 1, 2, 2, 4> -> OK
//
template <typename StaticDistributedTensor_, index_t... XSliceOrigins, index_t... XSliceLengths>
__host__ __device__ constexpr auto
slice_static_distributed_tensor_from_x(const StaticDistributedTensor_& src_tile,
                                       Sequence<XSliceOrigins...>,
                                       Sequence<XSliceLengths...>)
{
    using SrcDist     = typename StaticDistributedTensor_::StaticTileDistribution;
    using SrcDataType = typename StaticDistributedTensor_::DataType;
    using SrcEnc      = decltype(SrcDist::GetStaticTileDistributionEncoding());

    static_assert(sizeof...(XSliceOrigins) == sizeof...(XSliceLengths));
    constexpr auto all_length = Number<0>{};

    constexpr auto src_h_prefix_sum = SrcEnc::GetHDimLengthsPrefixSum();
    constexpr auto src_y_info =
        SrcEnc::GetSortedYInfo(); // TODO: c++20 can capture structure binding in lambda
    constexpr auto src_y_dims       = src_y_info[Number<0>{}];
    constexpr auto src_y_maps       = src_y_info[Number<1>{}];
    constexpr auto src_y_prefix_sum = src_y_info[Number<2>{}];

    constexpr auto x_slice_origins = Sequence<XSliceOrigins...>{};
    constexpr auto x_slice_lengths = Sequence<XSliceLengths...>{};

    constexpr auto sliced_hlen_yidx_ylen = [&]() constexpr
    {
        auto y_slice_sorted_origins = make_zero_multi_index<SrcDist::NDimY>();
        auto y_slice_lengths =
            to_array<index_t, SrcDist::NDimY>(SrcDist{}.GetYs2DDescriptor().GetLengths());

        // This lambda will modify some value outside, so c++ will not treat return value as
        // constexpr
        // TODO: ugly
        auto new_h_lengths = transform_tuples(
            [&](auto h_len, auto id) {
                if constexpr(x_slice_lengths[id] == all_length)
                {
                    // here seems must use constexpr, otherwise else... will evaluate.
                    return h_len;
                }
                else
                {
                    constexpr auto sliced_h =
                        reverse_slice_sequence(h_len, Number<x_slice_lengths[id]>{});

                    constexpr auto sliced_h_lens  = sliced_h[Number<0>{}];
                    constexpr auto sliced_h_index = sliced_h[Number<2>{}];

                    // update y_slice_lengths
                    constexpr auto uniformed_h_index =
                        sliced_h_index + Number<src_h_prefix_sum[id]>{};
                    constexpr auto found_y_index =
                        index_of_sequence(src_y_dims, Number<uniformed_h_index>{});
                    static_assert(found_y_index != -1);
                    static_for<0, sliced_h_index + 1, 1>{}([&](auto i) {
                        y_slice_lengths(src_y_maps[found_y_index - i]) =
                            sliced_h_lens[sliced_h_index - i];
                    });

                    // NOTE: this y_origin is for all dims, not only current dim
                    //       will later use pick to select target dim
                    constexpr auto y_origin = [&]() {
                        constexpr auto h_trans = make_merge_transform_v3_division_mod(h_len);
                        // h_trans.low_lengths_scan_.foo();
                        auto h_origin_ = make_zero_multi_index<h_trans.NDimLow>();
                        h_trans.CalculateLowerIndex(h_origin_,
                                                    Sequence<x_slice_origins[id].value>{});

                        auto y_origin_ = make_zero_multi_index<SrcDist::NDimY>();
                        static_for<0, sliced_h_index + 1, 1>{}([&](auto i) {
                            y_origin_(found_y_index - i) = h_origin_[sliced_h_index - i];
                        });
                        return y_origin_;
                    }();

                    constexpr auto y_picks =
                        typename arithmetic_sequence_gen<src_y_prefix_sum[id],
                                                         src_y_prefix_sum[id + 1],
                                                         1>::type{};

                    set_container_subset(
                        y_slice_sorted_origins, y_picks, get_container_subset(y_origin, y_picks));
                    // Modify
                    // TODO: some validation
                    return sliced_h_lens;
                }
            },
            typename SrcEnc::HsLengthss{},
            typename arithmetic_sequence_gen<0, SrcEnc::HsLengthss::Size(), 1>::type{});

        auto y_slice_origins = container_reorder_given_old2new(y_slice_sorted_origins, src_y_maps);
        return make_tuple(new_h_lengths, y_slice_origins, y_slice_lengths);
    }
    ();

    // TODO: use c++20 non-typed template, no need to write many temp variables...
    constexpr auto sliced_h_lengths       = sliced_hlen_yidx_ylen[Number<0>{}];
    constexpr auto sliced_y_origins_array = sliced_hlen_yidx_ylen[Number<1>{}];
    constexpr auto sliced_y_origins_size  = sliced_y_origins_array.Size();
    constexpr auto sliced_y_lengths_array = sliced_hlen_yidx_ylen[Number<2>{}];
    constexpr auto sliced_y_lengths_size  = sliced_y_lengths_array.Size();

    constexpr auto sliced_y_origins = TO_SEQUENCE(sliced_y_origins_array, sliced_y_origins_size);
    constexpr auto sliced_y_lengths = TO_SEQUENCE(sliced_y_lengths_array, sliced_y_lengths_size);

    using SlicedEnc =
        StaticTileDistributionEncoding<typename SrcEnc::RsLengths,
                                       decltype(sliced_h_lengths), // only need to change the
                                                                   // h_lengths type
                                       typename SrcEnc::Ps2RHssMajor,
                                       typename SrcEnc::Ps2RHssMinor,
                                       typename SrcEnc::Ys2RHsMajor,
                                       typename SrcEnc::Ys2RHsMinor>;

    auto sliced_tensor =
        make_static_distributed_tensor<SrcDataType>(make_static_tile_distribution(SlicedEnc{}));

    sliced_tensor.GetThreadBuffer() =
        src_tile.GetSlicedThreadData(sliced_y_origins, sliced_y_lengths);

    return sliced_tensor;
}

} // namespace tile_program
} // namespace ck
