// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/multi_index_transform_helper.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"

namespace ck {

namespace detail {

template <typename Lengths, typename Strides, index_t I, typename AccOld>
__host__ __device__ constexpr auto calculate_element_space_size_impl(const Lengths& lengths,
                                                                     const Strides& strides,
                                                                     Number<I> i,
                                                                     AccOld acc_old)
{
    auto acc_new = acc_old + (lengths[i] - Number<1>{}) * strides[i];

    if constexpr(i.value < Lengths::Size() - 1)
    {
        return calculate_element_space_size_impl(lengths, strides, i + Number<1>{}, acc_new);
    }
    else
    {
        return acc_new;
    }
}

} // namespace detail

/*
 * These functions create naive tensor descriptor
 */

// Lengths..., Strides... could be:
//   1) index_t, which is known at run-time, or
//   2) Number<>, which is known at compile-time
// element_space_size could be:
//   1) long_index_t, or
//   2) LongNumber<>
template <typename... Lengths,
          typename... Strides,
          index_t GuaranteedLastDimensionVectorLength                              = -1,
          index_t GuaranteedLastDimensionVectorStride                              = -1,
          typename enable_if<sizeof...(Lengths) == sizeof...(Strides), bool>::type = false>
__host__ __device__ constexpr auto
make_naive_tensor_descriptor(const Tuple<Lengths...>& lengths,
                             const Tuple<Strides...>& strides,
                             Number<GuaranteedLastDimensionVectorLength> = Number<-1>{},
                             Number<GuaranteedLastDimensionVectorStride> = Number<-1>{})
{
    constexpr index_t N = sizeof...(Lengths);

    const auto transforms = make_tuple(make_embed_transform(lengths, strides));

    constexpr auto low_dim_hidden_idss = make_tuple(Sequence<0>{});

    constexpr auto up_dim_hidden_idss =
        make_tuple(typename arithmetic_sequence_gen<1, N + 1, 1>::type{});

    constexpr auto visible_dim_hidden_ids = typename arithmetic_sequence_gen<1, N + 1, 1>::type{};

    const auto element_space_size =
        detail::calculate_element_space_size_impl(lengths, strides, Number<0>{}, LongNumber<1>{});

    using GuaranteedVectorLengths =
        typename sequence_merge<typename uniform_sequence_gen<N, -1>::type,
                                Sequence<GuaranteedLastDimensionVectorLength>>::type;

    using GuaranteedVectorStrides =
        typename sequence_merge<typename uniform_sequence_gen<N, -1>::type,
                                Sequence<GuaranteedLastDimensionVectorStride>>::type;

    return TensorDescriptor<remove_cv_t<decltype(transforms)>,
                            remove_cv_t<decltype(low_dim_hidden_idss)>,
                            remove_cv_t<decltype(up_dim_hidden_idss)>,
                            remove_cv_t<decltype(visible_dim_hidden_ids)>,
                            remove_cv_t<decltype(element_space_size)>,
                            GuaranteedVectorLengths,
                            GuaranteedVectorStrides>{transforms, element_space_size};
}

// tensor descriptor with offset, the offset will not be added into element space size
// only have an information of the starting offset, and will impact on offset calculation
template <typename... Lengths,
          typename... Strides,
          typename Offset,
          index_t GuaranteedLastDimensionVectorLength                              = -1,
          index_t GuaranteedLastDimensionVectorStride                              = -1,
          typename enable_if<sizeof...(Lengths) == sizeof...(Strides), bool>::type = false>
__host__ __device__ constexpr auto
make_naive_tensor_descriptor_with_offset(const Tuple<Lengths...>& lengths,
                                         const Tuple<Strides...>& strides,
                                         const Offset& offset,
                                         Number<GuaranteedLastDimensionVectorLength> = Number<-1>{},
                                         Number<GuaranteedLastDimensionVectorStride> = Number<-1>{})
{
    const auto desc_0 = [&]() {
        const auto element_space_size = detail::calculate_element_space_size_impl(
            lengths, strides, Number<0>{}, LongNumber<1>{});

        const auto transforms = make_tuple(make_offset_transform(element_space_size, offset));

        constexpr auto low_dim_hidden_idss = make_tuple(Sequence<0>{});

        constexpr auto up_dim_hidden_idss = make_tuple(Sequence<1>{});

        constexpr auto visible_dim_hidden_ids = Sequence<1>{};

        using GuaranteedVectorLengths =
            typename sequence_merge<typename uniform_sequence_gen<1, -1>::type,
                                    Sequence<GuaranteedLastDimensionVectorLength>>::type;

        using GuaranteedVectorStrides =
            typename sequence_merge<typename uniform_sequence_gen<1, -1>::type,
                                    Sequence<GuaranteedLastDimensionVectorStride>>::type;

        return TensorDescriptor<remove_cv_t<decltype(transforms)>,
                                remove_cv_t<decltype(low_dim_hidden_idss)>,
                                remove_cv_t<decltype(up_dim_hidden_idss)>,
                                remove_cv_t<decltype(visible_dim_hidden_ids)>,
                                remove_cv_t<decltype(element_space_size)>,
                                GuaranteedVectorLengths,
                                GuaranteedVectorStrides>{transforms, element_space_size};
    }();

    constexpr index_t N = sizeof...(Lengths);

    return transform_tensor_descriptor(
        desc_0,
        make_tuple(make_embed_transform(lengths, strides)),
        make_tuple(Sequence<0>{}),
        make_tuple(typename arithmetic_sequence_gen<0, N, 1>::type{}));
}

// Lengths... could be:
//   1) index_t, which is known at run-time, or
//   2) Number<>, which is known at compile-time
// element_space_size could be:
//   1) long_index_t, or
//   2) LongNumber<>
template <typename... Lengths, index_t GuaranteedLastDimensionVectorLength = -1>
__host__ __device__ constexpr auto
make_naive_tensor_descriptor_packed(const Tuple<Lengths...>& lengths,
                                    Number<GuaranteedLastDimensionVectorLength> = Number<-1>{})
{
    constexpr index_t N = sizeof...(Lengths);

    const auto transforms = make_tuple(make_unmerge_transform(lengths));

    constexpr auto low_dim_hidden_idss = make_tuple(Sequence<0>{});

    constexpr auto up_dim_hidden_idss =
        make_tuple(typename arithmetic_sequence_gen<1, N + 1, 1>::type{});

    constexpr auto visible_dim_hidden_ids = typename arithmetic_sequence_gen<1, N + 1, 1>::type{};

    const auto element_space_size = container_reduce(lengths, math::multiplies{}, LongNumber<1>{});

    using GuaranteedVectorLengths =
        typename sequence_merge<typename uniform_sequence_gen<N, -1>::type,
                                Sequence<GuaranteedLastDimensionVectorLength>>::type;

    using GuaranteedVectorStrides =
        typename sequence_merge<typename uniform_sequence_gen<N, -1>::type, Sequence<1>>::type;

    return TensorDescriptor<remove_cv_t<decltype(transforms)>,
                            remove_cv_t<decltype(low_dim_hidden_idss)>,
                            remove_cv_t<decltype(up_dim_hidden_idss)>,
                            remove_cv_t<decltype(visible_dim_hidden_ids)>,
                            remove_cv_t<decltype(element_space_size)>,
                            GuaranteedVectorLengths,
                            GuaranteedVectorStrides>{transforms, element_space_size};
}

template <typename... Lengths,
          typename... Strides,
          typename Offset,
          index_t GuaranteedLastDimensionVectorLength                              = -1,
          typename enable_if<sizeof...(Lengths) == sizeof...(Strides), bool>::type = false>
__host__ __device__ constexpr auto make_naive_tensor_descriptor_packed_with_offset(
    const Tuple<Lengths...>& lengths,
    const Offset& offset,
    Number<GuaranteedLastDimensionVectorLength> = Number<-1>{})
{
    const auto desc_0 = [&]() {
        const auto element_space_size =
            container_reduce(lengths, math::multiplies{}, LongNumber<1>{});

        const auto transforms = make_tuple(make_offset_transform(element_space_size, offset));

        constexpr auto low_dim_hidden_idss = make_tuple(Sequence<0>{});

        constexpr auto up_dim_hidden_idss = make_tuple(Sequence<1>{});

        constexpr auto visible_dim_hidden_ids = Sequence<1>{};

        using GuaranteedVectorLengths =
            typename sequence_merge<typename uniform_sequence_gen<1, -1>::type,
                                    Sequence<GuaranteedLastDimensionVectorLength>>::type;

        using GuaranteedVectorStrides =
            typename sequence_merge<typename uniform_sequence_gen<1, -1>::type, Sequence<1>>::type;

        return TensorDescriptor<remove_cv_t<decltype(transforms)>,
                                remove_cv_t<decltype(low_dim_hidden_idss)>,
                                remove_cv_t<decltype(up_dim_hidden_idss)>,
                                remove_cv_t<decltype(visible_dim_hidden_ids)>,
                                remove_cv_t<decltype(element_space_size)>,
                                GuaranteedVectorLengths,
                                GuaranteedVectorStrides>{transforms, element_space_size};
    }();

    constexpr index_t N = sizeof...(Lengths);

    return transform_tensor_descriptor(
        desc_0,
        make_tuple(make_unmerge_transform(lengths)),
        make_tuple(Sequence<0>{}),
        make_tuple(typename arithmetic_sequence_gen<0, N, 1>::type{}));
}

// Lengths... could be:
//   1) index_t, which is known at run-time, or
//   2) Number<>, which is known at compile-time
// align could be:
//   1) index_t, or
//   2) Number<>
template <typename... Lengths, typename Align>
__host__ __device__ constexpr auto
make_naive_tensor_descriptor_aligned(const Tuple<Lengths...>& lengths, Align align)
{
    constexpr auto I1 = Number<1>{};

    constexpr index_t N = sizeof...(Lengths);

    const auto stride_n_minus_2 = math::integer_least_multiple(lengths[Number<N - 1>{}], align);

    auto strides = generate_tuple(
        [&](auto i) {
            if constexpr(i.value == N - 1)
            {
                return I1;
            }
            else if constexpr(i.value == N - 2)
            {
                return Number<stride_n_minus_2>{};
            }
            else
            {
                return container_reduce(lengths,
                                        math::multiplies{},
                                        Number<stride_n_minus_2>{},
                                        i + I1,
                                        Number<N - 1>{},
                                        I1);
            }
        },
        Number<N>{});

    return make_naive_tensor_descriptor(lengths, strides);
}

} // namespace ck
