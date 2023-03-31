// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/utility/sequence_helper.hpp"
#include "ck/tensor_description/multi_index_transform.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"

namespace ck {

// Transforms: Tuple<transforms...>
// LowerDimensionHiddenIdss : Tuple<Sequence<...>, ...>
// UpperDimensionHiddenIdss : Tuple<Sequence<...>, ...>
// TopDimensionHiddenIds> : Sequence<...>
template <typename Transforms,
          typename LowerDimensionHiddenIdss,
          typename UpperDimensionHiddenIdss,
          typename TopDimensionHiddenIds,
          typename ElementSpaceSize>
struct TensorDescriptor : public TensorAdaptor<Transforms,
                                               LowerDimensionHiddenIdss,
                                               UpperDimensionHiddenIdss,
                                               Sequence<0>,
                                               TopDimensionHiddenIds>
{
    using Base = TensorAdaptor<Transforms,
                               LowerDimensionHiddenIdss,
                               UpperDimensionHiddenIdss,
                               Sequence<0>,
                               TopDimensionHiddenIds>;

    using ElementSpaceSizeType = ElementSpaceSize;

    constexpr static index_t ntransform_  = Base::GetNumOfTransform();
    constexpr static index_t ndim_hidden_ = Base::GetNumOfHiddenDimension();
    constexpr static index_t ndim_top_    = Base::GetNumOfTopDimension();

    using TopIndex    = MultiIndex<ndim_top_>;
    using HiddenIndex = MultiIndex<ndim_hidden_>;

    public:
    __host__ __device__ constexpr TensorDescriptor() = default;

    __host__ __device__ constexpr TensorDescriptor(const Transforms& transforms,
                                                   ElementSpaceSize element_space_size)
        : Base{transforms}, element_space_size_{element_space_size}

    {
        static_assert(Transforms::Size() == ntransform_ &&
                          LowerDimensionHiddenIdss::Size() == ntransform_ &&
                          UpperDimensionHiddenIdss::Size() == ntransform_,
                      "wrong! inconsistent # of transformations");

        // TODO check dependency of dimensions is valid
    }

    // construct from TensorAdaptor bass class
    __host__ __device__ constexpr TensorDescriptor(const Base& adaptor,
                                                   ElementSpaceSize element_space_size)
        : Base{adaptor}, element_space_size_{element_space_size}
    {
    }

    __host__ __device__ static constexpr index_t GetNumOfDimension()
    {
        return Base::GetNumOfTopDimension();
    }

    template <index_t IDim>
    __host__ __device__ constexpr auto GetLength(Number<IDim> idim) const
    {
        return Base::GetTopDimensionLength(idim);
    }

    __host__ __device__ constexpr auto GetLengths() const { return Base::GetTopDimensionLengths(); }

    __host__ __device__ constexpr auto GetElementSpaceSize() const { return element_space_size_; }

    template <typename Idx>
    __host__ __device__ constexpr index_t CalculateOffset(const Idx& idx) const
    {
        return Base::CalculateBottomIndex(idx)[Number<0>{}];
    }

    // TODO make these private
    __host__ __device__ constexpr const auto& GetTransforms() const
    {
        return Base::GetTransforms();
    }

    __host__ __device__ static constexpr auto GetLowerDimensionHiddenIdss()
    {
        return Base::GetLowerDimensionHiddenIdss();
    }

    __host__ __device__ static constexpr auto GetUpperDimensionHiddenIdss()
    {
        return Base::GetUpperDimensionHiddenIdss();
    }

    __host__ __device__ static constexpr auto GetTopDimensionHiddenIds()
    {
        return Base::GetTopDimensionHiddenIds();
    }

    __host__ __device__ static constexpr bool IsKnownAtCompileTime()
    {
        return Base::IsKnownAtCompileTime() && is_known_at_compile_time<ElementSpaceSize>::value;
    }

    __host__ __device__ void Print() const
    {
        printf("{");
        printf("TensorDescriptor, ");
        Base::Print();
        printf("}");
    }

    // TODO make these private
    ElementSpaceSize element_space_size_;
};

template <index_t NDimHidden, typename TopDimensionHiddenIds>
struct TensorCoordinate
    : public TensorAdaptorCoordinate<NDimHidden, Sequence<0>, TopDimensionHiddenIds>
{
    using Base = TensorAdaptorCoordinate<NDimHidden, Sequence<0>, TopDimensionHiddenIds>;

    // TODO make these private
    static constexpr index_t ndim_top_ = TopDimensionHiddenIds::Size();

    using HiddenIndex = MultiIndex<NDimHidden>;
    using TopIndex    = MultiIndex<ndim_top_>;

    public:
    __host__ __device__ constexpr TensorCoordinate() = default;

    __host__ __device__ constexpr TensorCoordinate(const HiddenIndex& idx_hidden) : Base{idx_hidden}
    {
    }

    // construct from TensorAdaptorCoordinte base class
    __host__ __device__ constexpr TensorCoordinate(const Base& adaptor_coord) : Base{adaptor_coord}
    {
    }

    __host__ __device__ constexpr auto GetIndex() const { return Base::GetTopIndex(); }

    __host__ __device__ constexpr index_t GetOffset() const
    {
        return Base::GetBottomIndex()[Number<0>{}];
    }

    __host__ __device__ constexpr const auto& GetHiddenIndex() const
    {
        return Base::GetHiddenIndex();
    }

    __host__ __device__ auto& GetHiddenIndex() { return Base::GetHiddenIndex(); }
};

template <index_t NTransform, index_t NDimTop, typename UpdateLowerIndexHack>
struct TensorCoordinateStep
{
    using TopIndex = MultiIndex<NDimTop>;

    public:
    __host__ __device__ constexpr TensorCoordinateStep() = default;

    __host__ __device__ constexpr TensorCoordinateStep(const TopIndex& idx_diff_top,
                                                       const MultiIndex<NTransform>& do_transforms)
        : idx_diff_top_{idx_diff_top}, do_transforms_{do_transforms}
    {
    }

    __host__ __device__ constexpr const auto& GetIndexDiff() const { return GetTopIndexDiff(); }

    __host__ __device__ constexpr const auto& GetTopIndexDiff() const { return idx_diff_top_; }

    TopIndex idx_diff_top_;
    MultiIndex<NTransform> do_transforms_;

    // HACK: control UpdateLowerIndex()
    static constexpr UpdateLowerIndexHack update_lower_index_hack_;
};

// TODO: How to fix this? It uses an struct instead of lambda because lambda
// doesn't have constructor, and to put it outside the scope where it is used
// (transform_tensor_descriptor) because template cannot be defined inside a function
// template
template <typename NewTransforms>
struct lambda_get_up_dim_num
{
    template <typename I>
    __host__ __device__ constexpr auto operator()(I) const
    {
        using Tran = remove_reference_t<decltype(NewTransforms{}.At(I{}))>;
        return Number<Tran::GetNumOfUpperDimension()>{};
    }
};

template <typename OldTensorDescriptor,
          typename NewTransforms,
          typename NewLowerDimensionOldTopIdss,
          typename NewUpperDimensionNewTopIdss>
__host__ __device__ constexpr auto
transform_tensor_descriptor(const OldTensorDescriptor& old_tensor_desc,
                            const NewTransforms& new_transforms,
                            NewLowerDimensionOldTopIdss,
                            NewUpperDimensionNewTopIdss)
{
    // sanity check
    {
        static_assert(NewTransforms::Size() == NewLowerDimensionOldTopIdss::Size() &&
                          NewTransforms::Size() == NewUpperDimensionNewTopIdss::Size(),
                      "wrong! inconsitent number of transform");

        constexpr auto all_old_top_ids = unpack([](auto... xs) { return merge_sequences(xs...); },
                                                NewLowerDimensionOldTopIdss{});

        constexpr auto all_new_top_ids = unpack([](auto... xs) { return merge_sequences(xs...); },
                                                NewUpperDimensionNewTopIdss{});

        static_assert(is_valid_sequence_map<decltype(all_old_top_ids)>::value &&
                          is_valid_sequence_map<decltype(all_new_top_ids)>::value,
                      "wrong!");
    }

    // lower dimension's hidden idss
    // convert lower dimension top idss (tuple of sequences) to hidden idss (tuple of
    // sequences)
    constexpr auto low_dim_hidden_idss = transform_tuples(
        // convert lower dimension top ids (a sequence) to hidden ids (a sequence)
        [](auto low_dim_top_ids) constexpr {
            return transform_sequences(
                // convert lower dimension top id to hidden id
                [](auto low_dim_top_id) constexpr {
                    return OldTensorDescriptor::GetTopDimensionHiddenIds()[low_dim_top_id];
                },
                low_dim_top_ids);
        },
        NewLowerDimensionOldTopIdss{});

    constexpr index_t num_new_transform = NewTransforms::Size();

    // upper dimension's hidden idss
    constexpr index_t old_hidden_dim_number = OldTensorDescriptor::GetNumOfHiddenDimension();

    constexpr auto up_dim_numbers =
        generate_sequence(lambda_get_up_dim_num<NewTransforms>{}, Number<num_new_transform>{});

    constexpr auto up_dim_numbers_scan = merge_sequences(
        Sequence<0>{}, inclusive_scan_sequence(up_dim_numbers, math::plus<index_t>{}, Number<0>{}));

    constexpr auto up_dim_hidden_idss = generate_tuple(
        [ old_hidden_dim_number, up_dim_numbers_scan ](auto i) constexpr {
            return
                typename arithmetic_sequence_gen<old_hidden_dim_number + up_dim_numbers_scan[i],
                                                 old_hidden_dim_number + up_dim_numbers_scan[i + 1],
                                                 1>::type{};
        },
        Number<num_new_transform>{});

    // new top dimension's hidden ids
    constexpr auto unordered_new_top_dim_hidden_ids = unpack(
        [](auto... xs) constexpr { return merge_sequences(xs...); }, up_dim_hidden_idss);

    constexpr auto new_top_dim_unordered2ordered = unpack(
        [](auto... xs) constexpr { return merge_sequences(xs...); }, NewUpperDimensionNewTopIdss{});

    constexpr auto new_top_dim_hidden_ids =
        unordered_new_top_dim_hidden_ids.ReorderGivenOld2New(new_top_dim_unordered2ordered);

    // put everything together
    const auto all_transforms = container_concat(old_tensor_desc.GetTransforms(), new_transforms);

    constexpr auto all_low_dim_hidden_idss =
        container_concat(OldTensorDescriptor::GetLowerDimensionHiddenIdss(), low_dim_hidden_idss);

    constexpr auto all_up_dim_hidden_idss =
        container_concat(OldTensorDescriptor::GetUpperDimensionHiddenIdss(), up_dim_hidden_idss);

    const auto element_space_size = old_tensor_desc.GetElementSpaceSize();

    return TensorDescriptor<remove_cv_t<decltype(all_transforms)>,
                            remove_cv_t<decltype(all_low_dim_hidden_idss)>,
                            remove_cv_t<decltype(all_up_dim_hidden_idss)>,
                            remove_cv_t<decltype(new_top_dim_hidden_ids)>,
                            remove_cv_t<decltype(element_space_size)>>{all_transforms,
                                                                       element_space_size};
}

template <typename TensorDesc, typename TopIndex>
__host__ __device__ constexpr auto make_tensor_coordinate(const TensorDesc& tensor_desc,
                                                          const TopIndex& idx_top)
{
    const auto adaptor_coord = make_tensor_adaptor_coordinate(tensor_desc, idx_top);

    constexpr index_t ndim_hidden = TensorDesc::GetNumOfHiddenDimension();
    constexpr auto top_dim_ids    = TensorDesc::GetTopDimensionHiddenIds();

    return TensorCoordinate<ndim_hidden, remove_cvref_t<decltype(top_dim_ids)>>{adaptor_coord};
}

template <bool JudgeDoTransforms = true, typename TensorDesc, typename TensorCoord, typename Index>
__host__ __device__ constexpr void
move_tensor_coordinate(const TensorDesc& tensor_desc, TensorCoord& coord, const Index& coord_step)
{
    move_tensor_adaptor_coordinate(tensor_desc, coord, coord_step);
}

template <typename TensorDesc, typename TensorCoord>
__host__ __device__ constexpr bool
coordinate_has_valid_offset_assuming_top_index_is_valid(const TensorDesc& tensor_desc,
                                                        const TensorCoord& coord)
{
    bool valid = true;

    constexpr index_t ntransform = TensorDesc::GetNumOfTransform();

    const auto& idx_hidden = coord.GetHiddenIndex();

    static_for<ntransform - 1, -1, -1>{}([&tensor_desc, &idx_hidden, &valid](auto itran) {
        const auto tran = tensor_desc.GetTransforms().At(itran);

        // check validity, only if current transformation does not always has a valid mapping
        if constexpr(!decltype(tran)::IsValidUpperIndexAlwaysMappedToValidLowerIndex())
        {
            const auto idx_up = get_container_subset(
                idx_hidden, TensorDesc::GetUpperDimensionHiddenIdss().At(itran));

            // Comment: using valid = valid && .. will result in weird control flow in ISA
            valid &= tran.IsValidUpperIndexMappedToValidLowerIndex(idx_up);
        }
    });

    return valid;
}

template <typename TensorDesc, typename TensorCoord>
__host__ __device__ constexpr bool coordinate_has_valid_offset(const TensorDesc& tensor_desc,
                                                               const TensorCoord& coord)
{
    // check top index
    const auto& idx_top = coord.GetTopIndex();

    bool is_top_index_valid = true;

    static_for<0, TensorDesc::GetNumOfDimension(), 1>{}(
        [&is_top_index_valid, &idx_top, &tensor_desc](auto i) {
            is_top_index_valid =
                is_top_index_valid && (idx_top[i] >= 0 && idx_top[i] < tensor_desc.GetLength(i));
        });

    // check other hidden index
    return is_top_index_valid &&
           coordinate_has_valid_offset_assuming_top_index_is_valid(tensor_desc, coord);
}

} // namespace ck
