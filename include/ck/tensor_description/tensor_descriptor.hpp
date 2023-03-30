// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/utility/sequence_helper.hpp"
#include "ck/tensor_description/multi_index_transform.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"

namespace ck {

template <index_t NDimHidden, typename TopDimensionHiddenIds>
struct TensorCoordinate;

template <index_t NTransform, index_t NDimTop, typename UpdateLowerIndexHack>
struct TensorCoordinateStep;

// Transforms: Tuple<transforms...>
// LowerDimensionHiddenIdss : Tuple<Sequence<...>, ...>
// UpperDimensionHiddenIdss : Tuple<Sequence<...>, ...>
// TopDimensionHiddenIds> : Sequence<...>
template <typename Transforms,
          typename LowerDimensionHiddenIdss,
          typename UpperDimensionHiddenIdss,
          typename TopDimensionHiddenIds,
          typename ElementSpaceSize>
struct TensorDescriptor
{
    using ElementSpaceSizeType = ElementSpaceSize;

    // TODO make these private
    __host__ __device__ static constexpr index_t GetNumOfTransform() { return Transforms::Size(); }

    __host__ __device__ static constexpr index_t GetNumOfTopDimension()
    {
        return TopDimensionHiddenIds::Size();
    }

    __host__ __device__ static constexpr index_t GetNumOfHiddenDimension()
    {
        constexpr auto all_low_dim_ids = unpack(
            [](auto&&... xs) constexpr { return merge_sequences(xs...); },
            LowerDimensionHiddenIdss{});

        constexpr auto all_up_dim_ids = unpack(
            [](auto&&... xs) constexpr { return merge_sequences(xs...); },
            UpperDimensionHiddenIdss{});

        constexpr auto all_dim_ids = merge_sequences(all_low_dim_ids, all_up_dim_ids);

        using unique_sort_all_dim_ids = typename sequence_unique_sort<decltype(all_dim_ids),
                                                                      math::less<index_t>,
                                                                      math::equal<index_t>>::type;

        return unique_sort_all_dim_ids::Size();
    }

    __host__ __device__ static constexpr auto InitializeElementSize(const Transforms& transforms)
    {
        const auto lengths = generate_tuple(
            [&](auto idim_top) {
                constexpr index_t idim_hidden = TopDimensionHiddenIds::At(idim_top);

                constexpr auto tmp = GetTransformAndItsUpperDimension(Number<idim_hidden>{});

                constexpr index_t itran   = tmp[Number<0>{}];
                constexpr index_t idim_up = tmp[Number<1>{}];
                constexpr bool found      = tmp[Number<2>{}];

                static_assert(found == true,
                              "wrong! not found matching transformation and upper-dimension");

                const auto length =
                    transforms[Number<itran>{}].GetUpperLengths()[Number<idim_up>{}];

                return length;
            },
            Number<ndim_top_>{});

        // TODO: make container_reduce support tuple of Number and index_t
        return container_reduce(lengths, math::multiplies{}, Number<1>{});
    }

    template <index_t IDimHidden>
    __host__ __device__ static constexpr auto GetTransformAndItsUpperDimension(Number<IDimHidden>)
    {
        // FIXME: length of IDimHidden ==0 is not known, since info about lower dim length are not
        // saved in transformation
        static_assert(IDimHidden > 0, "wrong! not implemented");

        index_t itran_found   = 0;
        index_t idim_up_found = 0;
        bool found            = false;

        static_for<0, ntransform_, 1>{}([&](auto itran) {
            constexpr auto up_dim_ids = UpperDimensionHiddenIdss{}[itran];

            static_for<0, up_dim_ids.Size(), 1>{}([&](auto idim_up) {
                if constexpr(up_dim_ids[idim_up] == IDimHidden)
                {
                    itran_found   = itran;
                    idim_up_found = idim_up;
                    found         = true;
                }
            });
        });

        return make_tuple(itran_found, idim_up_found, found);
    }

    constexpr static index_t ntransform_  = GetNumOfTransform();
    constexpr static index_t ndim_top_    = GetNumOfTopDimension();
    constexpr static index_t ndim_hidden_ = GetNumOfHiddenDimension();

    using TopIndex    = MultiIndex<ndim_top_>;
    using HiddenIndex = MultiIndex<ndim_hidden_>;
    using Coordinate  = TensorCoordinate<ndim_hidden_, TopDimensionHiddenIds>;

    // may be index_t or Number<>
    using ElementSize = remove_cv_t<decltype(InitializeElementSize(Transforms{}))>;

    public:
#if 1 // workaround compiler complaint about constexpr
    __host__ __device__ constexpr TensorDescriptor() = default;
#else
    __host__ __device__ constexpr TensorDescriptor()
        : transforms_{}, element_size_{}, element_space_size_{}
    {
    }
#endif

    __host__ __device__ constexpr TensorDescriptor(const Transforms& transforms,
                                                   ElementSpaceSize element_space_size)
        : transforms_{transforms},
          element_size_{InitializeElementSize(transforms)},
          element_space_size_{element_space_size}

    {
        static_assert(Transforms::Size() == ntransform_ &&
                          LowerDimensionHiddenIdss::Size() == ntransform_ &&
                          UpperDimensionHiddenIdss::Size() == ntransform_,
                      "wrong! inconsistent # of transformations");

        // TODO check dependency of dimensions is valid
    }

    __host__ __device__ static constexpr index_t GetNumOfDimension()
    {
        return GetNumOfTopDimension();
    }

    template <index_t IDim>
    __host__ __device__ constexpr auto GetLength(Number<IDim>) const
    {
        static_assert(IDim >= 0 && IDim < ndim_top_, "wrong! out of range");

        constexpr index_t idim_hidden = TopDimensionHiddenIds::At(Number<IDim>{});

        constexpr auto tmp = GetTransformAndItsUpperDimension(Number<idim_hidden>{});

        constexpr index_t itran   = tmp[Number<0>{}];
        constexpr index_t idim_up = tmp[Number<1>{}];
        constexpr bool found      = tmp[Number<2>{}];

        static_assert(found == true,
                      "wrong! not found matching transformation and upper-dimension");

        return transforms_[Number<itran>{}].GetUpperLengths()[Number<idim_up>{}];
    }

    __host__ __device__ constexpr auto GetLengths() const
    {
        // FIXME: use Tuple of reference instead
        return generate_sequence_v2([&](auto I) { return GetLength(I); }, Number<ndim_top_>{});
    }

    __host__ __device__ constexpr auto GetElementSize() const { return element_size_; }

    __host__ __device__ constexpr auto GetElementSpaceSize() const { return element_space_size_; }

    template <typename Idx>
    __host__ __device__ constexpr index_t CalculateOffset(const Idx& idx) const
    {
        static_assert(Idx::Size() == GetNumOfDimension(), "wrong! inconsistent # of dimension");

        return make_tensor_coordinate(*this, idx).GetOffset();
    }

    // TODO make these private
    __host__ __device__ constexpr const auto& GetTransforms() const { return transforms_; }

    __host__ __device__ static constexpr auto GetLowerDimensionHiddenIdss()
    {
        return LowerDimensionHiddenIdss{};
    }

    __host__ __device__ static constexpr auto GetUpperDimensionHiddenIdss()
    {
        return UpperDimensionHiddenIdss{};
    }

    __host__ __device__ static constexpr auto GetTopDimensionHiddenIds()
    {
        return TopDimensionHiddenIds{};
    }

    __host__ __device__ static constexpr bool IsKnownAtCompileTime()
    {
        bool is_known = true;

        static_for<0, Transforms::Size(), 1>{}([&](auto i) {
            is_known &= remove_cvref_t<decltype(Transforms{}[i])>::IsKnownAtCompileTime();
        });

        return is_known && is_known_at_compile_time<ElementSize>::value &&
               is_known_at_compile_time<ElementSpaceSize>::value;
    }

    __host__ __device__ void Print() const
    {
        printf("{");
        printf("TensorDescriptor, ");
        static_for<0, ntransform_, 1>{}([&](auto i) {
            printf("transforms: ");
            transforms_[i].Print();
            printf("LowerDimensionIds:");
            LowerDimensionHiddenIdss{}.At(i).Print();
            printf("UpperDimensionIds:");
            UpperDimensionHiddenIdss{}.At(i).Print();
        });
        printf("}");

        TopDimensionHiddenIds::Print();
    }

    // TODO make these private
    Transforms transforms_;
    ElementSize element_size_;
    ElementSpaceSize element_space_size_;
};

template <index_t NDimHidden, typename TopDimensionHiddenIds>
struct TensorCoordinate
{
    // TODO make these private
    static constexpr index_t ndim_top_ = TopDimensionHiddenIds::Size();

    using HiddenIndex = MultiIndex<NDimHidden>;
    using TopIndex    = MultiIndex<ndim_top_>;

    public:
    __host__ __device__ constexpr TensorCoordinate() = default;

    __host__ __device__ constexpr TensorCoordinate(const HiddenIndex& idx_hidden)
        : idx_hidden_{idx_hidden}
    {
    }

    __host__ __device__ constexpr auto GetIndex() const { return GetTopIndex(); }

    __host__ __device__ constexpr index_t GetOffset() const { return idx_hidden_[Number<0>{}]; }

    // TODO make these private
    __host__ __device__ constexpr const auto& GetHiddenIndex() const { return idx_hidden_; }

    __host__ __device__ auto& GetHiddenIndex() { return idx_hidden_; }

    __host__ __device__ constexpr auto GetTopIndex() const
    {
        return get_container_subset(idx_hidden_, TopDimensionHiddenIds{});
    }

    // TODO make these private
    HiddenIndex idx_hidden_;
};

template <index_t NTransform, index_t NDimTop, typename UpdateLowerIndexHack>
struct TensorCoordinateStep
{
    // TODO make these private
    using TopIndex = MultiIndex<NDimTop>;

    public:
    __host__ __device__ constexpr TensorCoordinateStep() = default;

    __host__ __device__ constexpr TensorCoordinateStep(const TopIndex& idx_diff_top,
                                                       const MultiIndex<NTransform>& do_transforms)
        : idx_diff_top_{idx_diff_top}, do_transforms_{do_transforms}
    {
    }

    __host__ __device__ constexpr const auto& GetIndexDiff() const { return GetTopIndexDiff(); }

    // TODO make these private
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
    static_assert(TensorDesc::GetNumOfDimension() == TopIndex::Size(),
                  "wrong! # of dimension inconsistent");

    constexpr index_t ntransform  = TensorDesc::GetNumOfTransform();
    constexpr index_t ndim_hidden = TensorDesc::GetNumOfHiddenDimension();
    constexpr auto top_dim_ids    = TensorDesc::GetTopDimensionHiddenIds();

    MultiIndex<ndim_hidden> idx_hidden;

    // initialize top index
    set_container_subset(idx_hidden, top_dim_ids, idx_top);

    // calculate hidden index
    static_for<ntransform, 0, -1>{}([&tensor_desc, &idx_hidden](auto itran_p1) {
        auto itran              = itran_p1 - Number<1>{};
        const auto& tran        = tensor_desc.GetTransforms().At(itran);
        constexpr auto dims_low = TensorDesc::GetLowerDimensionHiddenIdss().At(itran);
        constexpr auto dims_up  = TensorDesc::GetUpperDimensionHiddenIdss().At(itran);

        const auto idx_up = get_container_subset(idx_hidden, dims_up);

        MultiIndex<dims_low.Size()> idx_low;

        tran.CalculateLowerIndex(idx_low, idx_up);

        set_container_subset(idx_hidden, dims_low, idx_low);
    });

    return TensorCoordinate<ndim_hidden, decltype(top_dim_ids)>{idx_hidden};
}

// UpdateLowerIndexHack: Sequence<...>
// HACK: control UpdateLowerIndex
template <typename TensorDesc, typename TopIndex, typename UpdateLowerIndexHack>
__host__ __device__ constexpr auto
make_tensor_coordinate_step(const TensorDesc&, const TopIndex& idx_diff_top, UpdateLowerIndexHack)
{
    static_assert(TensorDesc::GetNumOfDimension() == TopIndex::Size(),
                  "wrong! # of dimension inconsistent");

    constexpr index_t ntransform  = TensorDesc::GetNumOfTransform();
    constexpr index_t ndim_hidden = TensorDesc::GetNumOfHiddenDimension();
    constexpr index_t ndim_top    = TensorDesc::GetNumOfTopDimension();
    constexpr auto top_dim_ids    = TensorDesc::GetTopDimensionHiddenIds();

    static_assert(UpdateLowerIndexHack::Size() == ntransform, "wrong!");

    // use index_t for boolean type
    auto do_transforms    = make_zero_multi_index<ntransform>();
    auto is_non_zero_diff = make_zero_multi_index<ndim_hidden>();

    // decide do_transform by checkout non-zero index diff components
    MultiIndex<TopIndex::Size()> non_zero_diff_pick_top;

    static_for<0, ndim_top, 1>{}(
        [&](auto i) { non_zero_diff_pick_top(i) = (idx_diff_top[i] != 0); });

    set_container_subset(is_non_zero_diff, top_dim_ids, non_zero_diff_pick_top);

    static_for<ntransform - 1, -1, -1>{}([&](auto itran) {
        constexpr auto dims_low = TensorDesc::GetLowerDimensionHiddenIdss().At(itran);
        constexpr auto dims_up  = TensorDesc::GetUpperDimensionHiddenIdss().At(itran);

        const auto non_zero_diff_pick_up = get_container_subset(is_non_zero_diff, dims_up);

        MultiIndex<dims_low.Size()> non_zero_diff_pick_low;

        // if any of upper index diff components is non-zero, then
        //   1) Need to do this transform
        //   2) all components of lower index diff will assume to be non-zero and need to be
        //   computed
        const bool idx_diff_up_has_non_zero = container_reduce(
            non_zero_diff_pick_up, [](auto a, auto b) constexpr { return a or b; }, false);

        do_transforms(itran) = idx_diff_up_has_non_zero;

        static_for<0, dims_low.Size(), 1>{}(
            [&](auto i) { non_zero_diff_pick_low(i) = idx_diff_up_has_non_zero; });

        set_container_subset(is_non_zero_diff, dims_low, non_zero_diff_pick_low);
    });

    return TensorCoordinateStep<ntransform, ndim_top, UpdateLowerIndexHack>{idx_diff_top,
                                                                            do_transforms};
}

template <typename TensorDesc, typename TopIndex>
__host__ __device__ constexpr auto make_tensor_coordinate_step(const TensorDesc&,
                                                               const TopIndex& idx_diff_top)
{
    constexpr index_t ntransform = TensorDesc::GetNumOfTransform();

    return make_tensor_coordinate_step(
        TensorDesc{}, idx_diff_top, typename uniform_sequence_gen<ntransform, 0>::type{});
}

template <typename TensorDesc,
          typename TensorCoord,
          index_t NTransform,
          index_t NDimTop,
          typename UpdateLowerIndexHack>
__host__ __device__ constexpr void move_tensor_coordinate(
    const TensorDesc& tensor_desc,
    TensorCoord& coord,
    const TensorCoordinateStep<NTransform, NDimTop, UpdateLowerIndexHack>& coord_step)
{
    constexpr index_t ndim_hidden = TensorDesc::GetNumOfHiddenDimension();
    constexpr index_t ntransform  = TensorDesc::GetNumOfTransform();

    // this is what needs to be calculated
    auto idx_diff_hidden = make_zero_multi_index<ndim_hidden>();

    // initialize top index diff
    set_container_subset(
        idx_diff_hidden, TensorDesc::GetTopDimensionHiddenIds(), coord_step.GetTopIndexDiff());

    // this is what needs to be updated
    auto& idx_hidden = coord.GetHiddenIndex();

    // update top index
    auto idx_hidden_pick_top =
        get_container_subset(idx_hidden, TensorDesc::GetTopDimensionHiddenIds());

    idx_hidden_pick_top += coord_step.GetIndexDiff();

    set_container_subset(idx_hidden, TensorDesc::GetTopDimensionHiddenIds(), idx_hidden_pick_top);

    // update rest of hidden index
    static_for<ntransform - 1, -1, -1>{}([&](auto itran) {
        if(coord_step.do_transforms_[itran])
        {
            const auto& tran        = tensor_desc.GetTransforms().At(itran);
            constexpr auto dims_low = TensorDesc::GetLowerDimensionHiddenIdss().At(itran);
            constexpr auto dims_up  = TensorDesc::GetUpperDimensionHiddenIdss().At(itran);

            const auto idx_up_new  = get_container_subset(idx_hidden, dims_up);
            auto idx_low           = get_container_subset(idx_hidden, dims_low);
            const auto idx_diff_up = get_container_subset(idx_diff_hidden, dims_up);

            MultiIndex<dims_low.Size()> idx_diff_low;

            // HACK: control UpdateLowerIndex for Merge using hack
            constexpr index_t Hack = decltype(coord_step.update_lower_index_hack_)::At(itran);

            tran.UpdateLowerIndex(idx_diff_low, idx_diff_up, idx_low, idx_up_new, Number<Hack>{});

            set_container_subset(idx_diff_hidden, dims_low, idx_diff_low);
            set_container_subset(idx_hidden, dims_low, idx_low);
        }
    });
}

template <typename TensorDesc, typename TensorCoord, typename Index>
__host__ __device__ constexpr void
move_tensor_coordinate(const TensorDesc& tensor_desc, TensorCoord& coord, const Index& coord_step)
{
    move_tensor_coordinate(tensor_desc, coord, make_tensor_coordinate_step(coord_step));
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

template <typename TensorDesc>
using TensorCoordinate_t = decltype(make_tensor_coordinate(
    TensorDesc{}, MultiIndex<remove_cvref_t<TensorDesc>::GetNumOfDimension()>{}));

template <typename TensorDesc>
using TensorCoordinateStep_t = decltype(make_tensor_coordinate_step(
    TensorDesc{}, MultiIndex<remove_cvref_t<TensorDesc>::GetNumOfDimension()>{}));

} // namespace ck
