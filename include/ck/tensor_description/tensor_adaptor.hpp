// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"

namespace ck {

// Transforms: Tuple<transforms...>
// LowerDimensionHiddenIdss : Tuple<Sequence<...>, ...>
// UpperDimensionHiddenIdss : Tuple<Sequence<...>, ...>
// BottomDimensionHiddenIds : Sequence<...>
// TopDimensionHiddenIds : Sequence<...>
template <typename Transforms,
          typename LowerDimensionHiddenIdss,
          typename UpperDimensionHiddenIdss,
          typename BottomDimensionHiddenIds,
          typename TopDimensionHiddenIds>
struct TensorAdaptor
{
    __host__ __device__ static constexpr index_t GetNumOfTransform() { return Transforms::Size(); }

    __host__ __device__ constexpr const auto& GetTransforms() const { return transforms_; }

    __host__ __device__ static constexpr auto GetLowerDimensionHiddenIdss()
    {
        return LowerDimensionHiddenIdss{};
    }

    __host__ __device__ static constexpr auto GetUpperDimensionHiddenIdss()
    {
        return UpperDimensionHiddenIdss{};
    }

    __host__ __device__ static constexpr auto GetBottomDimensionHiddenIds()
    {
        return BottomDimensionHiddenIds{};
    }

    __host__ __device__ static constexpr auto GetTopDimensionHiddenIds()
    {
        return TopDimensionHiddenIds{};
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
        // FIXME: length of bottom dimension is not known, since info about lower dim length are not
        // saved in transformation
        static_assert(IDimHidden >= ndim_bottom_, "wrong! not implemented");

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

    __host__ __device__ static constexpr index_t GetNumOfBottomDimension()
    {
        return BottomDimensionHiddenIds::Size();
    }

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

    constexpr static index_t ntransform_  = GetNumOfTransform();
    constexpr static index_t ndim_hidden_ = GetNumOfHiddenDimension();
    constexpr static index_t ndim_bottom_ = GetNumOfBottomDimension();
    constexpr static index_t ndim_top_    = GetNumOfTopDimension();

    using HiddenIndex = MultiIndex<ndim_hidden_>;
    using BottomIndex = MultiIndex<ndim_bottom_>;
    using TopIndex    = MultiIndex<ndim_top_>;

    // may be index_t or Number<>
    using ElementSize = remove_cv_t<decltype(InitializeElementSize(Transforms{}))>;

    public:
    __host__ __device__ constexpr TensorAdaptor() = default;

    __host__ __device__ constexpr TensorAdaptor(const Transforms& transforms)
        : transforms_{transforms}, element_size_{InitializeElementSize(transforms)}
    {
        static_assert(Transforms::Size() == ntransform_ &&
                          LowerDimensionHiddenIdss::Size() == ntransform_ &&
                          UpperDimensionHiddenIdss::Size() == ntransform_,
                      "wrong! inconsistent # of transformations");

        // TODO check dependency of dimensions is valid
    }

    __host__ __device__ constexpr auto GetElementSize() const { return element_size_; }

    // FIXME: this logic is wrong when getting bottome dimension lengths
    template <index_t IDimHidden>
    __host__ __device__ constexpr auto GetHiddenDimensionLength(Number<IDimHidden>) const
    {
        static_assert(IDimHidden >= 0 && IDimHidden < ndim_hidden_, "wrong! out of range");

        constexpr auto tmp = GetTransformAndItsUpperDimension(Number<IDimHidden>{});

        constexpr index_t itran   = tmp[Number<0>{}];
        constexpr index_t idim_up = tmp[Number<1>{}];
        constexpr bool found      = tmp[Number<2>{}];

        static_assert(found == true,
                      "wrong! not found matching transformation and upper-dimension");

        return transforms_[Number<itran>{}].GetUpperLengths()[Number<idim_up>{}];
    }

    template <index_t IDimTop>
    __host__ __device__ constexpr auto GetTopDimensionLength(Number<IDimTop> idim_top) const
    {
        return GetHiddenDimensionLength(TopDimensionHiddenIds::At(idim_top));
    }

#if 0
    // FIXME: GetHiddenDimensionLength is wrong when getting bottome dimension lengths
    template <index_t IDimBottom>
    __host__ __device__ constexpr index_t
    GetBottomDimensionLength(Number<IDimBottom> idim_bottom) const
    {
        return GetHiddenDimensionLength(TopDimensionHiddenIds::At(idim_bottom));
    }
#endif

    __host__ __device__ constexpr auto GetTopDimensionLengths() const
    {
        return generate_tuple([&](auto i) { return GetTopDimensionLength(i); },
                              Number<ndim_top_>{});
    }

#if 0
    // FIXME: GetHiddenDimensionLength is wrong when getting bottome dimension lengths
    __host__ __device__ constexpr auto GetBottomDimensionLengths() const
    {
        return generate_tuple([&](auto i) { return GetBottomDimensionLength(i); },
                              Number<ndim_bottom_>{});
    }
#endif

    template <typename TopIdx>
    __host__ __device__ constexpr auto CalculateBottomIndex(const TopIdx& idx_top) const
    {
        static_assert(TopIdx::Size() == TopDimensionHiddenIds::Size(),
                      "wrong! # of dimension inconsistent");

        constexpr index_t ntransform  = GetNumOfTransform();
        constexpr index_t ndim_hidden = GetNumOfHiddenDimension();

        MultiIndex<ndim_hidden> idx_hidden;

        // initialize uppest index
        set_container_subset(idx_hidden, GetTopDimensionHiddenIds(), idx_top);

        // calculate hidden index
        static_for<ntransform, 0, -1>{}([&](auto itran_p1) {
            auto itran              = itran_p1 - Number<1>{};
            const auto& tran        = GetTransforms().At(itran);
            constexpr auto dims_low = GetLowerDimensionHiddenIdss().At(itran);
            constexpr auto dims_up  = GetUpperDimensionHiddenIdss().At(itran);

            const auto idx_up = get_container_subset(idx_hidden, dims_up);

            MultiIndex<dims_low.Size()> idx_low;

            tran.CalculateLowerIndex(idx_low, idx_up);

            set_container_subset(idx_hidden, dims_low, idx_low);
        });

        return get_container_subset(idx_hidden, BottomDimensionHiddenIds{});
    }

    __host__ __device__ static constexpr bool IsStatic()
    {
        bool is_known = true;

        static_for<0, Transforms::Size(), 1>{}([&](auto i) {
            is_known &= remove_cvref_t<decltype(Transforms{}[i])>::IsKnownAtCompileTime();
        });

        return is_known && is_known_at_compile_time<ElementSize>::value;
    }

    __host__ __device__ static constexpr bool IsKnownAtCompileTime() { return IsStatic(); }

    __host__ __device__ static constexpr auto GetTopDimensionSafeVectorLengthStrides(
        const Array<index_t, ndim_hidden_>& guaranteed_vector_lengths,
        const Array<index_t, ndim_hidden_>& guaranteed_vector_strides)
    {
        auto vector_lengths = guaranteed_vector_lengths;
        auto vector_strides = guaranteed_vector_strides;

        static_for<0, GetNumOfTransform(), 1>{}([&](auto itran) {
            constexpr auto low_dims = GetLowerDimensionHiddenIdss().At(itran);
            constexpr auto up_dims  = GetUpperDimensionHiddenIdss().At(itran);

            const auto up_guaranteed_vector_lengths =
                get_container_subset(guaranteed_vector_lengths, up_dims);
            const auto up_guaranteed_vector_strides =
                get_container_subset(guaranteed_vector_strides, up_dims);

            // only need type of transform
            auto [up_vector_lengths, up_vector_strides] =
                Transforms{}.At(itran).CalculateUpperDimensionSafeVectorLengthStrides(
                    get_container_subset(vector_lengths, low_dims),
                    get_container_subset(vector_strides, low_dims));

            if constexpr(up_dims.Size() > 0)
            {
                for(index_t i = 0; i < up_dims.Size(); ++i)
                {
                    up_vector_lengths(i) = (up_guaranteed_vector_lengths[i] != -1)
                                               ? up_guaranteed_vector_lengths[i]
                                               : up_vector_lengths[i];

                    up_vector_strides(i) = (up_guaranteed_vector_strides[i] != -1)
                                               ? up_guaranteed_vector_strides[i]
                                               : up_vector_strides[i];
                }
            }

            set_container_subset(vector_lengths, up_dims, up_vector_lengths);
            set_container_subset(vector_strides, up_dims, up_vector_strides);
        });

        constexpr auto top_dims = TopDimensionHiddenIds{};

        return make_tuple(get_container_subset(vector_lengths, top_dims),
                          get_container_subset(vector_strides, top_dims));
    }

    __host__ __device__ void Print() const
    {
        printf("TensorAdaptor{");

        //
        printf("transforms: ");
        print(transforms_);
        printf(", ");

        //
        printf("LowerDimensionHiddenIds: ");
        print(LowerDimensionHiddenIdss{});
        printf(", ");

        //
        printf("UpperDimensionHiddenIds: ");
        print(UpperDimensionHiddenIdss{});
        printf(", ");

        //
        printf("BottomDimensionHiddenIds: ");
        print(BottomDimensionHiddenIds{});
        printf(", ");

        //
        printf("TopDimensionHiddenIds: ");
        print(TopDimensionHiddenIds{});

        printf("}");
    }

    private:
    Transforms transforms_;
    ElementSize element_size_;
};

// Transforms: Tuple<transforms...>
// LowerDimensionOldTopIdss: Tuple<Sequence<...>, ...>
// UpperDimensionNewTopIdss: Tuple<Sequence<...>, ...>
template <typename Transforms, typename LowerDimensionOldTopIdss, typename UpperDimensionNewTopIdss>
__host__ __device__ constexpr auto make_single_stage_tensor_adaptor(const Transforms& transforms,
                                                                    LowerDimensionOldTopIdss,
                                                                    UpperDimensionNewTopIdss)
{
    constexpr index_t ntransform = Transforms::Size();

    static_assert(LowerDimensionOldTopIdss::Size() == ntransform &&
                      UpperDimensionNewTopIdss::Size() == ntransform,
                  "wrong!");

    // sanity check on LowerDimensionOldTopIdss and UpperDimensionNewTopIdss
    constexpr auto all_low_dim_old_top_ids = unpack(
        [](auto&&... xs) constexpr { return merge_sequences(xs...); }, LowerDimensionOldTopIdss{});

    constexpr auto all_up_dim_new_top_ids = unpack(
        [](auto&&... xs) constexpr { return merge_sequences(xs...); }, UpperDimensionNewTopIdss{});

    static_assert(is_valid_sequence_map<decltype(all_low_dim_old_top_ids)>::value &&
                      is_valid_sequence_map<decltype(all_up_dim_new_top_ids)>::value,
                  "wrong!");

    constexpr index_t ndim_old_top = all_low_dim_old_top_ids.Size();
    constexpr index_t ndim_new_top = all_up_dim_new_top_ids.Size();

    // low_dim_hidden_idss
    constexpr auto low_dim_hidden_idss = LowerDimensionOldTopIdss{};

    // up_dim_hidden_idss: shift UpperDimensionNewTopIdss by ndim_bottom
    constexpr auto up_dim_hidden_idss = generate_tuple(
        [](auto itran) { return UpperDimensionNewTopIdss{}[itran] + Number<ndim_old_top>{}; },
        Number<ntransform>{});

    // bottom_dim_hidden_ids
    constexpr auto bottom_dim_hidden_ids =
        typename arithmetic_sequence_gen<0, ndim_old_top, 1>::type{};

    // top_dim_hidden_ids
    constexpr auto top_dim_hidden_ids =
        typename arithmetic_sequence_gen<0, ndim_new_top, 1>::type{} + Number<ndim_old_top>{};

    return TensorAdaptor<remove_cvref_t<Transforms>,
                         remove_cvref_t<decltype(low_dim_hidden_idss)>,
                         remove_cvref_t<decltype(up_dim_hidden_idss)>,
                         remove_cvref_t<decltype(bottom_dim_hidden_ids)>,
                         remove_cvref_t<decltype(top_dim_hidden_ids)>>{transforms};
}

// TODO: How to fix this? It uses an struct instead of lambda because lambda
// doesn't have constructor, and to put it outside the scope where it is used
// (transform_tensor_adaptor) because template cannot be defined inside a function
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

template <typename OldTensorAdaptor,
          typename NewTransforms,
          typename NewLowerDimensionOldTopIdss,
          typename NewUpperDimensionNewTopIdss>
__host__ __device__ constexpr auto
transform_tensor_adaptor(const OldTensorAdaptor& old_tensor_adaptor,
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
                    return OldTensorAdaptor::GetTopDimensionHiddenIds()[low_dim_top_id];
                },
                low_dim_top_ids);
        },
        NewLowerDimensionOldTopIdss{});

    constexpr index_t num_new_transform = NewTransforms::Size();

    // upper dimension's hidden idss
    constexpr index_t old_hidden_dim_number = OldTensorAdaptor::GetNumOfHiddenDimension();

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
    const auto all_transforms =
        container_concat(old_tensor_adaptor.GetTransforms(), new_transforms);

    constexpr auto all_low_dim_hidden_idss =
        container_concat(OldTensorAdaptor::GetLowerDimensionHiddenIdss(), low_dim_hidden_idss);

    constexpr auto all_up_dim_hidden_idss =
        container_concat(OldTensorAdaptor::GetUpperDimensionHiddenIdss(), up_dim_hidden_idss);

    return TensorAdaptor<remove_cvref_t<decltype(all_transforms)>,
                         remove_cvref_t<decltype(all_low_dim_hidden_idss)>,
                         remove_cvref_t<decltype(all_up_dim_hidden_idss)>,
                         remove_cvref_t<decltype(OldTensorAdaptor::GetBottomDimensionHiddenIds())>,
                         remove_cvref_t<decltype(new_top_dim_hidden_ids)>>{all_transforms};
}

template <typename TensorAdaptor0, typename TensorAdaptor1>
__host__ __device__ constexpr auto chain_tensor_adaptors(const TensorAdaptor0& adaptor0,
                                                         const TensorAdaptor1& adaptor1)
{
    static_assert(TensorAdaptor0::GetNumOfTopDimension() ==
                      TensorAdaptor1::GetNumOfBottomDimension(),
                  "wrong!");

    // all_transforms = transform0 + transform1
    const auto all_transforms =
        container_concat(adaptor0.GetTransforms(), adaptor1.GetTransforms());

    // shift
    constexpr index_t adaptor0_max_hidden_id = [&]() {
        index_t adaptor0_max_hidden_id_ = NumericLimits<index_t>::Min();

        static_for<0, TensorAdaptor0::GetNumOfTransform(), 1>{}([&](auto itran) {
            constexpr index_t ndim_low =
                TensorAdaptor0{}.GetTransforms()[itran].GetNumOfLowerDimension();

            static_for<0, ndim_low, 1>{}([&](auto idim_low) {
                adaptor0_max_hidden_id_ =
                    math::max(adaptor0_max_hidden_id_,
                              TensorAdaptor0::GetLowerDimensionHiddenIdss()[itran][idim_low].value);
            });

            constexpr index_t ndim_up =
                TensorAdaptor0{}.GetTransforms()[itran].GetNumOfUpperDimension();

            static_for<0, ndim_up, 1>{}([&](auto idim_up) {
                adaptor0_max_hidden_id_ =
                    math::max(adaptor0_max_hidden_id_,
                              TensorAdaptor0::GetUpperDimensionHiddenIdss()[itran][idim_up].value);
            });
        });

        return adaptor0_max_hidden_id_;
    }();

    constexpr index_t adaptor1_min_hidden_id = [&]() {
        index_t adaptor1_min_hidden_id_ = NumericLimits<index_t>::Max();

        static_for<0, TensorAdaptor1::GetNumOfTransform(), 1>{}([&](auto itran) {
            constexpr index_t ndim_low =
                TensorAdaptor1{}.GetTransforms()[itran].GetNumOfLowerDimension();

            // get the min of all lower dimenions, but not bottom dimension (because their id will
            // be matched with top id from adaptor0)
            static_for<0, ndim_low, 1>{}([&](auto idim_low) {
                constexpr index_t low_dim_hidden_id =
                    TensorAdaptor1::GetLowerDimensionHiddenIdss()[itran][idim_low].value;

                bool is_bottom_dim = false;
                static_for<0, TensorAdaptor1::GetNumOfBottomDimension(), 1>{}([&](auto i) {
                    if constexpr(low_dim_hidden_id ==
                                 TensorAdaptor1::GetBottomDimensionHiddenIds()[i])
                    {
                        is_bottom_dim = true;
                    }
                });

                if(!is_bottom_dim)
                {
                    adaptor1_min_hidden_id_ = math::min(adaptor1_min_hidden_id_, low_dim_hidden_id);
                }
            });

            constexpr index_t ndim_up =
                TensorAdaptor1{}.GetTransforms()[itran].GetNumOfUpperDimension();

            // get the min of all upper dimensions
            static_for<0, ndim_up, 1>{}([&](auto idim_up) {
                adaptor1_min_hidden_id_ =
                    math::min(adaptor1_min_hidden_id_,
                              TensorAdaptor1::GetUpperDimensionHiddenIdss()[itran][idim_up].value);
            });
        });

        return adaptor1_min_hidden_id_;
    }();

    constexpr index_t adaptor1_hidden_id_shift =
        adaptor0_max_hidden_id + 1 - adaptor1_min_hidden_id;

    constexpr index_t ndim_bottom_1 = TensorAdaptor1::GetNumOfBottomDimension();

    // all_low_dim_hidden_idss =
    // low_dim_hidden_idss_0 + match_hidden_id_for_1(shift_hidden_id_for_1(low_dim_hiden_idss_1))
    constexpr auto low_dim_hidden_idss_1 = generate_tuple(
        // generate sequence of ids for a transform
        [&](auto itran) {
            constexpr auto ndim_low_1 = TensorAdaptor1::GetLowerDimensionHiddenIdss()[itran].Size();

            constexpr auto low_dim_hidden_ids_1 =
                TensorAdaptor1::GetLowerDimensionHiddenIdss()[itran];

            // sequence in, sequence out
            constexpr auto low_dim_hidden_ids_1_mod = [&]() constexpr
            {
                auto low_dim_hidden_ids_1_mod_ = to_multi_index(low_dim_hidden_ids_1);

                // shift hidden id so every dim id is unique
                static_for<0, ndim_low_1, 1>{}([&](auto idim_low_1) {
                    low_dim_hidden_ids_1_mod_(idim_low_1) += adaptor1_hidden_id_shift;
                });

                // match hidden id
                static_for<0, ndim_low_1, 1>{}([&](auto idim_low_1) {
                    static_for<0, ndim_bottom_1, 1>{}([&](auto idim_bottom_1) {
                        // if this low dim is bottom dim, then do id matching
                        if constexpr(low_dim_hidden_ids_1[idim_low_1] ==
                                     TensorAdaptor1::GetBottomDimensionHiddenIds()[idim_bottom_1])
                        {
                            low_dim_hidden_ids_1_mod_(idim_low_1) =
                                TensorAdaptor0::GetTopDimensionHiddenIds()[idim_bottom_1];
                        }
                    });
                });

                return low_dim_hidden_ids_1_mod_;
            }
            ();

            return generate_sequence_v2(
                [&](auto i) constexpr { return Number<low_dim_hidden_ids_1_mod[i]>{}; },
                Number<ndim_low_1>{});
        },
        Number<TensorAdaptor1::GetNumOfTransform()>{});

    constexpr auto all_low_dim_hidden_idss =
        container_concat(TensorAdaptor0::GetLowerDimensionHiddenIdss(), low_dim_hidden_idss_1);

    // all_up_dim_hidden_idss =
    // up_dim_hidden_idss_0 + shift_hidden_id_for_1(up_dim_hiden_idss_1)
    constexpr auto up_dim_hidden_idss_1 = generate_tuple(
        // generate sequence of ids for a transform
        [&](auto itran) {
            constexpr auto ndim_up_1 = TensorAdaptor1::GetUpperDimensionHiddenIdss()[itran].Size();

            constexpr auto up_dim_hidden_ids_1 =
                TensorAdaptor1::GetUpperDimensionHiddenIdss()[itran];

            // sequence in, constexpr tuple out
            constexpr auto up_dim_hidden_ids_1_mod = [&]() constexpr
            {
                auto up_dim_hidden_ids_1_mod_ = to_multi_index(up_dim_hidden_ids_1);

                // shift hidden id
                static_for<0, ndim_up_1, 1>{}([&](auto idim_up_1) {
                    up_dim_hidden_ids_1_mod_(idim_up_1) += adaptor1_hidden_id_shift;
                });

                return up_dim_hidden_ids_1_mod_;
            }
            ();

            // constexpr tuple to sequence
            return generate_sequence_v2(
                [&](auto i) constexpr { return Number<up_dim_hidden_ids_1_mod[i]>{}; },
                Number<ndim_up_1>{});
        },
        Number<TensorAdaptor1::GetNumOfTransform()>{});

    constexpr auto all_up_dim_hidden_idss =
        container_concat(TensorAdaptor0::GetUpperDimensionHiddenIdss(), up_dim_hidden_idss_1);

    // bottom_dim_hidden_ids = bottom_dim_hidden_ids_0
    constexpr auto bottom_dim_hidden_ids = TensorAdaptor0::GetBottomDimensionHiddenIds();

    // top_dim_hidden_ids = shift_hidden_id(top_dim_hidden_ids_1)
    constexpr auto top_dim_hidden_ids =
        TensorAdaptor1::GetTopDimensionHiddenIds() + Number<adaptor1_hidden_id_shift>{};

    // put everything together
    return TensorAdaptor<remove_cvref_t<decltype(all_transforms)>,
                         remove_cvref_t<decltype(all_low_dim_hidden_idss)>,
                         remove_cvref_t<decltype(all_up_dim_hidden_idss)>,
                         remove_cvref_t<decltype(bottom_dim_hidden_ids)>,
                         remove_cvref_t<decltype(top_dim_hidden_ids)>>{all_transforms};
}

template <typename X, typename... Xs, typename enable_if<sizeof...(Xs) >= 2, bool>::type = false>
__host__ __device__ constexpr auto chain_tensor_adaptors(const X& x, const Xs&... xs)
{
    return chain_tensor_adaptors(x, chain_tensor_adaptors(xs...));
}

} // namespace ck
