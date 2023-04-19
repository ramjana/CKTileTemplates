// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <initializer_list>

#include "functional2.hpp"
#include "sequence.hpp"

namespace ck {

template <typename TData, index_t NSize>
struct Array
{
    using type      = Array;
    using data_type = TData;

    TData mData[NSize];

    __host__ __device__ constexpr Array() : mData{} {}

    __host__ __device__ constexpr Array(std::initializer_list<TData> ilist)
    {
        constexpr index_t list_size = std::initializer_list<TData>{}.size();

        static_assert(list_size <= NSize, "out of bound");

        index_t i = 0;
        for(const TData& val : ilist)
        {
            mData[i] = val;
            ++i;
        }

        for(; i < NSize; ++i)
        {
            mData[i] = TData();
        }
    }

    __host__ __device__ static constexpr index_t Size() { return NSize; }

    __host__ __device__ constexpr const TData& At(index_t i) const { return mData[i]; }

    __host__ __device__ constexpr TData& At(index_t i) { return mData[i]; }

    __host__ __device__ constexpr const TData& operator[](index_t i) const { return mData[i]; }

    __host__ __device__ constexpr TData& operator[](index_t i) { return mData[i]; }

    // TODO: remove
    __host__ __device__ constexpr TData& operator()(index_t i) { return mData[i]; }

    template <typename T>
    __host__ __device__ constexpr auto operator=(const T& a)
    {
        static_assert(T::Size() == Size(), "wrong! size not the same");

#if 0
        static_for<0, Size(), 1>{}([&](auto i) { operator()(i) = a[i]; });
#else
        for(index_t i = 0; i < NSize; ++i)
        {
            mData[i] = a[i];
        }
#endif

        return *this;
    }
};

// empty Array
template <typename TData>
struct Array<TData, 0>
{
    using type      = Array;
    using data_type = TData;

    __host__ __device__ static constexpr index_t Size() { return 0; }
};

template <typename X, typename... Xs>
__host__ __device__ constexpr auto make_array(X&& x, Xs&&... xs)
{
    using data_type = remove_cvref_t<X>;
    return Array<data_type, sizeof...(Xs) + 1>{std::forward<X>(x), std::forward<Xs>(xs)...};
}

// make empty array
template <typename X>
__host__ __device__ constexpr auto make_array()
{
    return Array<X, 0>{};
}

} // namespace ck
