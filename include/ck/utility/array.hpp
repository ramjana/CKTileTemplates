// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

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

        index_t i   = 0;
        TData vlast = TData{};

        for(const TData& val : ilist)
        {
            mData[i] = val;
            vlast    = val;
            ++i;
        }

        for(; i < NSize; ++i)
        {
            mData[i] = vlast;
        }
    }

    __host__ __device__ static constexpr index_t Size() { return NSize; }

    template <index_t I>
    __host__ __device__ constexpr const TData& At() const
    {
        return mData[I];
    }

    template <index_t I>
    __host__ __device__ constexpr TData& At()
    {
        return mData[I];
    }

    __host__ __device__ constexpr const TData& At(index_t i) const { return mData[i]; }

    __host__ __device__ constexpr TData& At(index_t i) { return mData[i]; }

    __host__ __device__ constexpr const TData& operator[](index_t i) const { return mData[i]; }

    __host__ __device__ constexpr TData& operator()(index_t i) { return mData[i]; }

    template <typename T>
    __host__ __device__ constexpr auto operator=(const T& a)
    {
        static_assert(T::Size() == Size(), "wrong! size not the same");

        for(index_t i = 0; i < NSize; ++i)
        {
            mData[i] = a[i];
        }

        return *this;
    }
};

// empty Array
template <typename TData>
struct Array<TData, 0>
{
    using type      = Array;
    using data_type = TData;

    __host__ __device__ constexpr Array() {}

    __host__ __device__ static constexpr index_t Size() { return 0; }
};

template <typename T, typename... Xs>
__host__ __device__ constexpr auto make_array(Xs&&... xs)
{
    using data_type = remove_cvref_t<T>;

    return Array<data_type, sizeof...(Xs)>{std::forward<Xs>(xs)...};
}

template <typename F, index_t N>
__host__ __device__ constexpr auto generate_array(F&& f, Number<N>)
{
    using T = remove_cvref_t<decltype(f(0))>;

    return unpack([&f](auto&&... is) { return Array<T, N>{f(is)...}; },
                  typename arithmetic_sequence_gen<0, N, 1>::type{});
}

template <typename T, index_t N, typename X>
__host__ __device__ constexpr auto to_array(const X& x)
{
    STATIC_ASSERT(N <= X::Size(), "");

    Array<T, N> arr;

    static_for<0, N, 1>{}([&x, &arr](auto i) { arr(i) = x[i]; });

    return arr;
}

} // namespace ck
