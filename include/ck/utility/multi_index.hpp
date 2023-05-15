// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "common_header.hpp"

#if CK_EXPERIMENTAL_USE_DYNAMICALLY_INDEXED_MULTI_INDEX
#include "array_multi_index.hpp"
#else
#include "statically_indexed_array_multi_index.hpp"
#endif

namespace ck {

template <typename T, index_t N>
__host__ __device__ void print_multi_index(const Array<T, N>& a)
{
    printf("{");
    printf("Array, ");
    printf("size %d,", N);
    for(index_t i = 0; i < N; ++i)
    {
        printf("%d ", static_cast<index_t>(a[i]));
    }
    printf("}");
}

template <typename... Xs>
__host__ __device__ void print_multi_index(const Tuple<Xs...>& x)
{
    printf("{");
    printf("Tuple, ");
    printf("size %d,", index_t{sizeof...(Xs)});
    static_for<0, sizeof...(Xs), 1>{}(
        [&](auto i) { printf("%d ", static_cast<index_t>(x.At(i))); });
    printf("}");
}

} // namespace ck
