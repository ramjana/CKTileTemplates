// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "ck/utility/enable_if.hpp"
#include "ck/utility/c_style_pointer_cast.hpp"
#include "ck/utility/data_type.hpp"

namespace ck {

template <typename DataType_>
struct ThreadPrivateBuffer
{
    using DataType = DataType_;
    using T        = DataType;

    using d1_t   = typename vector_type_maker<T, 1>::type;
    using d2_t   = typename vector_type_maker<T, 2>::type;
    using d4_t   = typename vector_type_maker<T, 4>::type;
    using d8_t   = typename vector_type_maker<T, 8>::type;
    using d16_t  = typename vector_type_maker<T, 16>::type;
    using d32_t  = typename vector_type_maker<T, 32>::type;
    using d64_t  = typename vector_type_maker<T, 64>::type;
    using d128_t = typename vector_type_maker<T, 128>::type;

    static constexpr index_t kMaxBufferSize_ = 32;

    //
    union
    {
        d1_t p_d1_[kMaxBufferSize_];
        d2_t p_d2_[kMaxBufferSize_ / 2];
        d4_t p_d4_[kMaxBufferSize_ / 4];
        d8_t p_d8_[kMaxBufferSize_ / 8];
        d16_t p_d8_[kMaxBufferSize_ / 16];
    } data_;
};

} // namespace ck
