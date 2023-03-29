// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "enable_if.hpp"
#include "c_style_pointer_cast.hpp"

namespace ck {

template <typename DataType_>
struct ThreadPrivateBuffer
{
    using DataType = DataType_;

    static constexpr index_t kMaxBufferSize_ = 32;

    //
    DataType p_data_[kMaxBufferSize_];
};

} // namespace ck
