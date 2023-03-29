
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "ck/utility/data_type.hpp"
#include "ck/utility/enable_if.hpp"
#include "ck/utility/c_style_pointer_cast.hpp"

namespace ck {

// T may be scalar or vector
// X may be scalar or vector
// T and X have same scalar type
// X contains multiple T
// FIXME: InvalidElementUseNumericalZeroValue and invalid_element_value_ should be a property of
//        transforms of TensorView/Tensor
template <AddressSpaceEnum BufferAddressSpace,
          typename T,
          typename BufferSizeType,
          bool InvalidElementUseNumericalZeroValue>
struct BufferView;

} // namespace ck
