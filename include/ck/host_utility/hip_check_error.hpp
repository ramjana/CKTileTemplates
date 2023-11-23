// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <sstream>
#include <stdexcept>

#include <hip/hip_runtime.h>

#define HIP_CHECK_ERROR(val)                                                                      \
    do                                                                                            \
    {                                                                                             \
        hipError_t _tmpVal;                                                                       \
        if((_tmpVal = (val)) != hipSuccess)                                                       \
        {                                                                                         \
            std::ostringstream ss;                                                                \
            ss << "HIP runtime error: " << hipGetErrorString(_tmpVal) << ". " << __FILE__ << ": " \
               << __LINE__ << "in function: " << __func__;                                        \
            throw std::runtime_error(ss.str());                                                   \
        }                                                                                         \
    } while(0)
