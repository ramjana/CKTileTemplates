// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/library/utility/host_tensor.hpp"

template <typename CDataType>
void reference_batched_masking(Tensor<CDataType>& c_b_m_n)
{
    const int M = c_b_m_n.mDesc.GetLengths()[1];
    const int N = c_b_m_n.mDesc.GetLengths()[2];

    auto f = [&](auto batch){
        for(int n = 0; n < N; ++n){
            for(int m = 0; m < M; ++m){
                if(n > m){
                    c_b_m_n(batch, m, n) = -ck::NumericLimits<CDataType>::Infinity();
                }
            }
        }
    };

    make_ParallelTensorFunctor(f, c_b_m_n.mDesc.GetLengths()[0])(
        std::thread::hardware_concurrency());
}
