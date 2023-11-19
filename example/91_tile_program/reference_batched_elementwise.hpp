// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <functional>

#include "ck/utility/common_header.hpp"
#include "ck/library/utility/host_tensor.hpp"

template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CDataType,
          typename AElementOp      = ck::identity,
          typename BElementOp      = ck::identity,
          typename BinaryElementOp = std::plus<AccDataType>>
void reference_batched_elementwise(const Tensor<ADataType>& a_b_m_n,
                                   const Tensor<BDataType>& b_b_m_n,
                                   Tensor<CDataType>& c_b_m_n,
                                   const AElementOp& a_element_op           = {},
                                   const BElementOp& b_element_op           = {},
                                   const BinaryElementOp& binary_element_op = {})
{
    const int N = c_b_m_n.mDesc.GetLengths()[2];

    auto f = [&](auto batch, auto m) {
        for(int n = 0; n < N; ++n)
        {
            auto v_a = ck::type_convert<AccDataType>(a_element_op(a_b_m_n(batch, m, n)));
            auto v_b = ck::type_convert<AccDataType>(b_element_op(b_b_m_n(batch, m, n)));

            c_b_m_n(batch, m, n) = ck::type_convert<CDataType>(binary_element_op(v_a, v_b));
        }
    };

    make_ParallelTensorFunctor(f, c_b_m_n.mDesc.GetLengths()[0], c_b_m_n.mDesc.GetLengths()[1])(
        std::thread::hardware_concurrency());
}
