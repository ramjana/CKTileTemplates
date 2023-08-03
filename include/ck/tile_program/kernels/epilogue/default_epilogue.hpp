#pragma once


#include <cstring>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/cluster_descriptor.hpp"
#include "ck/tensor/tensor_view.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/host_utility/device_prop.hpp"

#include "ck/tile_program/tile_program.hpp"
#include "ck/tile_program/meta_data_buffer.hpp"
#include "ck/tile_program/block_tensor_distribution.hpp"
#include "ck/tile_program/block_tensor_window.hpp"
#include "ck/tile_program/static_block_distributed_tensor.hpp"
#include "ck/tile_program/load_block_distributed_tensor.hpp"
#include "ck/tile_program/store_block_distributed_tensor.hpp"
#include "ck/tile_program/block_gemm_impl_cr_as_bs.hpp"
#include "ck/tile_program/block_elementwise.hpp"
#include "ck/tile_program/kernels/gemm_global_load_tile_encoding_predef.hpp"

// #include "ck/library/utility/check_err.hpp"
// #include "ck/library/utility/device_memory.hpp"
// #include "ck/library/utility/fill.hpp"
// #include "ck/library/utility/host_tensor.hpp"
// #include "ck/library/utility/host_tensor_generator.hpp"

namespace ck::tile_program {

template<typename ProblemDesc_,
        typename Policy_>
struct DefaultEpilogue
{
    using ProblemDesc = ProblemDesc_;
    using Policy = Policy_;


    struct Arguments {
        CType * p_c;
    };

    const Arguments & args;
    char * p_smem;

    DefaultEpilogue(const Arguments & args_, char * p_smem_)
        : args(args_), p_smem(p_smem_)
    {}

    static std::string Name()
    {
        return "";
    }

    template<typename AccTile, typename CTileWindow>
    __device__
    void operator()(const AccTile & acc_tile, CTileWindow & c_tile)
    {

    }
};

}