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

template<
    typename Arch_,
    index_t NumPrefetchA_ = 1,
    index_t NumPrefetchB_ = 1,
    bool SkipLdsA_ = false,
    bool SkipLdsB_ = false
>
struct GemmMainloopXdlopsTraits
{
    // tweak the behavior of the main loop
    using Arch = Arch_;
    static constexpr index_t NumPrefetchA = NumPrefetchA_;
    static constexpr index_t NumPrefetchB = NumPrefetchB_;
    static constexpr bool SkipLdsA = SkipLdsA_;
    static constexpr bool SkipLdsB = SkipLdsB_;
};

template<
         typename ProblemDesc_,
         typename Policy_,
         >
struct GemmMainloop{
    using ProblemDesc = ProblemDesc_;
    using Policy = Policy_;
    using BlockTileDesc = typename ProblemDesc::BlockTileDesc;

    using  AType      =   typename ProblemDesc::AType;
    using  BType      =   typename ProblemDesc::BType;
    using  CType      =   typename ProblemDesc::CType;

    using  ALayout    =   typename ProblemDesc::ALayout;
    using  BLayout    =   typename ProblemDesc::BLayout;
    using  CLayout    =   typename ProblemDesc::CLayout;

    static constexpr auto MPerBlock = BlockTileDesc::MPerBlock;
    static constexpr auto NPerBlock = BlockTileDesc::NPerBlock;
    static constexpr auto KPerBlock = BlockTileDesc::KPerBlock;

    static constexpr auto BlockSize = BlockTileDesc::BlockSize;

    static constexpr auto AlignmentA = ProblemDesc::AlignmentA;
    static constexpr auto AlignmentB = ProblemDesc::AlignmentB;
    static constexpr auto AlignmentC = ProblemDesc::AlignmentC;


    struct Arguments {
        AType * p_a;
        BType * p_b;
    };

    const Arguments& args;
    char * p_smem;  // shared memory workspace

    __host__ __device__
    static std::string Name()
    {
        return Policy::Name();
    }

    __device__
    GemmMainloop(const Arguments & args_, char * p_smem_) :
        args(args_), p_smem(p_smem_)
    {}

    __device__
    constexpr auto MakeLdsCopyWindowA()
    {
        AType* p_a_lds = static_cast<AType*>(static_cast<void*>(p_smem));
        constexpr auto a_lds_block_desc = Policy::MakeALdsBlockDescriptor();
        auto a_lds_block = make_tensor_view<AddressSpaceEnum::Lds>(p_a_lds, a_lds_block_desc);

        return a_lds_block;
    }

    __device__
    constexpr auto MakeLdsCopyWindowB()
    {
        constexpr index_t a_lds_block_space_size_aligned = ck::math::integer_divide_ceil(
                   sizeof(AType) * Policy::MakeALdsBlockDescriptor().GetElementSpaceSize(), 16) * 16;
        BType* p_b_lds = static_cast<BType*>(
            static_cast<void*>(p_smem + a_lds_block_space_size_aligned));
        constexpr auto b_lds_block_desc = Policy::MakeBLdsBlockDescriptor();
        auto b_lds_block = make_tensor_view<AddressSpaceEnum::Lds>(p_b_lds, b_lds_block_desc);

        return b_lds_block;
    }

    __host__ __device__
    static constexpr index_t GetLdsSize()
    {
        return Policy::GetLdsSize();
    }

    template<typename ALengths_, typename AStrides_, typename ACoords_>
    __device__
    static constexpr auto MakeAGlobalTileWindow(const ALengths_ & lengths, const AStrides_ & strides, const ACoords_ & coords)
    {
        // TODO: RCR Layout
        const auto a_dram_grid = make_naive_tensor_view<AddressSpaceEnum::Global>(
            args.p_a, lengths, strides, Number<32>{}, Number<1>{});
        auto a_global_tile_window = make_block_window(a_dram_grid, coords, Policy::MakeAGlobalTileDistribution());
        return a_global_tile_window;
    }

    
    template<typename BLengths_, typename BStrides_, typename BCoords_>
    __device__
    constexpr auto MakeBGlobalTileWindow(const BLengths_ & lengths, const BStrides_ & strides, const BCoords_ & coords)
    {
        const auto b_dram_grid = make_naive_tensor_view<AddressSpaceEnum::Global>(
            args.p_b, lengths, strides, Number<32>{}, Number<1>{});
        auto b_global_tile_window = make_block_window(b_dram_grid, coords, Policy::MakeBGlobalTileDistribution());
        return b_global_tile_window;
    }

    __device__
    constexpr auto MakeAccTile()
    {

    }

    template <typename AGlobalTileWindow, typename BGlobalTileWindow, typename AWindowStep, typename BWindowStep, typename AccTile>
    __device__
    constexpr void operator()(
        AGlobalTileWindow & a_global_tile_window,
        BGlobalTileWindow & b_global_tile_window,
        const AWindowStep & a_step,
        const BWindowStep & b_step,
        AccTile & acc_tile,
        int num_loops)
    {
        auto a_lds_block = MakeLdsCopyWindowA(p_smem);
        auto b_lds_block = MakeLdsCopyWindowB(p_smem);

        // FIXME
        auto a_lds_gemm_window = a_copy_lds_window;
        auto b_lds_gemm_window = b_copy_lds_window;

        auto a_block_tile = load_block_tile(a_global_tile_window);
        auto b_block_tile = load_block_tile(b_global_tile_window);

        // move to 1
        move_block_window(a_global_tile_window, a_step);
        move_block_window(b_global_tile_window, b_step);

        // Initialize C
        block_elementwise_inout([](auto& acc) { acc = 0; }, acc_tile);
        store_block_tile(a_copy_lds_window, a_block_tile);
        store_block_tile(b_copy_lds_window, b_block_tile);
        num_loop--;

        Policy::BlockGemm gemm;

        while(num_loop > 0){
            a_block_tile = load_block_tile(a_global_tile_window);
            block_sync_lds();
            b_block_tile = load_block_tile(b_global_tile_window);

            gemm(acc_tile, a_lds_gemm_window, b_lds_gemm_window);

            block_sync_lds();

            move_block_window(a_global_tile_window, a_step);
            move_block_window(b_global_tile_window, b_step);

            store_block_tile(a_copy_lds_window, a_block_tile);
            store_block_tile(b_copy_lds_window, b_block_tile);

            num_loop--;
        }
        // Tail
        {
            block_sync_lds();
            gemm(acc_tile, a_lds_gemm_window, b_lds_gemm_window);
        }
    }
};

}
