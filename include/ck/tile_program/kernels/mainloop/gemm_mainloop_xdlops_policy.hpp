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
         typename BlockGemm_>
struct GemmMainloopXdlopsPolicy {
    using ProblemDesc = ProblemDesc_;
    using BlockGemm = BlockGemm_;
    using BlockTileDesc = typename ProblemDesc::BlockTileDesc;

    using  AType      =   typename ProblemDesc::AType;
    using  BType      =   typename ProblemDesc::BType;

    using  ALayout    =   typename ProblemDesc::ALayout;
    using  BLayout    =   typename ProblemDesc::BLayout;

    static constexpr auto MPerBlock = BlockTileDesc::MPerBlock;
    static constexpr auto NPerBlock = BlockTileDesc::NPerBlock;
    static constexpr auto KPerBlock = BlockTileDesc::KPerBlock;

    static constexpr auto BlockSize = BlockTileDesc::BlockSize;

    static constexpr auto AlignmentA = ProblemDesc::AlignmentA;
    static constexpr auto AlignmentB = ProblemDesc::AlignmentB;


    using TileCoordType = MultiIndex<2>;    // TODO: dim?
    using AGlobalTileEnc =
            typename gemm_global_load_tile_encoding_dispatch_with_layout<ALayout, MPerBlock, KPerBlock, BlockSize, AlignmentA>::type;
    using BGlobalTileEnc =
            typename gemm_global_load_tile_encoding_dispatch_with_layout<BLayout, KPerBlock, NPerBlock, BlockSize, AlignmentB>::type;

    __host__ __device__
    static std::string Name()
    {
        return std::string("xdlops");   // TODO:
    }

    __host__ __device__ static constexpr auto MakeALdsBlockDescriptor()
    {
        using namespace ck;
#if 0
        constexpr auto a_lds_block_desc =
            make_naive_tensor_descriptor_packed(make_tuple(kMPerBlock, kKPerBlock), Number<32>{});

        return a_lds_block_desc;
#elif 0
        constexpr auto a_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(Number<kKPerBlock / 8>{}, Number<kMPerBlock>{}, Number<8>{}),
            make_tuple(Number<(kMPerBlock + 1) * 8>{}, Number<8>{}, Number<1>{}),
            Number<8>{},
            Number<1>{});

        constexpr auto a_lds_block_desc = transform_tensor_descriptor(
            a_lds_block_desc_0,
            make_tuple(make_pass_through_transform(kMPerBlock),
                       make_merge_transform(make_tuple(kKPerBlock / 8, 8))),
            make_tuple(Sequence<1>{}, Sequence<0, 2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return a_lds_block_desc;
#elif 1
        constexpr auto a_lds_block_desc_d1_d2_d3 = make_naive_tensor_descriptor_packed(
            make_tuple(kMPerBlock / 2, 2, kKPerBlock), Number<32>{});

        constexpr auto a_lds_block_desc_d4_d5_d6 = transform_tensor_descriptor(
            a_lds_block_desc_d1_d2_d3,
            make_tuple(make_xor_transform(make_tuple(kMPerBlock / 2, kKPerBlock), 8),
                       make_pass_through_transform(2)),
            make_tuple(Sequence<0, 2>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

        constexpr auto a_lds_block_desc_m_k = transform_tensor_descriptor(
            a_lds_block_desc_d4_d5_d6,
            make_tuple(make_merge_transform(make_tuple(kMPerBlock / 2, 2)),
                       make_pass_through_transform(kKPerBlock)),
            make_tuple(Sequence<0, 1>{}, Sequence<2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return a_lds_block_desc_m_k;
#endif
    }

    __host__ __device__ static constexpr auto MakeBLdsBlockDescriptor()
    {
        using namespace ck;
#if 0
        // 2D layout [N, K]
        constexpr auto b_lds_block_desc =
            make_naive_tensor_descriptor_packed(make_tuple(kNPerBlock, kKPerBlock), Number<32>{});

        return b_lds_block_desc;
#elif 0
        // [K0, M, K1] layout with padding
        constexpr auto b_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(Number<kKPerBlock / 8>{}, Number<kNPerBlock>{}, Number<8>{}),
            make_tuple(Number<(kNPerBlock + 1) * 8>{}, Number<8>{}, Number<1>{}),
            Number<8>{},
            Number<1>{});

        constexpr auto b_lds_block_desc = transform_tensor_descriptor(
            b_lds_block_desc_0,
            make_tuple(make_pass_through_transform(kNPerBlock),
                       make_merge_transform(make_tuple(kKPerBlock / 8, 8))),
            make_tuple(Sequence<1>{}, Sequence<0, 2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return b_lds_block_desc;
#elif 1
        // XOR layout
        constexpr auto b_lds_block_desc_d1_d2_d3 = make_naive_tensor_descriptor_packed(
            make_tuple(kNPerBlock / 2, 2, kKPerBlock), Number<32>{});

        constexpr auto b_lds_block_desc_d4_d5_d6 = transform_tensor_descriptor(
            b_lds_block_desc_d1_d2_d3,
            make_tuple(make_xor_transform(make_tuple(kNPerBlock / 2, kKPerBlock), 8),
                       make_pass_through_transform(2)),
            make_tuple(Sequence<0, 2>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

        constexpr auto b_lds_block_desc_n_k = transform_tensor_descriptor(
            b_lds_block_desc_d4_d5_d6,
            make_tuple(make_merge_transform(make_tuple(kNPerBlock / 2, 2)),
                       make_pass_through_transform(kKPerBlock)),
            make_tuple(Sequence<0, 1>{}, Sequence<2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return b_lds_block_desc_n_k;
#endif
    }

    __host__ __device__
    static constexpr index_t GetLdsSize()
    {
        return ck::math::integer_divide_ceil(
                   sizeof(AType) * MakeALdsBlockDescriptor().GetElementSpaceSize(), 16) *
                   16 +
               sizeof(BType) * MakeBLdsBlockDescriptor().GetElementSpaceSize();
    }

    __host__ __device__
    static constexpr auto MakeAGlobalTileDistribution()
    {
        return make_static_tile_distribution(AGlobalTileEnc{}){};
    }

    __host__ __device__
    static constexpr auto MakeBGlobalTileDistribution()
    {
        return make_static_tile_distribution(BGlobalTileEnc{}){};
    }

#if 0
    __device__
    static constexpr auto MakeGlobalCopyWindowA(const Arguments & args, const ProblemDesc & pd)
    {
        // TODO: RCR Layout
        const auto a_dram_grid = make_naive_tensor_view<AddressSpaceEnum::Global>(
            args.p_a, make_tuple(pd.m, pd.n), make_tuple(pd.stride_a, 1), Number<32>{}, Number<1>{});

        constexpr auto a_copy_dram_window_dstr = make_static_tile_distribution(AGlobalTileEnc{});

        auto a_copy_dram_window = make_block_window(a_dram_grid, tile_coord_a, a_copy_dram_window_dstr);
        return a_copy_dram_window;
    }

    __device__
    constexpr auto MakeGlobalCopyWindowB(const Arguments & args, const ProblemDesc & pd)
    {
        // TODO: RCR Layout        
        const auto b_dram_grid = make_naive_tensor_view<AddressSpaceEnum::Global>(
            args.p_b, make_tuple(pd.k, pd.n), make_tuple(1, pd.stride_b), Number<32>{}, Number<1>{});

        constexpr auto b_copy_dram_window_dstr = make_static_block_tensor_distribution(BGlobalTileEnc{});

        auto b_copy_dram_window = make_block_window(b_dram_grid, tile_coord_b, b_copy_dram_window_dstr);
        return b_copy_dram_window;
    }
#endif
};

}
