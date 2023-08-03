
#pragma once

#include <string>

namespace ck::tile_program {

template<typename Mainloop_,
        typename Epilogue_,
        typename ProblemDesc_>
struct GemmKernel {
    using Mainloop = Mainloop_;
    using Epilogue = Epilogue_;
    using ProblemDesc = ProblemDesc_;
    using BlockTileDesc = typename ProblemDesc::BlockTileDesc;

    using  ALayout    =   typename ProblemDesc::ALayout;
    using  BLayout    =   typename ProblemDesc::BLayout;
    using  CLayout    =   typename ProblemDesc::CLayout;

    struct Arguments {
        typename Mainloop::Arguments    mainloop_args;
        typename Epilogue::Arguments    epilogue_args;
        ProblemDesc                     pd;
    };

    const Arguments & args;
    char * p_smem;

    __host__ __device__
    GemmKernel(const Arguments & args_, char * p_smem_)
        : args(args_), p_smem(p_smem_)
    {}

    __host__ __device__
    constexpr auto GetBlockTileDesc() const
    {
        // we may consider use the instance passed in from args
        // or instantiate one here (if the type of tile_desc is empty)
        if constexpr (std::is_empty_v<BlockTileDesc>) {
            return BlockTileDesc{};
        }
        else {
            return args.pd.td;
        }
    }

    __host__ __device__
    static std::string Name()
    {
        return std::string("gemm_") +
            ProblemDesc::Name();
    }

    __host__ __device__
    constexpr auto GetALengths() const {
        return make_tuple(args.pd.m, args.pd.k);
    }
    __host__ __device__
    constexpr auto GetAStrides() const {
        if constexpr(ALayout == ck::tensor_layout::gemm::RowMajor)
            return make_tuple(args.pd.stride_a, 1);
        else
            return make_tuple(1, args.pd.stride_a);
    }

    __host__ __device__
    constexpr auto GetBLengths() const {
        return make_tuple(args.pd.k, args.pd.n);
    }
    __host__ __device__
    constexpr auto GetBStrides() const {
        if constexpr(ALayout == ck::tensor_layout::gemm::RowMajor)
            return make_tuple(1, args.pd.stride_b);
        else
            return make_tuple(args.pd.stride_b, 1);
    }

    __device__
    constexpr void operator()()
    {
        const auto & pd = args.pd;
        auto m = pd.m;
        auto n = pd.n;
        auto k = pd.k;
        const auto & td  = GetBlockTileDesc();

        auto tile_idices = td.TileToSpatial(get_block_id(), m, n, k);

        auto a_tile_coord = MultiIndex<2>(tile_idices[Number<0>{}], tile_idices[Number<2>{}]); // m_k
        auto b_tile_coord = MultiIndex<2>(tile_idices[Number<2>{}], tile_idices[Number<1>{}]); // k_n
        auto c_tile_coord = MultiIndex<2>(tile_idices[Number<0>{}], tile_idices[Number<1>{}]); // m_n

        // instantiate mainloop
        Mainloop mop(args.mainloop_args, p_smem);

        auto a_tile = mop.MakeAGlobalTileWindow(GetALengths(), GetAStrides(), a_tile_coord);
        auto b_tile = mop.MakeBGlobalTileWindow(GetBLengths(), GetCStrides(), b_tile_coord);
        auto acc_tile = mop.MakeCTile();

        index_t num_loops = (k + KPerBlock - 1 / KPerBlock);

        // mainloop working on a/b tile, write to acc_tile
        mop(a_tile,         // a_tile read from
            {0, KPerBlock}, // a_step while moving tile window
            b_tile,         // b_tile read from 
            {KPerBlock, 0}, // b_step while moving tile window
            acc_tile,       // acc_tile write to
            num_loops);     // num loops

        // instantiate epilogue
        Epilogue epi(args.epilogue_args, p_smem);
        auto c_tile = epi.MakeCGlobalTileWindow(GetCLengths(), GetCStrides(), c_tile_coord);

        // epilogue read acc_tile and write to global c_tile
        epi(acc_tile, c_tile);
    }
};

}
