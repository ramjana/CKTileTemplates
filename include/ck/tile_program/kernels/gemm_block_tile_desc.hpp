#pragma once

#include "tla/tla_internal.hpp"
#include "tla/container/sequence.hpp"
#include "gemm_tile_mapping.hpp"
#include <string>

namespace ck::tile_program {

// This is arch independent ?
template<typename BlockTileDesc_>
struct GemmTileMappingLinearMN
{
    using BlockTileDesc = BlockTileDesc_;
    using BlockTile = typename BlockTileDesc::BlockTile;
    static constexpr auto MPerBlock = BlockTile::template At<0>();
    static constexpr auto NPerBlock = BlockTile::template At<1>();

    __host__ __device__
    static constexpr auto GetTotalTiles(index_t m_, index_t n_, index_t /*k*/)
    {
        return ((m_ + MPerBlock - 1) / MPerBlock) *
                ((n_ + NPerBlock - 1) / NPerBlock);
    }

    __device__
    static constexpr auto TileToSpatial(index_t tile_idx_, index_t m_, index_t n_, index_t /*k*/)
    {
        // n first, then m
        // auto m_tiles = (m_ + MPerBlock - 1) / MPerBlock;
        auto n_tiles = (n_ + NPerBlock - 1) / NPerBlock;
        auto n_tile_idx = tile_idx_ % n_tiles;
        auto m_tile_idx = tile_idx_ / n_tiles;
        return make_tuple(m_tile_idx, n_tile_idx, 0);
    }

    // __host__ __device__
    // dim3 GetGridDims(index_t m_, index_t n_, index_t k_) const{
    //     return dim3(GetTotalTiles(m_, n_, k_), 1, 1);
    // }

    __host__ __device__
    static std::string Name()
    {
        return std::string("");
    }
};

template<typename BlockTileDesc_>
struct GemmTileMappingSwizzleMNSubm
{
    using BlockTileDesc = BlockTileDesc_;
    using BlockTile = typename BlockTileDesc::BlockTile;
    static constexpr auto MPerBlock = BlockTile::template At<0>();
    static constexpr auto NPerBlock = BlockTile::template At<1>();

    __host__ __device__
    static constexpr auto GetTotalTiles(index_t m_, index_t n_, index_t /*k*/)
    {
        return ((m_ + MPerBlock - 1) / MPerBlock) *
                ((n_ + NPerBlock - 1) / NPerBlock);
    }

    __device__
    template<index_t SUBM = 8>
    static constexpr auto TileToSpatial(index_t tile_idx_, index_t m_, index_t n_, index_t /*k*/)
    {
        static constexpr index_t tile_swizzle_sub_m = SUBM;
        auto n_tiles = (n_ + NPerBlock - 1) / NPerBlock;
        auto n_tile_idx = tile_idx_ % n_tiles;
        auto m_tile_idx = tile_idx_ / n_tiles;

        auto m_tiles = (m_ + MPerBlock - 1) / MPerBlock;

        uint32_t tile_swizzle_sub_m_rem = m_tiles % tile_swizzle_sub_m;

        const auto sub_m_adapt = (m_tile_idx < (m_tiles - tile_swizzle_sub_m_rem))
                                     ? tile_swizzle_sub_m
                                     : tile_swizzle_sub_m_rem;

        uint32_t m_tile_idx_sub0, m_tile_idx_sub1;
        m_tile_idx_sub0 = m_tile_idx / tile_swizzle_sub_m;
        m_tile_idx_sub1 = m_tile_idx % tile_swizzle_sub_m;

        uint32_t tile_idx_local = n_tile_idx + m_tile_idx_sub1 * n_tiles;

        uint32_t m_tile_idx_with_adapt, n_tile_idx_with_adapt;

        n_tile_idx_with_adapt = tile_idx_local / sub_m_adapt;
        m_tile_idx_with_adapt = tile_idx_local % sub_m_adapt;
        return make_tuple(m_tile_idx_with_adapt + m_tile_idx_sub0 * tile_swizzle_sub_m,
                          n_tile_idx_with_adapt, 0);
    }

    // __host__ __device__
    // dim3 GetGridDims(index_t m_, index_t n_, index_t k_) const{
    //     return dim3(GetTotalTiles(m_, n_, k_), 1, 1);
    // }

    __host__ __device__
    static std::string Name()
    {
        return std::string("_s");
    }
};

enum class GemmTileMappingEnum {
    LINEAR_M_N = 0,
    SWIZZLE_M_N_SUBM = 1,
};

// arch independent, but need tensor core
template<typename BlockTile_,  // global m/n/k per block
        typename MmaTile_,     // each mma compute size in M/N/K
        typename WaveTile_,    // waves responsible for M/N/K
        GemmTileMappingEnum MappingEnum_,
        index_t WaveSize = 64>
struct GemmBlockTileDesc;

#define _GEMM_BLOCK_TILE_DESC_TYPE(mapping_enum_) \
        GemmBlockTileDesc<BlockTile_,MmaTile_,WaveTile_,mapping_enum_,WaveSize>

// NOTE: better use aggregate initializatoin for this structure
//       and use "-Wno-missing-braces" to turn off warning for aggregate constructor for base class
#define _GEMM_BLOCK_TILE_DESC_DISPATCH(mapping_enum_, mapping_type_)                   \
    template<typename BlockTile_,                                                      \
            typename MmaTile_,                                                         \
            typename WaveTile_,                                                        \
            index_t WaveSize_ = 64>                                                    \
    struct _GEMM_BLOCK_TILE_DESC_TYPE(mapping_enum_) :                                 \
            mapping_type_<_GEMM_BLOCK_TILE_DESC_TYPE(mapping_enum_)> {                 \
        using Type = GemmBlockTileDesc;                                               \
        using BlockTile = BlockTile_;                                                 \
        using MmaTile = MmaTile_;                                                     \
        using WaveTile = WaveTile_;                                                   \
        using WaveRepeat = decltype(BlockTile{} / (WaveTile{} * MmaTile{}));          \
        static constexpr auto MPerBlock = BlockTile::template At<0>();              \
        static constexpr auto NPerBlock = BlockTile::template At<1>();              \
        static constexpr auto KPerBlock = BlockTile::template At<2>();              \
        static constexpr auto MmaM = MmaTile::template At<0>();                      \
        static constexpr auto MmaN = MmaTile::template At<1>();                      \
        static constexpr auto MmaK = MmaTile::template At<2>();                      \
        static constexpr auto WaveM = WaveTile::template At<0>();                    \
        static constexpr auto WaveN = WaveTile::template At<1>();                    \
        static constexpr auto WaveK = WaveTile::template At<2>();                    \
        static constexpr GemmTileMappingEnum MappingEnum = mapping_enum_;           \
        static constexpr index_t WaveSize = WaveSize_;                                \
        static constexpr index_t BlockSize = WaveSize * reduce_on_sequence(WaveTile{}, math::multiplies{}, Number<1>{});        \
        using MappingType = mapping_type_<_GEMM_BLOCK_TILE_DESC_TYPE(mapping_enum_)>;                 \
        __host__ __device__                                                                                 \
        static std::string Name() {                                                \
            return std::to_string(MPerBlock) + std::string("x") +                          \
                std::to_string(NPerBlock) + std::string("x") +                             \
                std::to_string(KPerBlock) + std::string("_") +                             \
                std::to_string(MmaM) + std::string("x") +                                   \
                std::to_string(MmaN) + std::string("x") +                                   \
                std::to_string(MmaK) + std::string("_") +                                   \
                std::to_string(WaveM) + std::string("x") +                                  \
                std::to_string(WaveN) + std::string("x") +                                  \
                std::to_string(WaveK) +                                                     \
                MappingType::Name();                                                        \
        }                                                                                    \
    }

_GEMM_BLOCK_TILE_DESC_DISPATCH(GemmTileMappingEnum::LINEAR_M_N, GemmTileMappingLinearMN);
_GEMM_BLOCK_TILE_DESC_DISPATCH(GemmTileMappingEnum::SWIZZLE_M_N_SUBM, GemmTileMappingSwizzleMNSubm);

#if 0
template<typename BlockTile_, typename WaveTile_>
using gemm_tile_desc_xdlops_fp16_32x32x8 = GemmBlockTileDesc<BlockTile_, Sequence<32, 32, 8>, WaveTile_>;    // using V_MFMA_F32_32x32x8F16

template<typename BlockTile_, typename WaveTile_>
using gemm_tile_desc_xdlops_fp16_16x16x16 = GemmBlockTileDesc<BlockTile_, Sequence<16, 16, 16>, WaveTile_>;    // using V_MFMA_F32_32x32x8F16

template<typename BlockTile_, typename WaveTile_>
using gemm_tile_desc_xdlops_fp32_32x32x2 = GemmBlockTileDesc<BlockTile_, Sequence<32, 32, 2>, WaveTile_>;    // using V_MFMA_F32_32x32x2F16

// use following pre-defined tiles
using gemm_tile_desc_xdlops_fp16_256x128x32 = gemm_tile_desc_xdlops_fp16_32x32x8<Sequence<256, 128, 32>, Sequence<2, 2, 1>>;
using gemm_tile_desc_xdlops_fp16_128x256x32 = gemm_tile_desc_xdlops_fp16_32x32x8<Sequence<128, 256, 32>, Sequence<2, 2, 1>>;
using gemm_tile_desc_xdlops_fp16_128x128x32 = gemm_tile_desc_xdlops_fp16_32x32x8<Sequence<128, 128, 32>, Sequence<2, 2, 1>>;
using gemm_tile_desc_xdlops_fp16_128x64x32  = gemm_tile_desc_xdlops_fp16_32x32x8<Sequence<128, 64, 32>, Sequence<2, 2, 1>>;
using gemm_tile_desc_xdlops_fp16_64x128x32  = gemm_tile_desc_xdlops_fp16_32x32x8<Sequence<64, 128, 32>, Sequence<2, 2, 1>>;

using gemm_tile_desc_xdlops_fp32_256x128x32 = gemm_tile_desc_xdlops_fp32_32x32x2<Sequence<256, 128, 32>, Sequence<2, 2, 1>>;
using gemm_tile_desc_xdlops_fp32_128x256x32 = gemm_tile_desc_xdlops_fp32_32x32x2<Sequence<128, 256, 32>, Sequence<2, 2, 1>>;
using gemm_tile_desc_xdlops_fp32_128x128x32 = gemm_tile_desc_xdlops_fp32_32x32x2<Sequence<128, 128, 32>, Sequence<2, 2, 1>>;
using gemm_tile_desc_xdlops_fp32_128x64x32  = gemm_tile_desc_xdlops_fp32_32x32x2<Sequence<128, 64, 32>, Sequence<2, 2, 1>>;
using gemm_tile_desc_xdlops_fp32_64x128x32  = gemm_tile_desc_xdlops_fp32_32x32x2<Sequence<64, 128, 32>, Sequence<2, 2, 1>>;
#endif


// TODO: auto gen these
template<GemmTileMappingEnum MappingEnum_ = GemmTileMappingEnum::SWIZZLE_M_N_SUBM>
using gemm_block_tile_desc_xdlops_fp16_256x128x32_32x32x8_2x2x1 = GemmBlockTileDesc<Sequence<256, 128, 32>, Sequence<32, 32, 8>, Sequence<2, 2, 1>, MappingEnum_>;


}
