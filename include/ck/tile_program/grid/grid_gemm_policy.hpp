// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <type_traits>
#include <utility>

#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"

#include "ck/utility/multi_index.hpp"

namespace ck {
namespace detail {
template <typename Type, typename = void>
struct is_block2tile_map : std::bool_constant<false>
{
};

template <typename Type>
struct is_block2tile_map<Type,
                         std::void_t<decltype(std::declval<const std::remove_cvref_t<Type>&>()(
                             std::declval<index_t>()))>>
    : std::bool_constant<std::is_same_v<MultiIndex<2>,
                                        decltype(std::declval<const std::remove_cvref_t<Type>&>()(
                                            std::declval<index_t>()))>>
{
};

template <typename Type>
inline constexpr bool is_block2tile_map_v = is_block2tile_map<Type>::value;

template <typename Descriptor>
class DescToBlock2TileMapAdaptor
{
    static_assert(
        std::is_same_v<
            MultiIndex<2>,
            std::remove_cvref_t<decltype(std::declval<const Descriptor&>().CalculateBottomIndex(
                std::declval<MultiIndex<1>>()))>>);

    Descriptor descriptor_;

    public:
    explicit constexpr DescToBlock2TileMapAdaptor(Descriptor descriptor)
        : descriptor_(std::move(descriptor))
    {
    }

    __host__ __device__ MultiIndex<2> operator()(index_t block_id) const
    {
        return descriptor_.CalculateBottomIndex(make_multi_index(block_id));
    }
};

template <typename Descriptor>
__host__ __device__ static auto make_desc_to_block2tile_map_adaptor(Descriptor&& descriptor)
{
    return DescToBlock2TileMapAdaptor<remove_cvref_t<Descriptor>>{
        std::forward<Descriptor>(descriptor)};
}
} // namespace detail

namespace tile_program {
namespace grid {

struct GridGemmDefaultPolicy
{
    __host__ __device__ static constexpr auto MakeBlock2TileMap(index_t NumTilesM,
                                                                index_t NumTilesN)
    {
        return ck::detail::make_desc_to_block2tile_map_adaptor(
            make_cluster_descriptor(make_tuple(NumTilesM, NumTilesN)));
    }
};

template <index_t MaxRows = 8>
struct GridGemmMAdaptPolicy
{
    __host__ __device__ static constexpr auto MakeBlock2TileMap(index_t NumTilesM,
                                                                index_t NumTilesN)
    {
        return [=](index_t block_id) {
            index_t idx_N0 = block_id % NumTilesN;
            index_t idx_M0 = block_id / NumTilesN;

            const auto M01_adapt =
                (idx_M0 < NumTilesM - NumTilesM % MaxRows) ? MaxRows : NumTilesM % MaxRows;

            index_t idx_M00          = idx_M0 / MaxRows;
            index_t idx_M01          = idx_M0 % MaxRows;
            index_t idx_N0_M01_local = idx_N0 + idx_M01 * NumTilesN;

            return make_multi_index(idx_N0_M01_local % M01_adapt + idx_M00 * MaxRows,
                                    idx_N0_M01_local / M01_adapt);
        };
    }
};

template <index_t MaxCols = 8>
struct GridGemmNAdaptPolicy
{
    __host__ __device__ static constexpr auto MakeBlock2TileMap(index_t NumTilesM,
                                                                index_t NumTilesN)
    {
        return [=](index_t block_id) {
            const ck::index_t NumBlocksInSingleCompleteArea = NumTilesM * MaxCols;

            const ck::index_t MaxNumCompleteArea = NumTilesN / MaxCols;
            const ck::index_t MaxCompleteAreaBoundary =
                MaxNumCompleteArea * NumBlocksInSingleCompleteArea;

            const ck::index_t LastCols =
                (block_id < MaxCompleteAreaBoundary ? MaxCols : NumTilesN - MaxNumCompleteArea);
            const ck::index_t NumRemainedBlocks = block_id % NumBlocksInSingleCompleteArea;

            const ck::index_t idxM = NumRemainedBlocks / LastCols;
            const ck::index_t idxN =
                ((block_id - NumRemainedBlocks) / NumTilesM) + (NumRemainedBlocks % LastCols);

            return make_multi_index(idxM, idxN);
        };
    }
};

} // namespace grid
} // namespace tile_program
} // namespace ck
