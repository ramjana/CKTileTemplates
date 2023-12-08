// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace ck {
namespace tile_program {
namespace block {

struct MaskDisabledPredicate
{
    __host__ __device__ constexpr bool operator()(index_t /*m*/, index_t /*n*/) const
    {
        return false;
    };

    __host__ __device__ constexpr bool
    IsTileSkippable(index_t /*m*/, index_t /*n*/, index_t /*m_tile*/, index_t /*n_tile*/) const
    {
        return false;
    }
};

struct MaskUpperTriangleFromTopLeftPredicate
{
    __host__ __device__ constexpr bool operator()(index_t m, index_t n) const { return n > m; }

    __host__ __device__ constexpr bool
    IsTileSkippable(index_t m, index_t n, index_t m_tile, index_t /*n_tile*/) const
    {
        return operator()(m + m_tile - 1, n);
    }
};

// eg: m = 3, n = 5 => offset = 2
//    so matrix(n > m + offset) = 0
//      1  2  3  4  5
//    1 *  *  *  0  0
//    2 *  *  *  *  0
//    3 *  *  *  *  *
struct MaskUpperTriangleFromBottomRightPredicate
{
    __host__ __device__ void SetDiagonalOffset(const index_t diagonal_offset)
    {
        diagonal_offset_ = diagonal_offset;
    }
    __host__ __device__ constexpr bool operator()(index_t m, index_t n) const
    {
        return n > (m - diagonal_offset_);
    }

    __host__ __device__ constexpr bool IsTileSkippable(index_t m_tile_orig,
                                                       index_t n_tile_orig,
                                                       index_t m_tile_size,
                                                       index_t /*n_tile_size*/) const
    {
        return operator()(m_tile_orig + m_tile_size - 1, n_tile_orig);
    }

    private:
    index_t diagonal_offset_;
};

struct MaskLocalAttentionPredicate
{
    __host__ __device__ void SetParameters(const index_t diagonal_offset,
                                           const index_t left_window_size,
                                           const index_t right_window_size)
    {
        diagonal_offset_   = diagonal_offset;
        left_window_size_  = left_window_size;
        right_window_size_ = right_window_size;
    }
    __host__ __device__ constexpr bool operator()(index_t m, index_t n) const
    {
        if(left_window_size_ < 0)
        {
            return n > (m - diagonal_offset_) + right_window_size_;
        }
        else
        {
            return (n > (m - diagonal_offset_) + right_window_size_) ||
                   (n < (m - diagonal_offset_) - left_window_size_);
        }
    }

    __host__ __device__ constexpr bool IsTileSkippable(index_t m_tile_orig,
                                                       index_t n_tile_orig,
                                                       index_t m_tile_size,
                                                       index_t n_tile_size) const
    {
        // block_bottom_left_point: (m_tile_orig + m_tile_size - 1, n_tile_orig)
        // block_top_right_point:   (m_tile_orig, n_tile_orig + n_tile_size - 1)
        if(left_window_size_ < 0)
        {
            return operator()(m_tile_orig + m_tile_size - 1, n_tile_orig);
        }
        else
        {
            return (n_tile_orig >
                    ((m_tile_orig + m_tile_size - 1) - diagonal_offset_ + right_window_size_)) ||
                   ((n_tile_orig + n_tile_size - 1) <
                    (m_tile_orig - diagonal_offset_ - left_window_size_));
        }
    }

    private:
    index_t diagonal_offset_;
    index_t left_window_size_;
    index_t right_window_size_;
};

// to track the points which need to be set to -inf on C0
// Note: no need to reset M padding value, because they will not be stored out.
template <typename MaskOutPredicate_>
struct C0MatrixMask_impl
{
    using MaskOutPredicate = MaskOutPredicate_;

    __host__ __device__
    C0MatrixMask_impl(index_t MRaw, index_t NRaw, index_t LWSize = -1, index_t RWSize = -1)
        : NRaw_(NRaw), predicate_(MaskOutPredicate{})
    {
        if constexpr(std::is_same_v<MaskOutPredicate, MaskUpperTriangleFromBottomRightPredicate>)
        {
            predicate_.SetDiagonalOffset(MRaw - NRaw);
        }
        if constexpr(std::is_same_v<MaskOutPredicate, MaskLocalAttentionPredicate>)
        {
            predicate_.SetParameters(MRaw - NRaw, LWSize, RWSize);
        }
    }

    __host__ __device__ constexpr bool IsNOutOfBound(/*index_t m, */ index_t n) const
    {
        return n >= NRaw_;
    }

    __host__ __device__ constexpr bool IsMaskedElement(index_t m, index_t n) const
    {
        return predicate_(m, n) || IsNOutOfBound(n);
    }

    __host__ __device__ constexpr bool
    IsTileSkippable(index_t m, index_t n, index_t m_tile, index_t n_tile) const
    {
        return predicate_.IsTileSkippable(m, n, m_tile, n_tile);
    }

    private:
    // index_t MRaw_;
    index_t NRaw_;
    MaskOutPredicate predicate_;
};

} // namespace block
} // namespace tile_program
} // namespace ck
