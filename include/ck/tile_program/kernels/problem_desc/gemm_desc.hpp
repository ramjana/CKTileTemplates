#pragma once

#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include <string>

namespace ck::tile_program {

// arch independent problem descriptor for gemm
template<
    typename AType_,
    typename BType_,
    typename CType_,
    typename AccType_,
    typename ALayout_,  // one of tensor_layout.hpp
    typename BLayout_,
    typename CLayout_,
    index_t  AlignmentA_,   // in unit of element
    index_t  AlignmentB_,
    index_t  AlignmentC_,
    typename BlockTileDesc_,
    bool = std::is_empty_v<BlockTileDesc_>>
struct GemmDesc;

// NOTE: we can also inherit GemmDesc from BlockTileDesc.
//       but aggregate initialization require empty {} if BlockTileDesc is empty.
//       so to avoid confusion if using {} to initialize this structure
//       here use template specialization depends on emptiness of BlockTileDesc

template<
    typename AType_,
    typename BType_,
    typename CType_,
    typename AccType_,
    typename ALayout_,
    typename BLayout_,
    typename CLayout_,
    index_t  AlignmentA_,
    index_t  AlignmentB_,
    index_t  AlignmentC_,
    typename BlockTileDesc_>
struct GemmDesc<
    AType_,
    BType_,
    CType_,
    AccType_,
    ALayout_,
    BLayout_,
    CLayout_,
    AlignmentA_,
    AlignmentB_,
    AlignmentC_,
    BlockTileDesc_
    true>
{
    using  AType      =   AType_;
    using  BType      =   BType_;
    using  CType      =   CType_;
    using  AccType    =   AccType_;
    using  ALayout    =   ALayout_;
    using  BLayout    =   BLayout_;
    using  CLayout    =   CLayout_;
    static constexpr index_t AlignmentA    = AlignmentA_;   // in unit of element
    static constexpr index_t AlignmentB    = AlignmentB_;
    static constexpr index_t AlignmentC    = AlignmentC_;
    using BlockTileDesc =  BlockTileDesc_;

    index_t m;
    index_t n;
    index_t k;
    index_t stride_a;   // in unit of element
    index_t stride_b;
    index_t stride_c;

    __host__ __device__
    static constexpr std::string Name()
    {
        return std::string(ALayout::simple_name) +
                std::string(BLayout::simple_name) +
                std::string(CLayout::simple_name) +
                std::string("_") +
            DataTypeToStr<AType>::get() + 
            DataTypeToStr<BType>::get() + 
            DataTypeToStr<CType>::get() + 
            DataTypeToStr<AccType>::get() + 
            std::string("_") +
            BlockTileDesc::Name();
    }
};

template<
    typename AType_,
    typename BType_,
    typename CType_,
    typename AccType_,
    typename ALayout_,
    typename BLayout_,
    typename CLayout_,
    index_t  AlignmentA_,
    index_t  AlignmentB_,
    index_t  AlignmentC_,
    typename BlockTileDesc_>
struct GemmDesc<
    AType_,
    BType_,
    CType_,
    AccType_,
    ALayout_,
    BLayout_,
    CLayout_,
    AlignmentA_,
    AlignmentB_,
    AlignmentC_,
    BlockTileDesc_
    false>
{
    using  AType      =   AType_;
    using  BType      =   BType_;
    using  CType      =   CType_;
    using  AccType    =   AccType_;
    using  ALayout    =   ALayout_;
    using  BLayout    =   BLayout_;
    using  CLayout    =   CLayout_;
    static constexpr index_t AlignmentA    = AlignmentA_;
    static constexpr index_t AlignmentB    = AlignmentB_;
    static constexpr index_t AlignmentC    = AlignmentC_;
    using BlockTileDesc =  BlockTileDesc_;

    index_t m;
    index_t n;
    index_t k;
    index_t stride_a;
    index_t stride_b;
    index_t stride_c;
    BlockTileDesc td;   // !!

    __host__ __device__
    static std::string Name()
    {
        return std::string(ALayout::simple_name) +
                std::string(BLayout::simple_name) +
                std::string(CLayout::simple_name) +
                std::string("_") +
            DataTypeToStr<AType>::get() + 
            DataTypeToStr<BType>::get() + 
            DataTypeToStr<CType>::get() + 
            DataTypeToStr<AccType>::get() + 
            std::string("_") +
            BlockTileDesc::Name();
    }
};

}
