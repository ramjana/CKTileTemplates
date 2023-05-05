#include "ck/utility/common_header.hpp"

#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/cluster_descriptor.hpp"
#include "ck/tensor/tensor_view.hpp"

#include "ck/host_utility/device_prop.hpp"

#include "ck/library/utility/device_memory.hpp"

#include "tile_program.hpp"
#include "ck/tile_program/meta_data_buffer.hpp"
#include "ck/tile_program/block_tensor_distribution.hpp"

#if 0
__global__ void foo(int* p)
{
    using namespace ck;

    constexpr auto encoded_adaptor = []() {
        constexpr index_t kMaxNumTransforms = 10;
        constexpr index_t kMaxMetaDataSize  = 128;
        constexpr index_t kMaxNumDims       = 10;

        using Name     = IndexTransformEnum;
        using MetaData = MetaDataBuffer<kMaxMetaDataSize>;
        using NumDim   = index_t;
        using Ids      = Array<index_t, kMaxNumDims>;

        index_t N = 128;
        index_t C = 192;

        index_t Y             = 3;
        index_t X             = 3;
        index_t InLeftPadH    = 1;
        index_t InRightPadH   = 1;
        index_t InLeftPadW    = 1;
        index_t InRightPadW   = 1;
        index_t ConvStrideH   = 1;
        index_t ConvStrideW   = 1;
        index_t ConvDilationH = 1;
        index_t ConvDilationW = 1;

        index_t Hi = 17;
        index_t Wi = 17;
        index_t Ho = 17;
        index_t Wo = 17;

        auto trans = Array<Tuple<Name, MetaData, NumDim, Ids, NumDim, Ids>, kMaxNumTransforms>{};

        index_t num_tran = 0;

        trans[num_tran++] = {IndexTransformEnum::Pad,
                             MetaData{index_t{Hi}, index_t{InLeftPadH}, index_t{InRightPadH}},
                             NumDim{1},
                             Ids{2},
                             NumDim{1},
                             Ids{4}};

        trans[num_tran++] = {IndexTransformEnum::Pad,
                             MetaData{index_t{Wi}, index_t{InLeftPadW}, index_t{InRightPadW}},
                             NumDim{1},
                             Ids{3},
                             NumDim{1},
                             Ids{5}};

        trans[num_tran++] = {
            IndexTransformEnum::Embed,
            MetaData{Array<index_t, 2>{Y, Ho}, Array<index_t, 2>{ConvDilationH, ConvStrideH}},
            NumDim{1},
            Ids{4},
            NumDim{2},
            Ids{6, 7}};

        trans[num_tran++] = {
            IndexTransformEnum::Embed,
            MetaData{Array<index_t, 2>{X, Wo}, Array<index_t, 2>{ConvDilationW, ConvStrideW}},
            NumDim{1},
            Ids{5},
            NumDim{2},
            Ids{8, 9}};

        trans[num_tran++] = {IndexTransformEnum::Merge,
                             MetaData{Array<index_t, 3>{N, Ho, Wo}},
                             NumDim{3},
                             Ids{0, 7, 9},
                             NumDim{1},
                             Ids{10}};

        trans[num_tran++] = {IndexTransformEnum::Merge,
                             MetaData{Array<index_t, 3>{C, Y, X}},
                             NumDim{3},
                             Ids{1, 6, 8},
                             NumDim{1},
                             Ids{11}};

        index_t num_bottom_dim = 4;

        auto bottom_dim_ids = Ids{0, 1, 2, 3};

        index_t num_top_dim = 2;

        auto top_dim_ids = Ids{10, 11};

        return make_tuple(
            trans, num_tran, bottom_dim_ids, num_bottom_dim, top_dim_ids, num_top_dim);
    }();

    constexpr auto adaptor = CONSTRUCT_TENSOR_ADAPTOR_FROM_ENCODING(encoded_adaptor);

#if 0
    adaptor.foo();
#elif 0
    adaptor.Print();
#endif

    p[4] = adaptor.template GetTopDimensionLength<0>();
    p[5] = adaptor.template GetTopDimensionLength<1>();
    p[6] = adaptor.GetElementSize();
}
#else
__global__ void foo(int* /*p*/)
{
    using namespace ck;

    ck::tile_program::block::make_block_distribution(
        make_tuple(Sequence<2, 4, 16>{}, Sequence<4, 8>{}),
        Sequence<0>{},
        Sequence<1>{},
        Sequence<0, 1>{},
        Sequence<2, 0>{},
        Sequence<0, 1>{},
        Sequence<2, 1>{},
        Sequence<0, 1>{});
}
#endif

int main()
{
    DeviceMem res(1024);

    foo<<<dim3{1}, dim3{1}, 0, nullptr>>>(static_cast<int*>(res.GetDeviceBuffer()));
}
