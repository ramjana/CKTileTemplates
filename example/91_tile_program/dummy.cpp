#include "ck/utility/common_header.hpp"

#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/cluster_descriptor.hpp"
#include "ck/tensor/tensor_view.hpp"

#include "ck/host_utility/device_prop.hpp"

#include "ck/library/utility/device_memory.hpp"

#include "tile_program.hpp"
#include "ck/tile_program/meta_data_buffer.hpp"

#if 1
// convert constexpr array to sequence
#define TO_SEQUENCE(arr, n)                                                \
    [&arr, &n] {                                                           \
        static_assert(arr.Size() >= n, "wrong! out of bound");             \
                                                                           \
        static_assert(n < 6, "not implemented");                           \
                                                                           \
        if constexpr(n == 0)                                               \
        {                                                                  \
            return ck::Sequence<>();                                       \
        }                                                                  \
        else if constexpr(n == 1)                                          \
        {                                                                  \
            return ck::Sequence<arr[0]>();                                 \
        }                                                                  \
        else if constexpr(n == 2)                                          \
        {                                                                  \
            return ck::Sequence<arr[0], arr[1]>();                         \
        }                                                                  \
        else if constexpr(n == 3)                                          \
        {                                                                  \
            return ck::Sequence<arr[0], arr[1], arr[2]>();                 \
        }                                                                  \
        else if constexpr(n == 4)                                          \
        {                                                                  \
            return ck::Sequence<arr[0], arr[1], arr[2], arr[3]>();         \
        }                                                                  \
        else if constexpr(n == 5)                                          \
        {                                                                  \
            return ck::Sequence<arr[0], arr[1], arr[2], arr[3], arr[4]>(); \
        }                                                                  \
    }()
#endif

#if 0
#define CONVERT_ENCODED_TRANSFORMS_TO_TRANSFORMS(encoded_transforms, num_transform)               \
    [&encoded_transforms, &num_transform]() {                                                     \
        return generate_tuple(                                                                    \
            [&encoded_transforms](auto i) {                                                       \
                constexpr auto name        = encoded_transforms[i].template At<0>();              \
                constexpr auto meta_data   = encoded_transforms[i].template At<1>();              \
                constexpr auto num_low_dim = encoded_transforms[i].template At<2>();              \
                constexpr auto low_dims    = encoded_transforms[i].template At<3>();              \
                constexpr auto num_up_dim  = encoded_transforms[i].template At<4>();              \
                constexpr auto up_dims     = encoded_transforms[i].template At<5>();              \
                                                                                                  \
                constexpr auto low_dim_ids = TO_SEQUENCE(low_dims, num_low_dim);                  \
                constexpr auto up_dim_ids  = TO_SEQUENCE(up_dims, num_up_dim);                    \
                                                                                                  \
                constexpr auto tran = [&name, &meta_data]() {                                     \
                    if constexpr(name == IndexTransformEnum::PassThrough)                         \
                    {                                                                             \
                        index_t pos  = 0;                                                         \
                        auto low_len = meta_data.template Pop<index_t>(pos);                      \
                                                                                                  \
                        return make_pass_through_transform(low_len);                              \
                    }                                                                             \
                    else if constexpr(name == IndexTransformEnum::Pad)                            \
                    {                                                                             \
                        index_t pos    = 0;                                                       \
                        auto low_len   = meta_data.template Pop<index_t>(pos);                    \
                        auto left_pad  = meta_data.template Pop<index_t>(pos);                    \
                        auto right_pad = meta_data.template Pop<index_t>(pos);                    \
                                                                                                  \
                        return make_pad_transform(low_len, left_pad, right_pad);                  \
                    }                                                                             \
                    else if constexpr(name == IndexTransformEnum::Merge)                          \
                    {                                                                             \
                        index_t pos   = 0;                                                        \
                        auto low_lens = meta_data.template Pop<Array<index_t, num_low_dim>>(pos); \
                                                                                                  \
                        return make_merge_transform(low_lens);                                    \
                    }                                                                             \
                }();                                                                              \
                                                                                                  \
                return make_tuple(tran, low_dim_ids, up_dim_ids);                                 \
            },                                                                                    \
            Number<num_transform>{});                                                             \
    }()
#endif

enum IndexTransformEnum
{
    PassThrough,
    Pad,
    Embed,
    Merge,
    UnMerge,
    Undefined,
};

__global__ void foo(int* p)
{
    using namespace ck;

    // constexpr auto I0 = Number<0>{};
    // constexpr auto I1 = Number<1>{};
    // constexpr auto I2 = Number<2>{};
    // constexpr auto I3 = Number<3>{};

    constexpr auto encode = []() {
        constexpr index_t kMaxNumTransforms = 10;
        constexpr index_t kMaxMetaDataSize  = 128;
        constexpr index_t kMaxNumDims       = 10;

        using Name     = IndexTransformEnum;
        using MetaData = MetaDataBuffer<kMaxMetaDataSize>;
        using NumDim   = index_t;
        using Ids      = ck::Array<index_t, kMaxNumDims>;

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

        auto trans =
            ck::Array<Tuple<Name, MetaData, NumDim, Ids, NumDim, Ids>, kMaxNumTransforms>{};

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
            num_tran, trans, num_bottom_dim, bottom_dim_ids, num_top_dim, top_dim_ids);
    }();

    constexpr index_t num_transform    = encode.template At<0>();
    constexpr auto encoded_transforms  = encode.template At<1>();
    constexpr index_t num_bottom_dim   = encode.template At<2>();
    constexpr auto encoded_bottom_dims = encode.template At<3>();
    constexpr index_t num_top_dim      = encode.template At<4>();
    constexpr auto encoded_top_dims    = encode.template At<5>();

#if 0
    constexpr auto transforms =
        CONVERT_ENCODED_TRANSFORMS_TO_TRANSFORMS(encoded_transforms, num_transform);

    //      transforms.foo();

    if constexpr(encoded_transforms[0][I0] == IndexTransformEnum::PassThrough)
    {
        p[0] = transforms[I0][I0].GetUpperLengths()[I0];
    }

    if constexpr(encoded_transforms[1][I0] == IndexTransformEnum::Pad)
    {
        p[1] = transforms[I1][I0].GetUpperLengths()[I0];
    }

    if constexpr(encoded_transforms[2][I0] == IndexTransformEnum::Merge)
    {
        p[2] = transforms[I2][I0].GetUpperLengths()[I0];
    }

#endif

    constexpr auto adaptor = [&encoded_transforms,
                              &num_transform,
                              &encoded_bottom_dims,
                              &num_bottom_dim,
                              &encoded_top_dims,
                              &num_top_dim]() {
        constexpr auto trans = [&num_transform,
                                &encoded_transforms,
                                &encoded_bottom_dims,
                                &num_bottom_dim,
                                &encoded_top_dims,
                                &num_top_dim]() {
            return generate_tuple(
                [&encoded_transforms](auto i) constexpr {
                    constexpr auto name        = encoded_transforms[i].template At<0>();
                    constexpr auto meta_data   = encoded_transforms[i].template At<1>();
                    constexpr auto num_low_dim = encoded_transforms[i].template At<2>();
                    constexpr auto num_up_dim  = encoded_transforms[i].template At<4>();

                    if constexpr(name == IndexTransformEnum::PassThrough)
                    {
                        index_t pos  = 0;
                        auto low_len = meta_data.template Pop<index_t>(pos);

                        return make_pass_through_transform(low_len);
                    }
                    else if constexpr(name == IndexTransformEnum::Pad)
                    {
                        index_t pos    = 0;
                        auto low_len   = meta_data.template Pop<index_t>(pos);
                        auto left_pad  = meta_data.template Pop<index_t>(pos);
                        auto right_pad = meta_data.template Pop<index_t>(pos);

                        return make_pad_transform(low_len, left_pad, right_pad);
                    }
                    else if constexpr(name == IndexTransformEnum::Embed)
                    {
                        index_t pos       = 0;
                        auto up_lens      = meta_data.template Pop<Array<index_t, num_up_dim>>(pos);
                        auto coefficients = meta_data.template Pop<Array<index_t, num_up_dim>>(pos);

                        return make_embed_transform(up_lens, coefficients);
                    }
                    else if constexpr(name == IndexTransformEnum::Merge)
                    {
                        index_t pos   = 0;
                        auto low_lens = meta_data.template Pop<Array<index_t, num_low_dim>>(pos);

                        return make_merge_transform(low_lens);
                    }
                },
                Number<num_transform>{});
        }();

        constexpr auto low_dim_idss = [&num_transform, &encoded_transforms]() {
            return generate_tuple(
                [&encoded_transforms](auto i) {
                    constexpr auto num_low_dim = encoded_transforms[i].template At<2>();
                    constexpr auto low_dims    = encoded_transforms[i].template At<3>();

                    return TO_SEQUENCE(low_dims, num_low_dim);
                },
                Number<num_transform>());
        }();

        constexpr auto up_dim_idss = [&num_transform, &encoded_transforms]() {
            return generate_tuple(
                [&encoded_transforms](auto i) {
                    constexpr auto num_up_dim = encoded_transforms[i].template At<4>();
                    constexpr auto up_dims    = encoded_transforms[i].template At<5>();

                    return TO_SEQUENCE(up_dims, num_up_dim);
                },
                Number<num_transform>());
        }();

        constexpr auto bottom_dim_ids = TO_SEQUENCE(encoded_bottom_dims, num_bottom_dim);
        constexpr auto top_dim_ids    = TO_SEQUENCE(encoded_top_dims, num_top_dim);

        return TensorAdaptor<remove_cvref_t<decltype(trans)>,
                             remove_cvref_t<decltype(low_dim_idss)>,
                             remove_cvref_t<decltype(up_dim_idss)>,
                             remove_cvref_t<decltype(bottom_dim_ids)>,
                             remove_cvref_t<decltype(top_dim_ids)>>{trans};
    }();

#if 0
    adaptor.foo();
#elif 0
    adaptor.Print();
#endif

    p[4] = adaptor.template GetTopDimensionLength<0>();
    p[5] = adaptor.template GetTopDimensionLength<1>();
    p[6] = adaptor.GetElementSize();
}

int main()
{
    DeviceMem res(1024);

    foo<<<dim3{1}, dim3{1}, 0, nullptr>>>(static_cast<int*>(res.GetDeviceBuffer()));
}
