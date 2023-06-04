#include <string_view>
#include <tuple>
#include <array>
#include <utility>
#include <type_traits>
#include <cstring>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/cluster_descriptor.hpp"
#include "ck/tensor/tensor_view.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"
#include "ck/host_utility/device_prop.hpp"

#include "tile_program.hpp"
#include "ck/tile_program/meta_data_buffer.hpp"
#include "ck/tile_program/block_tensor_distribution.hpp"
#include "ck/tile_program/block_tensor_window.hpp"
#include "ck/tile_program/static_block_distributed_tensor.hpp"
#include "ck/tile_program/load_block_distributed_tensor.hpp"
#include "ck/tile_program/store_block_distributed_tensor.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/fill.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"

template <typename T>
void reference_im2col(Tensor<T>& in_mtx_host_ref,
                      const Tensor<T>& in_host,
                      int /*N*/,
                      int /*K*/,
                      int C,
                      int /*Y*/,
                      int X,
                      int Hi,
                      int Wi,
                      int Ho,
                      int Wo,
                      int ConvStrideH,
                      int ConvStrideW,
                      int ConvDilationH,
                      int ConvDilationW,
                      int InLeftPadH,
                      int InLeftPadW,
                      int /*InRightPadH*/,
                      int /*InRightPadW*/)
{
    int GemmM = in_mtx_host_ref.GetLengths()[0];
    int GemmK = in_mtx_host_ref.GetLengths()[1];

    for(int gemm_m = 0; gemm_m < GemmM; ++gemm_m)
    {
        int mtmp = gemm_m;
        int n    = mtmp / (Ho * Wo);
        mtmp -= n * Ho * Wo;
        int ho = mtmp / Wo;
        int wo = mtmp - ho * Wo;

        for(int gemm_k = 0; gemm_k < GemmK; ++gemm_k)
        {
            int ktmp = gemm_k;
            int y    = ktmp / (X * C);
            ktmp -= y * X * C;
            int x = ktmp / C;
            int c = ktmp - x * C;

            int hi = y * ConvDilationH + ho * ConvStrideH - InLeftPadH;
            int wi = x * ConvDilationW + wo * ConvStrideW - InLeftPadW;

            bool inbound = (hi >= 0 && hi < Hi && wi >= 0 && wi < Wi);

            in_mtx_host_ref(gemm_m, gemm_k) = inbound ? in_host(n, hi, wi, c) : 0;
        }
    }
}

struct CopierStrategy
{
};

// program
template <ck::index_t NDimSpatial,
          typename T,
          // tuning parameter
          ck::index_t kMPerTile,
          ck::index_t kKPerTile>
struct Im2Col
{
    template <typename Server, typename CopierStrategy>
    __host__ __device__ void
    operator()(Server& ps,
               const std::array<ck::index_t, NDimSpatial + 2>& a_n_wis_c_lengths,
               const std::array<ck::index_t, NDimSpatial + 2>& /* a_n_wis_c_strides */,
               const std::array<ck::index_t, NDimSpatial + 2>& b_k_xs_c_lengths,
               const std::array<ck::index_t, NDimSpatial + 2>& /* b_k_xs_c_strides */,
               const std::array<ck::index_t, NDimSpatial + 2>& c_n_wos_k_lengths,
               const std::array<ck::index_t, NDimSpatial + 2>& /* c_n_wos_k_strides */,
               const std::array<ck::index_t, NDimSpatial>& conv_filter_strides,
               const std::array<ck::index_t, NDimSpatial>& conv_filter_dilations,
               const std::array<ck::index_t, NDimSpatial>& input_left_pads,
               const std::array<ck::index_t, NDimSpatial>& input_right_pads,
               //
               const std::array<ck::index_t, 2> a_gemmm_gemmk_lengths,
               const std::array<ck::index_t, 2> a_gemmm_gemmk_strides,
               //
               const T* p_a_img,
               T* p_a_mtx,
               // strategy
               const CopierStrategy& copier_strategy)
    {
        using namespace ck;

        const index_t N = a_n_wis_c_lengths[0];
        const index_t C = a_n_wis_c_lengths[3];

        const index_t Hi = a_n_wis_c_lengths[1];
        const index_t Wi = a_n_wis_c_lengths[2];

        const index_t Ho = c_n_wos_k_lengths[1];
        const index_t Wo = c_n_wos_k_lengths[2];

        const index_t Y = b_k_xs_c_lengths[1];
        const index_t X = b_k_xs_c_lengths[2];

        const index_t ConvStrideH = conv_filter_strides[0];
        const index_t ConvStrideW = conv_filter_strides[1];

        const index_t ConvDilationH = conv_filter_dilations[0];
        const index_t ConvDilationW = conv_filter_dilations[1];

        const index_t InLeftPadH = input_left_pads[0];
        const index_t InLeftPadW = input_left_pads[1];

        const index_t InRightPadH = input_right_pads[0];
        const index_t InRightPadW = input_right_pads[1];

        const auto a_img_buf = make_dynamic_buffer<AddressSpaceEnum::Global, const T, index_t>(
            p_a_img, N * Hi * Wi * C);

        const auto a_n_hi_wi_c = make_naive_tensor_view_packed(a_img_buf, make_tuple(N, Hi, Wi, C));

        const auto a_n_hip_wip_c = transform_tensor_view(
            a_n_hi_wi_c,
            make_tuple(make_pass_through_transform(N),
                       make_pad_transform(Hi, InLeftPadH, InRightPadH),
                       make_pad_transform(Wi, InLeftPadW, InRightPadW),
                       make_pass_through_transform(C)),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

        const auto a_n_y_ho_x_wo_c = transform_tensor_view(
            a_n_hip_wip_c,
            make_tuple(
                make_pass_through_transform(N),
                make_embed_transform(make_tuple(Y, Ho), make_tuple(ConvDilationH, ConvStrideH)),
                make_embed_transform(make_tuple(X, Wo), make_tuple(ConvDilationW, ConvStrideW)),
                make_pass_through_transform(C)),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3, 4>{}, Sequence<5>{}));

#if 1
        const auto src_gemmm_gemmk =
            transform_tensor_view(a_n_y_ho_x_wo_c,
                                  make_tuple(ps(make_merge_transform(make_tuple(N, Ho, Wo))),
                                             ps(make_merge_transform(make_tuple(Y, X, C)))),
                                  make_tuple(Sequence<0, 2, 4>{}, Sequence<1, 3, 5>{}),
                                  make_tuple(Sequence<0>{}, Sequence<1>{}));
#else
        const auto src_gemmm_gemmk =
            transform_tensor_view(a_n_y_ho_x_wo_c,
                                  make_tuple(make_merge_transform(make_tuple(N, Ho, Wo)),
                                             make_merge_transform(make_tuple(Y, X, C))),
                                  make_tuple(Sequence<0, 2, 4>{}, Sequence<1, 3, 5>{}),
                                  make_tuple(Sequence<0>{}, Sequence<1>{}));
#endif

        auto a_mtx_buf = make_dynamic_buffer<AddressSpaceEnum::Global, T, index_t>(
            p_a_mtx, N * Ho * Wo * Y * X * C);

        auto dst_gemmm_gemmk =
            make_naive_tensor_view(a_mtx_buf,
                                   make_tuple(a_gemmm_gemmk_lengths[0], a_gemmm_gemmk_lengths[1]),
                                   make_tuple(a_gemmm_gemmk_strides[0], a_gemmm_gemmk_strides[1]));

        const auto numGemmM = a_gemmm_gemmk_lengths[0];
        const auto numGemmK = a_gemmm_gemmk_lengths[1];

        const auto id_block = ps.get_block_1d_id();

        const auto num_tile_m = ps.read_first_lane(numGemmM / kMPerTile);

#if 1
        const auto block2tile = ps(make_cluster_descriptor(make_tuple(num_tile_m)));
#else
        const auto block2tile = make_cluster_descriptor(make_tuple(num_tile_m));
#endif

        const auto i_gemmm_gemmk = block2tile.CalculateBottomIndex(make_multi_index(id_block));

        const auto iGemmM = ps.read_first_lane(i_gemmm_gemmk[0]) * kMPerTile;

        (void)copier_strategy;

        // FIXME: use strategy to generate
        constexpr auto src_block_dstr =
            ck::tile_program::block::make_static_block_tensor_distribution(
                make_tuple(Sequence<2, 4, 16>{}, Sequence<4, 8>{}),
                Sequence<0>{},
                Sequence<1>{},
                Sequence<0, 1>{},
                Sequence<2, 0>{},
                Sequence<0, 1>{},
                Sequence<0, 1>{},
                Sequence<0, 1>{});

        // FIXME: make dst's block distribution different from src's
        constexpr auto dst_block_dstr = src_block_dstr;

        auto src_block_window = ck::tile_program::block::make_block_window(
            src_gemmm_gemmk, {iGemmM, 0}, src_block_dstr);

        auto dst_block_window = ck::tile_program::block::make_block_window(
            dst_gemmm_gemmk, {iGemmM, 0}, dst_block_dstr);

#if 1 // debug
        {
            // FIXME: set these override vector inforamtion correctly
            constexpr auto guaranteed_vector_alignments =
                to_array<index_t, 17>(typename uniform_sequence_gen<17, -1>::type{});

            auto guaranteed_vector_lengths =
                to_array<index_t, 17>(typename uniform_sequence_gen<17, -1>::type{});

            auto guaranteed_vector_strides =
                to_array<index_t, 17>(typename uniform_sequence_gen<17, -1>::type{});

            guaranteed_vector_lengths(0) = 32;
            guaranteed_vector_lengths(4) = 8;

            guaranteed_vector_strides(0) = 1;

            const auto [src_aligns, src_lengths, src_strides] =
                src_gemmm_gemmk.GetTensorDescriptor()
                    .GetHiddenDimensionSafeVectorAlignmentLengthStrides(
                        guaranteed_vector_alignments,
                        guaranteed_vector_lengths,
                        guaranteed_vector_strides);

            if(ps.get_block_1d_id() == 0 && ps.get_thread_local_1d_id() == 0)
            {
                printf("src\n");
                print_array("aligns ", src_aligns);
                print_array("lengths", src_lengths);
                print_array("strides", src_strides);
            }
        }
#endif

        index_t iGemmK = 0;

        do
        {
            const auto src_block_tile = ck::tile_program::block::load_block_tile(src_block_window);

            // FIXME: use shuffle API
            const auto dst_block_tile = src_block_tile;

            ck::tile_program::block::store_block_tile(dst_block_window, dst_block_tile);

            ck::tile_program::block::move_block_window(src_block_window, {0, kKPerTile});
            ck::tile_program::block::move_block_window(dst_block_window, {0, kKPerTile});

            iGemmK += kKPerTile;
        } while(iGemmK < numGemmK);
    }
};

int main()
{
    using DataType = ck::half_t;

    constexpr ck::index_t NumDimSpatial = 2;

    ck::index_t N  = 32;
    ck::index_t K  = 1;
    ck::index_t C  = 192;
    ck::index_t Y  = 3;
    ck::index_t X  = 3;
    ck::index_t Hi = 28;
    ck::index_t Wi = 28;
    ck::index_t Ho = 14;
    ck::index_t Wo = 14;

    std::array<ck::index_t, NumDimSpatial + 2> in_lengths{N, Hi, Wi, C};
    std::array<ck::index_t, NumDimSpatial + 2> in_strides{0, 0, 0, 1};

    std::array<ck::index_t, NumDimSpatial + 2> wei_lengths{K, Y, X, C};
    std::array<ck::index_t, NumDimSpatial + 2> wei_strides{0, 0, 0, 1};

    std::array<ck::index_t, NumDimSpatial + 2> out_lengths{N, Ho, Wo, K};
    std::array<ck::index_t, NumDimSpatial + 2> out_strides{0, 0, 0, 1};

    std::partial_sum(rbegin(in_lengths),
                     std::prev(rend(in_lengths)),
                     std::next(rbegin(in_strides)),
                     std::multiplies<>{});
    std::partial_sum(rbegin(wei_lengths),
                     std::prev(rend(wei_lengths)),
                     std::next(rbegin(wei_strides)),
                     std::multiplies<>{});
    std::partial_sum(rbegin(out_lengths),
                     std::prev(rend(out_lengths)),
                     std::next(rbegin(out_strides)),
                     std::multiplies<>{});

    std::array<ck::index_t, NumDimSpatial> filter_strides{2, 2};
    std::array<ck::index_t, NumDimSpatial> filter_dilations{1, 1};
    std::array<ck::index_t, NumDimSpatial> input_left_pads{1, 1};
    std::array<ck::index_t, NumDimSpatial> input_right_pads{1, 1};

    // matrix
    std::array<ck::index_t, 2> in_mtx_lengths{N * Ho * Wo, Y * X * C};
    std::array<ck::index_t, 2> in_mtx_strides{0, 1};

    std::partial_sum(rbegin(in_mtx_lengths),
                     std::prev(rend(in_mtx_lengths)),
                     std::next(rbegin(in_mtx_strides)),
                     std::multiplies<>{});

    // host verify
    Tensor<DataType> in_host(in_lengths, in_strides);
    Tensor<DataType> in_mtx_host_ref(in_mtx_lengths, in_mtx_strides);
    Tensor<DataType> in_mtx_host_dev(in_mtx_lengths, in_mtx_strides);

    std::cout << in_host.GetElementSpaceSize() << std::endl;
    std::cout << in_mtx_host_ref.GetElementSpaceSize() << std::endl;

    ck::utils::FillUniformDistributionIntegerValue<DataType>{-5.f, 5.f}(in_host);

    reference_im2col(in_mtx_host_ref,
                     in_host,
                     N,
                     K,
                     C,
                     Y,
                     X,
                     Hi,
                     Wi,
                     Ho,
                     Wo,
                     filter_strides[0],
                     filter_strides[1],
                     filter_dilations[0],
                     filter_dilations[1],
                     input_left_pads[0],
                     input_left_pads[1],
                     input_right_pads[0],
                     input_right_pads[1]);

    DeviceMem in_buf(sizeof(DataType) * in_host.GetElementSpaceSize());
    DeviceMem in_mtx_buf(sizeof(DataType) * in_mtx_host_ref.GetElementSpaceSize());

    std::cout << in_mtx_host_ref.GetElementSpaceSize() << std::endl;

    in_buf.ToDevice(in_host.mData.data());

    constexpr ck::index_t kGemmMPerBlock = 128;
    constexpr ck::index_t kGemmKPerBlock = 32;

    constexpr ck::index_t kBlockSize = 256;
    ck::index_t kGridSize            = (N * Ho * Wo) / kGemmMPerBlock;

    std::cout << "grid size " << kGridSize << std::endl;

    launch(ProgramServer{},
           Im2Col<2, DataType, kGemmMPerBlock, kGemmKPerBlock>{},
           kGridSize,
           kBlockSize,
           in_lengths,
           in_strides,
           wei_lengths,
           wei_strides,
           out_lengths,
           out_strides,
           filter_strides,
           filter_dilations,
           input_left_pads,
           input_right_pads,
           //
           in_mtx_lengths,
           in_mtx_strides,
           //
           static_cast<DataType*>(in_buf.GetDeviceBuffer()),
           static_cast<DataType*>(in_mtx_buf.GetDeviceBuffer()),
           CopierStrategy{});

    in_mtx_buf.FromDevice(in_mtx_host_dev.mData.data());

    return ck::utils::check_err(in_mtx_host_dev, in_mtx_host_ref);
}
