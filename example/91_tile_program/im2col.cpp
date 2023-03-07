#include "tile_program.hpp"

#include "ck/utility/common_header.hpp"
#include "ck/utility/thread_group.hpp"

#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/cluster_descriptor.hpp"
#include "ck/tensor/tensor.hpp"

#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v4r1.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"

#include "ck/tensor_operation/operator_transform/transform_conv_fwd_to_gemm.hpp"
#include "ck/tensor_operation/gpu/device/convolution_forward_specialization.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"
#include "ck/host_utility/device_prop.hpp"

#include "ck/library/utility/device_memory.hpp"

namespace ck {

template <typename ThreadGroup,
          typename SrcElementwiseOperation,
          typename DstElementwiseOperation,
          InMemoryDataOperationEnum DstInMemOp,
          typename BlockSliceLengths,
          typename ThreadClusterLengths,
          typename ThreadClusterArrangeOrder,
          typename SrcTensor,
          typename DstTensor,
          typename SrcDimAccessOrder,
          typename DstDimAccessOrder,
          index_t SrcVectorDim,
          index_t DstVectorDim,
          index_t SrcScalarPerVector,
          index_t DstScalarPerVector,
          index_t SrcScalarStrideInVector,
          index_t DstScalarStrideInVector,
          bool ThreadTransferSrcResetCoordinateAfterRun,
          bool ThreadTransferDstResetCoordinateAfterRun>
struct Copier
{
    using SrcDesc = typename SrcTensor::TensorDescriptor;
    using DstDesc = typename DstTensor::TensorDescriptor;

    static constexpr ck::index_t nDim = remove_reference_t<SrcDesc>::GetNumOfDimension();

    using Index = MultiIndex<nDim>;

    __host__ __device__ constexpr Copier() : block_copy_{}, src_tensor_{}, dst_tensor_{} {}

    __device__ constexpr Copier(const SrcTensor& src_tensor,
                                const Index& src_block_slice_origin,
                                const SrcElementwiseOperation& src_element_op,
                                DstTensor& dst_tensor,
                                const Index& dst_block_slice_origin,
                                const DstElementwiseOperation& dst_element_op)
        : block_copy_{src_tensor.desc_,
                      src_block_slice_origin,
                      src_element_op,
                      dst_tensor.desc_,
                      dst_block_slice_origin,
                      dst_element_op},
          src_tensor_{src_tensor.buf_.p_data_, src_tensor.desc_},
          dst_tensor_{dst_tensor.buf_.p_data_, dst_tensor.desc_}
    {
    }

    __host__ void operator()() {}

    __device__ void operator()()
    {
        block_copy_.Run(
            src_tensor_.desc_, src_tensor_.buf_, dst_tensor_.desc_, dst_tensor_.buf_, Number<0>{});
    }

    __host__ void move_src_window(const Index&) {}

    __device__ void move_src_window(const Index& step)
    {
        block_copy_.MoveSrcSliceWindow(src_tensor_.desc_, step);
    }

    __host__ void move_dst_window(const Index&) {}

    __device__ void move_dst_window(const Index& step)
    {
        block_copy_.MoveDstSliceWindow(dst_tensor_.desc_, step);
    }

    // member
    ThreadGroupTensorSliceTransfer_v4r1<ThreadGroup,
                                        SrcElementwiseOperation,
                                        DstElementwiseOperation,
                                        DstInMemOp,
                                        BlockSliceLengths,
                                        ThreadClusterLengths,
                                        ThreadClusterArrangeOrder,
                                        typename SrcTensor::DataType,
                                        typename SrcTensor::DataType,
                                        SrcDesc,
                                        DstDesc,
                                        SrcDimAccessOrder,
                                        DstDimAccessOrder,
                                        SrcVectorDim,
                                        DstVectorDim,
                                        SrcScalarPerVector,
                                        DstScalarPerVector,
                                        SrcScalarStrideInVector,
                                        DstScalarStrideInVector,
                                        ThreadTransferSrcResetCoordinateAfterRun,
                                        ThreadTransferDstResetCoordinateAfterRun>
        block_copy_;

    SrcTensor src_tensor_;
    DstTensor dst_tensor_;
};

} // namespace ck

struct CopierStrategy
{
};

template <ck::index_t BlockSize>
struct MyProgramServer : public ProgramServer
{
    template <typename SrcTensor, typename DstTensor, typename Index, typename Strategy>
    __host__ auto make_copier(const SrcTensor& src_tensor,
                              const Index& src_window_origin,
                              DstTensor& dst_tensor,
                              const Index& dst_window_origin,
                              const Index& window_lengths,
                              const Strategy& strategy)
    {
        using namespace ck;

        return Copier<ThisThreadBlock<BlockSize>,
                      tensor_operation::element_wise::PassThrough,
                      tensor_operation::element_wise::PassThrough,
                      InMemoryDataOperationEnum::Set,
                      Sequence<128, 16>, // BlockSliceLengths,
                      Sequence<16, 16>,
                      Sequence<0, 1>,
                      SrcTensor,
                      DstTensor,
                      Sequence<0, 1>,
                      Sequence<0, 1>,
                      1,
                      1,
                      1,
                      1,
                      1,
                      1,
                      true,
                      true>{};
    }

    template <typename SrcTensor, typename DstTensor, typename Index, typename Strategy>
    __device__ auto make_copier(const SrcTensor& src_tensor,
                                const Index& src_window_origin,
                                DstTensor& dst_tensor,
                                const Index& dst_window_origin,
                                const Index& window_lengths,
                                const Strategy& strategy)
    {
        using namespace ck;

        return Copier<ThisThreadBlock<BlockSize>,
                      tensor_operation::element_wise::PassThrough,
                      tensor_operation::element_wise::PassThrough,
                      InMemoryDataOperationEnum::Set,
                      Sequence<128, 16>, // BlockSliceLengths,
                      Sequence<16, 16>,
                      Sequence<0, 1>,
                      SrcTensor,
                      DstTensor,
                      Sequence<0, 1>,
                      Sequence<0, 1>,
                      1,
                      1,
                      1,
                      1,
                      1,
                      1,
                      true,
                      true>{src_tensor,
                            src_window_origin,
                            tensor_operation::element_wise::PassThrough{},
                            dst_tensor,
                            dst_window_origin,
                            tensor_operation::element_wise::PassThrough{}};
    }
};

// program
template <ck::index_t NDimSpatial,
          typename ALayout,
          typename T,
          // tuning parameter
          ck::index_t kMPerTile,
          ck::index_t kKPerTile>
struct Im2Col
{
    template <typename Server, typename CopierStrategy>
    __host__ __device__ void
    operator()(Server& ps,
               const std::array<ck::index_t, NDimSpatial + 3>& a_g_n_c_wis_lengths,
               const std::array<ck::index_t, NDimSpatial + 3>& a_g_n_c_wis_strides,
               const std::array<ck::index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,
               const std::array<ck::index_t, NDimSpatial + 3>& b_g_k_c_xs_strides,
               const std::array<ck::index_t, NDimSpatial + 3>& c_g_n_k_wos_lengths,
               const std::array<ck::index_t, NDimSpatial + 3>& c_g_n_k_wos_strides,
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

        constexpr auto I0 = Number<0>{};

        const index_t N = a_g_n_c_wis_lengths[1];
        const index_t C = a_g_n_c_wis_lengths[2];

        const index_t Hi = a_g_n_c_wis_lengths[3];
        const index_t Wi = a_g_n_c_wis_lengths[4];

        const index_t Ho = c_g_n_k_wos_lengths[3];
        const index_t Wo = c_g_n_k_wos_lengths[4];

        const index_t ConvStrideH = conv_filter_strides[0];
        const index_t ConvStrideW = conv_filter_strides[1];

        const index_t Y = b_g_k_c_xs_lengths[3];
        const index_t X = b_g_k_c_xs_lengths[4];

        const index_t ConvDilationH = conv_filter_dilations[0];
        const index_t ConvDilationW = conv_filter_dilations[1];

        const index_t InLeftPadH = input_left_pads[0];
        const index_t InLeftPadW = input_left_pads[1];

        const index_t InRightPadH = input_right_pads[0];
        const index_t InRightPadW = input_right_pads[1];

        const auto a_n_hi_wi_c = make_naive_tensor_packed<AddressSpaceEnum::Global, true>(
            make_tuple(N, Hi, Wi, C), p_a_img);

        const auto a_n_hip_wip_c = transform_tensor(
            a_n_hi_wi_c,
            make_tuple(make_pass_through_transform(N),
                       make_pad_transform(Hi, InLeftPadH, InRightPadH),
                       make_pad_transform(Wi, InLeftPadW, InRightPadW),
                       make_pass_through_transform(C)),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

        const auto a_n_y_ho_x_wo_c = transform_tensor(
            a_n_hip_wip_c,
            make_tuple(
                make_pass_through_transform(N),
                make_embed_transform(make_tuple(Y, Ho), make_tuple(ConvDilationH, ConvStrideH)),
                make_embed_transform(make_tuple(X, Wo), make_tuple(ConvDilationW, ConvStrideW)),
                make_pass_through_transform(C)),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3, 4>{}, Sequence<5>{}));

        const auto src_gemmm_gemmk =
            transform_tensor(a_n_y_ho_x_wo_c,
                             make_tuple(ps(make_merge_transform(make_tuple(N, Ho, Wo))),
                                        ps(make_merge_transform(make_tuple(Y, X, C)))),
                             make_tuple(Sequence<0, 2, 4>{}, Sequence<1, 3, 5>{}),
                             make_tuple(Sequence<0>{}, Sequence<1>{}));

        auto dst = make_naive_tensor<AddressSpaceEnum::Global, true>(
            make_tuple(a_gemmm_gemmk_lengths[0], a_gemmm_gemmk_lengths[1]),
            make_tuple(a_gemmm_gemmk_strides[0], a_gemmm_gemmk_strides[1]),
            p_a_mtx);

        const auto num_gemmm = a_gemmm_gemmk_lengths[0];
        const auto num_gemmk = a_gemmm_gemmk_lengths[1];

        const auto id_block = ps.get_block_1d_id();

        const auto num_tile_m = ps.read_first_lane(num_gemmm / kMPerTile);

        const auto block2tile = ps(make_cluster_descriptor(make_tuple(num_tile_m)));

        const auto id_tile = block2tile.CalculateBottomIndex(make_tuple(id_block));

        const auto id_tile_m = ps.read_first_lane(id_tile[I0]);

#if 0
        auto window_src = make_window(src,
                                      make_tuple(kMPerTile, kKPerTile),
                                      make_tuple(id_tile_m * kMPerTile, 0),
                                      src_window_map_strategy);

        auto window_dst = make_window(dst,
                                      make_tuple(kMPerTile, kKPerTile),
                                      make_tuple(id_tile_m * kMPerTile, 0),
                                      dst_window_map_strategy);

        ck::index_t id_gemmk = 0;

        do
        {
            const auto src_vgpr_block = load(window_src, src_window_load_strategy);

            store(src_vgpr_block, window_dst, dst_window_store_strategy);

            move_window(window_src, make_tuple(0, kKPerTile));
            move_window(window_dst, make_tuple(0, kKPerTile));
        } while(id_gemmk < num_gemmk - kKPerTile);
#else
        auto copier = ps.make_copier(src,
                                     make_tuple(id_tile_m * kMPerTile, 0),
                                     dst,
                                     make_tuple(id_tile_m * kMPerTile, 0),
                                     make_tuple(kMPerTile, kKPerTile),
                                     copier_strategy);

        blockiwise_copy = ThreadGripTe<xxxx>;

        ck::index_t id_gemmk = 0;

        do
        {
            copier();

            copier.move_src_window(make_tuple(0, kKPerTile));
            copier.move_dst_window(make_tuple(0, kKPerTile));

            id_gemmk += kKPerTile;
        } while(id_gemmk < num_gemmk - kKPerTile);
#endif
    }
};

int main()
{
    using DataType = float;

    constexpr ck::index_t NumDimSpatial = 2;

    ck::index_t G  = 1;
    ck::index_t N  = 256;
    ck::index_t K  = 192;
    ck::index_t C  = 192;
    ck::index_t Y  = 3;
    ck::index_t X  = 3;
    ck::index_t Hi = 28;
    ck::index_t Wi = 28;
    ck::index_t Ho = 28;
    ck::index_t Wo = 28;

    std::array<ck::index_t, NumDimSpatial + 3> in_lengths{G, N, Hi, Wi, C};
    std::array<ck::index_t, NumDimSpatial + 3> in_strides{0, 0, 0, 0, 1};

    std::array<ck::index_t, NumDimSpatial + 3> wei_lengths{G, K, Y, X, C};
    std::array<ck::index_t, NumDimSpatial + 3> wei_strides{0, 0, 0, 0, 1};

    std::array<ck::index_t, NumDimSpatial + 3> out_lengths{G, N, Ho, Wo, K};
    std::array<ck::index_t, NumDimSpatial + 3> out_strides{0, 0, 0, 0, 1};

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

    // transpose GNHWC/GKYXC/GNHWK to GNCHW/GKCYX/GNCHW
    std::rotate(
        rbegin(in_lengths), std::next(rbegin(in_lengths)), std::next(rbegin(in_lengths), 3));
    std::rotate(
        rbegin(in_strides), std::next(rbegin(in_strides)), std::next(rbegin(in_strides), 3));
    std::rotate(
        rbegin(wei_lengths), std::next(rbegin(wei_lengths)), std::next(rbegin(wei_lengths), 3));
    std::rotate(
        rbegin(wei_strides), std::next(rbegin(wei_strides)), std::next(rbegin(wei_strides), 3));
    std::rotate(
        rbegin(out_lengths), std::next(rbegin(out_lengths)), std::next(rbegin(out_lengths), 3));
    std::rotate(
        rbegin(out_strides), std::next(rbegin(out_strides)), std::next(rbegin(out_strides), 3));

    std::array<ck::index_t, NumDimSpatial> filter_strides{1, 1};
    std::array<ck::index_t, NumDimSpatial> filter_dilations{1, 1};
    std::array<ck::index_t, NumDimSpatial> input_left_pads{1, 1};
    std::array<ck::index_t, NumDimSpatial> input_right_pads{1, 1};

    // matrix
    std::array<ck::index_t, 2> in_mtx_lengths{N * Ho * Wo, C * Y * X};
    std::array<ck::index_t, 2> in_mtx_strides{0, 1};

    std::partial_sum(rbegin(in_mtx_lengths),
                     std::prev(rend(in_mtx_lengths)),
                     std::next(rbegin(in_mtx_strides)),
                     std::multiplies<>{});

    DeviceMem in(sizeof(DataType) * G * N * Hi * Wi * C);
    DeviceMem in_mtx(sizeof(DataType) * G * N * Ho * Wo * C * Y * X);

    launch(MyProgramServer<256>{},
           Im2Col<2, ck::tensor_layout::convolution::GNHWC, float, 128, 16>{},
           1,
           1,
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
           static_cast<DataType*>(in.GetDeviceBuffer()),
           static_cast<DataType*>(in_mtx.GetDeviceBuffer()),
           CopierStrategy{});

    return 0;
}
