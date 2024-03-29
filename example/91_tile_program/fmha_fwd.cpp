#include <cstring>
#include <ostream>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/cluster_descriptor.hpp"
#include "ck/tensor/tensor_view.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/fill.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"

#include "ck/tile_program/block_tile_pipeline/block_fmha_pipeline_qkvs.hpp"
#include "ck/tile_program/block_tile_pipeline/block_fmha_pipeline_qkvs_default_policy.hpp"
#include "ck/tile_program/block_tile_pipeline/block_fmha_pipeline_qr_ks_vs.hpp"
#include "ck/tile_program/block_tile_pipeline/block_fmha_pipeline_qr_ks_vs_default_policy.hpp"
#include "ck/tile_program/block_tile_pipeline/block_fmha_pipeline_problem.hpp"
#include "ck/tile_program/tile/tile_fmha_shape.hpp"

#include "reference_batched_gemm.hpp"
#include "reference_batched_softmax.hpp"
#include "fmha_fwd_kernel.hpp"
#include "fmha_fwd_tile_partitioner.hpp"
#include "fmha_fwd_epilogue.hpp"

#if 1
using QDataType           = ck::half_t;
using KDataType           = ck::half_t;
using VDataType           = ck::half_t;
using SaccDataType        = float;      // data type for first gemm accumulation
using SMPLComputeDataType = float;      // data type for reduction, softmax
using PDataType           = ck::half_t; // data type for A matrix of second gemm
using OaccDataType        = float;      // data type for second gemm accumulation
using ODataType           = ck::half_t;
#else
using QDataType           = ck::bhalf_t;
using KDataType           = ck::bhalf_t;
using VDataType           = ck::bhalf_t;
using SaccDataType        = float;       // data type for first gemm accumulation
using SMPLComputeDataType = float;       // data type for reduction, softmax
using PDataType           = ck::bhalf_t; // data type for A matrix of second gemm
using OaccDataType        = float;       // data type for second gemm accumulation
using ODataType           = ck::bhalf_t;
#endif

//                                                 M0   N0  K0   N1  K1  K0L
// using FmhaShape = ck::tile_program::TileFmhaShape<128,  64, 64, 128, 64>;
// using FmhaShape = ck::tile_program::TileFmhaShape<128, 256, 32, 128, 32>;
using VLayout = ck::tensor_layout::gemm::RowMajor; // (bs, nhead) seqlen * hdim
// using VLayout = ck::tensor_layout::gemm::ColumnMajor; // (bs, nhead) hdim * seqlen

using FmhaBlockTileHdim64  = ck::Sequence<128, 64, 32, 64, 32, 64>;
using FmhaBlockTileHdim128 = ck::Sequence<128, 128, 32, 128, 32, 128>;
using FmhaBlockWarps       = ck::Sequence<4, 1, 1>;
using FmhaWarpTile         = ck::Sequence<32, 32, 16>;
using FmhaShapeHDim64      = ck::tile_program::TileFmhaShape<FmhaBlockTileHdim64,
                                                        FmhaBlockWarps,
                                                        FmhaWarpTile,
                                                        FmhaBlockWarps,
                                                        FmhaWarpTile,
                                                        VLayout>;
using FmhaShapeHDim128     = ck::tile_program::TileFmhaShape<FmhaBlockTileHdim128,
                                                         FmhaBlockWarps,
                                                         FmhaWarpTile,
                                                         FmhaBlockWarps,
                                                         FmhaWarpTile,
                                                         VLayout>;

using FmhaTilePartitionerHDim64  = FmhaFwdTilePartitioner<FmhaShapeHDim64>;
using FmhaTilePartitionerHDim128 = FmhaFwdTilePartitioner<FmhaShapeHDim128>;
using FmhaPipelineProblemHDim64 =
    ck::tile_program::block::BlockFmhaPipelineProblem<QDataType,
                                                      KDataType,
                                                      VDataType,
                                                      SaccDataType,
                                                      SMPLComputeDataType,
                                                      PDataType,
                                                      OaccDataType,
                                                      ODataType,
                                                      256, // BlockSize
                                                      FmhaShapeHDim64>;
using FmhaPipelineProblemHDim128 =
    ck::tile_program::block::BlockFmhaPipelineProblem<QDataType,
                                                      KDataType,
                                                      VDataType,
                                                      SaccDataType,
                                                      SMPLComputeDataType,
                                                      PDataType,
                                                      OaccDataType,
                                                      ODataType,
                                                      256, // BlockSize
                                                      FmhaShapeHDim128>;
// using FmhaPipeline        = ck::tile_program::block::BlockFmhaPipelineQKVS<FmhaPipelineProblem>;
using FmhaPipelineHDim64 =
    ck::tile_program::block::BlockFmhaPipelineQRKSVS<FmhaPipelineProblemHDim64>;
using FmhaPipelineHDim128 =
    ck::tile_program::block::BlockFmhaPipelineQRKSVS<FmhaPipelineProblemHDim128>;

using FmhaEpilogue     = FmhaFwdEpilogue<FmhaFwdEpilogueProblem<OaccDataType, ODataType>>;
using FmhaKernelHDim64 = FmhaFwdKernel<FmhaTilePartitionerHDim64, FmhaPipelineHDim64, FmhaEpilogue>;
using FmhaKernelHDim128 =
    FmhaFwdKernel<FmhaTilePartitionerHDim128, FmhaPipelineHDim128, FmhaEpilogue>;

template <typename FmhaKernel>
float invoker_fmha_kernel(const void* q_ptr,
                          const void* k_ptr,
                          const void* v_ptr,
                          void* o_ptr,
                          ck::index_t batch,
                          ck::index_t nhead,
                          ck::index_t seqlen_q,
                          ck::index_t seqlen_k,
                          ck::index_t hdim_q,
                          ck::index_t hdim_v,
                          float scale,
                          bool i_perm,
                          bool o_perm)
{
    dim3 kGridSize            = FmhaKernel::GridSize(batch, nhead, seqlen_q, hdim_v);
    constexpr dim3 kBlockSize = FmhaKernel::BlockSize();

    constexpr ck::index_t kWarpPerCu    = 8; // 2 warps per SIMD
    constexpr ck::index_t kWarpPerBlock = kBlockSize.x / warpSize;
    constexpr ck::index_t kBlockPerCu   = kWarpPerCu / kWarpPerBlock;

    constexpr bool is_v_rowmajor =
        ck::is_same_v<typename FmhaKernel::VLayout, ck::tensor_layout::gemm::RowMajor>;

    // batch * nhead * seqlen * hdim or batch * seqlen * nhead * hdim
    auto kargs = FmhaKernel::MakeKargs(
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        seqlen_q, // seqlen_q
        seqlen_k, // seqlen_k
        hdim_q,   // hdim_q
        hdim_v,   // hdim_v
        scale,
        i_perm ? hdim_q : nhead * hdim_q, // stride_q
        i_perm ? hdim_q : nhead * hdim_q, // stride_k
        [&]() {
            if constexpr(is_v_rowmajor)
                return i_perm ? hdim_v : nhead * hdim_v;
            else
                return i_perm ? seqlen_k : nhead * seqlen_k;
        }(),                                 // stride_v
        o_perm ? hdim_v : nhead * hdim_v,    // stride_o
        i_perm ? seqlen_q * hdim_q : hdim_q, // nhead_stride_q
        i_perm ? seqlen_k * hdim_q : hdim_q, // nhead_stride_k
        [&]() {
            if constexpr(is_v_rowmajor)
                return i_perm ? seqlen_k * hdim_v : hdim_v;
            else
                return i_perm ? hdim_v * seqlen_k : seqlen_k;
        }(),                                 // nhead_stride_v
        o_perm ? seqlen_q * hdim_v : hdim_v, // nhead_stride_o
        nhead * seqlen_q * hdim_q,           // batch_stride_q
        nhead * seqlen_k * hdim_q,           // batch_stride_k
        nhead * hdim_v * seqlen_k,           // batch_stride_v
        nhead * seqlen_q * hdim_v);          // batch_stride_o

    float ave_time = launch_kernel<kBlockSize.x, kBlockPerCu>(StreamConfig{nullptr, true},
                                                              FmhaKernel{},
                                                              kGridSize,
                                                              kBlockSize,
                                                              0,
                                                              kargs); // BatchStrideO
    return ave_time;
}

int main(int argc, char* argv[])
{
    int do_validation    = 1;
    ck::index_t batch    = 2;
    ck::index_t nhead    = 8;
    ck::index_t seqlen_q = 3328;
    ck::index_t seqlen_k = 4096;
    ck::index_t hdim_q   = 128;
    ck::index_t hdim_v   = 128;

    float scale = .0f;

    bool i_perm = true; // if true, will be batch * nhead * seqlen * hdim
    bool o_perm = true; // if false, will be batch * seqlen * nhead * hdim

    if(argc >= 2)
        do_validation = std::stoi(argv[1]);

    if(argc >= 8)
    {
        batch    = std::stoi(argv[2]);
        nhead    = std::stoi(argv[3]);
        seqlen_q = std::stoi(argv[4]);
        seqlen_k = std::stoi(argv[5]);
        hdim_q   = std::stoi(argv[6]);
        hdim_v   = std::stoi(argv[7]);
    }
    if(argc >= 9)
        scale = std::stof(argv[8]);
    if(argc >= 10)
        i_perm = static_cast<bool>(std::stoi(argv[9]));
    if(argc >= 11)
        o_perm = static_cast<bool>(std::stoi(argv[10]));

    if(scale == .0f)
        scale = 1.0 / ck::math::sqrt(static_cast<float>(hdim_q)); // TODO: q ? v ?

    auto get_lengths = [&](bool permute,
                           ck::index_t b /*batch*/,
                           ck::index_t h /*nhead*/,
                           ck::index_t s /*seqlen*/,
                           ck::index_t d /*hdim*/) {
        if(permute)
            return std::array<ck::index_t, 4>{b, h, s, d};
        else
            return std::array<ck::index_t, 4>{b, s, h, d};
    };

    constexpr bool is_v_rowmajor =
        ck::is_same_v<typename FmhaKernelHDim64::VLayout, ck::tensor_layout::gemm::RowMajor>;

    // host verify
    Tensor<QDataType> q_host(get_lengths(i_perm, batch, nhead, seqlen_q, hdim_q));
    Tensor<KDataType> k_host(get_lengths(i_perm, batch, nhead, seqlen_k, hdim_q));
    Tensor<VDataType> v_host(is_v_rowmajor ? get_lengths(i_perm, batch, nhead, seqlen_k, hdim_v)
                                           : get_lengths(i_perm, batch, nhead, hdim_v, seqlen_k));
    Tensor<ODataType> o_host(get_lengths(o_perm, batch, nhead, seqlen_q, hdim_v));

#if 0
    ck::utils::FillUniformDistributionIntegerValue<QDataType>{-2.f, 2.f}(q_host);
    ck::utils::FillUniformDistributionIntegerValue<KDataType>{-2.f, 2.f}(k_host);
    ck::utils::FillUniformDistributionIntegerValue<VDataType>{-2.f, 2.f}(v_host);
#else
    ck::utils::FillUniformDistribution<QDataType>{0.f, 1.f}(q_host);
    ck::utils::FillUniformDistribution<KDataType>{0.f, 1.f}(k_host);
    ck::utils::FillUniformDistribution<VDataType>{-.5f, .5f}(v_host);
#endif

    DeviceMem q_buf(sizeof(QDataType) * q_host.GetElementSpaceSize());
    DeviceMem k_buf(sizeof(KDataType) * k_host.GetElementSpaceSize());
    DeviceMem v_buf(sizeof(VDataType) * v_host.GetElementSpaceSize());
    DeviceMem o_buf(sizeof(ODataType) * o_host.GetElementSpaceSize());

    q_buf.ToDevice(q_host.mData.data());
    k_buf.ToDevice(k_host.mData.data());
    v_buf.ToDevice(v_host.mData.data());

    std::cout << "batch:" << batch << ", nhead:" << nhead << ", seqlen_q:" << seqlen_q
              << ", seqlen_k:" << seqlen_k << ", hdim_q:" << hdim_q << ", hdim_v:" << hdim_v
              << ", scale:" << scale << ", i_perm:" << i_perm << ", o_perm:" << o_perm
              << ", v:" << std::string(FmhaKernelHDim64::VLayout::name) << std::flush << std::endl;

    float ave_time = 0;
    if(hdim_q == hdim_v && hdim_q == 64)
        ave_time = invoker_fmha_kernel<FmhaKernelHDim64>(q_buf.GetDeviceBuffer(),
                                                         k_buf.GetDeviceBuffer(),
                                                         v_buf.GetDeviceBuffer(),
                                                         o_buf.GetDeviceBuffer(),
                                                         batch,
                                                         nhead,
                                                         seqlen_q,
                                                         seqlen_k,
                                                         hdim_q,
                                                         hdim_v,
                                                         scale,
                                                         i_perm,
                                                         o_perm);
    else if(hdim_q == hdim_v && hdim_q == 128)
        ave_time = invoker_fmha_kernel<FmhaKernelHDim128>(q_buf.GetDeviceBuffer(),
                                                          k_buf.GetDeviceBuffer(),
                                                          v_buf.GetDeviceBuffer(),
                                                          o_buf.GetDeviceBuffer(),
                                                          batch,
                                                          nhead,
                                                          seqlen_q,
                                                          seqlen_k,
                                                          hdim_q,
                                                          hdim_v,
                                                          scale,
                                                          i_perm,
                                                          o_perm);
    else
    {
        std::cout << "not support hdim, will not run" << std::endl;
        return -1;
    }

    std::size_t flop = std::size_t(2) * batch * nhead * seqlen_q * seqlen_k * hdim_q +
                       std::size_t(2) * batch * nhead * seqlen_q * hdim_v * seqlen_k;

    std::size_t num_btype = sizeof(QDataType) * batch * nhead * seqlen_q * hdim_q +
                            sizeof(KDataType) * batch * nhead * seqlen_k * hdim_q +
                            sizeof(VDataType) * batch * nhead * hdim_v * seqlen_k +
                            sizeof(ODataType) * batch * nhead * seqlen_q * hdim_v;

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s"
              << std::endl;

    if(do_validation)
    {
        Tensor<QDataType> q_host_ref({batch * nhead, seqlen_q, hdim_q});
        Tensor<KDataType> k_host_ref({batch * nhead, seqlen_k, hdim_q});
        const auto v_lengths = std::array<ck::index_t, 3>{batch * nhead, hdim_v, seqlen_k};
        const auto v_strides = is_v_rowmajor
                                   ? std::array<ck::index_t, 3>{hdim_v * seqlen_k, 1, hdim_v}
                                   : std::array<ck::index_t, 3>{hdim_v * seqlen_k, seqlen_k, 1};
        Tensor<VDataType> v_host_ref(v_lengths, v_strides);
        Tensor<ODataType> o_host_ref({batch * nhead, seqlen_q, hdim_v});
        Tensor<ODataType> o_host_result_ref(get_lengths(o_perm, batch, nhead, seqlen_q, hdim_v));

        Tensor<SMPLComputeDataType> s_host_ref({batch * nhead, seqlen_q, seqlen_k});
        Tensor<PDataType> p_host_ref({batch * nhead, seqlen_q, seqlen_k});

        // clang-format off
        // permute
        if(i_perm) q_host.ForEach([&](auto& self, auto idx) { q_host_ref(idx[0] * nhead + idx[1], idx[2], idx[3]) = self(idx); });
        else       q_host.ForEach([&](auto& self, auto idx) { q_host_ref(idx[0] * nhead + idx[2], idx[1], idx[3]) = self(idx); });

        if(i_perm) k_host.ForEach([&](auto& self, auto idx) { k_host_ref(idx[0] * nhead + idx[1], idx[2], idx[3]) = self(idx); });
        else       k_host.ForEach([&](auto& self, auto idx) { k_host_ref(idx[0] * nhead + idx[2], idx[1], idx[3]) = self(idx); });

        if constexpr (is_v_rowmajor) {
            //                              v_host ：b, h, s, d, v_host_ref : batch*hdim*seq
            if(i_perm) v_host.ForEach([&](auto& self, auto idx) { v_host_ref(idx[0] * nhead + idx[1], idx[3], idx[2]) = self(idx); });
            //                              v_host : b, s, h, d, v_host_ref : batch*hdim*seq
            else       v_host.ForEach([&](auto& self, auto idx) { v_host_ref(idx[0] * nhead + idx[2], idx[3], idx[1]) = self(idx); });
        }
        else {
            if(i_perm) v_host.ForEach([&](auto& self, auto idx) { v_host_ref(idx[0] * nhead + idx[1], idx[2], idx[3]) = self(idx); });
            else       v_host.ForEach([&](auto& self, auto idx) { v_host_ref(idx[0] * nhead + idx[2], idx[1], idx[3]) = self(idx); });
        }

        // reference
        reference_batched_gemm<QDataType, KDataType, SaccDataType, SMPLComputeDataType>(
            q_host_ref, k_host_ref, s_host_ref,
            [](const QDataType& x) { return x; },
            [](const KDataType& x) { return x; },
            [&scale](const SaccDataType& x) { return scale * x; });
        reference_batched_softmax<SMPLComputeDataType, SMPLComputeDataType, PDataType>(s_host_ref,
                                                                                       p_host_ref);
        reference_batched_gemm<PDataType, VDataType, OaccDataType, ODataType>(
            p_host_ref, v_host_ref, o_host_ref);

        // permute
        if(o_perm) o_host_result_ref.ForEach([&](auto& self, auto idx) { self(idx) = o_host_ref(idx[0] * nhead + idx[1], idx[2], idx[3]); });
        else       o_host_result_ref.ForEach([&](auto& self, auto idx) { self(idx) = o_host_ref(idx[0] * nhead + idx[2], idx[1], idx[3]); });
        // clang-format on

        o_buf.FromDevice(o_host.mData.data());
        return !ck::utils::check_err(o_host, o_host_result_ref);
    }
    else
    {
        return 0;
    }
}
