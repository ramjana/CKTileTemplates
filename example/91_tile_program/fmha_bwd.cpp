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

#include "ck/tile_program/block_tile_pipeline/block_fmha_bwd_pipeline.hpp"
#include "ck/tile_program/block_tile_pipeline/block_fmha_bwd_pipeline_default_policy.hpp"
#include "ck/tile_program/block_tile_pipeline/block_fmha_bwd_pipeline_problem.hpp"
#include "ck/tile_program/tile/tile_fmha_bwd_shape.hpp"

#include "reference_batched_gemm.hpp"
#include "reference_batched_softmax.hpp"
#include "fmha_bwd_kernel.hpp"
#include "fmha_bwd_tile_partitioner.hpp"
#include "fmha_bwd_epilogue.hpp"

using QDataType           = ck::half_t;
using KDataType           = ck::half_t;
using VDataType           = ck::half_t;
using GemmDataType        = ck::half_t;
using LSEDataType         = float;
using AccDataType         = float; // data type for gemm accumulation
using SMPLComputeDataType = float; // data type for reduction, softmax
using DDataType           = float;
using ZDataType           = unsigned short;
using ODataType           = ck::half_t;
using OGradDataType       = ck::half_t;
using QGradDataType       = ck::half_t;
using KGradDataType       = ck::half_t;
using VGradDataType       = ck::half_t;

// GEMM0: Q@K^T=S
// GEMM1: P^T@dO=dV(This was chosen as G1 to match fwd, but N1 must be equal to headdim_v)
// GEMM2: dO@V^T=dP(This was chosen as G2 because of the calculation order)
// GEMM3: dS^T@Q=dK(Similar to G1, but N3 must be equal to headdim_qk)
// GEMM4: dS@K=dQ(N4 must be equal to headdim_qk)
// Is it necessary to distinguish between K0~K4?
// clang-format off
// ######################################|  M0|  N0|  K0|  K1|  K2|  K3|  K4| QKHD|  VHD|
using FmhaBlockTileHdim32  = ck::Sequence< 128, 128,  32,  32,  32,  32,  32,   32,   32>;
using FmhaBlockTileHdim64  = ck::Sequence<  64, 128,  32,  32,  32,  32,  32,   64,   64>;
using FmhaBlockTileHdim128 = ck::Sequence<  64, 128,  32,  32,  32,  32,  32,  128,  128>;
// clang-format on
using FmhaBlockWarps0 = ck::Sequence<1, 4, 1>;
using FmhaBlockWarps1 = ck::Sequence<4, 1, 1>;
using FmhaBlockWarps2 = ck::Sequence<2, 2, 1>;
using FmhaWarpTile0   = ck::Sequence<32, 32, 16>;
using FmhaWarpTile1   = ck::Sequence<16, 16, 16>;
// TODO: simplify Gemm0~4BlockWarps in TileFmhaBwdShape
//       G0&G2 -> GSdP
//       G1&G3 -> GdKV
//       G4    -> GdQ
using FmhaShapeHDim32 = ck::tile_program::TileFmhaBwdShape<FmhaBlockTileHdim32,
                                                           FmhaBlockWarps0,
                                                           FmhaWarpTile0,
                                                           FmhaBlockWarps1,
                                                           FmhaWarpTile0,
                                                           FmhaBlockWarps0,
                                                           FmhaWarpTile0,
                                                           FmhaBlockWarps1,
                                                           FmhaWarpTile0,
                                                           FmhaBlockWarps1,
                                                           FmhaWarpTile0>;
using FmhaShapeHDim64 = ck::tile_program::TileFmhaBwdShape<FmhaBlockTileHdim64,
                                                           FmhaBlockWarps0,
                                                           FmhaWarpTile0,
                                                           FmhaBlockWarps1,
                                                           FmhaWarpTile0,
                                                           FmhaBlockWarps0,
                                                           FmhaWarpTile0,
                                                           FmhaBlockWarps1,
                                                           FmhaWarpTile0,
                                                           FmhaBlockWarps2,
                                                           FmhaWarpTile0>;

using FmhaBwdTilePartitionerHDim32 = FmhaBwdTilePartitioner<FmhaShapeHDim32>;
using FmhaBwdTilePartitionerHDim64 = FmhaBwdTilePartitioner<FmhaShapeHDim64>;
using FmhaBwdPipelineProblemHDim32 =
    ck::tile_program::block::BlockFmhaBwdPipelineProblem<QDataType,
                                                         KDataType,
                                                         VDataType,
                                                         GemmDataType,
                                                         LSEDataType,
                                                         AccDataType,
                                                         SMPLComputeDataType,
                                                         DDataType,
                                                         ZDataType,
                                                         ODataType,
                                                         OGradDataType,
                                                         QGradDataType,
                                                         KGradDataType,
                                                         VGradDataType,
                                                         256, // BlockSize
                                                         FmhaShapeHDim32>;
using FmhaBwdPipelineProblemHDim64 =
    ck::tile_program::block::BlockFmhaBwdPipelineProblem<QDataType,
                                                         KDataType,
                                                         VDataType,
                                                         GemmDataType,
                                                         LSEDataType,
                                                         AccDataType,
                                                         SMPLComputeDataType,
                                                         DDataType,
                                                         ZDataType,
                                                         ODataType,
                                                         OGradDataType,
                                                         QGradDataType,
                                                         KGradDataType,
                                                         VGradDataType,
                                                         256, // BlockSize
                                                         FmhaShapeHDim64>;

using FmhaBwdPipelineHDim32 =
    ck::tile_program::block::BlockFmhaBwdPipeline<FmhaBwdPipelineProblemHDim32>;
using FmhaBwdPipelineHDim64 =
    ck::tile_program::block::BlockFmhaBwdPipeline<FmhaBwdPipelineProblemHDim64>;

using FmhaBWDEpilogue = FmhaBwdEpilogue<
    FmhaBwdEpilogueProblem<AccDataType, QGradDataType, KGradDataType, VGradDataType>>;
using FmhaBwdKernelHDim32 =
    FmhaBwdKernel<FmhaBwdTilePartitionerHDim32, FmhaBwdPipelineHDim32, FmhaBWDEpilogue>;
using FmhaBwdKernelHDim64 =
    FmhaFwdKernel<FmhaBwdTilePartitionerHDim64, FmhaBwdPipelineHDim64, FmhaBWDEpilogue>;

template <typename FmhaBwdKernel>
float invoker_fmha_bwd_kernel(const void* q_ptr,
                              const void* k_ptr,
                              const void* v_ptr,
                              const void* o_ptr,
                              const void* lse_ptr,
                              const void* do_ptr,
                              // void* z_ptr,
                              // void* d_ptr,
                              void* dq_ptr,
                              void* dk_ptr,
                              void* dv_ptr,
                              ck::index_t batch,
                              ck::index_t nhead,
                              ck::index_t seqlen_q,
                              ck::index_t seqlen_k,
                              ck::index_t hdim_q,
                              ck::index_t hdim_v,
                              float scale,
                              bool i_perm,
                              bool o_perm,
                              bool time_kernel = true)
{
    dim3 kGridSize            = FmhaBwdKernel::GridSize(batch, nhead, seqlen_k);
    constexpr dim3 kBlockSize = FmhaBwdKernel::BlockSize();

    // constexpr ck::index_t kWarpPerCu    = 8; // 2 warps per SIMD
    // constexpr ck::index_t kWarpPerBlock = kBlockSize.x / warpSize;
    // constexpr ck::index_t kBlockPerCu   = kWarpPerCu / kWarpPerBlock;
    constexpr ck::index_t kBlockPerCu = 1;

    // batch * nhead * seqlen * hdim or batch * seqlen * nhead * hdim
    auto kargs = FmhaBwdKernel::MakeKargs(q_ptr,
                                          k_ptr,
                                          v_ptr,
                                          o_ptr,
                                          lse_ptr,
                                          do_ptr,
                                          // d_ptr,
                                          // z_ptr,
                                          dq_ptr,
                                          dk_ptr,
                                          dv_ptr,
                                          seqlen_q, // seqlen_q
                                          seqlen_k, // seqlen_k
                                          hdim_q,   // hdim_q
                                          hdim_v,   // hdim_v
                                          scale,
                                          i_perm ? hdim_q : nhead * hdim_q,    // stride_q
                                          i_perm ? hdim_q : nhead * hdim_q,    // stride_k
                                          i_perm ? hdim_v : nhead * hdim_v,    // stride_v
                                          o_perm ? hdim_v : nhead * hdim_v,    // stride_o
                                          i_perm ? seqlen_q * hdim_q : hdim_q, // nhead_stride_q
                                          i_perm ? seqlen_k * hdim_q : hdim_q, // nhead_stride_k
                                          i_perm ? seqlen_k * hdim_v : hdim_v, // nhead_stride_v
                                          o_perm ? seqlen_q * hdim_v : hdim_v, // nhead_stride_o
                                          seqlen_q,                            // nhead_stride_lse
                                          nhead * seqlen_q * hdim_q,           // batch_stride_q
                                          nhead * seqlen_k * hdim_q,           // batch_stride_k
                                          nhead * seqlen_k * hdim_v,           // batch_stride_v
                                          nhead * seqlen_q * hdim_v,           // batch_stride_o
                                          nhead * seqlen_q);                   // batch_stride_lse

    float ave_time = launch_kernel<kBlockSize.x, kBlockPerCu>(
        StreamConfig{nullptr, time_kernel}, FmhaBwdKernel{}, kGridSize, kBlockSize, 0, kargs);
    return ave_time;
}

template <typename TensorQ,
          typename TensorK,
          typename TensorV,
          typename TensorS,
          typename TensorPHP,
          typename TensorPLP,
          typename TensorO,
          typename TensorLSE>
// typename TensorZ>
void run_fmha_fwd_host(const TensorQ& q_g_m_k,
                       const TensorK& k_g_n_k,
                       const float alpha,
                       const TensorV& v_g_o_n,
                       TensorS& s_g_m_n,
                       TensorPHP& p_hp_g_m_n,
                       TensorPLP& p_lp_g_m_n,
                       TensorO& o_g_m_o,
                       TensorLSE& lse_g_m)
// TensorPHP& p_drop_hp_g_m_n,
// TensorZ& z_g_m_n,
// ZDataType p_dropout_in_uint8_t,
// float rp_dropout)
{
    // S = alpha * Q * K^T
    reference_batched_gemm<QDataType, KDataType, AccDataType, SMPLComputeDataType>(
        q_g_m_k,
        k_g_n_k,
        s_g_m_n,
        [](const QDataType& x) { return x; },
        [](const KDataType& x) { return x; },
        [&alpha](const AccDataType& x) { return alpha * x; });
    // TODO: masking
    // P = Softmax(S)
    reference_batched_softmax<SMPLComputeDataType,
                              SMPLComputeDataType,
                              SMPLComputeDataType,
                              LSEDataType>(s_g_m_n, p_hp_g_m_n, lse_g_m);

    p_hp_g_m_n.ForEach(
        [&](auto& self, auto idx) { p_lp_g_m_n(idx) = ck::type_convert<GemmDataType>(self(idx)); });

    // TODO: dropout
    // O = P * V
    reference_batched_gemm<GemmDataType, VDataType, AccDataType, ODataType>(
        p_lp_g_m_n, v_g_o_n, o_g_m_o);
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

    // host verify
    Tensor<QDataType> q_host(get_lengths(i_perm, batch, nhead, seqlen_q, hdim_q));
    Tensor<KDataType> k_host(get_lengths(i_perm, batch, nhead, seqlen_k, hdim_q));
    Tensor<VDataType> v_host(get_lengths(i_perm, batch, nhead, seqlen_k, hdim_v));
    Tensor<ODataType> o_host(get_lengths(o_perm, batch, nhead, seqlen_q, hdim_v));
    Tensor<LSEDataType> lse_host(std::array<ck::index_t, 3>{batch, nhead, seqlen_q});
    // Tensor<ZDataType> z_host(std::array<ck::index_t, 4>{batch, nhead, seqlen_q, seqlen_k});
    // Tensor<DDataType> d_host(std::array<ck::index_t, 3>{batch, nhead, seqlen_q});
    Tensor<QGradDataType> dq_host(get_lengths(i_perm, batch, nhead, seqlen_q, hdim_q));
    Tensor<KGradDataType> dk_host(get_lengths(i_perm, batch, nhead, seqlen_k, hdim_q));
    Tensor<VGradDataType> dv_host(get_lengths(i_perm, batch, nhead, seqlen_k, hdim_v));
    Tensor<OGradDataType> do_host(get_lengths(o_perm, batch, nhead, seqlen_q, hdim_v));

#if 0
    ck::utils::FillUniformDistributionIntegerValue<QDataType>{-2.f, 2.f}(q_host);
    ck::utils::FillUniformDistributionIntegerValue<KDataType>{-2.f, 2.f}(k_host);
    ck::utils::FillUniformDistributionIntegerValue<VDataType>{-2.f, 2.f}(v_host);
    ck::utils::FillUniformDistributionIntegerValue<OGradDataType>{-2.f, 2.f}(do_host);
#else
    ck::utils::FillUniformDistribution<QDataType>{0.f, 1.f}(q_host);
    ck::utils::FillUniformDistribution<KDataType>{0.f, 1.f}(k_host);
    ck::utils::FillUniformDistribution<VDataType>{-.5f, .5f}(v_host);
    ck::utils::FillUniformDistribution<OGradDataType>{-.5f, .5f}(do_host);
#endif

    DeviceMem q_buf(sizeof(QDataType) * q_host.GetElementSpaceSize());
    DeviceMem k_buf(sizeof(KDataType) * k_host.GetElementSpaceSize());
    DeviceMem v_buf(sizeof(VDataType) * v_host.GetElementSpaceSize());
    DeviceMem o_buf(sizeof(ODataType) * o_host.GetElementSpaceSize());
    DeviceMem lse_buf(sizeof(LSEDataType) * lse_host.GetElementSpaceSize());
    // DeviceMem z_buf(sizeof(ZDataType) * z_host.GetElementSpaceSize());
    // DeviceMem d_buf(sizeof(DDataType) * d_host.GetElementSpaceSize());
    DeviceMem dq_buf(sizeof(QGradDataType) * dq_host.GetElementSpaceSize());
    DeviceMem dk_buf(sizeof(KGradDataType) * dk_host.GetElementSpaceSize());
    DeviceMem dv_buf(sizeof(VGradDataType) * dv_host.GetElementSpaceSize());
    DeviceMem do_buf(sizeof(OGradDataType) * do_host.GetElementSpaceSize());

    q_buf.ToDevice(q_host.mData.data());
    k_buf.ToDevice(k_host.mData.data());
    v_buf.ToDevice(v_host.mData.data());
    do_buf.ToDevice(do_host.mData.data());

    std::cout << "batch:" << batch << ", nhead:" << nhead << ", seqlen_q:" << seqlen_q
              << ", seqlen_k:" << seqlen_k << ", hdim_q:" << hdim_q << ", hdim_v:" << hdim_v
              << ", scale:" << scale << ", i_perm:" << i_perm << ", o_perm:" << o_perm << std::flush
              << std::endl;

    float ave_time = 0;
    if(hdim_q == hdim_v && hdim_q == 32)
        ave_time = invoker_fmha_bwd_kernel<FmhaKernelHDim32>(q_buf.GetDeviceBuffer(),
                                                             k_buf.GetDeviceBuffer(),
                                                             v_buf.GetDeviceBuffer(),
                                                             o_buf.GetDeviceBuffer(),
                                                             lse_buf.GetDeviceBuffer(),
                                                             do_buf.GetDeviceBuffer(),
                                                             // z_buf.GetDeviceBuffer(),
                                                             // d_buf.GetDeviceBuffer(),
                                                             dq_buf.GetDeviceBuffer(),
                                                             dk_buf.GetDeviceBuffer(),
                                                             dv_buf.GetDeviceBuffer(),
                                                             batch,
                                                             nhead,
                                                             seqlen_q,
                                                             seqlen_k,
                                                             hdim_q,
                                                             hdim_v,
                                                             scale,
                                                             i_perm,
                                                             o_perm);
    else if(hdim_q == hdim_v && hdim_q == 64)
        ave_time = invoker_fmha_bwd_kernel<FmhaKernelHDim64>(q_buf.GetDeviceBuffer(),
                                                             k_buf.GetDeviceBuffer(),
                                                             v_buf.GetDeviceBuffer(),
                                                             o_buf.GetDeviceBuffer(),
                                                             lse_buf.GetDeviceBuffer(),
                                                             do_buf.GetDeviceBuffer(),
                                                             // z_buf.GetDeviceBuffer(),
                                                             // d_buf.GetDeviceBuffer(),
                                                             dq_buf.GetDeviceBuffer(),
                                                             dk_buf.GetDeviceBuffer(),
                                                             dv_buf.GetDeviceBuffer(),
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

    std::size_t flop = std::size_t(3) * std::size_t(2) * batch * nhead * seqlen_q * seqlen_k *
                           hdim_q + // Q@K^T/dS^T@Q/dS@K
                       std::size_t(2) * std::size_t(2) * batch * nhead * seqlen_q * seqlen_k *
                           hdim_v; // dO@V^T/P^T@dO

    std::size_t num_btype = sizeof(QDataType) * batch * nhead * seqlen_q * hdim_q +
                            sizeof(KDataType) * batch * nhead * seqlen_k * hdim_q +
                            sizeof(VDataType) * batch * nhead * seqlen_k * hdim_v +
                            sizeof(ODataType) * batch * nhead * seqlen_q * hdim_v +
                            sizeof(OGradDataType) * batch * nhead * seqlen_q * hdim_v +
                            sizeof(QGradDataType) * batch * nhead * seqlen_q * hdim_q +
                            sizeof(KGradDataType) * batch * nhead * seqlen_k * hdim_q +
                            sizeof(VGradDataType) * batch * nhead * seqlen_k * hdim_v +
                            sizeof(LSEGradDataType) * batch * nhead * seqlen_q;

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s"
              << std::endl;

    if(do_validation)
    {
        Tensor<QDataType> q_host_ref({batch * nhead, seqlen_q, hdim_q}); // q_g_m_k
        Tensor<KDataType> k_host_ref({batch * nhead, seqlen_k, hdim_q}); // k_g_n_k
        Tensor<VDataType> v_host_ref({batch * nhead, hdim_v, seqlen_k}); // v_g_o_n
        Tensor<ODataType> o_host_ref({batch * nhead, seqlen_q, hdim_v}); // o_g_m_o
        Tensor<LSEDataType> lse_host_ref({batch * nhead, seqlen_q});     // lse_g_m
        // Tensor<ZDataType> z_host_ref({batch * nhead, seqlen_q, seqlen_k}); // z_g_m_n
        Tensor<QGradDataType> dq_host_ref({batch * nhead, seqlen_q, hdim_q}); // dq_g_m_k
        Tensor<KGradDataType> dk_host_ref({batch * nhead, seqlen_k, hdim_q}); // dk_g_n_k
        Tensor<VGradDataType> dv_host_ref({batch * nhead, seqlen_k, hdim_v}); // dv_g_n_o
        Tensor<OGradDataType> do_host_ref({batch * nhead, seqlen_q, hdim_v}); // do_g_m_o

        Tensor<SMPLComputeDataType> s_host_ref({batch * nhead, seqlen_q, seqlen_k}); // s_g_m_n
        Tensor<GemmDataType> ds_host_ref({batch * nhead, seqlen_q, seqlen_k});       // ds_g_m_n
        Tensor<SMPLComputeDataType> p_hp_host_ref(
            {batch * nhead, seqlen_q, seqlen_k}); // p_hp_g_m_n high precision
        Tensor<GemmDataType> p_lp_host_ref(
            {batch * nhead, seqlen_q, seqlen_k}); // p_lp_g_m_n low precision
        Tensor<SMPLComputeDataType> dp_hp_host_ref(
            {batch * nhead, seqlen_q, seqlen_k}); // dp_hp_g_m_n high precision

        Tensor<QGradDataType> dq_host_result_ref(
            get_lengths(i_perm, batch, nhead, seqlen_q, hdim_q)); // dq_g_m_k
        Tensor<KGradDataType> dk_host_result_ref(
            get_lengths(i_perm, batch, nhead, seqlen_k, hdim_q)); // dk_g_n_k
        Tensor<VGradDataType> dv_host_result_ref(
            get_lengths(i_perm, batch, nhead, seqlen_k, hdim_v)); // dv_g_n_o

        // permute
        if(i_perm)
            q_host.ForEach([&](auto& self, auto idx) {
                q_host_ref(idx[0] * nhead + idx[1], idx[2], idx[3]) = self(idx);
            });
        else
            q_host.ForEach([&](auto& self, auto idx) {
                q_host_ref(idx[0] * nhead + idx[2], idx[1], idx[3]) = self(idx);
            });

        if(i_perm)
            k_host.ForEach([&](auto& self, auto idx) {
                k_host_ref(idx[0] * nhead + idx[1], idx[2], idx[3]) = self(idx);
            });
        else
            k_host.ForEach([&](auto& self, auto idx) {
                k_host_ref(idx[0] * nhead + idx[2], idx[1], idx[3]) = self(idx);
            });

        // v_host ：b, h, s, d, v_host_ref : batch*hdim*seq
        if(i_perm)
            v_host.ForEach([&](auto& self, auto idx) {
                v_host_ref(idx[0] * nhead + idx[1], idx[3], idx[2]) = self(idx);
            });
        // v_host : b, s, h, d, v_host_ref : batch*hdim*seq
        else
            v_host.ForEach([&](auto& self, auto idx) {
                v_host_ref(idx[0] * nhead + idx[2], idx[3], idx[1]) = self(idx);
            });

        // reference
        run_fmha_fwd_host(q_host_ref,
                          k_host_ref,
                          scale,
                          v_host_ref,
                          s_host_ref,
                          p_hp_host_ref,
                          p_lp_host_ref,
                          o_host_ref,
                          lse_host_ref);

        // permute
        if(o_perm)
            o_buf.ForEach([&](auto& self, auto idx) {
                self(idx) = o_host_ref(idx[0] * nhead + idx[1], idx[2], idx[3]);
            });
        else
            o_buf.ForEach([&](auto& self, auto idx) {
                self(idx) = o_host_ref(idx[0] * nhead + idx[2], idx[1], idx[3]);
            });
        lse_buf.ForEach([&](auto& self, auto idx) {
            self(idx) = lse_host_ref(idx[0] * nhead + idx[1], idx[2]);
        });

        o_buf.ToDevice(o_host.mData.data());
        lse_buf.ToDevice(lse_host.mData.data());

        if(hdim_q == hdim_v && hdim_q == 32)
            invoker_fmha_bwd_kernel<FmhaKernelHDim32>(q_buf.GetDeviceBuffer(),
                                                      k_buf.GetDeviceBuffer(),
                                                      v_buf.GetDeviceBuffer(),
                                                      o_buf.GetDeviceBuffer(),
                                                      lse_buf.GetDeviceBuffer(),
                                                      do_buf.GetDeviceBuffer(),
                                                      // z_buf.GetDeviceBuffer(),
                                                      // d_buf.GetDeviceBuffer(),
                                                      dq_buf.GetDeviceBuffer(),
                                                      dk_buf.GetDeviceBuffer(),
                                                      dv_buf.GetDeviceBuffer(),
                                                      batch,
                                                      nhead,
                                                      seqlen_q,
                                                      seqlen_k,
                                                      hdim_q,
                                                      hdim_v,
                                                      scale,
                                                      i_perm,
                                                      o_perm,
                                                      false);
        else if(hdim_q == hdim_v && hdim_q == 64)
            invoker_fmha_bwd_kernel<FmhaKernelHDim64>(q_buf.GetDeviceBuffer(),
                                                      k_buf.GetDeviceBuffer(),
                                                      v_buf.GetDeviceBuffer(),
                                                      o_buf.GetDeviceBuffer(),
                                                      lse_buf.GetDeviceBuffer(),
                                                      do_buf.GetDeviceBuffer(),
                                                      // z_buf.GetDeviceBuffer(),
                                                      // d_buf.GetDeviceBuffer(),
                                                      dq_buf.GetDeviceBuffer(),
                                                      dk_buf.GetDeviceBuffer(),
                                                      dv_buf.GetDeviceBuffer(),
                                                      batch,
                                                      nhead,
                                                      seqlen_q,
                                                      seqlen_k,
                                                      hdim_q,
                                                      hdim_v,
                                                      scale,
                                                      i_perm,
                                                      o_perm,
                                                      false);

        // dP_dropout = dO * V^T
        // dP = dO * V^T w/o dropout
        auto v_t_host_ref = v_host_ref.Transpose({0, 2, 1}); // v_g_o_n -> v_g_n_o
        reference_batched_gemm<OGradDataType, VDataType, AccDataType, SMPLComputeDataType>(
            do_host_ref, v_t_host_ref, dp_hp_host_ref); // dp_g_m_n = do_g_m_o@v_g_n_o

        // TODO: dP = dP_dropout x Z

        // dS_i_j = P_i_j .* (dP_i_j - dO_i dot O_i)
        ds_host_ref.ForEach([&](auto& self, auto idx_gmn) {
            AccDataType do_dot_o = 0;
            for(int o = 0; o < hdim_v; o++)
            {
                auto idx_gmo = idx_gmn;
                idx_gmo[2]   = o;
                do_dot_o += ck::type_convert<AccDataType>(do_host_ref(idx_gmo)) *
                            ck::type_convert<AccDataType>(o_host_ref(idx_gmo));
            }
            self(idx_gmn) = ck::type_convert<GemmDataType>(
                ck::type_convert<AccDataType>(p_hp_host_ref(idx_gmn)) *
                (ck::type_convert<AccDataType>(dp_hp_host_ref(idx_gmn)) - do_dot_o));
        });

        // dV = P_drop^T * dO
        // dV = P^T * dO w/o dropout
        auto p_t_lp_host_ref = p_lp_host_ref.Transpose({0, 2, 1}); // p_lp_g_m_n -> p_lp_g_n_m
        auto do_t_host_ref   = do_host_ref.Transpose({0, 2, 1});   // do_g_m_o -> do_g_o_m
        reference_batched_gemm<GemmDataType, OGradDataType, AccDataType, VGradDataType>(
            p_t_lp_host_ref, do_t_host_ref, dv_g_n_o); // dv_g_n_o = p_lp_g_n_m@do_g_o_m

        // dQ = alpha * dS * K
        reference_batched_gemm<GemmDataType, KDataType, AccDataType, QGradDataType>(
            ds_host_ref,
            k_host_ref,
            dq_host_ref,
            [](const GemmDataType& x) { return x; },
            [](const KDataType& x) { return x; },
            [&alpha](const AccDataType& x) { return alpha * x; }); // dq_g_m_k = ds_g_m_n@k_g_k_n

        // dK = alpha * dS^T * Q
        auto ds_t_host_ref = ds_host_ref.Transpose({0, 2, 1}); // ds_g_m_n -> ds_g_n_m
        auto q_t_host_ref  = q_host_ref.Transpose({0, 2, 1});  // q_g_m_k -> q_g_k_m
        reference_batched_gemm<GemmDataType, QDataType, AccDataType, KGradDataType>(
            ds_t_host_ref,
            q_t_host_ref,
            dk_host_ref,
            [](const GemmDataType& x) { return x; },
            [](const QDataType& x) { return x; },
            [&alpha](const AccDataType& x) { return alpha * x; }); // dk_g_n_k = ds_g_n_m@q_g_k_m

        // permute
        if(i_perm)
            dq_host_result_ref.ForEach([&](auto& self, auto idx) {
                self(idx) = dq_host_ref(idx[0] * nhead + idx[1], idx[2], idx[3]);
            });
        else
            dq_host_result_ref.ForEach([&](auto& self, auto idx) {
                self(idx) = dq_host_ref(idx[0] * nhead + idx[2], idx[1], idx[3]);
            });

        if(i_perm)
            dk_host_result_ref.ForEach([&](auto& self, auto idx) {
                self(idx) = dk_host_ref(idx[0] * nhead + idx[1], idx[2], idx[3]);
            });
        else
            dk_host_result_ref.ForEach([&](auto& self, auto idx) {
                self(idx) = dk_host_ref(idx[0] * nhead + idx[2], idx[1], idx[3]);
            });

        if(i_perm)
            dv_host_result_ref.ForEach([&](auto& self, auto idx) {
                self(idx) = dv_host_ref(idx[0] * nhead + idx[1], idx[2], idx[3]);
            });
        else
            dv_host_result_ref.ForEach([&](auto& self, auto idx) {
                self(idx) = dv_host_ref(idx[0] * nhead + idx[2], idx[1], idx[3]);
            });

        dq_buf.FromDevice(dq_host.mData.data());
        dk_buf.FromDevice(dk_host.mData.data());
        dv_buf.FromDevice(dv_host.mData.data());
        return !(ck::utils::check_err(dq_host, dq_host_result_ref) &
                 ck::utils::check_err(dk_host, dk_host_result_ref) &
                 ck::utils::check_err(dv_host, dv_host_result_ref));
    }
    else
    {
        return 0;
    }
}
