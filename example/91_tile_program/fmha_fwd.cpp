#include <cstring>
#include <cstdlib>
#include <iomanip>
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

#include "reference_gemm.hpp"
#include "reference_softmax.hpp"
#include "fmha_fwd_kernel.hpp"
#include "fmha_fwd_tile_partitioner.hpp"
#include "fmha_fwd_epilogue.hpp"

using QDataType           = ck::half_t;
using KDataType           = ck::half_t;
using VDataType           = ck::half_t;
using SaccDataType        = float;      // data type for first gemm accumulation
using SMPLComputeDataType = float;      // data type for reduction, softmax
using PDataType           = ck::half_t; // data type for A matrix of second gemm
using OaccDataType        = float;      // data type for second gemm accumulation
using ODataType           = ck::half_t;

//                                                 M0   N0  K0   N1  K1  K0L
// using FmhaShape = ck::tile_program::TileFmhaShape<128,  64, 64, 128, 64>;
// using FmhaShape = ck::tile_program::TileFmhaShape<128, 256, 32, 128, 32>;
using FmhaBlockTile  = ck::Sequence<128, 128, 32, 128, 32, 128>;
using FmhaBlockWarps = ck::Sequence<4, 1, 1>;
using FmhaWarpTile   = ck::Sequence<32, 32, 16>;
using FmhaShape      = ck::tile_program::
    TileFmhaShape<FmhaBlockTile, FmhaBlockWarps, FmhaWarpTile, FmhaBlockWarps, FmhaWarpTile>;

using FmhaTilePartitioner = FmhaFwdTilePartitioner<FmhaShape>;
using FmhaPipelineProblem = ck::tile_program::block::BlockFmhaPipelineProblem<QDataType,
                                                                              KDataType,
                                                                              VDataType,
                                                                              SaccDataType,
                                                                              SMPLComputeDataType,
                                                                              PDataType,
                                                                              OaccDataType,
                                                                              ODataType,
                                                                              256, // BlockSize
                                                                              FmhaShape>;
// using FmhaPipeline        = ck::tile_program::block::BlockFmhaPipelineQKVS<FmhaPipelineProblem>;
using FmhaPipeline = ck::tile_program::block::BlockFmhaPipelineQRKSVS<FmhaPipelineProblem>;

using FmhaEpilogue = FmhaFwdEpilogue<FmhaFwdEpilogueProblem<OaccDataType, ODataType>>;
using FmhaKernel   = FmhaFwdKernel<FmhaTilePartitioner, FmhaPipeline, FmhaEpilogue>;

struct Options
{
    bool do_validation   = true;
    ck::index_t batch    = 2;
    ck::index_t nhead    = 8;
    ck::index_t seqlen_q = 3328;
    ck::index_t seqlen_k = 4096;
    ck::index_t hdim_q   = 128;
    ck::index_t hdim_v   = 128;

    float scale = .0f;

    // for following flag values, if their value is true, the shape input/output tensor will be
    // [batch, nhead, seqlen, hdim]; otherwise: [batch, seqlen, nhead, hdim]
    bool i_perm = true;
    bool o_perm = true;

    bool parse(int argc, char* argv[])
    {
        if(argc >= 2)
            do_validation = static_cast<bool>(std::stoi(argv[1]));

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

        return true;
    }
};

template <std::size_t Dim>
using TensorShape = std::array<ck::index_t, Dim>;

TensorShape<4> get_shape(bool permute,
                         ck::index_t b /*batch*/,
                         ck::index_t h /*nhead*/,
                         ck::index_t s /*seqlen*/,
                         ck::index_t d /*hdim*/)
{
    if(permute)
        return TensorShape<4>{b, h, s, d};
    else
        return TensorShape<4>{b, s, h, d};
}

template <std::size_t Dim>
ck::index_t get_stride(const TensorShape<Dim>& shape, ck::index_t axis)
{
    return std::accumulate(std::rbegin(shape),
                           std::next(std::rbegin(shape), Dim - axis - 1),
                           static_cast<ck::index_t>(1),
                           std::multiplies<ck::index_t>{});
}

int main(int argc, char* argv[])
{
    Options options;
    if(!options.parse(argc, argv))
    {
        std::cerr << "failed to parse command line arguments" << std::endl;
        return EXIT_FAILURE;
    }

    const auto q_shape =
        get_shape(options.i_perm, options.batch, options.nhead, options.seqlen_q, options.hdim_q);
    const auto k_shape =
        get_shape(options.i_perm, options.batch, options.nhead, options.seqlen_k, options.hdim_q);
    const auto v_shape =
        get_shape(options.i_perm, options.batch, options.nhead, options.hdim_v, options.seqlen_k);
    const auto o_shape =
        get_shape(options.o_perm, options.batch, options.nhead, options.seqlen_q, options.hdim_v);

    // host verify
    Tensor<QDataType> q_host(q_shape);
    Tensor<KDataType> k_host(k_shape);
    Tensor<VDataType> v_host(v_shape);
    Tensor<ODataType> o_host(o_shape);

#if 0
    ck::utils::FillUniformDistributionIntegerValue<QDataType>{-2.f, 2.f}(q_host);
    ck::utils::FillUniformDistributionIntegerValue<KDataType>{-2.f, 2.f}(k_host);
    ck::utils::FillUniformDistributionIntegerValue<VDataType>{-2.f, 2.f}(v_host);
#else
    ck::utils::FillUniformDistribution<QDataType>{0.f, 1.f}(q_host);
    ck::utils::FillUniformDistribution<KDataType>{0.f, 1.f}(k_host);
    ck::utils::FillUniformDistribution<VDataType>{-.5f, .5f}(v_host);
#endif

    DeviceMem q_buf(q_host.GetElementSpaceSizeInBytes());
    DeviceMem k_buf(k_host.GetElementSpaceSizeInBytes());
    DeviceMem v_buf(v_host.GetElementSpaceSizeInBytes());
    DeviceMem o_buf(o_host.GetElementSpaceSizeInBytes());

    q_buf.ToDevice(q_host.mData.data());
    k_buf.ToDevice(k_host.mData.data());
    v_buf.ToDevice(v_host.mData.data());

    dim3 kGridSize =
        FmhaKernel::GridSize(options.batch, options.nhead, options.seqlen_q, options.hdim_v);
    constexpr dim3 kBlockSize = FmhaKernel::BlockSize();

    std::cout << "batch: " << options.batch << ", nhead: " << options.nhead
              << ", seqlen_q: " << options.seqlen_q << ", seqlen_k: " << options.seqlen_k
              << ", hdim_q: " << options.hdim_q << ", hdim_v: " << options.hdim_v
              << ", scale: " << options.scale << ", i_perm: " << std::boolalpha << options.i_perm
              << ", o_perm: " << std::boolalpha << options.o_perm << ", grid_size: " << kGridSize.x
              << "x" << kGridSize.y << "x" << kGridSize.z << std::endl;

    constexpr ck::index_t kWarpPerCu    = 8; // 2 warps per SIMD
    constexpr ck::index_t kWarpPerBlock = kBlockSize.x / warpSize;
    constexpr ck::index_t kBlockPerCu   = kWarpPerCu / kWarpPerBlock;

    // if value of i_perm/o_perm is true, the tensor shape is [batch, nhead, seqlen, hdim];
    // otherwise, shape is [batch, seqlen, nhead, hdim]. which means we are choosing
    // stride/nhead_stride base on the axis of seqlen/nhead (axis=1 or axis=2)
    auto kargs = FmhaKernel::MakeKargs(q_buf.GetDeviceBuffer(),
                                       k_buf.GetDeviceBuffer(),
                                       v_buf.GetDeviceBuffer(),
                                       o_buf.GetDeviceBuffer(),
                                       options.seqlen_q, // seqlen_q
                                       options.seqlen_k, // seqlen_k
                                       options.hdim_q,   // hdim_q
                                       options.hdim_v,   // hdim_v
                                       options.scale,
                                       get_stride(q_shape, 1 + options.i_perm),  // stride_q
                                       get_stride(k_shape, 1 + options.i_perm),  // stride_k
                                       get_stride(v_shape, 1 + options.i_perm),  // stride_v
                                       get_stride(o_shape, 1 + options.o_perm),  // stride_o
                                       get_stride(q_shape, 1 + !options.i_perm), // nhead_stride_q
                                       get_stride(k_shape, 1 + !options.i_perm), // nhead_stride_k
                                       get_stride(v_shape, 1 + !options.i_perm), // nhead_stride_v
                                       get_stride(o_shape, 1 + !options.o_perm), // nhead_stride_o
                                       get_stride(q_shape, 0),                   // batch_stride_q
                                       get_stride(k_shape, 0),                   // batch_stride_k
                                       get_stride(v_shape, 0),                   // batch_stride_v
                                       get_stride(o_shape, 0));                  // batch_stride_o

    float ave_time = launch_kernel<kBlockSize.x, kBlockPerCu>(StreamConfig{nullptr, true},
                                                              FmhaKernel{},
                                                              kGridSize,
                                                              kBlockSize,
                                                              0,
                                                              kargs); // BatchStrideO

    std::size_t flop = std::size_t(2) * options.batch * options.nhead * options.seqlen_q *
                           options.seqlen_k * options.hdim_q +
                       std::size_t(2) * options.batch * options.nhead * options.seqlen_q *
                           options.hdim_v * options.seqlen_k;

    std::size_t num_btype =
        sizeof(QDataType) * options.batch * options.nhead * options.seqlen_q * options.hdim_q +
        sizeof(KDataType) * options.batch * options.nhead * options.seqlen_k * options.hdim_q +
        sizeof(VDataType) * options.batch * options.nhead * options.hdim_v * options.seqlen_k +
        sizeof(ODataType) * options.batch * options.nhead * options.seqlen_q * options.hdim_v;

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s"
              << std::endl;

    if(!options.do_validation)
    {
        return EXIT_SUCCESS;
    }

    o_buf.FromDevice(o_host.mData.data());

    for(ck::index_t b = 0; b < options.batch; ++b)
    {
        for(ck::index_t h = 0; h < options.nhead; ++h)
        {
            Tensor<QDataType> q_host_ref({options.seqlen_q, options.hdim_q});
            Tensor<KDataType> k_host_ref({options.seqlen_k, options.hdim_q});
            Tensor<VDataType> v_host_ref({options.hdim_v, options.seqlen_k});
            Tensor<ODataType> o_host_ref({options.seqlen_q, options.hdim_v});

            Tensor<SMPLComputeDataType> s_host_ref({options.seqlen_q, options.seqlen_k});
            Tensor<PDataType> p_host_ref({options.seqlen_q, options.seqlen_k});

            // clang-format off
            // permute
            if(options.i_perm) q_host_ref.ForEach([&](auto& self, auto idx) { self(idx) = q_host(b, h, idx[0], idx[1]); });
            else               q_host_ref.ForEach([&](auto& self, auto idx) { self(idx) = q_host(b, idx[0], h, idx[1]); });

            if(options.i_perm) k_host_ref.ForEach([&](auto& self, auto idx) { self(idx) = k_host(b, h, idx[0], idx[1]); });
            else               k_host_ref.ForEach([&](auto& self, auto idx) { self(idx) = k_host(b, idx[0], h, idx[1]); });

            if(options.i_perm) v_host_ref.ForEach([&](auto& self, auto idx) { self(idx) = v_host(b, h, idx[0], idx[1]); });
            else               v_host_ref.ForEach([&](auto& self, auto idx) { self(idx) = v_host(b, idx[0], h, idx[1]); });

            // reference
            reference_gemm<QDataType, KDataType, SaccDataType, SMPLComputeDataType>(
                q_host_ref, k_host_ref, s_host_ref,
                ck::identity{}, ck::identity{},
                [scale=options.scale](const SaccDataType& x) { return scale * x; });
            reference_softmax<SMPLComputeDataType, SMPLComputeDataType, PDataType>(s_host_ref,
                                                                                   p_host_ref);
            reference_gemm<PDataType, VDataType, OaccDataType, ODataType>(
                p_host_ref, v_host_ref, o_host_ref);

            Tensor<ODataType> o_host_per_head({options.seqlen_q, options.hdim_v});
            // permute
            if(options.o_perm) o_host_per_head.ForEach([&](auto& self, auto idx) { self(idx) = o_host(b, h, idx[0], idx[1]); });
            else               o_host_per_head.ForEach([&](auto& self, auto idx) { self(idx) = o_host(b, idx[0], h, idx[1]); });
            // clang-format on

            if(!ck::utils::check_err(o_host_per_head, o_host_ref))
            {
                return EXIT_FAILURE;
            }
        }
    }
}
