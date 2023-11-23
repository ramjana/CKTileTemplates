#include <array>
#include <cstdlib>
#include <cstring>
#include <cstdlib>
#include <numeric>
#include <optional>
#include <ostream>
#include <random>

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
#include "ck/library/utility/literals.hpp"

#include "ck/tile_program/block_tile_pipeline/block_fmha_pipeline_qkvs.hpp"
#include "ck/tile_program/block_tile_pipeline/block_fmha_pipeline_qkvs_default_policy.hpp"
#include "ck/tile_program/block_tile_pipeline/block_fmha_pipeline_qr_ks_vs.hpp"
#include "ck/tile_program/block_tile_pipeline/block_fmha_pipeline_qr_ks_vs_default_policy.hpp"
#include "ck/tile_program/block_tile_pipeline/block_fmha_pipeline_problem.hpp"
#include "ck/tile_program/tile/tile_fmha_shape.hpp"

#include "reference_batched_elementwise.hpp"
#include "reference_batched_gemm.hpp"
#include "reference_batched_softmax.hpp"
#include "fmha_fwd_kernel.hpp"
#include "fmha_fwd_tile_partitioner.hpp"
#include "fmha_fwd_epilogue.hpp"

using QDataType           = ck::half_t;
using KDataType           = ck::half_t;
using VDataType           = ck::half_t;
using BiasDataType        = ck::half_t;
using SaccDataType        = float;      // data type for first gemm accumulation
using SMPLComputeDataType = float;      // data type for reduction, softmax
using PDataType           = ck::half_t; // data type for A matrix of second gemm
using OaccDataType        = float;      // data type for second gemm accumulation
using ODataType           = ck::half_t;

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
                                                      BiasDataType,
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
                                                      BiasDataType,
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

enum class Mode : unsigned
{
    Batch,
    Group
};

inline std::ostream& operator<<(std::ostream& stream, Mode mode)
{
    return stream << (mode == Mode::Batch ? "batch" : "group");
}

template <typename FmhaKernel>
float invoker_fmha_kernel(Mode mode,
                          const void* q_ptr,
                          const void* k_ptr,
                          const void* v_ptr,
                          const void* bias_ptr,
                          void* o_ptr,
                          const void* seqstart_q_ptr,
                          const void* seqstart_k_ptr,
                          const void* seqlen_k_ptr,
                          ck::index_t problem_count,
                          ck::index_t nhead,
                          ck::index_t seqlen_q,
                          ck::index_t seqlen_k,
                          ck::index_t hdim_q,
                          ck::index_t hdim_v,
                          float scale,
                          bool i_perm,
                          bool o_perm,
                          bool use_bias)
{
    dim3 kGridSize            = FmhaKernel::GridSize(problem_count, nhead, seqlen_q, hdim_v);
    constexpr dim3 kBlockSize = FmhaKernel::BlockSize();

    constexpr ck::index_t kWarpPerCu    = 8; // 2 warps per SIMD
    constexpr ck::index_t kWarpPerBlock = kBlockSize.x / warpSize;
    constexpr ck::index_t kBlockPerCu   = kWarpPerCu / kWarpPerBlock;

    constexpr bool is_v_rowmajor =
        ck::is_same_v<typename FmhaKernel::VLayout, ck::tensor_layout::gemm::RowMajor>;
    /// NOTE: we broadcast bias from [1, 1, seqlen_q, seqlen_k] to [batch, nhead, seqlen_q,
    ///       seqlen_k] in this example, hence both the 'batch_stride_bias' & 'nhead_stride_bias'
    ///       are 0.
    // setup stride_* arguments
    const ck::index_t stride_q = (i_perm ? hdim_q : nhead * hdim_q);
    const ck::index_t stride_k = (i_perm ? hdim_q : nhead * hdim_q);
    const ck::index_t stride_v = [&]() {
        if constexpr(is_v_rowmajor)
            return i_perm ? hdim_v : nhead * hdim_v;
        else
            return i_perm ? seqlen_k : nhead * seqlen_k;
    }();
    const ck::index_t stride_bias = (i_perm ? seqlen_k : 1 * seqlen_k);
    const ck::index_t stride_o    = (o_perm ? hdim_v : nhead * hdim_v);
    // setup nhead_stride_* arguments
    const ck::index_t nhead_stride_q = (i_perm ? seqlen_q * hdim_q : hdim_q);
    const ck::index_t nhead_stride_k = (i_perm ? seqlen_k * hdim_q : hdim_q);
    const ck::index_t nhead_stride_v = [&]() {
        if constexpr(is_v_rowmajor)
            return i_perm ? seqlen_k * hdim_v : hdim_v;
        else
            return i_perm ? hdim_v * seqlen_k : seqlen_k;
    }();
    const ck::index_t nhead_stride_bias = (i_perm ? 0 * seqlen_q * seqlen_k : 0 * seqlen_k);
    const ck::index_t nhead_stride_o    = (o_perm ? seqlen_q * hdim_v : hdim_v);
    // setup batch_stride_* arguments
    const ck::index_t batch_stride_q    = (nhead * seqlen_q * hdim_q);
    const ck::index_t batch_stride_k    = (nhead * seqlen_k * hdim_q);
    const ck::index_t batch_stride_v    = (nhead * hdim_v * seqlen_k);
    const ck::index_t batch_stride_bias = (0 * nhead * seqlen_q * seqlen_k);
    const ck::index_t batch_stride_o    = (nhead * seqlen_q * hdim_v);

    const auto launch_flow =
        [](bool condition, auto true_action, auto false_action, auto launcher) {
            if(condition)
            {
                return launcher(true_action());
            }
            else
            {
                return launcher(false_action());
            }
        };

    return launch_flow(
        mode == Mode::Batch,
        [&] {
            std::optional<std::tuple<const void*, ck::index_t, ck::index_t, ck::index_t>> bias;
            if(use_bias)
            {
                bias = std::make_tuple(bias_ptr, stride_bias, nhead_stride_bias, batch_stride_bias);
            }

            return FmhaKernel::MakeKargs(q_ptr,
                                         k_ptr,
                                         v_ptr,
                                         o_ptr,
                                         seqlen_q,
                                         seqlen_k,
                                         hdim_q,
                                         hdim_v,
                                         scale,
                                         stride_q,
                                         stride_k,
                                         stride_v,
                                         stride_o,
                                         nhead_stride_q,
                                         nhead_stride_k,
                                         nhead_stride_v,
                                         nhead_stride_o,
                                         batch_stride_q,
                                         batch_stride_k,
                                         batch_stride_v,
                                         batch_stride_o,
                                         bias);
        },
        [&] {
            std::optional<std::tuple<const void*, ck::index_t, ck::index_t>> bias;
            if(use_bias)
            {
                bias = std::make_tuple(bias_ptr, stride_bias, nhead_stride_bias);
            }

            return FmhaKernel::MakeKargs(q_ptr,
                                         k_ptr,
                                         v_ptr,
                                         o_ptr,
                                         seqstart_q_ptr,
                                         seqstart_k_ptr,
                                         seqlen_k_ptr,
                                         hdim_q,
                                         hdim_v,
                                         scale,
                                         stride_q,
                                         stride_k,
                                         stride_v,
                                         stride_o,
                                         nhead_stride_q,
                                         nhead_stride_k,
                                         nhead_stride_v,
                                         nhead_stride_o,
                                         bias);
        },
        [&](auto kargs) {
            return launch_kernel<kBlockSize.x, kBlockPerCu>(StreamConfig{nullptr, true},
                                                            FmhaKernel{},
                                                            kGridSize,
                                                            kBlockSize,
                                                            0,
                                                            kargs); // BatchStrideO
        });
}

static constexpr ck::index_t seqlen_alignment = 128;

struct Options
{
    bool do_validation         = true;
    Mode mode                  = Mode::Batch;
    ck::index_t original_batch = 2;
    ck::index_t nhead          = 8;
    ck::index_t seqlen_q       = 3328;
    ck::index_t seqlen_k       = 4096;
    ck::index_t hdim_q         = 128;
    ck::index_t hdim_v         = 128;

    float scale = .0f;

    // for following flag values, if their value is true, the shape input/output tensor will be
    // [batch, nhead, seqlen, hdim]; otherwise: [batch, seqlen, nhead, hdim]
    bool i_perm = true;
    bool o_perm = true;

    bool use_bias = false;

    bool parse(int argc, char* argv[])
    {
        if(argc >= 2)
            do_validation = static_cast<bool>(std::stoi(argv[1]));
        if(argc >= 3)
            mode = static_cast<Mode>(std::min(1ul, std::max(0ul, std::stoul(argv[2]))));

        if(argc >= 9)
        {
            original_batch = std::stoi(argv[3]);
            nhead          = std::stoi(argv[4]);
            seqlen_q       = std::stoi(argv[5]);
            seqlen_k       = std::stoi(argv[6]);
            hdim_q         = std::stoi(argv[7]);
            hdim_v         = std::stoi(argv[8]);
        }
        if(argc >= 10)
            scale = std::stof(argv[9]);
        if(argc >= 11)
            i_perm = static_cast<bool>(std::stoi(argv[10]));
        if(argc >= 12)
            o_perm = static_cast<bool>(std::stoi(argv[11]));
        if(argc >= 13)
            use_bias = static_cast<bool>(std::stoi(argv[12]));

        if(scale == .0f)
            scale = 1.0 / ck::math::sqrt(static_cast<float>(hdim_q)); // TODO: q ? v ?

        return validate();
    }

    bool validate()
    {
        if(seqlen_q % seqlen_alignment || seqlen_k % seqlen_alignment)
        {
            return false;
        }

        return true;
    }

    ck::index_t batch() const noexcept { return mode == Mode::Batch ? original_batch : 1; }

    ck::index_t problem_count() const noexcept { return original_batch; }
};

std::array<ck::index_t, 4> get_lengths(bool permute,
                                       ck::index_t b /*batch*/,
                                       ck::index_t h /*nhead*/,
                                       ck::index_t s /*seqlen*/,
                                       ck::index_t d /*hdim*/)
{
    if(permute)
        return {b, h, s, d};
    else
        return {b, s, h, d};
}

int main(int argc, char* argv[])
{
    Options options;
    if(!options.parse(argc, argv))
    {
        std::cerr << "get invalid command line arguments" << std::endl;
        return EXIT_FAILURE;
    }

    // accumulation numbers for performance evaluation
    std::size_t flop = 0, num_byte = 0;

    std::vector<int32_t> seqstart_q_host;
    std::vector<int32_t> seqstart_k_host;
    {
        int32_t next_seqstart_q = 0;
        int32_t next_seqstart_k = 0;

        seqstart_q_host.push_back(next_seqstart_q);
        seqstart_k_host.push_back(next_seqstart_k);

        std::mt19937 random_engine(0);
        std::uniform_int_distribution<int32_t> gen_seqlen_q_factor(
            1, options.seqlen_q / seqlen_alignment);
        std::uniform_int_distribution<int32_t> gen_seqlen_k_factor(
            1, options.seqlen_k / seqlen_alignment);

        for(ck::index_t p = 0; p < options.problem_count(); ++p)
        {
            const auto [real_seqlen_q, real_seqlen_k] = [&]() {
                if(options.mode == Mode::Batch)
                {
                    return std::make_tuple(options.seqlen_q, options.seqlen_k);
                }
                else
                {
                    int32_t next_seqlen_q = gen_seqlen_q_factor(random_engine) * seqlen_alignment;

                    // only randomize seqlen_k if it was set to a different value than seqlen_q
                    // originally
                    if(options.seqlen_q == options.seqlen_k)
                    {
                        return std::make_tuple(next_seqlen_q, options.seqlen_k);
                    }
                    else
                    {
                        int32_t next_seqlen_k =
                            gen_seqlen_k_factor(random_engine) * seqlen_alignment;

                        return std::make_tuple(next_seqlen_q, next_seqlen_k);
                    }
                }
            }();

            next_seqstart_q += real_seqlen_q;
            next_seqstart_k += real_seqlen_k;

            seqstart_q_host.push_back(next_seqstart_q);
            seqstart_k_host.push_back(next_seqstart_k);

            using namespace ck::literals;

            flop += options.nhead * (2_uz * real_seqlen_q * real_seqlen_k * options.hdim_q +
                                     2_uz * real_seqlen_q * options.hdim_v * real_seqlen_k);

            num_byte += options.nhead * (sizeof(QDataType) * real_seqlen_q * options.hdim_q +
                                         sizeof(KDataType) * real_seqlen_k * options.hdim_q +
                                         sizeof(VDataType) * options.hdim_v * real_seqlen_k +
                                         sizeof(ODataType) * real_seqlen_q * options.hdim_v);
        }
    }

    const ck::index_t shape_seqlen_q =
        (options.mode == Mode::Batch ? options.seqlen_q : seqstart_q_host.back());
    const ck::index_t shape_seqlen_k =
        (options.mode == Mode::Batch ? options.seqlen_k : seqstart_k_host.back());

    constexpr bool is_v_rowmajor =
        ck::is_same_v<typename FmhaKernelHDim64::VLayout, ck::tensor_layout::gemm::RowMajor>;

    // host memory for storing all the tensor elements
    Tensor<QDataType> q_host(get_lengths(
        options.i_perm, options.batch(), options.nhead, shape_seqlen_q, options.hdim_q));
    Tensor<KDataType> k_host(get_lengths(
        options.i_perm, options.batch(), options.nhead, shape_seqlen_k, options.hdim_q));
    Tensor<VDataType> v_host(
        is_v_rowmajor
            ? get_lengths(
                  options.i_perm, options.batch(), options.nhead, shape_seqlen_k, options.hdim_v)
            : get_lengths(
                  options.i_perm, options.batch(), options.nhead, options.hdim_v, shape_seqlen_k));
    // use bias shape = [1, 1, shape_seqlen_q, shape_seqlen_k]. if use_bias=false, the bias_host
    // will not be used for verification at all (but will be copied to device anyway).
    Tensor<KDataType> bias_host(
        options.use_bias
            ? get_lengths(options.i_perm, 1, 1, shape_seqlen_q, shape_seqlen_k)
            : std::array<ck::index_t, 4>{1, 1, 1, 1} /* dummy shape for simplifying code */);
    Tensor<ODataType> o_host(get_lengths(
        options.o_perm, options.batch(), options.nhead, shape_seqlen_q, options.hdim_v));

    // intialize tensors
#if 0
    ck::utils::FillUniformDistributionIntegerValue<QDataType>{-2.f, 2.f}(q_host);
    ck::utils::FillUniformDistributionIntegerValue<KDataType>{-2.f, 2.f}(k_host);
    ck::utils::FillUniformDistributionIntegerValue<VDataType>{-2.f, 2.f}(v_host);
    ck::utils::FillUniformDistributionIntegerValue<BiasDataType>{-2.f, 2.f}(bias_host);
#else
    ck::utils::FillUniformDistribution<QDataType>{0.f, 1.f}(q_host);
    ck::utils::FillUniformDistribution<KDataType>{0.f, 1.f}(k_host);
    ck::utils::FillUniformDistribution<VDataType>{-.5f, .5f}(v_host);
    ck::utils::FillUniformDistribution<BiasDataType>{0.f, 1.f}(bias_host);
#endif

    DeviceMem q_buf(q_host.GetElementSpaceSizeInBytes());
    DeviceMem k_buf(k_host.GetElementSpaceSizeInBytes());
    DeviceMem v_buf(v_host.GetElementSpaceSizeInBytes());
    DeviceMem bias_buf(bias_host.GetElementSpaceSizeInBytes());
    DeviceMem o_buf(o_host.GetElementSpaceSizeInBytes());
    DeviceMem seqstart_q(seqstart_q_host.size() * sizeof(int32_t));
    DeviceMem seqstart_k(seqstart_k_host.size() * sizeof(int32_t));

    q_buf.ToDevice(q_host.data());
    k_buf.ToDevice(k_host.data());
    v_buf.ToDevice(v_host.data());
    bias_buf.ToDevice(bias_host.data());
    seqstart_q.ToDevice(seqstart_q_host.data());
    seqstart_k.ToDevice(seqstart_k_host.data());

    std::cout << "mode:" << options.mode << ", batch:" << options.original_batch
              << ", nhead:" << options.nhead << ", seqlen_q:" << options.seqlen_q
              << ", seqlen_k:" << options.seqlen_k << ", hdim_q:" << options.hdim_q
              << ", hdim_v:" << options.hdim_v << ", scale:" << options.scale
              << ", i_perm:" << options.i_perm << ", o_perm:" << options.o_perm
              << ", use_bias:" << options.use_bias
              << ", v:" << std::string(FmhaKernelHDim64::VLayout::name) << std::endl;

    float ave_time = 0;
    if(options.hdim_q == options.hdim_v && options.hdim_q == 64)
        ave_time = invoker_fmha_kernel<FmhaKernelHDim64>(options.mode,
                                                         q_buf.GetDeviceBuffer(),
                                                         k_buf.GetDeviceBuffer(),
                                                         v_buf.GetDeviceBuffer(),
                                                         bias_buf.GetDeviceBuffer(),
                                                         o_buf.GetDeviceBuffer(),
                                                         seqstart_q.GetDeviceBuffer(),
                                                         seqstart_k.GetDeviceBuffer(),
                                                         nullptr,
                                                         options.problem_count(),
                                                         options.nhead,
                                                         shape_seqlen_q,
                                                         shape_seqlen_k,
                                                         options.hdim_q,
                                                         options.hdim_v,
                                                         options.scale,
                                                         options.i_perm,
                                                         options.o_perm,
                                                         options.use_bias);
    else if(options.hdim_q == options.hdim_v && options.hdim_q == 128)
        ave_time = invoker_fmha_kernel<FmhaKernelHDim128>(options.mode,
                                                          q_buf.GetDeviceBuffer(),
                                                          k_buf.GetDeviceBuffer(),
                                                          v_buf.GetDeviceBuffer(),
                                                          bias_buf.GetDeviceBuffer(),
                                                          o_buf.GetDeviceBuffer(),
                                                          seqstart_q.GetDeviceBuffer(),
                                                          seqstart_k.GetDeviceBuffer(),
                                                          nullptr,
                                                          options.problem_count(),
                                                          options.nhead,
                                                          shape_seqlen_q,
                                                          shape_seqlen_k,
                                                          options.hdim_q,
                                                          options.hdim_v,
                                                          options.scale,
                                                          options.i_perm,
                                                          options.o_perm,
                                                          options.use_bias);
    else
    {
        std::cerr << "not support hdim, will not run" << std::endl;
        return EXIT_FAILURE;
    }

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_byte / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s"
              << std::endl;

    if(!options.do_validation)
    {
        return EXIT_SUCCESS;
    }

    o_buf.FromDevice(o_host.data());

    for(ck::index_t p = 0; p < options.problem_count(); ++p)
    {
        const ck::index_t real_seqlen_q = seqstart_q_host[p + 1] - seqstart_q_host[p];
        const ck::index_t real_seqlen_k = seqstart_k_host[p + 1] - seqstart_k_host[p];

        // adjust matrix index according to the mode
        const ck::index_t b            = (options.mode == Mode::Batch ? p : 0);
        const ck::index_t query_offset = (options.mode == Mode::Batch ? 0 : seqstart_q_host[p]);
        const ck::index_t key_offset   = (options.mode == Mode::Batch ? 0 : seqstart_k_host[p]);

        const auto v_host_ref_lengths =
            std::array<ck::index_t, 3>{options.nhead, options.hdim_v, real_seqlen_k};
        const auto v_host_ref_strides =
            is_v_rowmajor
                ? std::array<ck::index_t, 3>{options.hdim_v * real_seqlen_k, 1, options.hdim_v}
                : std::array<ck::index_t, 3>{options.hdim_v * real_seqlen_k, real_seqlen_k, 1};

        Tensor<QDataType> q_host_ref({options.nhead, real_seqlen_q, options.hdim_q});
        Tensor<KDataType> k_host_ref({options.nhead, real_seqlen_k, options.hdim_q});
        Tensor<VDataType> v_host_ref(v_host_ref_lengths, v_host_ref_strides);
        Tensor<ODataType> o_host_ref({options.nhead, real_seqlen_q, options.hdim_v});

        Tensor<SMPLComputeDataType> s_host_ref({options.nhead, real_seqlen_q, real_seqlen_k});
        Tensor<PDataType> p_host_ref({options.nhead, real_seqlen_q, real_seqlen_k});

        // clang-format off
        // permute
        if(options.i_perm) q_host_ref.ForEach([&](auto& self, auto idx) { self(idx) = q_host(b, idx[0], idx[1] + query_offset, idx[2]); });
        else               q_host_ref.ForEach([&](auto& self, auto idx) { self(idx) = q_host(b, idx[1] + query_offset, idx[0], idx[2]); });

        if(options.i_perm) k_host_ref.ForEach([&](auto& self, auto idx) { self(idx) = k_host(b, idx[0], idx[1] + key_offset, idx[2]); });
        else               k_host_ref.ForEach([&](auto& self, auto idx) { self(idx) = k_host(b, idx[1] + key_offset, idx[0], idx[2]); });

        if constexpr (is_v_rowmajor) {
            //                                                                v_host_ref: [nhead, hdim, seq], v_host: [b, h, s, d] 
            if(options.i_perm) v_host_ref.ForEach([&](auto& self, auto idx) { self(idx) = v_host(b, idx[0], idx[2] + key_offset, idx[1]); });
            //                                                                v_host_ref: [nhead, hdim, seq], v_host: [b, s, h, d]
            else               v_host_ref.ForEach([&](auto& self, auto idx) { self(idx) = v_host(b, idx[2] + key_offset, idx[0], idx[1]); });
        }
        else {
            if(options.i_perm) v_host_ref.ForEach([&](auto& self, auto idx) { self(idx) = v_host(b, idx[0], idx[1], idx[2] + key_offset); });
            else               v_host_ref.ForEach([&](auto& self, auto idx) { self(idx) = v_host(b, idx[1], idx[0], idx[2] + key_offset); });
        }

        // reference
        reference_batched_gemm<QDataType, KDataType, SaccDataType, SMPLComputeDataType>(
            q_host_ref, k_host_ref, s_host_ref,
            ck::identity{}, ck::identity{},
            [scale = options.scale](SaccDataType x) { return scale * x; });
        if(options.use_bias)
        {
            Tensor<BiasDataType> bias_host_ref({1, real_seqlen_q, real_seqlen_k});
            if(options.i_perm)
                bias_host_ref.ForEach([&](auto& self, auto idx) { self(idx) = bias_host(0, 0, idx[1] + query_offset, idx[2] + key_offset); });
            else
                bias_host_ref.ForEach([&](auto& self, auto idx) { self(idx) = bias_host(0, idx[1] + query_offset, 0, idx[2] + key_offset); });
            
            // broadcast from [1, real_seqlen_q, real_seqlen_k] to [nhead, real_seqlen_q, real_seqlen_k]
            reference_batched_elementwise<SMPLComputeDataType, BiasDataType, SMPLComputeDataType, SMPLComputeDataType>(
                s_host_ref, bias_host_ref, s_host_ref);
        }
        reference_batched_softmax<SMPLComputeDataType, SMPLComputeDataType, PDataType>(s_host_ref,
                                                                                       p_host_ref);
        reference_batched_gemm<PDataType, VDataType, OaccDataType, ODataType>(
            p_host_ref, v_host_ref, o_host_ref);

        Tensor<ODataType> o_host_result({options.nhead, real_seqlen_q, options.hdim_v});
        // permute
        if(options.o_perm) o_host_result.ForEach([&](auto& self, auto idx) { self(idx) = o_host(b, idx[0], idx[1] + query_offset, idx[2]); });
        else               o_host_result.ForEach([&](auto& self, auto idx) { self(idx) = o_host(b, idx[1] + query_offset, idx[0], idx[2]); });
        // clang-format on

        if(!ck::utils::check_err(o_host_result, o_host_ref))
        {
            std::cerr << "mismatch found at batch: " << p << std::endl;
            return EXIT_FAILURE;
        }
    }
}
