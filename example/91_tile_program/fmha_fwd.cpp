#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <numeric>
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

#include "reference_batched_gemm.hpp"
#include "reference_batched_softmax.hpp"
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

static constexpr ck::index_t seqlen_alignment = 128;

enum class Mode : unsigned
{
    Batch,
    Group
};

inline std::ostream& operator<<(std::ostream& stream, Mode mode)
{
    return stream << (mode == Mode::Batch ? "batch" : "group");
}

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

        if(scale == .0f)
            scale = 1.0 / ck::math::sqrt(static_cast<float>(hdim_q)); // TODO: q ? v ?

        // group mode is only available if i_perm=false & o_perm=false
        if(mode == Mode::Group && !(!i_perm && !o_perm))
        {
            mode = Mode::Batch;
        }

        if(seqlen_q % seqlen_alignment || seqlen_k % seqlen_alignment)
        {
            return false;
        }

        return true;
    }

    ck::index_t shape_batch() const noexcept { return mode == Mode::Batch ? original_batch : 1; }

    ck::index_t work_batch() const noexcept { return original_batch; }
};

template <std::size_t Dim>
using TensorShape = std::array<ck::index_t, Dim>;

template <std::size_t Dim>
std::ostream& operator<<(std::ostream& stream, const TensorShape<Dim>& shape)
{
    stream << "[";
    if(!shape.empty())
    {
        stream << shape[0];
        for(std::size_t idx = 1; idx < shape.size(); ++idx)
        {
            stream << ", " << shape[idx];
        }
    }
    return stream << "]";
}

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
        std::cerr << "get invalid command line arguments" << std::endl;
        return EXIT_FAILURE;
    }

    // accumulation numbers for performance evaluation
    std::size_t flop = 0, num_byte = 0;

    // decide tensor size & prepare group mode kernel arguments
    ck::index_t num_elements_q = 0;
    ck::index_t num_elements_k = 0;
    ck::index_t num_elements_v = 0;
    ck::index_t num_elements_o = 0;

    std::vector<ck::index_t> seqstart_q_host;
    std::vector<ck::index_t> seqstart_k_host;
    {
        ck::index_t next_seqstart_q = 0;
        ck::index_t next_seqstart_k = 0;

        seqstart_q_host.push_back(next_seqstart_q);
        seqstart_k_host.push_back(next_seqstart_k);

        std::mt19937 random_engine(0);
        std::uniform_int_distribution<ck::index_t> gen_seqlen_q_factor(
            1, options.seqlen_q / seqlen_alignment);
        std::uniform_int_distribution<ck::index_t> gen_seqlen_k_factor(
            1, options.seqlen_k / seqlen_alignment);

        for(ck::index_t b = 0; b < options.work_batch(); ++b)
        {
            const auto [real_seqlen_q, real_seqlen_k] = [&]() {
                if(options.mode == Mode::Batch)
                {
                    return std::make_tuple(options.seqlen_q, options.seqlen_k);
                }
                else
                {
                    ck::index_t next_seqlen_q =
                        gen_seqlen_q_factor(random_engine) * seqlen_alignment;

                    // only randomize seqlen_k if it was set to a different value than seqlen_q
                    // originally
                    if(options.seqlen_q == options.seqlen_k)
                    {
                        return std::make_tuple(next_seqlen_q, options.seqlen_k);
                    }
                    else
                    {
                        ck::index_t next_seqlen_k =
                            gen_seqlen_k_factor(random_engine) * seqlen_alignment;

                        return std::make_tuple(next_seqlen_q, next_seqlen_k);
                    }
                }
            }();

            next_seqstart_q += real_seqlen_q;
            next_seqstart_k += real_seqlen_k;

            seqstart_q_host.push_back(next_seqstart_q);
            seqstart_k_host.push_back(next_seqstart_k);

            num_elements_q += (options.nhead * real_seqlen_q * options.hdim_q);
            num_elements_k += (options.nhead * real_seqlen_k * options.hdim_q);
            num_elements_v += (options.nhead * options.hdim_v * real_seqlen_k);
            num_elements_o += (options.nhead * real_seqlen_q * options.hdim_v);

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

    const auto q_shape = get_shape(
        options.i_perm, options.shape_batch(), options.nhead, shape_seqlen_q, options.hdim_q);
    const auto k_shape = get_shape(
        options.i_perm, options.shape_batch(), options.nhead, shape_seqlen_k, options.hdim_q);
    const auto v_shape = get_shape(
        options.i_perm, options.shape_batch(), options.nhead, options.hdim_v, shape_seqlen_k);
    const auto o_shape = get_shape(
        options.o_perm, options.shape_batch(), options.nhead, shape_seqlen_q, options.hdim_v);

    // host memory for storing all the tensor elements
    std::vector<QDataType> q_block(num_elements_q);
    std::vector<KDataType> k_block(num_elements_k);
    std::vector<VDataType> v_block(num_elements_v);
    std::vector<ODataType> o_block(num_elements_o);

    // intialize tensors
#if 0
    ck::utils::FillUniformDistributionIntegerValue<QDataType>{-2.f, 2.f}(q_block);
    ck::utils::FillUniformDistributionIntegerValue<KDataType>{-2.f, 2.f}(k_block);
    ck::utils::FillUniformDistributionIntegerValue<VDataType>{-2.f, 2.f}(v_block);
#else
    ck::utils::FillUniformDistribution<QDataType>{0.f, 1.f}(q_block);
    ck::utils::FillUniformDistribution<KDataType>{0.f, 1.f}(k_block);
    ck::utils::FillUniformDistribution<VDataType>{-.5f, .5f}(v_block);
#endif

    // view for easy access the tensors
    TensorView<const QDataType> q_host(q_block.data(), q_shape);
    TensorView<const KDataType> k_host(k_block.data(), k_shape);
    TensorView<const VDataType> v_host(v_block.data(), v_shape);
    TensorView<ODataType> o_host(o_block.data(), o_shape);

    DeviceMem q_buf(q_host.GetElementSpaceSizeInBytes());
    DeviceMem k_buf(k_host.GetElementSpaceSizeInBytes());
    DeviceMem v_buf(v_host.GetElementSpaceSizeInBytes());
    DeviceMem o_buf(o_host.GetElementSpaceSizeInBytes());
    DeviceMem seqstart_q(seqstart_q_host.size() * sizeof(ck::index_t));
    DeviceMem seqstart_k(seqstart_k_host.size() * sizeof(ck::index_t));

    q_buf.ToDevice(q_host.mData.data());
    k_buf.ToDevice(k_host.mData.data());
    v_buf.ToDevice(v_host.mData.data());
    seqstart_q.ToDevice(seqstart_q_host.data());
    seqstart_k.ToDevice(seqstart_k_host.data());

    dim3 kGridSize =
        FmhaKernel::GridSize(options.work_batch(), options.nhead, shape_seqlen_q, options.hdim_v);
    constexpr dim3 kBlockSize = FmhaKernel::BlockSize();

    std::cout << "mode:" << options.mode << ", batch:" << options.original_batch
              << ", nhead:" << options.nhead << ", seqlen_q:" << options.seqlen_q
              << ", seqlen_k:" << options.seqlen_k << ", hdim_q:" << options.hdim_q
              << ", hdim_v:" << options.hdim_v << ", scale:" << options.scale
              << ", i_perm:" << std::boolalpha << options.i_perm << ", o_perm:" << std::boolalpha
              << options.o_perm << ", grid_size:" << kGridSize.x << "x" << kGridSize.y << "x"
              << kGridSize.z << std::endl;

    constexpr ck::index_t kWarpPerCu    = 8; // 2 warps per SIMD
    constexpr ck::index_t kWarpPerBlock = kBlockSize.x / warpSize;
    constexpr ck::index_t kBlockPerCu   = kWarpPerCu / kWarpPerBlock;

    // if value of i_perm/o_perm is true, the tensor shape is [batch, nhead, seqlen, hdim];
    // otherwise, shape is [batch, seqlen, nhead, hdim]. which means we are choosing
    // stride/nhead_stride base on the axis of seqlen/nhead (axis=1 or axis=2)
    auto kargs = [&]() {
        const ck::index_t stride_q       = get_stride(q_shape, 1 + options.i_perm);
        const ck::index_t stride_k       = get_stride(k_shape, 1 + options.i_perm);
        const ck::index_t stride_v       = get_stride(v_shape, 1 + options.i_perm);
        const ck::index_t stride_o       = get_stride(o_shape, 1 + options.o_perm);
        const ck::index_t nhead_stride_q = get_stride(q_shape, 1 + !options.i_perm);
        const ck::index_t nhead_stride_k = get_stride(k_shape, 1 + !options.i_perm);
        const ck::index_t nhead_stride_v = get_stride(v_shape, 1 + !options.i_perm);
        const ck::index_t nhead_stride_o = get_stride(o_shape, 1 + !options.o_perm);

        if(options.mode == Mode::Batch)
        {
            return FmhaKernel::MakeKargs(q_buf.GetDeviceBuffer(),
                                         k_buf.GetDeviceBuffer(),
                                         v_buf.GetDeviceBuffer(),
                                         o_buf.GetDeviceBuffer(),
                                         shape_seqlen_q,
                                         shape_seqlen_k,
                                         options.hdim_q,
                                         options.hdim_v,
                                         options.scale,
                                         stride_q,
                                         stride_k,
                                         stride_v,
                                         stride_o,
                                         nhead_stride_q,
                                         nhead_stride_k,
                                         nhead_stride_v,
                                         nhead_stride_o,
                                         get_stride(q_shape, 0),  // batch_stride_q
                                         get_stride(k_shape, 0),  // batch_stride_k
                                         get_stride(v_shape, 0),  // batch_stride_v
                                         get_stride(o_shape, 0)); // batch_stride_o
        }
        else
        {
            return FmhaKernel::MakeKargs(q_buf.GetDeviceBuffer(),
                                         k_buf.GetDeviceBuffer(),
                                         v_buf.GetDeviceBuffer(),
                                         o_buf.GetDeviceBuffer(),
                                         seqstart_q.GetDeviceBuffer(),
                                         seqstart_k.GetDeviceBuffer(),
                                         nullptr,
                                         options.hdim_q,
                                         options.hdim_v,
                                         options.scale,
                                         stride_q,
                                         stride_k,
                                         stride_v,
                                         stride_o,
                                         nhead_stride_q,
                                         nhead_stride_k,
                                         nhead_stride_v,
                                         nhead_stride_o);
        }
    }();

    float ave_time = launch_kernel<kBlockSize.x, kBlockPerCu>(StreamConfig{nullptr, true},
                                                              FmhaKernel{},
                                                              kGridSize,
                                                              kBlockSize,
                                                              0,
                                                              kargs); // BatchStrideO

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_byte / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s"
              << std::endl;

    if(!options.do_validation)
    {
        return EXIT_SUCCESS;
    }

    o_buf.FromDevice(o_host.mData.data());

    for(ck::index_t wb = 0; wb < options.work_batch(); ++wb)
    {
        const ck::index_t real_seqlen_q = seqstart_q_host[wb + 1] - seqstart_q_host[wb];
        const ck::index_t real_seqlen_k = seqstart_k_host[wb + 1] - seqstart_k_host[wb];

        // adjust matrix index according to the mode
        const ck::index_t b            = (options.mode == Mode::Batch ? wb : 0);
        const ck::index_t query_offset = (options.mode == Mode::Batch ? 0 : seqstart_q_host[wb]);
        const ck::index_t key_offset   = (options.mode == Mode::Batch ? 0 : seqstart_k_host[wb]);

        Tensor<QDataType> q_host_ref({options.nhead, real_seqlen_q, options.hdim_q});
        Tensor<KDataType> k_host_ref({options.nhead, real_seqlen_k, options.hdim_q});
        Tensor<VDataType> v_host_ref({options.nhead, options.hdim_v, real_seqlen_k});
        Tensor<ODataType> o_host_ref({options.nhead, real_seqlen_q, options.hdim_v});

        Tensor<SMPLComputeDataType> s_host_ref({options.nhead, real_seqlen_q, real_seqlen_k});
        Tensor<PDataType> p_host_ref({options.nhead, real_seqlen_q, real_seqlen_k});

        // clang-format off
        // permute
        if(options.i_perm) q_host_ref.ForEach([&](auto& self, auto idx) { self(idx) = q_host(b, idx[0], idx[1], idx[2]); });
        else               q_host_ref.ForEach([&](auto& self, auto idx) { self(idx) = q_host(b, idx[1] + query_offset, idx[0], idx[2]); });

        if(options.i_perm) k_host_ref.ForEach([&](auto& self, auto idx) { self(idx) = k_host(b, idx[0], idx[1], idx[2]); });
        else               k_host_ref.ForEach([&](auto& self, auto idx) { self(idx) = k_host(b, idx[1] + key_offset, idx[0], idx[2]); });

        if(options.i_perm) v_host_ref.ForEach([&](auto& self, auto idx) { self(idx) = v_host(b, idx[0], idx[1], idx[2]); });
        else               v_host_ref.ForEach([&](auto& self, auto idx) { self(idx) = v_host(b, idx[1], idx[0], idx[2] + key_offset); });

        // reference
        reference_batched_gemm<QDataType, KDataType, SaccDataType, SMPLComputeDataType>(
            q_host_ref, k_host_ref, s_host_ref,
            ck::identity{}, ck::identity{},
            [scale = options.scale](const SaccDataType& x) { return scale * x; });
        reference_batched_softmax<SMPLComputeDataType, SMPLComputeDataType, PDataType>(s_host_ref, 
                                                                                    p_host_ref);
        reference_batched_gemm<PDataType, VDataType, OaccDataType, ODataType>(
            p_host_ref, v_host_ref, o_host_ref);

        Tensor<ODataType> o_host_result({options.nhead, real_seqlen_q, options.hdim_v});
        // permute
        if(options.o_perm) o_host_result.ForEach([&](auto& self, auto idx) { self(idx) = o_host(b, idx[0], idx[1], idx[2]); });
        else               o_host_result.ForEach([&](auto& self, auto idx) { self(idx) = o_host(b, idx[1] + query_offset, idx[0], idx[2]); });
        // clang-format on

        if(!ck::utils::check_err(o_host_result, o_host_ref))
        {
            return EXIT_FAILURE;
        }
    }
}
