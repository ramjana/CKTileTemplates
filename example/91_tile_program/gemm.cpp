#include <cstring>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/cluster_descriptor.hpp"
#include "ck/tensor/tensor_view.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/host_utility/device_prop.hpp"

#include "tile_program.hpp"
#include "ck/tile_program/meta_data_buffer.hpp"
#include "ck/tile_program/tile_distribution.hpp"
#include "ck/tile_program/tile_window.hpp"
#include "ck/tile_program/static_distributed_tensor.hpp"
#include "ck/tile_program/load_tile.hpp"
#include "ck/tile_program/store_tile.hpp"
#include "ck/tile_program/block_gemm_impl_cr_as_bs.hpp"
#include "ck/tile_program/tile_elementwise.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/fill.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"

#include "gemm_impl_naive_pipeline.hpp"
#include "gemm_impl_better_pipeline.hpp"
#include "gemm_impl_lds_allocator.hpp"
#include "gemm_impl_dram_to_lds_loader.hpp"

template <typename ADataType, typename BDataType, typename CDataType, typename AccDataType>
void reference_gemm(const Tensor<ADataType>& a_m_k,
                    const Tensor<BDataType>& b_n_k,
                    Tensor<CDataType>& c_m_n)
{
    auto f_mk_kn_mn = [&](auto m, auto n) {
        const int K = a_m_k.mDesc.GetLengths()[1];

        AccDataType v_acc = 0;

        for(int k = 0; k < K; ++k)
        {
            ADataType v_a = a_m_k(m, k);
            BDataType v_b = b_n_k(n, k);

            v_acc += ck::type_convert<AccDataType>(v_a) * ck::type_convert<AccDataType>(v_b);
        }

        c_m_n(m, n) = ck::type_convert<CDataType>(v_acc);
    };

    make_ParallelTensorFunctor(f_mk_kn_mn,
                               c_m_n.mDesc.GetLengths()[0],
                               c_m_n.mDesc.GetLengths()[1])(std::thread::hardware_concurrency());
}

int main(int argc, char* argv[])
{
    using ADataType = ck::half_t;
    using BDataType = ck::half_t;
    using CDataType = ck::half_t;

    ck::index_t M = 3328;
    ck::index_t N = 4096;
    ck::index_t K = 4096;

    if(argc == 4)
    {
        M = std::stoi(argv[1]);
        N = std::stoi(argv[2]);
        K = std::stoi(argv[3]);
    }

    std::array<ck::index_t, 2> a_lengths{M, K};
    std::array<ck::index_t, 2> a_strides{K, 1};

    std::array<ck::index_t, 2> b_lengths{N, K};
    std::array<ck::index_t, 2> b_strides{K, 1};

    std::array<ck::index_t, 2> c_lengths{M, N};
    std::array<ck::index_t, 2> c_strides{N, 1};

    // host verify
    Tensor<ADataType> a_host(a_lengths, a_strides);
    Tensor<BDataType> b_host(b_lengths, b_strides);
    Tensor<CDataType> c_host_ref(c_lengths, c_strides);
    Tensor<CDataType> c_host_dev(c_lengths, c_strides);

    ck::utils::FillUniformDistributionIntegerValue<ADataType>{-5.f, 5.f}(a_host);
    ck::utils::FillUniformDistributionIntegerValue<BDataType>{-5.f, 5.f}(b_host);

    // reference gemm
    reference_gemm<ADataType, ADataType, CDataType, float>(a_host, b_host, c_host_ref);

    DeviceMem a_buf(sizeof(ADataType) * a_host.GetElementSpaceSize());
    DeviceMem b_buf(sizeof(BDataType) * b_host.GetElementSpaceSize());
    DeviceMem c_buf(sizeof(CDataType) * c_host_dev.GetElementSpaceSize());

    a_buf.ToDevice(a_host.mData.data());
    b_buf.ToDevice(b_host.mData.data());

    constexpr ck::index_t kGemmMPerBlock = 256;
    constexpr ck::index_t kGemmNPerBlock = 128;
    constexpr ck::index_t kGemmKPerBlock = 32;

    constexpr ck::index_t kBlockSize = 256;
    ck::index_t kGridSize            = (M / kGemmMPerBlock) * (N / kGemmNPerBlock);

    std::cout << "grid size " << kGridSize << std::endl;

#if 1
    using LdsAllocator = LdsAllocator2d<ADataType,
                                        BDataType,
                                        CDataType,
                                        kBlockSize,
                                        kGemmMPerBlock,
                                        kGemmNPerBlock,
                                        kGemmKPerBlock>;
#elif 0
    using LdsAllocator     = LdsAllocator3dPad<ADataType,
                                           BDataType,
                                           CDataType,
                                           kBlockSize,
                                           kGemmMPerBlock,
                                           kGemmNPerBlock,
                                           kGemmKPerBlock>;
#elif 1
    using LdsAllocator = LdsAllocatorXor<ADataType,
                                         BDataType,
                                         CDataType,
                                         kBlockSize,
                                         kGemmMPerBlock,
                                         kGemmNPerBlock,
                                         kGemmKPerBlock>;
#endif

#if 1
    using Dram2LdsLoader = NaiveDram2LdsLoader<ADataType,
                                               BDataType,
                                               CDataType,
                                               kBlockSize,
                                               kGemmMPerBlock,
                                               kGemmNPerBlock,
                                               kGemmKPerBlock>;
#else
    using Dram2LdsLoader   = BetterDram2LdsLoader<ADataType,
                                                BDataType,
                                                CDataType,
                                                kBlockSize,
                                                kGemmMPerBlock,
                                                kGemmNPerBlock,
                                                kGemmKPerBlock>;
#endif

#if 1
    const auto gemm_kernel = GemmNaivePipeline<ADataType,
                                               BDataType,
                                               CDataType,
                                               ck::tensor_layout::gemm::RowMajor,
                                               ck::tensor_layout::gemm::ColumnMajor,
                                               ck::tensor_layout::gemm::RowMajor,
                                               ck::tensor_operation::element_wise::PassThrough,
                                               ck::tensor_operation::element_wise::PassThrough,
                                               ck::tensor_operation::element_wise::PassThrough,
                                               kBlockSize,
                                               kGemmMPerBlock,
                                               kGemmNPerBlock,
                                               kGemmKPerBlock,
                                               LdsAllocator,
                                               Dram2LdsLoader>{};
#else
    const auto gemm_kernel = GemmBetterPipeline<ADataType,
                                                BDataType,
                                                CDataType,
                                                ck::tensor_layout::gemm::RowMajor,
                                                ck::tensor_layout::gemm::ColumnMajor,
                                                ck::tensor_layout::gemm::RowMajor,
                                                ck::tensor_operation::element_wise::PassThrough,
                                                ck::tensor_operation::element_wise::PassThrough,
                                                ck::tensor_operation::element_wise::PassThrough,
                                                kBlockSize,
                                                kGemmMPerBlock,
                                                kGemmNPerBlock,
                                                kGemmKPerBlock,
                                                LdsAllocator,
                                                Dram2LdsLoader>{};
#endif

    float ave_time = launch(ProgramServer{},
                            gemm_kernel,
                            kGridSize,
                            kBlockSize,
                            static_cast<ADataType*>(a_buf.GetDeviceBuffer()),
                            static_cast<BDataType*>(b_buf.GetDeviceBuffer()),
                            static_cast<CDataType*>(c_buf.GetDeviceBuffer()),
                            M,
                            N,
                            K,
                            K,
                            K,
                            N,
                            ck::tensor_operation::element_wise::PassThrough{},
                            ck::tensor_operation::element_wise::PassThrough{},
                            ck::tensor_operation::element_wise::PassThrough{});

    c_buf.FromDevice(c_host_dev.mData.data());

    std::size_t flop = std::size_t(2) * M * N * K;
    std::size_t num_btype =
        sizeof(ADataType) * M * K + sizeof(BDataType) * K * N + sizeof(CDataType) * M * N;

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s"
              << std::endl;

    return ck::utils::check_err(c_host_dev, c_host_ref);
}
