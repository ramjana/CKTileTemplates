#include <cstring>

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

#include "reference_gemm.hpp"
#include "gemm.hpp"

// elementwise lambda
struct AElementFunction
{
    template <typename X>
    __host__ __device__ auto operator()(const X& x) const
    {
        return x;
    }
};

struct BElementFunction
{
    template <typename X>
    __host__ __device__ auto operator()(const X& x) const
    {
        return x;
    }
};

struct CElementFunction
{
    template <typename X>
    __host__ __device__ auto operator()(const X& x) const
    {
        return x;
    }
};

int main(int argc, char* argv[])
{
    using ADataType   = ck::half_t;
    using BDataType   = ck::half_t;
    using AccDataType = float;
    using CDataType   = ck::half_t;

    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    ck::index_t do_debug = 0;
    ck::index_t do_verification = 1;
    ck::index_t time_kernel = 1;
    ck::index_t initial_method =1;

    ck::index_t M = 3328;
    ck::index_t N = 4096;
    ck::index_t K = 4096;

   if(argc == 4){
        do_verification = std::stoi(argv[1]);
        time_kernel = std::stoi(argv[2]);
        initial_method = std::stoi(argv[3]);
    }
    else if(argc == 8)
    {
        do_verification = std::stoi(argv[1]);
        time_kernel = std::stoi(argv[2]);
        initial_method = std::stoi(argv[3]);
        M = std::stoi(argv[4]);
        N = std::stoi(argv[5]);
        K = std::stoi(argv[6]);
        do_debug = std::stoi(argv[7]);
    }

    const ck::index_t Lda = std::is_same_v<ALayout, ck::tensor_layout::gemm::RowMajor> ? K : M;
    const ck::index_t Ldb = std::is_same_v<BLayout, ck::tensor_layout::gemm::ColumnMajor> ? K : N;
    const ck::index_t Ldc = std::is_same_v<CLayout, ck::tensor_layout::gemm::RowMajor> ? N : M;

    const auto a_lengths = std::array<ck::index_t, 2>{M, K};
    const auto a_strides = std::is_same_v<ALayout, ck::tensor_layout::gemm::RowMajor>
                               ? std::array<ck::index_t, 2>{Lda, 1}
                               : std::array<ck::index_t, 2>{1, Lda};

    const auto b_lengths = std::array<ck::index_t, 2>{N, K};
    const auto b_strides = std::is_same_v<BLayout, ck::tensor_layout::gemm::ColumnMajor>
                               ? std::array<ck::index_t, 2>{Ldb, 1}
                               : std::array<ck::index_t, 2>{1, Ldb};

    const auto c_lengths = std::array<ck::index_t, 2>{M, N};
    const auto c_strides = std::is_same_v<CLayout, ck::tensor_layout::gemm::RowMajor>
                               ? std::array<ck::index_t, 2>{Ldc, 1}
                               : std::array<ck::index_t, 2>{1, Ldc};

    // host verify
    Tensor<ADataType> a_host(a_lengths, a_strides);
    Tensor<BDataType> b_host(b_lengths, b_strides);
    Tensor<CDataType> c_host_dev(c_lengths, c_strides);

    switch(initial_method)
    {
    case 0: break;
    case 1:
        printf("Initialization, A: RandomInt ; B: RandomInt\n");
        ck::utils::FillUniformDistributionIntegerValue<ADataType>{-3.f, 3.f}(a_host);
        ck::utils::FillUniformDistributionIntegerValue<BDataType>{-3.f, 3.f}(b_host);
        break;
    case 2:
        printf("Initialization, A: RandomFloat ; B: RandomFloat\n");
        ck::utils::FillUniformDistribution<ADataType>{-3.f, 3.f}(a_host);
        ck::utils::FillUniformDistribution<BDataType>{-3.f, 3.f}(b_host);
        break;
    case 3:
        printf("Initialization, A: Constant ; B: Constant\n");
        ck::utils::FillConstant<ADataType>{1.f}(a_host);
        ck::utils::FillConstant<BDataType>{1.f}(b_host);
        break;
    case 4:
        printf("Initialization, A: RandomInt ; B: Constant\n");
        ck::utils::FillUniformDistributionIntegerValue<ADataType>{-3.f, 3.f}(a_host);
        ck::utils::FillConstant<BDataType>{1.f}(b_host);
        break;
    case 5:
        printf("Initialization, A: Constant ; B: RandomInt\n");
        ck::utils::FillConstant<ADataType>{1.f}(a_host);
        ck::utils::FillUniformDistributionIntegerValue<BDataType>{-3.f, 3.f}(b_host);
        break;
    default:
        ck::utils::FillUniformDistributionIntegerValue<ADataType>{-2.f, 2.f}(a_host);
        ck::utils::FillUniformDistributionIntegerValue<BDataType>{-2.f, 2.f}(b_host);
    }

    if(do_debug){
#if 0
        printf("Print A matrix\n");
        for (int im = 0; im < M; im++)
        {
            for (int ik = 0; ik < K; ik++)
            {
                printf("%04x ", *(reinterpret_cast<uint16_t*>(&(a_host(im,ik)))));
                if(ik%8==7) printf("|");
            }
            printf("\n");
        }
#endif
        printf("Print B matrix\n");
        for (int in = 0; in < N; in++)
        {
            for (int ik = 0; ik < K; ik++)
            {
                printf("%04x ", *(reinterpret_cast<uint16_t*>(&(b_host(in,ik)))));
                if(ik%8==7) printf("|");
            }
            printf("\n");
        }
    }
    DeviceMem a_buf(sizeof(ADataType) * a_host.GetElementSpaceSize());
    DeviceMem b_buf(sizeof(BDataType) * b_host.GetElementSpaceSize());
    DeviceMem c_buf(sizeof(CDataType) * c_host_dev.GetElementSpaceSize());

    a_buf.ToDevice(a_host.mData.data());
    b_buf.ToDevice(b_host.mData.data());

    // Alignment
    constexpr ck::index_t kAAlignment = 32;
    constexpr ck::index_t kBAlignment = 32;
    constexpr ck::index_t kCAlignment = 32;

    constexpr ck::index_t kBlockSize = 256;

    constexpr ck::index_t kGemmMPerBlock = 128;
    constexpr ck::index_t kGemmNPerBlock = 128;
    constexpr ck::index_t kGemmKPerBlock = 32;
    //Tuning parameter, MMA tile, MMA type, Warp tile, split-k, data type, layout

    ck::index_t kGridSize = (M / kGemmMPerBlock) * (N / kGemmNPerBlock);

    std::cout << "grid size " << kGridSize << std::endl;

    constexpr ck::index_t kWarpPerCu    = 8; // 2 warps per SIMD
    constexpr ck::index_t kWarpPerBlock = kBlockSize / warpSize;
    constexpr ck::index_t kBlockPerCu   = kWarpPerCu / kWarpPerBlock;

    const auto gemm_kernel = Gemm<ADataType,
                                  BDataType,
                                  AccDataType,
                                  CDataType,
                                  ALayout,
                                  BLayout,
                                  CLayout,
                                  AElementFunction,
                                  BElementFunction,
                                  CElementFunction,
                                  kAAlignment,
                                  kBAlignment,
                                  kCAlignment,
                                  kBlockSize,
                                  kGemmMPerBlock,
                                  kGemmNPerBlock,
                                  kGemmKPerBlock>{};

    float ave_time =
        launch_kernel<kBlockSize, kBlockPerCu>(StreamConfig{nullptr, static_cast<bool>( time_kernel)},
                                               gemm_kernel,
                                               kGridSize,
                                               kBlockSize,
                                               0,
                                               static_cast<ADataType*>(a_buf.GetDeviceBuffer()),
                                               static_cast<BDataType*>(b_buf.GetDeviceBuffer()),
                                               static_cast<CDataType*>(c_buf.GetDeviceBuffer()),
                                               M,
                                               N,
                                               K,
                                               Lda,
                                               Ldb,
                                               Ldc,
                                               AElementFunction{},
                                               BElementFunction{},
                                               CElementFunction{});

    std::size_t flop = std::size_t(2) * M * N * K;
    std::size_t num_btype =
        sizeof(ADataType) * M * K + sizeof(BDataType) * K * N + sizeof(CDataType) * M * N;

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s"
              << std::endl;

    if(do_verification || do_debug){
        printf("Verfication: ON\n");
        Tensor<CDataType> c_host_ref(c_lengths, c_strides);
        // reference gemm
        reference_gemm<ADataType, ADataType, AccDataType, CDataType>(a_host, b_host, c_host_ref);

        c_buf.FromDevice(c_host_dev.mData.data());

        return !ck::utils::check_err(c_host_dev, c_host_ref);
    }
    printf("Verfication: OFF\n");
    return 0;
}
