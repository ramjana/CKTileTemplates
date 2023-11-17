// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <hip/hip_runtime.h>

#include "ck/ck.hpp"
#include "ck/stream_config.hpp"
#include "ck/host_utility/hip_check_error.hpp"

template <int MaxThreadPerBlock, int MinBlockPerCu, typename Kernel, typename... Args>
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(MaxThreadPerBlock, MinBlockPerCu)
#endif
    __global__ void kernel_wrapper(Kernel f, Args... args)
{
    f(args...);
}

template <typename... Args, typename F>
float launch_and_time_kernel(const StreamConfig& stream_config,
                             F kernel,
                             dim3 grid_dim,
                             dim3 block_dim,
                             std::size_t lds_byte,
                             Args... args)
{
#if CK_TIME_KERNEL
    if(stream_config.time_kernel_)
    {
#if DEBUG_LOG
        printf("%s: grid_dim {%d, %d, %d}, block_dim {%d, %d, %d} \n",
               __func__,
               grid_dim.x,
               grid_dim.y,
               grid_dim.z,
               block_dim.x,
               block_dim.y,
               block_dim.z);

        printf("Warm up 1 time\n");
#endif
        // warm up
        for(int i = 0; i < 10; i++)
        {
            kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args...);
        }

        const int nrepeat = 10;
#if DEBUG_LOG
        printf("Start running %d times...\n", nrepeat);
#endif
        hipEvent_t start, stop;

        hip_check_error(hipEventCreate(&start));
        hip_check_error(hipEventCreate(&stop));

        hip_check_error(hipDeviceSynchronize());
        hip_check_error(hipEventRecord(start, stream_config.stream_id_));

        for(int i = 0; i < nrepeat; ++i)
        {
            kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args...);
        }

        hip_check_error(hipEventRecord(stop, stream_config.stream_id_));
        hip_check_error(hipEventSynchronize(stop));

        float total_time = 0;

        hip_check_error(hipEventElapsedTime(&total_time, start, stop));

        return total_time / nrepeat;
    }
    else
    {
        kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args...);

        return 0;
    }
#else
    kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args...);

    return 0;
#endif
}

template <typename... Args, typename F, typename PreProcessFunc>
float launch_and_time_kernel_with_preprocess(const StreamConfig& stream_config,
                                             PreProcessFunc preprocess,
                                             F kernel,
                                             dim3 grid_dim,
                                             dim3 block_dim,
                                             std::size_t lds_byte,
                                             Args... args)
{
#if CK_TIME_KERNEL
    if(stream_config.time_kernel_)
    {
#if DEBUG_LOG
        printf("%s: grid_dim {%d, %d, %d}, block_dim {%d, %d, %d} \n",
               __func__,
               grid_dim.x,
               grid_dim.y,
               grid_dim.z,
               block_dim.x,
               block_dim.y,
               block_dim.z);

        printf("Warm up 1 time\n");
#endif
        // warm up
        preprocess();
        kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args...);

        const int nrepeat = 10;
#if DEBUG_LOG
        printf("Start running %d times...\n", nrepeat);
#endif
        hipEvent_t start, stop;

        hip_check_error(hipEventCreate(&start));
        hip_check_error(hipEventCreate(&stop));

        hip_check_error(hipDeviceSynchronize());
        hip_check_error(hipEventRecord(start, stream_config.stream_id_));

        for(int i = 0; i < nrepeat; ++i)
        {
            preprocess();
            kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args...);
        }

        hip_check_error(hipEventRecord(stop, stream_config.stream_id_));
        hip_check_error(hipEventSynchronize(stop));

        float total_time = 0;

        hip_check_error(hipEventElapsedTime(&total_time, start, stop));

        return total_time / nrepeat;
    }
    else
    {
        preprocess();
        kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args...);

        return 0;
    }
#else
    kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args...);

    return 0;
#endif
}

template <int MaxThreadPerBlock = CK_MAX_THREAD_PER_BLOCK,
          int MinBlockPerCu     = CK_MIN_BLOCK_PER_CU,
          typename KernelImpl,
          typename... Args>
float launch_kernel(const StreamConfig& stream_config,
                    KernelImpl kernel_impl,
                    dim3 grid_dim,
                    dim3 block_dim,
                    std::size_t dynamic_smem_byte,
                    Args... args)
{
    const auto kernel = kernel_wrapper<MaxThreadPerBlock, MinBlockPerCu, KernelImpl, Args...>;

    return launch_and_time_kernel(
        stream_config, kernel, grid_dim, block_dim, dynamic_smem_byte, kernel_impl, args...);
}
