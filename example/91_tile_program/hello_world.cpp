#include "tile_program.hpp"

#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_batched_gemm_gemm.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_batched_gemm_gemm_xdl_cshuffle_v1.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/host_utility/io.hpp"

#include "ck/library/utility/device_memory.hpp"

// program
struct HelloWorld
{
    __host__ __device__ void operator()(ProgramServer& ps, int x, int y, int* res)
    {
        // host will evaulate value inside ps(), and push the result into meta data buffer
        // device will read the value inside ps() from meta data buffer
        auto z = ps(x + y);

        res[0] = z;
    }
};

int main()
{
    int x = 100;
    int y = 101;

    DeviceMem res_dev_buf(sizeof(int));

    launch(ProgramServer{},
           HelloWorld{},
           1,
           1,
           x,
           y,
           static_cast<int*>(res_dev_buf.GetDeviceBuffer()));

    int res_host;

    res_dev_buf.FromDevice(&res_host);

    printf("res_host %d\n", res_host);

    return 0;
}
