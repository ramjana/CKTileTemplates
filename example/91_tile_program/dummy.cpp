#include "ck/utility/common_header.hpp"

#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/cluster_descriptor.hpp"
#include "ck/tensor/tensor_view.hpp"

#include "ck/host_utility/device_prop.hpp"

#include "ck/library/utility/device_memory.hpp"

#include "tile_program.hpp"
#include "ck/tile_program/meta_data_buffer.hpp"
#include "ck/tile_program/block_tensor_distribution.hpp"

__global__ void foo(int* p)
{
    using namespace ck;

    constexpr auto block_dstr_0 = ck::tile_program::block::make_static_block_tensor_distribution(
        make_tuple(Sequence<2, 4, 16>{}, Sequence<4, 8>{}),
        Sequence<0>{},
        Sequence<1>{},
        Sequence<0, 1>{},
        Sequence<2, 0>{},
        Sequence<0, 1>{},
        Sequence<0, 1>{},
        Sequence<0, 1>{});

    constexpr auto block_dstr_1 = ck::tile_program::block::make_block_tensor_distribution(
        make_tuple(Sequence<2, 4, 16>{}, Sequence<4, 8>{}),
        Sequence<0>{},
        Sequence<1>{},
        Sequence<0, 1>{},
        Sequence<2, 0>{},
        Sequence<0, 1>{},
        Sequence<0, 1>{},
        Sequence<0, 1>{});

    static_assert(block_dstr_0.GetWidLidYs2XsAdaptor().IsKnownAtCompileTime(), "");
    static_assert(block_dstr_1.GetYs2DidDescriptor().IsKnownAtCompileTime() == false, "");

    p[0] = block_dstr_0.GetWidLidYs2XsAdaptor().GetElementSize();
    p[1] = block_dstr_0.GetYs2DidDescriptor().GetElementSize();
    p[2] = block_dstr_0.GetYs2DidDescriptor().GetElementSpaceSize();

    p[3] = block_dstr_1.GetWidLidYs2XsAdaptor().GetElementSize();
    p[4] = block_dstr_1.GetYs2DidDescriptor().GetElementSize();
    p[5] = block_dstr_1.GetYs2DidDescriptor().GetElementSpaceSize();
}

int main()
{
    DeviceMem res(1024);

    foo<<<dim3{1}, dim3{1}, 0, nullptr>>>(static_cast<int*>(res.GetDeviceBuffer()));
}
