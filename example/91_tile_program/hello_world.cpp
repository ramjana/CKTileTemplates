#include "ck/ck.hpp"
#include "ck/utility/common_header.hpp"

#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/host_utility/io.hpp"

#include "ck/library/utility/device_memory.hpp"

#include "ck/tile_program/tile/tile_distribution.hpp"
#include "ck/tile_program/tile/tile_window.hpp"
#include "ck/tile_program/tile/load_tile.hpp"
#include "ck/tile_program/tile/store_tile.hpp"
#include "ck/tile_program/tile/tile_elementwise.hpp"
#include "ck/tile_program/tile/shuffle_distributed_tensor.hpp"

struct HelloWorld
{
    __device__ void operator()(int x, int y, int* res)
    {
        using namespace ck;

#if 0
        auto r0 = x + y;
        auto r1 = x - y;

        res[0] = r0;
        res[1] = r1;

#elif 0
        (void)x;
        (void)y;

        constexpr auto old_array = Array<index_t, 4>{100, 101, 102, 104};

        constexpr auto map = [] {
            Map<int, int> map_;

            map_(0) = 3;
            map_(1) = 2;
            map_(2) = 1;
            map_(3) = 0;

            return map_;
        }();

        constexpr auto new_array = container_reorder_given_new2old(old_array, map);

        res[0] = new_array[0];
        res[1] = new_array[1];
        res[2] = new_array[2];
        res[3] = new_array[3];
#else
        (void)x;
        (void)y;
        (void)res;

        using namespace ck;
        using namespace ck::tile_program;

        // 2x2 wave
        constexpr auto in_dstr = make_static_tile_distribution(
            StaticTileDistributionEncoding<Sequence<>,
                                           Tuple<Sequence<2, 2, 4, 2, 4>, Sequence<2, 2, 32>>,
                                           Tuple<Sequence<1, 2>, Sequence<1, 2>>,
                                           Tuple<Sequence<1, 1>, Sequence<3, 2>>,
                                           Sequence<1, 2, 1, 1>,
                                           Sequence<0, 0, 2, 4>>{});

        constexpr auto out_dstr = make_static_tile_distribution(
            StaticTileDistributionEncoding<Sequence<>,
                                           Tuple<Sequence<2, 2, 4, 2, 4>, Sequence<2, 2, 32>>,
                                           Tuple<Sequence<1, 2>, Sequence<1, 2>>,
                                           Tuple<Sequence<1, 1>, Sequence<3, 2>>,
                                           Sequence<1, 2, 1, 1>,
                                           Sequence<0, 0, 4, 2>>{});

        auto in_tensor = make_static_distributed_tensor<half_t>(in_dstr);

        in_tensor.Initialize(get_thread_id());

        auto out_tensor = make_static_distributed_tensor<half_t>(out_dstr);

        shuffle_distributed_tensor(out_tensor, in_tensor);

        printf("tid %d, %f\n",
               get_thread_id(),
               type_convert<float>(out_tensor.GetThreadBuffer()[Number<0>{}]));
#endif
    }
};

int main()
{
    int x = 100;
    int y = 101;

    DeviceMem res_dev_buf(4 * sizeof(int));

    launch_kernel(StreamConfig{},
                  HelloWorld{},
                  1,
                  1,
                  0,
                  x,
                  y,
                  static_cast<int*>(res_dev_buf.GetDeviceBuffer()));

    int res_host[4];

    res_dev_buf.FromDevice(&res_host);

    printf("res[0]: %d\n", res_host[0]);
    printf("res[1]: %d\n", res_host[1]);
    printf("res[2]: %d\n", res_host[2]);
    printf("res[3]: %d\n", res_host[3]);

    return 0;
}
