
#include <hip/hip_runtime.h>

#include "ck/ck.hpp"
#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"

// Meta data for GPU
// TODO: do we need to take care of data alignment in code or it's done by compiler?
template <ck::index_t kSize>
struct MetaData
{
    char p_data_[kSize];

    ck::index_t size_ = 0;
    ck::index_t pos_  = 0;

    __host__ __device__ void reset()
    {
        size_ = 0;
        pos_  = 0;
    }

    __device__ void reset_pos() { pos_ = 0; }

    // push meta data on host
    // TODO: correct forwarding?
    template <typename T>
    __host__ auto push(T&& a)
    {
        assert(size_ + sizeof(Type) <= kSize);

        using Type = ck::remove_cvref_t<T>;

        *reinterpret_cast<Type*>(p_data_ + size_) = a;

        size_ += sizeof(Type);

        return ck::forwarder{}(a);
    }

    // pull meta data on device
    // TODO: correct forwarding?
    template <typename T>
    __device__ auto pull()
    {
        using Type = ck::remove_cvref_t<T>;

        Type a = *reinterpret_cast<Type*>(p_data_ + pos_);

        pos_ += sizeof(Type);

        return a;
    }
};

// namespace tp (for tile programming)
struct ProgramServer
{
    // meta data on device
    MetaData<1024> meta_data_;

    __host__ void cpu_init() { meta_data_.reset(); }

    __device__ void gpu_init() { meta_data_.reset_pos(); }

    // push meta data on host
    template <typename T>
    __host__ auto operator()(T&& a)
    {
        return ck::forwarder{}(meta_data_.push(a));
    }

    // push meta data on host
    template <typename T>
    __device__ auto operator()(T&&)
    {
        return ck::forwarder{}(meta_data_.pull<T>());
    }

    __host__ static ck::index_t get_block_1d_id() { return -1; }

    __host__ static ck::index_t get_grid_size() { return -1; }

    __device__ static ck::index_t get_block_1d_id() { return ck::get_block_1d_id(); }

    __device__ static ck::index_t get_grid_size() { return ck::get_grid_size(); }

    // TODO: correct forwarding?
    template <typename T>
    __host__ static constexpr auto read_first_lane(T&& a)
    {
        return ck::forwarder{}(a);
    }

    template <typename T>
    __device__ static constexpr auto read_first_lane(T&& a)
    {
        return __builtin_amdgcn_readfirstlane(a);
    }
};

template <typename Server, typename Program, typename... Xs>
__global__ void gpu_program_wrapper(Server server, Program f, Xs... xs)
{
    server.gpu_init();
    f(server, xs...);
}

template <typename Server, typename Program, typename... Xs>
void launch(Server server, Program f, dim3 grid_dim, dim3 block_dim, Xs... xs)
{
    server.cpu_init();

    f(server, xs...);

    printf("meta data size %d\n", server.meta_data_.size_);

    gpu_program_wrapper<Server, Program><<<grid_dim, block_dim, 0, nullptr>>>(server, f, xs...);
}