#include <iostream>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "data_type.hpp"
#include "element_wise_operation.hpp"
#include "conv_fwd_util.hpp"
#include "conv_util.hpp"

namespace {

bool test_conv1D_nwc()
{
    using namespace std::placeholders;
    using namespace ck::utils;
    namespace ctl = ck::tensor_layout::convolution;

    ck::utils::conv::ConvParams params;
    params.num_dim_spatial        = 1;
    params.N                      = 2;
    params.K                      = 16;
    params.C                      = 4;
    params.filter_spatial_lengths = std::vector<ck::index_t>{3};
    params.input_spatial_lengths  = std::vector<ck::index_t>{16};
    params.conv_filter_strides    = std::vector<ck::index_t>{1};
    params.conv_filter_dilations  = std::vector<ck::index_t>{1};
    params.input_left_pads        = std::vector<ck::index_t>{1};
    params.input_right_pads       = std::vector<ck::index_t>{1};

    std::vector<test::conv::DeviceConvFwdNoOpPtr> conv_ptrs;
    test::conv::get_test_convolution_fwd_instance<1>(conv_ptrs);
    conv::ConvFwdOpInstance<float, float, float, ctl::NWC, ctl::KCX, ctl::NWK> conv_instance(
        params);

    auto reference_conv_fwd_fun = std::bind(
        conv::run_reference_convolution_forward<1, float, float, float>, params, _1, _2, _3);
    OpInstanceRunEngine<float, float, float> run_engine(conv_instance, reference_conv_fwd_fun);
    run_engine.SetAtol(1e-5);
    run_engine.SetRtol(1e-4);
    return run_engine.Test(conv_ptrs);
}

template <typename T>
bool test_conv1d_nwc_instances(const std::vector<test::conv::DeviceConvFwdNoOpPtr>& conv_ptrs)
{
    using namespace std::placeholders;
    using namespace ck::utils;
    namespace ctl = ck::tensor_layout::convolution;

    ck::utils::conv::ConvParams params;
    params.num_dim_spatial        = 1;
    params.filter_spatial_lengths = std::vector<ck::index_t>{3};
    params.input_spatial_lengths  = std::vector<ck::index_t>{71};
    params.conv_filter_strides    = std::vector<ck::index_t>{2};
    params.conv_filter_dilations  = std::vector<ck::index_t>{1};
    params.input_left_pads        = std::vector<ck::index_t>{1};
    params.input_right_pads       = std::vector<ck::index_t>{1};

    conv::ConvFwdOpInstance<T, T, T, ctl::NWC, ctl::KCX, ctl::NWK> conv_instance(params);

    auto reference_conv_fwd_fun =
        std::bind(conv::run_reference_convolution_forward<1, T, T, T>, params, _1, _2, _3);
    OpInstanceRunEngine<T, T, T> run_engine(conv_instance, reference_conv_fwd_fun);
    return run_engine.Test(conv_ptrs);
}

bool test_conv1d_nwc_bf16_instances()
{
    return test_conv1d_nwc_instances<ck::bhalf_t>(
        ck::utils::conv::ConvolutionFwdInstances<ck::bhalf_t, ck::bhalf_t, ck::bhalf_t>::Get<1>());
}

bool test_conv1d_nwc_f16_instances()
{
    return test_conv1d_nwc_instances<ck::half_t>(
        ck::utils::conv::ConvolutionFwdInstances<ck::half_t, ck::half_t, ck::half_t>::Get<1>());
}

bool test_conv1d_nwc_f32_instances()
{
    return test_conv1d_nwc_instances<float>(
        ck::utils::conv::ConvolutionFwdInstances<float, float, float>::Get<1>());
}

bool test_conv1d_nwc_int8_instances()
{
    return test_conv1d_nwc_instances<int8_t>(
        ck::utils::conv::ConvolutionFwdInstances<int8_t, int8_t, int8_t>::Get<1>());
}

} // anonymous namespace

int main()
{
    bool res{true};
    res = test_conv1D_nwc();
    std::cout << "test_conv1D_nwc ..... " << (res ? "SUCCESS" : "FAILURE") << std::endl;

    res = test_conv1d_nwc_bf16_instances();
    std::cout << "\nTestConv1DNWCBF16Instances ..... " << (res ? "SUCCESS" : "FAILURE")
              << std::endl;
    res = test_conv1d_nwc_f16_instances();
    std::cout << "\ntest_conv1d_nwc_f16_instances ..... " << (res ? "SUCCESS" : "FAILURE")
              << std::endl;
    res = test_conv1d_nwc_f32_instances();
    std::cout << "\ntest_conv1d_nwc_f32_instances ..... " << (res ? "SUCCESS" : "FAILURE")
              << std::endl;
    res = test_conv1d_nwc_int8_instances();
    std::cout << "\ntest_conv1d_nwc_int8_instances ..... " << (res ? "SUCCESS" : "FAILURE")
              << std::endl;

    return res ? 0 : 1;
}