list(APPEND gpu_list gfx908 gfx90a gfx940 gfx941 gfx942)
set(target 0)
foreach(gpu IN LISTS GPU_TARGETS)
 if(gpu IN_LIST gpu_list AND target EQUAL 0)
   add_example_executable(example_conv2d_fwd_xdl_perlayer_quantization_int8 conv2d_fwd_xdl_perlayer_quantization_int8.cpp)
   add_example_executable(example_conv2d_fwd_xdl_perchannel_quantization_int8 conv2d_fwd_xdl_perchannel_quantization_int8.cpp)
   add_example_executable(example_conv2d_fwd_xdl_bias_relu_perlayer_quantization_int8 conv2d_fwd_xdl_bias_relu_perlayer_quantization_int8.cpp)
   add_example_executable(example_conv2d_fwd_xdl_bias_relu_perchannel_quantization_int8 conv2d_fwd_xdl_bias_relu_perchannel_quantization_int8.cpp)
   set(target 1)
 endif()
endforeach()

  # Conv perlayer quantization
  add_example_executable(example_conv2d_fwd_dl_perlayer_quantization_int8 conv2d_fwd_dl_perlayer_quantization_int8.cpp)
  # Conv perchannel quantization
  add_example_executable(example_conv2d_fwd_dl_perchannel_quantization_int8 conv2d_fwd_dl_perchannel_quantization_int8.cpp)
  # Conv + bias + relu perlayer quantization
  add_example_executable(example_conv2d_fwd_dl_bias_relu_perlayer_quantization_int8 conv2d_fwd_dl_bias_relu_perlayer_quantization_int8.cpp)
  # Conv + bias + relu perchannel quantization
  add_example_executable(example_conv2d_fwd_dl_bias_relu_perchannel_quantization_int8 conv2d_fwd_dl_bias_relu_perchannel_quantization_int8.cpp)
  # Conv + bias + tanh perlayer quantization
  add_example_executable(example_conv2d_fwd_dl_bias_tanh_perlayer_quantization_int8 conv2d_fwd_dl_bias_tanh_perlayer_quantization_int8.cpp)
  # Conv + bias + tanh perchannel quantization
  add_example_executable(example_conv2d_fwd_dl_bias_tanh_perchannel_quantization_int8 conv2d_fwd_dl_bias_tanh_perchannel_quantization_int8.cpp)
