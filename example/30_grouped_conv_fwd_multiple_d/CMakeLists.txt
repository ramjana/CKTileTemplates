list(APPEND gpu_list1 gfx908 gfx90a gfx940 gfx941 gfx942)
list(APPEND gpu_list2 gfx1100 gfx1101 gfx1102)

set(target 0)
foreach(gpu IN LISTS GPU_TARGETS)
    if(gpu IN_LIST gpu_list1 AND target EQUAL 0)
        add_custom_target(example_grouped_conv_fwd_multiple_d)

        add_example_executable(example_grouped_conv_fwd_bias_relu_add_xdl_fp16 grouped_conv_fwd_bias_relu_add_xdl_fp16.cpp)
        add_example_dependencies(example_grouped_conv_fwd_multiple_d example_grouped_conv_fwd_bias_relu_add_xdl_fp16)

        add_example_executable(example_grouped_conv_fwd_xdl_fp16 grouped_conv_fwd_xdl_fp16.cpp)
        add_example_dependencies(example_grouped_conv_fwd_multiple_d example_grouped_conv_fwd_xdl_fp16)

        add_example_executable(example_grouped_conv_fwd_bias_relu_add_xdl_fp32 grouped_conv_fwd_bias_relu_add_xdl_fp32.cpp)
        add_example_dependencies(example_grouped_conv_fwd_multiple_d example_grouped_conv_fwd_bias_relu_add_xdl_fp32)

        add_example_executable(example_grouped_conv_fwd_bias_relu_add_xdl_bf16 grouped_conv_fwd_bias_relu_add_xdl_bf16.cpp)
        add_example_dependencies(example_grouped_conv_fwd_multiple_d example_grouped_conv_fwd_bias_relu_add_xdl_bf16)

        add_example_executable(example_grouped_conv_fwd_bias_relu_add_xdl_int8 grouped_conv_fwd_bias_relu_add_xdl_int8.cpp)
        add_example_dependencies(example_grouped_conv_fwd_multiple_d example_grouped_conv_fwd_bias_relu_add_xdl_int8)

        if(USE_BITINT_EXTENSION_INT4)
            add_example_executable(example_grouped_conv_fwd_bias_relu_add_xdl_int4 grouped_conv_fwd_bias_relu_add_xdl_int4.cpp)
            add_example_dependencies(example_grouped_conv_fwd_multiple_d example_grouped_conv_fwd_bias_relu_add_xdl_int4)
        endif() # USE_BITINT_EXTENSION_INT4

        set(target 1)
    endif()
endforeach()

set(target 0)
foreach(gpu IN LISTS GPU_TARGETS)
    if(gpu IN_LIST gpu_list2 AND target EQUAL 0)
        add_example_executable(example_grouped_conv_fwd_bias_relu_add_wmma_fp16 grouped_conv_fwd_bias_relu_add_wmma_fp16.cpp)
        add_example_executable(example_grouped_conv_fwd_bias_relu_add_wmma_int8 grouped_conv_fwd_bias_relu_add_wmma_int8.cpp)
        set(target 1)
    endif()
endforeach()
