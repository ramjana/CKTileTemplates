list(APPEND gpu_list gfx908 gfx90a gfx940 gfx941 gfx942)
set(target 0)
foreach(gpu IN LISTS GPU_TARGETS)
    if(gpu IN_LIST gpu_list AND target EQUAL 0)
        add_custom_target(example_convnd_fwd_reduce_xdl)

        add_example_executable(example_convnd_fwd_max_xdl_int8 convnd_fwd_max_xdl_int8.cpp)
        add_example_dependencies(example_convnd_fwd_reduce_xdl example_convnd_fwd_max_xdl_int8)

        add_example_executable_no_testing(example_convnd_fwd_max_xdl_bf16 convnd_fwd_max_xdl_bf16.cpp)
        add_example_dependencies(example_convnd_fwd_reduce_xdl example_convnd_fwd_max_xdl_bf16)

        add_example_executable_no_testing(example_convnd_fwd_max_xdl_fp16 convnd_fwd_max_xdl_fp16.cpp)
        add_example_dependencies(example_convnd_fwd_reduce_xdl example_convnd_fwd_max_xdl_fp16)

        add_example_executable(example_convnd_fwd_max_xdl_fp32 convnd_fwd_max_xdl_fp32.cpp)
        add_example_dependencies(example_convnd_fwd_reduce_xdl example_convnd_fwd_max_xdl_fp32)

        if(USE_BITINT_EXTENSION_INT4)
            add_example_executable(example_convnd_fwd_max_xdl_int4 convnd_fwd_max_xdl_int4.cpp)
            add_example_dependencies(example_convnd_fwd_reduce_xdl example_convnd_fwd_max_xdl_int4)
        endif(USE_BITINT_EXTENSION_INT4)
        set(target 1)
    endif()
endforeach()
