list(APPEND gpu_list gfx908 gfx90a gfx940 gfx941 gfx942)
set(target 0)
foreach(gpu IN LISTS GPU_TARGETS)
 if(gpu IN_LIST gpu_list AND target EQUAL 0)
   add_custom_target(example_splitK_gemm_xdl)

   add_example_executable(example_splitK_gemm_xdl_fp32 splitK_gemm_xdl_fp32.cpp)
   add_example_dependencies(example_splitK_gemm_xdl example_splitK_gemm_xdl_fp32)

   add_example_executable(example_splitK_gemm_xdl_fp16 splitK_gemm_xdl_fp16.cpp)
   add_example_dependencies(example_splitK_gemm_xdl example_splitK_gemm_xdl_fp16)

   add_example_executable(example_splitK_gemm_xdl_bf16 splitK_gemm_xdl_bf16.cpp)
   add_example_dependencies(example_splitK_gemm_xdl example_splitK_gemm_xdl_bf16)

   add_example_executable(example_splitK_gemm_xdl_int8 splitK_gemm_xdl_int8.cpp)
   add_example_dependencies(example_splitK_gemm_xdl example_splitK_gemm_xdl_int8)

   if(USE_BITINT_EXTENSION_INT4)
       add_example_executable(example_splitK_gemm_xdl_int4 splitK_gemm_xdl_int4.cpp)
       add_example_dependencies(example_splitK_gemm_xdl example_splitK_gemm_xdl_int4)
   endif()

   set(target 1)
 endif()
endforeach()
