add_custom_target(example_gemm_scale_softmax_gemm)

add_example_executable(example_batched_gemm_scale_softmax_gemm_xdl_fp16 batched_gemm_scale_softmax_gemm_xdl_fp16.cpp)
add_example_dependencies(example_gemm_scale_softmax_gemm example_batched_gemm_scale_softmax_gemm_xdl_fp16)

add_example_executable(example_batched_gemm_scale_softmax_gemm_permute_xdl_fp16 batched_gemm_scale_softmax_gemm_permute_xdl_fp16.cpp)
add_example_dependencies(example_gemm_scale_softmax_gemm example_batched_gemm_scale_softmax_gemm_permute_xdl_fp16)

add_example_executable(example_grouped_gemm_scale_softmax_gemm_permute_xdl_fp16 grouped_gemm_scale_softmax_gemm_permute_xdl_fp16.cpp)
add_example_dependencies(example_gemm_scale_softmax_gemm example_grouped_gemm_scale_softmax_gemm_permute_xdl_fp16)

add_example_executable(example_batched_gemm_lower_triangle_scale_softmax_gemm_permute_xdl_fp16 batched_gemm_lower_triangle_scale_softmax_gemm_permute_xdl_fp16.cpp)
add_example_dependencies(example_gemm_scale_softmax_gemm example_batched_gemm_lower_triangle_scale_softmax_gemm_permute_xdl_fp16)

add_example_executable(example_grouped_gemm_lower_triangle_scale_softmax_gemm_permute_xdl_fp16 grouped_gemm_lower_triangle_scale_softmax_gemm_permute_xdl_fp16.cpp)
add_example_dependencies(example_gemm_scale_softmax_gemm example_grouped_gemm_lower_triangle_scale_softmax_gemm_permute_xdl_fp16)

add_example_executable(example_batched_gemm_scale_softmax_gemm_xdl_bf16 batched_gemm_scale_softmax_gemm_xdl_bf16.cpp)
add_example_dependencies(example_gemm_scale_softmax_gemm example_batched_gemm_scale_softmax_gemm_xdl_bf16)

add_example_executable(example_batched_gemm_scale_softmax_gemm_permute_xdl_bf16 batched_gemm_scale_softmax_gemm_permute_xdl_bf16.cpp)
add_example_dependencies(example_gemm_scale_softmax_gemm example_batched_gemm_scale_softmax_gemm_permute_xdl_bf16)

