set(DEVICE_CONTRACTION_BILINEAR_INSTANCES)

# FP32
list(APPEND DEVICE_CONTRACTION_BILINEAR_INSTANCES device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_f32_kknn_instance.cpp
                                                    device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_f32_knnn_instance.cpp
                                                    device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_f32_mknn_instance.cpp
                                                    device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_f32_mnnn_instance.cpp)

list(APPEND DEVICE_CONTRACTION_BILINEAR_INSTANCES device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_f32_compute_f16_kknn_instance.cpp
                                                    device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_f32_compute_f16_knnn_instance.cpp
                                                    device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_f32_compute_f16_mknn_instance.cpp
                                                    device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_f32_compute_f16_mnnn_instance.cpp)

list(APPEND DEVICE_CONTRACTION_BILINEAR_INSTANCES device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_f32_compute_bf16_kknn_instance.cpp
                                                    device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_f32_compute_bf16_knnn_instance.cpp
                                                    device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_f32_compute_bf16_mknn_instance.cpp
                                                    device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_f32_compute_bf16_mnnn_instance.cpp)

# FP64
list(APPEND DEVICE_CONTRACTION_BILINEAR_INSTANCES device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_f64_f64_f64_f64_kknn_instance.cpp
                                                    device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_f64_f64_f64_f64_knnn_instance.cpp
                                                    device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_f64_f64_f64_f64_mknn_instance.cpp
                                                    device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_f64_f64_f64_f64_mnnn_instance.cpp)

list(APPEND DEVICE_CONTRACTION_BILINEAR_INSTANCES device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_f64_f64_f64_f64_compute_f32_kknn_instance.cpp
                                                    device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_f64_f64_f64_f64_compute_f32_knnn_instance.cpp
                                                    device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_f64_f64_f64_f64_compute_f32_mknn_instance.cpp
                                                    device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_f64_f64_f64_f64_compute_f32_mnnn_instance.cpp)

# FP16
list(APPEND DEVICE_CONTRACTION_BILINEAR_INSTANCES device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_f16_f16_f16_f16_compute_f32_kknn_instance.cpp
                                                    device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_f16_f16_f16_f16_compute_f32_knnn_instance.cpp
                                                    device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_f16_f16_f16_f16_compute_f32_mknn_instance.cpp
                                                    device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_f16_f16_f16_f16_compute_f32_mnnn_instance.cpp)

# BF16
list(APPEND DEVICE_CONTRACTION_BILINEAR_INSTANCES device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_bf16_bf16_bf16_bf16_compute_f32_kknn_instance.cpp
                                                    device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_bf16_bf16_bf16_bf16_compute_f32_knnn_instance.cpp
                                                    device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_bf16_bf16_bf16_bf16_compute_f32_mknn_instance.cpp
                                                    device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_bf16_bf16_bf16_bf16_compute_f32_mnnn_instance.cpp)

add_instance_library(device_contraction_bilinear_instance ${DEVICE_CONTRACTION_BILINEAR_INSTANCES})
