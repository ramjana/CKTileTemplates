add_gtest_executable(test_layout test_layout.cpp)
target_link_libraries(test_layout PRIVATE utility)
add_gtest_executable(test_tensor test_tensor.cpp)
target_link_libraries(test_tensor PRIVATE utility)
add_gtest_executable(test_copy test_copy.cpp)
target_link_libraries(test_copy PRIVATE utility)
add_gtest_executable(test_partition test_partition.cpp)
target_link_libraries(test_partition PRIVATE utility)
if(GPU_TARGETS MATCHES "gfx908" OR GPU_TARGETS MATCHES "gfx90a" OR
   GPU_TARGETS MATCHES "gfx940" OR GPU_TARGETS MATCHES "gfx941" OR
   GPU_TARGETS MATCHES "gfx942")
    add_gtest_executable(test_gemm test_gemm.cpp)
    target_link_libraries(test_gemm PRIVATE utility)
endif()
