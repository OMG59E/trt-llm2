# GoogleTest Preparation - Code block copied from
# https://google.github.io/googletest/quickstart-cmake.html
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

find_library_create_target(nvonnxparser nvonnxparser SHARED ${TRT_OUT_DIR}
                           ${TRT_LIB_DIR})

include_directories(
  ${PROJECT_SOURCE_DIR}/tensorrt_llm/cutlass_extensions/include
  ${PROJECT_SOURCE_DIR}/include)

set(TOP_LEVEL_DIR "${PROJECT_SOURCE_DIR}/..")

add_custom_target(google-tests)

function(add_gtest test_name test_src)
  add_executable(${test_name} ${test_src})

  target_link_libraries(
    ${test_name} PUBLIC ${SHARED_TARGET} gtest_main gmock_main stdc++fs
                        nvonnxparser nvinfer_plugin)

  target_compile_features(${test_name} PRIVATE cxx_std_17)
  target_compile_definitions(${test_name}
                             PUBLIC TOP_LEVEL_DIR="${TOP_LEVEL_DIR}")
  gtest_discover_tests(${test_name} PROPERTIES ENVIRONMENT
                                               "CUDA_MODULE_LOADING=LAZY")
  add_dependencies(google-tests ${test_name})
endfunction()

add_gtest(attentionKernelTest runtime/transposeKVKernelTest.cpp)
add_gtest(gptDecoderTest runtime/gptDecoderTest.cpp)
add_gtest(gptDecoderBatchTest runtime/gptDecoderBatchTest.cpp)
add_gtest(gptSessionTest runtime/gptSessionTest.cpp)
add_gtest(allocatorTest common/allocatorTest.cpp)
add_gtest(memoryUtilsTest common/memoryUtilsTest.cu)
add_gtest(stringUtilsTest common/stringUtilsTest.cpp)
add_gtest(tllmExceptionTest common/tllmExceptionTest.cpp)
add_gtest(tensorTest common/tensorTest.cpp)
add_gtest(tllmRuntimeTest runtime/tllmRuntimeTest.cpp)
add_gtest(tllmBuffersTest runtime/tllmBuffersTest.cpp)
add_gtest(bufferManagerTest runtime/bufferManagerTest.cpp)
add_gtest(runtimeKernelTest runtime/runtimeKernelTest.cpp)

