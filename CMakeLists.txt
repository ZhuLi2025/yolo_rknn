cmake_minimum_required(VERSION 3.10)

# 在 RK3588 上原生编译，不走交叉工具链。
# 本机是 Ubuntu 20.04 的 g++ 9.4。代码仍锁在 C++14，不使用 filesystem / optional。
project(yolo_rknn VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Debug or Release" FORCE)
endif()

# FFmpeg、MPP、RGA、RKNN 都在系统库目录里。
set(CMAKE_BUILD_RPATH "/usr/lib;/usr/local/lib")
set(CMAKE_INSTALL_RPATH "/usr/lib;/usr/local/lib")

add_compile_options(-Wall -Wextra -Wno-unused-parameter -D_GNU_SOURCE)

# ---------------------------------------------------------------------------
# 依赖全部来自当前系统，不在工程里私自拷贝一份 SDK。
# 有意不链接 libswscale：像素缩放和色域转换必须走 RGA，软缩放库连进来容易被人误用。
# ---------------------------------------------------------------------------
function(yr_require_lib OUT_VAR)
    find_library(${OUT_VAR} ${ARGN})
    if(NOT ${OUT_VAR})
        message(FATAL_ERROR "找不到库: ${ARGN}")
    endif()
endfunction()

yr_require_lib(YR_MPP NAMES rockchip_mpp PATHS /usr/lib/aarch64-linux-gnu /usr/lib)
yr_require_lib(YR_RGA NAMES rga PATHS /usr/lib /usr/local/lib)
yr_require_lib(YR_AVFORMAT NAMES avformat PATHS /usr/local/lib /usr/lib/aarch64-linux-gnu)
yr_require_lib(YR_AVCODEC NAMES avcodec PATHS /usr/local/lib /usr/lib/aarch64-linux-gnu)
yr_require_lib(YR_AVUTIL NAMES avutil PATHS /usr/local/lib /usr/lib/aarch64-linux-gnu)
yr_require_lib(YR_DRM NAMES drm PATHS /usr/lib/aarch64-linux-gnu /usr/lib)

# RK3588 的用户态是 RKNN Runtime 2（librknnrt.so），不是 RK3399Pro 的 librknn_api.so。
yr_require_lib(YR_RKNN NAMES rknnrt PATHS /usr/lib /lib /usr/lib/aarch64-linux-gnu)
get_filename_component(YR_RKNN_DIR ${YR_RKNN} DIRECTORY)
message(STATUS "Using RKNN runtime: ${YR_RKNN}")

set(YR_SOURCES
    src/common/logger.cpp
    src/common/thread_util.cpp
    src/common/app_config.cpp
    src/common/dma_buf.cpp
    src/stream/rtsp_client.cpp
    src/decode/mpp_decoder.cpp
    src/preprocess/rga_preprocessor.cpp
    src/preprocess/rga_lock.cpp
    src/display/osd_painter.cpp
    src/display/drm_display.cpp
    src/infer/postprocess.cpp
    src/infer/yolov8_detector.cpp
    src/business/zoom_policy.cpp
    src/business/controller.cpp
    src/pipeline/pipeline.cpp
)

add_library(yolo_core STATIC ${YR_SOURCES})
target_include_directories(yolo_core
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    PRIVATE /usr/include/rockchip /usr/include/rga /usr/local/include /usr/include/libdrm
)
target_compile_definitions(yolo_core PUBLIC YR_VERSION=\"${PROJECT_VERSION}\")
target_link_libraries(yolo_core
    PUBLIC
        ${YR_MPP}
        ${YR_RGA}
        ${YR_RKNN}
        ${YR_AVFORMAT}
        ${YR_AVCODEC}
        ${YR_AVUTIL}
        ${YR_DRM}
        pthread
        dl
)

add_executable(yolo_rknn src/main.cpp)
target_link_libraries(yolo_rknn PRIVATE yolo_core)
set_target_properties(yolo_rknn PROPERTIES
    BUILD_RPATH "${YR_RKNN_DIR};/usr/local/lib"
    INSTALL_RPATH "${YR_RKNN_DIR};/usr/local/lib"
)

# 后处理与变焦策略不依赖硬件，可在没有码流、没有模型时回归。
add_executable(yolo_rknn_tests
    tests/test_core.cpp
    src/infer/postprocess.cpp
    src/business/zoom_policy.cpp
    src/display/osd_painter.cpp
)
target_include_directories(yolo_rknn_tests PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_compile_definitions(yolo_rknn_tests PRIVATE YR_VERSION=\"${PROJECT_VERSION}\")

enable_testing()
add_test(NAME core COMMAND yolo_rknn_tests)

install(TARGETS yolo_rknn RUNTIME DESTINATION bin)
install(FILES config/app.conf DESTINATION share/yolo_rknn)
