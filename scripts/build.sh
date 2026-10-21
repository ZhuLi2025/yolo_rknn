#!/bin/bash
# 在本机用系统 CMake 编译。3.16 支持 cmake --build -j。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "${ROOT}/build"
cd "${ROOT}/build"
cmake .. -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
cmake --build . -j "$(nproc)"
"${ROOT}/build/bin/yolo_rknn_tests"
echo "binary: ${ROOT}/build/bin/yolo_rknn"
