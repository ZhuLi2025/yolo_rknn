/**
 * @file osd_painter.hpp
 * @brief 在 ARGB 画布上画检测框和一行文字。调用方负责把画布清成透明。
 *
 * 像素是 DRM ARGB8888：小端内存里每个像素是 0xAARRGGBB。只描边和写字，不填满框的内部。
 */
#pragma once

#include "yolo_rknn/infer/detection.hpp"

#include <cstdint>
#include <vector>

namespace yolo_rknn {

struct OsdCanvas {
    std::uint8_t* pixels = nullptr;
    int width = 0;
    int height = 0;
    int stride_bytes = 0;
};

void osd_draw(OsdCanvas& canvas, const std::vector<Detection>& detections);

}  // namespace yolo_rknn
