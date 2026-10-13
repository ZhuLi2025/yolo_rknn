#include "yolo_rknn/display/osd_painter.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace yolo_rknn {
namespace {

constexpr int kScale = 2;
constexpr int kBorder = 3;

void fill_rect(OsdCanvas& canvas, int x, int y, int w, int h, std::uint32_t argb) {
    if (w <= 0 || h <= 0) {
        return;
    }
    const int x1 = std::max(0, x);
    const int y1 = std::max(0, y);
    const int x2 = std::min(canvas.width, x + w);
    const int y2 = std::min(canvas.height, y + h);
    for (int yy = y1; yy < y2; ++yy) {
        std::uint8_t* row = canvas.pixels + static_cast<std::size_t>(yy) * static_cast<std::size_t>(canvas.stride_bytes);
        std::uint32_t* pix = reinterpret_cast<std::uint32_t*>(row);
        for (int xx = x1; xx < x2; ++xx) {
            pix[xx] = argb;
        }
    }
}

void stroke_rect(OsdCanvas& canvas, int x, int y, int w, int h, int thick, std::uint32_t argb) {
    if (w <= 0 || h <= 0) {
        return;
    }
    fill_rect(canvas, x, y, w, thick, argb);
    fill_rect(canvas, x, y + h - thick, w, thick, argb);
    fill_rect(canvas, x, y, thick, h, argb);
    fill_rect(canvas, x + w - thick, y, thick, h, argb);
}

// 自绘 8x8 点阵，高位在左。只覆盖日志和 OSD 会用到的字符，缺字就跳过。
const std::uint8_t* glyph_rows(char ch) {
    if (ch >= 'A' && ch <= 'Z') {
        ch = static_cast<char>(ch - 'A' + 'a');
    }
    static const std::uint8_t k0[] = {0x3C, 0x66, 0x6E, 0x76, 0x66, 0x66, 0x3C, 0x00};
    static const std::uint8_t k1[] = {0x18, 0x38, 0x18, 0x18, 0x18, 0x18, 0x7E, 0x00};
    static const std::uint8_t k2[] = {0x3C, 0x66, 0x06, 0x1C, 0x30, 0x60, 0x7E, 0x00};
    static const std::uint8_t k3[] = {0x3C, 0x66, 0x06, 0x1C, 0x06, 0x66, 0x3C, 0x00};
    static const std::uint8_t k4[] = {0x0C, 0x1C, 0x3C, 0x6C, 0x7E, 0x0C, 0x0C, 0x00};
    static const std::uint8_t k5[] = {0x7E, 0x60, 0x7C, 0x06, 0x06, 0x66, 0x3C, 0x00};
    static const std::uint8_t k6[] = {0x1C, 0x30, 0x60, 0x7C, 0x66, 0x66, 0x3C, 0x00};
    static const std::uint8_t k7[] = {0x7E, 0x06, 0x0C, 0x18, 0x30, 0x30, 0x30, 0x00};
    static const std::uint8_t k8[] = {0x3C, 0x66, 0x66, 0x3C, 0x66, 0x66, 0x3C, 0x00};
    static const std::uint8_t k9[] = {0x3C, 0x66, 0x66, 0x3E, 0x06, 0x0C, 0x38, 0x00};
    static const std::uint8_t ka[] = {0x18, 0x3C, 0x66, 0x66, 0x7E, 0x66, 0x66, 0x00};
    static const std::uint8_t kb[] = {0x7C, 0x66, 0x66, 0x7C, 0x66, 0x66, 0x7C, 0x00};
    static const std::uint8_t kc[] = {0x3C, 0x66, 0x60, 0x60, 0x60, 0x66, 0x3C, 0x00};
    static const std::uint8_t kd[] = {0x78, 0x6C, 0x66, 0x66, 0x66, 0x6C, 0x78, 0x00};
    static const std::uint8_t ke[] = {0x7E, 0x60, 0x60, 0x7C, 0x60, 0x60, 0x7E, 0x00};
    static const std::uint8_t kf[] = {0x7E, 0x60, 0x60, 0x7C, 0x60, 0x60, 0x60, 0x00};
    static const std::uint8_t kg[] = {0x3C, 0x66, 0x60, 0x6E, 0x66, 0x66, 0x3C, 0x00};
    static const std::uint8_t kh[] = {0x66, 0x66, 0x66, 0x7E, 0x66, 0x66, 0x66, 0x00};
    static const std::uint8_t ki[] = {0x3C, 0x18, 0x18, 0x18, 0x18, 0x18, 0x3C, 0x00};
    static const std::uint8_t kj[] = {0x1E, 0x0C, 0x0C, 0x0C, 0x0C, 0x6C, 0x38, 0x00};
    static const std::uint8_t kk[] = {0x66, 0x6C, 0x78, 0x70, 0x78, 0x6C, 0x66, 0x00};
    static const std::uint8_t kl[] = {0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0x7E, 0x00};
    static const std::uint8_t km[] = {0x63, 0x77, 0x7F, 0x6B, 0x63, 0x63, 0x63, 0x00};
    static const std::uint8_t kn[] = {0x66, 0x76, 0x7E, 0x7E, 0x6E, 0x66, 0x66, 0x00};
    static const std::uint8_t ko[] = {0x3C, 0x66, 0x66, 0x66, 0x66, 0x66, 0x3C, 0x00};
    static const std::uint8_t kp[] = {0x7C, 0x66, 0x66, 0x7C, 0x60, 0x60, 0x60, 0x00};
    static const std::uint8_t kq[] = {0x3C, 0x66, 0x66, 0x66, 0x6A, 0x6C, 0x36, 0x00};
    static const std::uint8_t kr[] = {0x7C, 0x66, 0x66, 0x7C, 0x78, 0x6C, 0x66, 0x00};
    static const std::uint8_t ks[] = {0x3C, 0x66, 0x60, 0x3C, 0x06, 0x66, 0x3C, 0x00};
    static const std::uint8_t kt[] = {0x7E, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x00};
    static const std::uint8_t ku[] = {0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x3C, 0x00};
    static const std::uint8_t kv[] = {0x66, 0x66, 0x66, 0x66, 0x66, 0x3C, 0x18, 0x00};
    static const std::uint8_t kw[] = {0x63, 0x63, 0x63, 0x6B, 0x7F, 0x77, 0x63, 0x00};
    static const std::uint8_t kx[] = {0x66, 0x66, 0x3C, 0x18, 0x3C, 0x66, 0x66, 0x00};
    static const std::uint8_t ky[] = {0x66, 0x66, 0x66, 0x3C, 0x18, 0x18, 0x18, 0x00};
    static const std::uint8_t kz[] = {0x7E, 0x06, 0x0C, 0x18, 0x30, 0x60, 0x7E, 0x00};
    static const std::uint8_t kdot[] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x18, 0x00};
    static const std::uint8_t kminus[] = {0x00, 0x00, 0x00, 0x7E, 0x00, 0x00, 0x00, 0x00};
    static const std::uint8_t kpct[] = {0x62, 0x66, 0x0C, 0x18, 0x30, 0x66, 0x46, 0x00};
    static const std::uint8_t kcolon[] = {0x00, 0x18, 0x18, 0x00, 0x18, 0x18, 0x00, 0x00};
    static const std::uint8_t kspace[] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
    switch (ch) {
        case '0': return k0;
        case '1': return k1;
        case '2': return k2;
        case '3': return k3;
        case '4': return k4;
        case '5': return k5;
        case '6': return k6;
        case '7': return k7;
        case '8': return k8;
        case '9': return k9;
        case 'a': return ka;
        case 'b': return kb;
        case 'c': return kc;
        case 'd': return kd;
        case 'e': return ke;
        case 'f': return kf;
        case 'g': return kg;
        case 'h': return kh;
        case 'i': return ki;
        case 'j': return kj;
        case 'k': return kk;
        case 'l': return kl;
        case 'm': return km;
        case 'n': return kn;
        case 'o': return ko;
        case 'p': return kp;
        case 'q': return kq;
        case 'r': return kr;
        case 's': return ks;
        case 't': return kt;
        case 'u': return ku;
        case 'v': return kv;
        case 'w': return kw;
        case 'x': return kx;
        case 'y': return ky;
        case 'z': return kz;
        case '.': return kdot;
        case '-': return kminus;
        case '%': return kpct;
        case ':': return kcolon;
        case ' ': return kspace;
        default: return nullptr;
    }
}

int draw_char(OsdCanvas& canvas, int x, int y, char ch, std::uint32_t argb) {
    const std::uint8_t* rows = glyph_rows(ch);
    const int advance = 8 * kScale;
    if (rows == nullptr) {
        return advance;
    }
    for (int row = 0; row < 8; ++row) {
        for (int col = 0; col < 8; ++col) {
            if ((rows[row] & (0x80 >> col)) == 0) {
                continue;
            }
            fill_rect(canvas, x + col * kScale, y + row * kScale, kScale, kScale, argb);
        }
    }
    return advance;
}

int draw_text(OsdCanvas& canvas, int x, int y, const char* text, std::uint32_t argb) {
    int cursor = x;
    for (const char* p = text; *p != '\0'; ++p) {
        cursor += draw_char(canvas, cursor, y, *p, argb);
    }
    return cursor - x;
}

std::uint32_t box_color(int class_id) {
    static const std::uint32_t kColors[] = {0xFF00FF00u, 0xFF00FFFFu, 0xFFFFFF00u, 0xFFFF8000u};
    const int index = class_id >= 0 ? class_id : 0;
    return kColors[index & 3];
}

}  // namespace

void osd_draw(OsdCanvas& canvas, const std::vector<Detection>& detections) {
    if (canvas.pixels == nullptr || canvas.width <= 0 || canvas.height <= 0 || canvas.stride_bytes < canvas.width * 4) {
        return;
    }
    const int text_h = 8 * kScale;
    for (std::size_t i = 0; i < detections.size(); ++i) {
        const Detection& det = detections[i];
        int x = static_cast<int>(det.x);
        int y = static_cast<int>(det.y);
        int w = static_cast<int>(det.w);
        int h = static_cast<int>(det.h);
        if (w < 2 || h < 2) {
            continue;
        }
        const std::uint32_t color = box_color(det.class_id);
        stroke_rect(canvas, x, y, w, h, kBorder, color);

        char label[64];
        std::snprintf(label, sizeof(label), "%s %.0f%%", coco_name(det.class_id), det.score * 100.f);
        const int text_w = static_cast<int>(std::strlen(label)) * 8 * kScale;
        int tx = x;
        int ty = y - text_h - 2;
        if (ty < 0) {
            ty = y + kBorder;
        }
        fill_rect(canvas, tx, ty, text_w, text_h, 0xC0000000u);
        draw_text(canvas, tx, ty, label, 0xFFFFFFFFu);
    }
}

}  // namespace yolo_rknn
