#include "yolo_rknn/display/drm_display.hpp"

#include "yolo_rknn/common/dma_buf.hpp"
#include "yolo_rknn/common/logger.hpp"
#include "yolo_rknn/display/osd_painter.hpp"
#include "yolo_rknn/preprocess/rga_lock.hpp"

#include "im2d.h"

#include <xf86drm.h>
#include <xf86drmMode.h>

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <vector>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <utility>

namespace yolo_rknn {
namespace {

constexpr std::uint32_t fourcc(char a, char b, char c, char d) {
    return static_cast<std::uint32_t>(a) | (static_cast<std::uint32_t>(b) << 8) |
           (static_cast<std::uint32_t>(c) << 16) | (static_cast<std::uint32_t>(d) << 24);
}

constexpr std::uint32_t kNv12 = fourcc('N', 'V', '1', '2');
constexpr std::uint32_t kArgb = fourcc('A', 'R', '2', '4');
constexpr std::uint32_t kXrgb = fourcc('X', 'R', '2', '4');

// drm_plane_type：0 overlay，1 primary，2 cursor。
constexpr std::uint64_t kPlanePrimary = 1;
constexpr std::uint64_t kPlaneCursor = 2;

std::string errno_text(int err) { return std::strerror(err); }

bool plane_has_format(const drmModePlane* plane, std::uint32_t format) {
    for (std::uint32_t i = 0; i < plane->count_formats; ++i) {
        if (plane->formats[i] == format) {
            return true;
        }
    }
    return false;
}

std::uint64_t plane_type(int fd, std::uint32_t plane_id) {
    drmModeObjectProperties* props = drmModeObjectGetProperties(fd, plane_id, DRM_MODE_OBJECT_PLANE);
    std::uint64_t type = 0;
    if (props == nullptr) {
        return type;
    }
    for (std::uint32_t i = 0; i < props->count_props; ++i) {
        drmModePropertyRes* prop = drmModeGetProperty(fd, props->props[i]);
        if (prop != nullptr && std::strcmp(prop->name, "type") == 0) {
            type = props->prop_values[i];
        }
        if (prop != nullptr) {
            drmModeFreeProperty(prop);
        }
    }
    drmModeFreeObjectProperties(props);
    return type;
}

void fit_on_mode(int src_w, int src_h, int mode_w, int mode_h, int* x, int* y, int* w, int* h) {
    if (src_w <= 0 || src_h <= 0 || mode_w <= 0 || mode_h <= 0) {
        *x = 0;
        *y = 0;
        *w = mode_w;
        *h = mode_h;
        return;
    }
    const double sx = static_cast<double>(mode_w) / static_cast<double>(src_w);
    const double sy = static_cast<double>(mode_h) / static_cast<double>(src_h);
    const double scale = sx < sy ? sx : sy;
    *w = static_cast<int>(static_cast<double>(src_w) * scale) & ~1;
    *h = static_cast<int>(static_cast<double>(src_h) * scale) & ~1;
    if (*w < 2) {
        *w = 2;
    }
    if (*h < 2) {
        *h = 2;
    }
    *x = ((mode_w - *w) / 2) & ~1;
    *y = ((mode_h - *h) / 2) & ~1;
}

struct ImportedNv12 {
    std::uint32_t fb = 0;
    std::uint32_t gem = 0;
    DecodedFrame frame;
};

}  // namespace

struct DrmDisplay::Impl {
    int fd = -1;
    std::uint32_t crtc_id = 0;
    std::uint32_t connector_id = 0;
    std::uint32_t video_plane = 0;
    std::uint32_t osd_plane = 0;
    drmModeModeInfo mode;
    bool mode_valid = false;
    bool crtc_on = false;

    std::uint32_t blank_handle = 0;
    std::uint32_t blank_fb = 0;

    ImportedNv12 shown[2];
    int shown_count = 0;

    std::uint32_t osd_handle = 0;
    std::uint32_t osd_fb = 0;
    int osd_prime = -1;
    void* osd_map = nullptr;
    std::uint64_t osd_map_size = 0;
    int osd_w = 0;
    int osd_h = 0;
    int osd_pitch = 0;
    bool osd_plane_set = false;
    bool osd_plane_warned = false;
    bool sync_warned = false;
    int placed_x = -1;
    int placed_y = -1;
    int placed_w = 0;
    int placed_h = 0;

    int dst_x = 0;
    int dst_y = 0;
    int dst_w = 0;
    int dst_h = 0;

    ~Impl() = default;
};

DrmDisplay::DrmDisplay(const AppConfig& config) : config_(config), impl_(new Impl) {
    std::memset(&impl_->mode, 0, sizeof(impl_->mode));
}

DrmDisplay::~DrmDisplay() {
    if (impl_ == nullptr) {
        return;
    }
    if (impl_->fd >= 0) {
        for (int i = 0; i < impl_->shown_count; ++i) {
            if (impl_->shown[i].fb != 0) {
                drmModeRmFB(impl_->fd, impl_->shown[i].fb);
            }
            if (impl_->shown[i].gem != 0) {
                drm_gem_close close_arg;
                std::memset(&close_arg, 0, sizeof(close_arg));
                close_arg.handle = impl_->shown[i].gem;
                ioctl(impl_->fd, DRM_IOCTL_GEM_CLOSE, &close_arg);
            }
        }
        if (impl_->osd_map != nullptr && impl_->osd_map != MAP_FAILED) {
            munmap(impl_->osd_map, static_cast<size_t>(impl_->osd_map_size));
        }
        if (impl_->osd_prime >= 0) {
            close(impl_->osd_prime);
        }
        if (impl_->osd_fb != 0) {
            drmModeRmFB(impl_->fd, impl_->osd_fb);
        }
        if (impl_->osd_handle != 0) {
            drm_gem_close close_arg;
            std::memset(&close_arg, 0, sizeof(close_arg));
            close_arg.handle = impl_->osd_handle;
            ioctl(impl_->fd, DRM_IOCTL_GEM_CLOSE, &close_arg);
        }
        if (impl_->blank_fb != 0) {
            drmModeRmFB(impl_->fd, impl_->blank_fb);
        }
        if (impl_->blank_handle != 0) {
            drm_gem_close close_arg;
            std::memset(&close_arg, 0, sizeof(close_arg));
            close_arg.handle = impl_->blank_handle;
            ioctl(impl_->fd, DRM_IOCTL_GEM_CLOSE, &close_arg);
        }
        close(impl_->fd);
    }
    delete impl_;
    impl_ = nullptr;
}

Status DrmDisplay::init() {
    const std::string& path = config_.display_device.empty() ? std::string("/dev/dri/card0") : config_.display_device;
    impl_->fd = open(path.c_str(), O_RDWR | O_CLOEXEC);
    if (impl_->fd < 0) {
        return Status::Fail("open " + path + ": " + errno_text(errno));
    }
    drmSetClientCap(impl_->fd, DRM_CLIENT_CAP_UNIVERSAL_PLANES, 1);

    drmModeRes* res = drmModeGetResources(impl_->fd);
    if (res == nullptr) {
        return Status::Fail("drmModeGetResources failed");
    }

    drmModeConnector* chosen = nullptr;
    for (int i = 0; i < res->count_connectors; ++i) {
        drmModeConnector* connector = drmModeGetConnector(impl_->fd, res->connectors[i]);
        if (connector == nullptr) {
            continue;
        }
        const bool ok = connector->connection == DRM_MODE_CONNECTED && connector->count_modes > 0;
        const bool hdmi = connector->connector_type == DRM_MODE_CONNECTOR_HDMIA;
        if (ok && (chosen == nullptr || hdmi)) {
            if (chosen != nullptr) {
                drmModeFreeConnector(chosen);
            }
            chosen = connector;
            if (hdmi) {
                break;
            }
        } else {
            drmModeFreeConnector(connector);
        }
    }
    if (chosen == nullptr) {
        drmModeFreeResources(res);
        return Status::Fail("no connected display with a mode. HDMI-A-1, HDMI-A-2 and DP-1 are unplugged");
    }

    impl_->connector_id = chosen->connector_id;
    impl_->mode = chosen->modes[0];
    for (int i = 0; i < chosen->count_modes; ++i) {
        if (chosen->modes[i].hdisplay == 1920 && chosen->modes[i].vdisplay == 1080) {
            impl_->mode = chosen->modes[i];
            break;
        }
    }
    impl_->mode_valid = true;

    std::uint32_t crtc_id = 0;
    if (chosen->encoder_id != 0) {
        drmModeEncoder* encoder = drmModeGetEncoder(impl_->fd, chosen->encoder_id);
        if (encoder != nullptr) {
            crtc_id = encoder->crtc_id;
            drmModeFreeEncoder(encoder);
        }
    }
    if (crtc_id == 0) {
        for (int e = 0; e < res->count_encoders && crtc_id == 0; ++e) {
            drmModeEncoder* encoder = drmModeGetEncoder(impl_->fd, res->encoders[e]);
            if (encoder == nullptr) {
                continue;
            }
            for (int c = 0; c < res->count_crtcs; ++c) {
                if ((encoder->possible_crtcs & (1u << c)) != 0) {
                    crtc_id = res->crtcs[c];
                    break;
                }
            }
            drmModeFreeEncoder(encoder);
        }
    }
    drmModeFreeConnector(chosen);
    if (crtc_id == 0) {
        drmModeFreeResources(res);
        return Status::Fail("display connector has no CRTC");
    }
    impl_->crtc_id = crtc_id;

    int crtc_index = -1;
    for (int i = 0; i < res->count_crtcs; ++i) {
        if (res->crtcs[i] == crtc_id) {
            crtc_index = i;
            break;
        }
    }
    drmModeFreeResources(res);
    if (crtc_index < 0) {
        return Status::Fail("CRTC is not in the resource list");
    }
    const std::uint32_t crtc_bit = 1u << crtc_index;

    drmModePlaneRes* planes = drmModeGetPlaneResources(impl_->fd);
    if (planes == nullptr) {
        return Status::Fail("drmModeGetPlaneResources failed");
    }
    std::uint32_t video_primary = 0;
    std::uint32_t video_any = 0;
    std::uint32_t osd_rgb = 0;
    std::uint32_t osd_any = 0;
    for (std::uint32_t i = 0; i < planes->count_planes; ++i) {
        drmModePlane* plane = drmModeGetPlane(impl_->fd, planes->planes[i]);
        if (plane == nullptr || (plane->possible_crtcs & crtc_bit) == 0) {
            drmModeFreePlane(plane);
            continue;
        }
        const std::uint64_t type = plane_type(impl_->fd, plane->plane_id);
        if (type == kPlaneCursor) {
            drmModeFreePlane(plane);
            continue;
        }
        const bool nv12 = plane_has_format(plane, kNv12);
        const bool argb = plane_has_format(plane, kArgb);
        if (nv12 && type == kPlanePrimary) {
            video_primary = plane->plane_id;
        } else if (nv12 && video_any == 0) {
            video_any = plane->plane_id;
        }
        if (argb && !nv12 && osd_rgb == 0) {
            osd_rgb = plane->plane_id;
        } else if (argb && osd_any == 0) {
            osd_any = plane->plane_id;
        }
        drmModeFreePlane(plane);
    }
    drmModeFreePlaneResources(planes);

    impl_->video_plane = video_primary != 0 ? video_primary : video_any;
    impl_->osd_plane = osd_rgb != 0 ? osd_rgb : osd_any;
    if (impl_->osd_plane == impl_->video_plane) {
        impl_->osd_plane = osd_any != impl_->video_plane ? osd_any : 0;
    }
    if (impl_->video_plane == 0 || impl_->osd_plane == 0) {
        return Status::Fail("this CRTC has no NV12 plane plus a separate ARGB plane");
    }

    drm_mode_create_dumb create_req;
    std::memset(&create_req, 0, sizeof(create_req));
    create_req.width = impl_->mode.hdisplay;
    create_req.height = impl_->mode.vdisplay;
    create_req.bpp = 32;
    if (ioctl(impl_->fd, DRM_IOCTL_MODE_CREATE_DUMB, &create_req) != 0) {
        return Status::Fail(std::string("create blank scanout: ") + errno_text(errno));
    }
    impl_->blank_handle = create_req.handle;
    if (drmModeAddFB(impl_->fd, impl_->mode.hdisplay, impl_->mode.vdisplay, 24, 32, create_req.pitch,
                     impl_->blank_handle, &impl_->blank_fb) != 0) {
        return Status::Fail("drmModeAddFB blank failed");
    }
    if (drmModeSetCrtc(impl_->fd, impl_->crtc_id, impl_->blank_fb, 0, 0, &impl_->connector_id, 1, &impl_->mode) !=
        0) {
        return Status::Fail(std::string("drmModeSetCrtc: ") + errno_text(errno));
    }
    impl_->crtc_on = true;
    YR_LOGI("display %ux%u crtc=%u video_plane=%u osd_plane=%u", impl_->mode.hdisplay, impl_->mode.vdisplay,
            impl_->crtc_id, impl_->video_plane, impl_->osd_plane);
    return Status::Ok();
}

bool DrmDisplay::place_osd_plane() {
    if (!impl_->crtc_on || impl_->osd_fb == 0 || impl_->dst_w < 2 || impl_->dst_h < 2) {
        return false;
    }
    // 视频层可以缩放 NV12。OSD 层是 RGB overlay，驱动里没有缩放器：
    // 源和目标只要差一个像素，drmModeSetPlane 就返回 ERANGE。
    if (impl_->osd_w != impl_->dst_w || impl_->osd_h != impl_->dst_h) {
        return false;
    }
    if (impl_->osd_plane_set && impl_->placed_x == impl_->dst_x && impl_->placed_y == impl_->dst_y &&
        impl_->placed_w == impl_->dst_w && impl_->placed_h == impl_->dst_h) {
        return true;
    }
    if (drmModeSetPlane(impl_->fd, impl_->osd_plane, impl_->crtc_id, impl_->osd_fb, 0, impl_->dst_x, impl_->dst_y,
                        static_cast<std::uint32_t>(impl_->dst_w), static_cast<std::uint32_t>(impl_->dst_h), 0, 0,
                        static_cast<std::uint32_t>(impl_->osd_w) << 16,
                        static_cast<std::uint32_t>(impl_->osd_h) << 16) != 0) {
        if (!impl_->osd_plane_warned) {
            YR_LOGW("OSD SetPlane: %s (src %dx%d dst %d,%d %dx%d)", errno_text(errno).c_str(), impl_->osd_w,
                    impl_->osd_h, impl_->dst_x, impl_->dst_y, impl_->dst_w, impl_->dst_h);
            impl_->osd_plane_warned = true;
        }
        return false;
    }
    impl_->osd_plane_set = true;
    impl_->placed_x = impl_->dst_x;
    impl_->placed_y = impl_->dst_y;
    impl_->placed_w = impl_->dst_w;
    impl_->placed_h = impl_->dst_h;
    return true;
}

void DrmDisplay::update_osd(const InferenceResult& result) {
    // 缓冲跟屏幕上的目标矩形一样大，不跟原图一样大。1280x720 放到 1920x1080 时，
    // 原图尺寸拿去 SetPlane 会被 RGB 平面拒绝。
    const int width = impl_->dst_w;
    const int height = impl_->dst_h;
    if (width < 2 || height < 2 || impl_->fd < 0) {
        return;
    }
    if (impl_->osd_fb == 0 || width != impl_->osd_w || height != impl_->osd_h) {
        if (impl_->osd_map != nullptr && impl_->osd_map != MAP_FAILED) {
            munmap(impl_->osd_map, static_cast<size_t>(impl_->osd_map_size));
            impl_->osd_map = nullptr;
        }
        if (impl_->osd_prime >= 0) {
            close(impl_->osd_prime);
            impl_->osd_prime = -1;
        }
        if (impl_->osd_fb != 0) {
            drmModeRmFB(impl_->fd, impl_->osd_fb);
            impl_->osd_fb = 0;
        }
        if (impl_->osd_handle != 0) {
            drm_gem_close close_arg;
            std::memset(&close_arg, 0, sizeof(close_arg));
            close_arg.handle = impl_->osd_handle;
            ioctl(impl_->fd, DRM_IOCTL_GEM_CLOSE, &close_arg);
            impl_->osd_handle = 0;
        }
        drm_mode_create_dumb create_req;
        std::memset(&create_req, 0, sizeof(create_req));
        create_req.width = static_cast<std::uint32_t>(width);
        create_req.height = static_cast<std::uint32_t>(height);
        create_req.bpp = 32;
        if (ioctl(impl_->fd, DRM_IOCTL_MODE_CREATE_DUMB, &create_req) != 0) {
            YR_LOGE("create OSD buffer: %s", errno_text(errno).c_str());
            return;
        }
        impl_->osd_handle = create_req.handle;
        impl_->osd_pitch = static_cast<int>(create_req.pitch);
        impl_->osd_map_size = create_req.size;
        std::uint32_t handles[4] = {impl_->osd_handle, 0, 0, 0};
        std::uint32_t pitches[4] = {create_req.pitch, 0, 0, 0};
        std::uint32_t offsets[4] = {0, 0, 0, 0};
        if (drmModeAddFB2(impl_->fd, static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height), kArgb,
                          handles, pitches, offsets, &impl_->osd_fb, 0) != 0) {
            YR_LOGE("drmModeAddFB2 OSD failed");
            return;
        }
        if (drmPrimeHandleToFD(impl_->fd, impl_->osd_handle, DRM_CLOEXEC, &impl_->osd_prime) != 0) {
            YR_LOGE("OSD prime fd failed");
            return;
        }
        drm_mode_map_dumb map_req;
        std::memset(&map_req, 0, sizeof(map_req));
        map_req.handle = impl_->osd_handle;
        if (ioctl(impl_->fd, DRM_IOCTL_MODE_MAP_DUMB, &map_req) != 0) {
            YR_LOGE("map OSD buffer: %s", errno_text(errno).c_str());
            return;
        }
        impl_->osd_map = mmap(nullptr, static_cast<size_t>(create_req.size), PROT_READ | PROT_WRITE, MAP_SHARED,
                              impl_->fd, static_cast<off_t>(map_req.offset));
        if (impl_->osd_map == MAP_FAILED) {
            impl_->osd_map = nullptr;
            YR_LOGE("mmap OSD buffer failed");
            return;
        }
        impl_->osd_w = width;
        impl_->osd_h = height;
        impl_->osd_plane_set = false;
    }

    // 整块清透明交给 RGA。CPU 只描边和写字，避免每帧 memset 一张 1080p。
    const int stride_px = impl_->osd_pitch / 4;
    rga_buffer_t dst = wrapbuffer_fd_t(impl_->osd_prime, impl_->osd_w, impl_->osd_h, stride_px, impl_->osd_h,
                                      RK_FORMAT_BGRA_8888);
    im_rect rect;
    rect.x = 0;
    rect.y = 0;
    rect.width = impl_->osd_w;
    rect.height = impl_->osd_h;
    IM_STATUS filled = IM_STATUS_FAILED;
    {
        RgaApiLock lock;
        filled = imfill(dst, rect, 0);
    }
    if (!dma_buf_begin_cpu_write(impl_->osd_prime) && !impl_->sync_warned) {
        YR_LOGW("OSD dma-buf sync failed; boxes may flicker if the scanout cache is stale");
        impl_->sync_warned = true;
    }
    if (filled != IM_STATUS_SUCCESS && filled != IM_STATUS_NOERROR) {
        std::memset(impl_->osd_map, 0, static_cast<size_t>(impl_->osd_map_size));
    }
    OsdCanvas canvas;
    canvas.pixels = static_cast<std::uint8_t*>(impl_->osd_map);
    canvas.width = impl_->osd_w;
    canvas.height = impl_->osd_h;
    canvas.stride_bytes = impl_->osd_pitch;
    std::vector<Detection> boxes = result.detections;
    if (result.src_width > 0 && result.src_height > 0) {
        const float sx = static_cast<float>(impl_->osd_w) / static_cast<float>(result.src_width);
        const float sy = static_cast<float>(impl_->osd_h) / static_cast<float>(result.src_height);
        for (std::size_t i = 0; i < boxes.size(); ++i) {
            boxes[i].x *= sx;
            boxes[i].y *= sy;
            boxes[i].w *= sx;
            boxes[i].h *= sy;
        }
    }
    osd_draw(canvas, boxes);
    dma_buf_end_cpu_write(impl_->osd_prime);
    place_osd_plane();
}

Status DrmDisplay::present(const DecodedFrame& frame) {
    if (!impl_->crtc_on || frame.fd < 0 || frame.width < 2 || frame.height < 2) {
        return Status::Fail("display is not ready or the frame has no NV12 fd");
    }
    std::uint32_t gem = 0;
    if (drmPrimeFDToHandle(impl_->fd, frame.fd, &gem) != 0) {
        return Status::Fail(std::string("import NV12 dma-buf: ") + errno_text(errno));
    }
    std::uint32_t handles[4] = {gem, gem, 0, 0};
    std::uint32_t pitches[4] = {static_cast<std::uint32_t>(frame.hor_stride),
                                static_cast<std::uint32_t>(frame.hor_stride), 0, 0};
    std::uint32_t offsets[4] = {0, static_cast<std::uint32_t>(frame.hor_stride) * static_cast<std::uint32_t>(frame.ver_stride),
                                0, 0};
    std::uint32_t fb = 0;
    if (drmModeAddFB2(impl_->fd, static_cast<std::uint32_t>(frame.width), static_cast<std::uint32_t>(frame.height), kNv12,
                      handles, pitches, offsets, &fb, 0) != 0) {
        drm_gem_close close_arg;
        std::memset(&close_arg, 0, sizeof(close_arg));
        close_arg.handle = gem;
        ioctl(impl_->fd, DRM_IOCTL_GEM_CLOSE, &close_arg);
        return Status::Fail(std::string("drmModeAddFB2 NV12: ") + errno_text(errno));
    }

    fit_on_mode(frame.width, frame.height, impl_->mode.hdisplay, impl_->mode.vdisplay, &impl_->dst_x, &impl_->dst_y,
                &impl_->dst_w, &impl_->dst_h);
    const std::uint32_t src_w = static_cast<std::uint32_t>(frame.width & ~1);
    const std::uint32_t src_h = static_cast<std::uint32_t>(frame.height & ~1);
    if (drmModeSetPlane(impl_->fd, impl_->video_plane, impl_->crtc_id, fb, 0, impl_->dst_x, impl_->dst_y, impl_->dst_w,
                        impl_->dst_h, 0, 0, src_w << 16, src_h << 16) != 0) {
        drmModeRmFB(impl_->fd, fb);
        drm_gem_close close_arg;
        std::memset(&close_arg, 0, sizeof(close_arg));
        close_arg.handle = gem;
        ioctl(impl_->fd, DRM_IOCTL_GEM_CLOSE, &close_arg);
        return Status::Fail(std::string("video SetPlane: ") + errno_text(errno));
    }

    // 显示控制器可能还在扫上一帧。只释放再往前的那一张，当前和上一张都握着。
    if (impl_->shown_count == 2) {
        if (impl_->shown[0].fb != 0) {
            drmModeRmFB(impl_->fd, impl_->shown[0].fb);
        }
        if (impl_->shown[0].gem != 0) {
            drm_gem_close close_arg;
            std::memset(&close_arg, 0, sizeof(close_arg));
            close_arg.handle = impl_->shown[0].gem;
            ioctl(impl_->fd, DRM_IOCTL_GEM_CLOSE, &close_arg);
        }
        impl_->shown[0] = std::move(impl_->shown[1]);
        impl_->shown_count = 1;
    }
    ImportedNv12 imported;
    imported.fb = fb;
    imported.gem = gem;
    imported.frame = frame;
    impl_->shown[impl_->shown_count] = std::move(imported);
    ++impl_->shown_count;

    place_osd_plane();
    return Status::Ok();
}

}  // namespace yolo_rknn
