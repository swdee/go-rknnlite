#include <cstddef>    // for NULL
#include <cstring>    // for memset
#include <cstdio>     // for printf in im2d_common.h
#include <sys/mman.h>          // mlock, munlock

#include "rga_resize.h"
#include <rga.h>
#include <im2d_buffer.h>       // importbuffer_virtualaddr, wrapbuffer_handle
#include <im2d_common.h>       // imcheck
#include <im2d_single.h>       // imresize

static bool   inited = false;
static rga_buffer_handle_t hsrc, hdst;
static rga_buffer_t        sbuf, dbuf;

extern "C" {

int resize_rga(void* src_va, int src_w, int src_h, int src_fmt,
               void* dst_va, int dst_w, int dst_h, int dst_fmt)
{
    // calculate sizes in bytes (4 B/pixel for BGRA8888)
    const int src_size = src_w * src_h * 4;
    const int dst_size = dst_w * dst_h * 4;

    // import your Mat.data pointers
    rga_buffer_handle_t hsrc = importbuffer_virtualaddr(src_va, src_size);
    rga_buffer_handle_t hdst = importbuffer_virtualaddr(dst_va, dst_size);
    if (!hsrc || !hdst) {
        // undo the mlocks before returning
        return -1;
    }

    // wrap into RGA buffers
    rga_buffer_t sbuf = wrapbuffer_handle(hsrc, src_w, src_h, src_fmt);
    rga_buffer_t dbuf = wrapbuffer_handle(hdst, dst_w, dst_h, dst_fmt);

    // full-image rects
    im_rect src_rect = {0, 0, src_w, src_h};
    im_rect dst_rect = {0, 0, dst_w, dst_h};

    // sanity-check
    IM_STATUS chk = imcheck(sbuf, dbuf, src_rect, dst_rect, INTER_LINEAR);
    if (chk != IM_STATUS_NOERROR) {
        releasebuffer_handle(hsrc);
        releasebuffer_handle(hdst);
        return (int)chk;
    }

    // hardware resize
    IM_STATUS st = imresize(sbuf, dbuf, 0, 0, INTER_LINEAR, 1, nullptr);

    // cleanup
    releasebuffer_handle(hsrc);
    releasebuffer_handle(hdst);

    // map SUCCESS→0, else error code
    if (st == IM_STATUS_SUCCESS) {
        return 0;
    } else {
        return (int)st;
    }
}


int resize_rga_init(void* src_va, int src_w, int src_h, int src_fmt,
                    void* dst_va, int dst_w, int dst_h, int dst_fmt)
{
    if (inited) return 0;

    int src_size = src_w * src_h * 4;
    int dst_size = dst_w * dst_h * 4;

    hsrc = importbuffer_virtualaddr(src_va, src_size);
    hdst = importbuffer_virtualaddr(dst_va, dst_size);
    if (!hsrc || !hdst) return -1;

    sbuf = wrapbuffer_handle(hsrc, src_w, src_h, src_fmt);
    dbuf = wrapbuffer_handle(hdst, dst_w, dst_h, dst_fmt);

    im_rect sr = {0,0,src_w,src_h}, dr = {0,0,dst_w,dst_h};
    if (imcheck(sbuf, dbuf, sr, dr, INTER_LINEAR) != IM_STATUS_NOERROR)
        return -1;

    inited = true;
    return 0;
}

int resize_rga_frame()
{
    IM_STATUS st = imresize(sbuf, dbuf, 0,0, INTER_LINEAR, 1, nullptr);
    return (st == IM_STATUS_SUCCESS) ? 0 : (int)st;
}


void resize_rga_deinit()
{
    if (!inited) return;
    releasebuffer_handle(hsrc);
    releasebuffer_handle(hdst);
    inited = false;
}



}  // extern "C"
