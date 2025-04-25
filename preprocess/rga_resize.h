#pragma once
#ifdef __cplusplus
extern "C" {
#endif

// resize_rga:
//  - src_va: pointer to BGRA8888 source pixels
//  - src_w, src_h: source width & height
//  - src_fmt: RK_FORMAT_BGRA_8888
//  - dst_va, dst_w, dst_h, dst_fmt: likewise for destination
// Returns 0 on success, or an IM_STATUS error code.
int resize_rga(void* src_va, int src_w, int src_h, int src_fmt,
               void* dst_va, int dst_w, int dst_h, int dst_fmt);


int resize_rga_init(void* src_va, int src_w, int src_h, int src_fmt,
                    void* dst_va, int dst_w, int dst_h, int dst_fmt);

int resize_rga_frame();

void resize_rga_deinit();

#ifdef __cplusplus
}
#endif
