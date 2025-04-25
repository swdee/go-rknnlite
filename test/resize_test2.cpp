#include <iostream>
#include <opencv2/opencv.hpp>

// RGA core and C API
#include <rga/rga.h>
#include <im2d_buffer.h>   // importbuffer_virtualaddr, wrapbuffer_handle
#include <im2d_common.h>   // imcheck
#include <im2d_single.h>   // imresize

int main(int argc, char** argv) {

    if (argc != 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <in.jpg> <out.jpg>\n";
        return 1;
    }

    const std::string in  = argv[1];
    const std::string out = argv[2];

    // Load BGR JPEG
    cv::Mat bgr = cv::imread(in, cv::IMREAD_COLOR);
    if (bgr.empty()) {
        std::cerr << "Error: failed to load '" << in << "'\n";
        return 2;
    }

    // Convert BGR to BGRA
    cv::Mat src;
    cv::cvtColor(bgr, src, cv::COLOR_BGR2BGRA);
    if (!src.isContinuous()) src = src.clone();

    // Prepare half-size BGRA output
    int dst_w = src.cols / 2;
    int dst_h = src.rows / 2;
    cv::Mat dst(dst_h, dst_w, CV_8UC4);
    if (!dst.isContinuous()) dst = dst.clone();

    // Two-arg import (matches sample’s importbuffer_virtualaddr(ptr,size))
    int src_size = src.cols * src.rows * 4; // 4 bytes/pixel for BGRA8888
    int dst_size = dst.cols * dst.rows * 4;
    rga_buffer_handle_t hsrc = importbuffer_virtualaddr(src.data, src_size);
    rga_buffer_handle_t hdst = importbuffer_virtualaddr(dst.data, dst_size);
    if (!hsrc || !hdst) {
        std::cerr << "importbuffer_virtualaddr failed\n";
        return 3;
    }

    // Wrap into rga_buffer_t
    rga_buffer_t sbuf = wrapbuffer_handle(
        hsrc, src.cols, src.rows, RK_FORMAT_BGRA_8888);
    rga_buffer_t dbuf = wrapbuffer_handle(
        hdst, dst.cols, dst.rows, RK_FORMAT_BGRA_8888);

    // Full-image rectangles
    im_rect src_rect = {0, 0, src.cols, src.rows};
    im_rect dst_rect = {0, 0, dst.cols, dst.rows};

    // Sanity-check
    IM_STATUS chk = imcheck(sbuf, dbuf, src_rect, dst_rect, INTER_LINEAR);
    if (chk != IM_STATUS_NOERROR) {
        std::cerr << "imcheck failed: " << chk << "\n";
        releasebuffer_handle(hsrc);
        releasebuffer_handle(hdst);
        return 4;
    }

    // Hardware resize (single-call API)
    IM_STATUS st = imresize(sbuf, dbuf, 0, 0, INTER_LINEAR, 1, nullptr);
    releasebuffer_handle(hsrc);
    releasebuffer_handle(hdst);
    if (st != IM_STATUS_SUCCESS) {
        std::cerr << "imresize failed: " << st << "\n";
        return 5;
    }

    // Convert BGRA→BGR and write out JPG
    cv::Mat out_bgr;
    cv::cvtColor(dst, out_bgr, cv::COLOR_BGRA2BGR);
    if (!cv::imwrite(out, out_bgr)) {
        std::cerr << "Error: failed to write '" << out << "'\n";
        return 6;
    }

    std::cout << "Resized " << in
              << " (" << src.cols << "×" << src.rows << ") → "
              << out
              << " (" << dst.cols << "×" << dst.rows << ")\n";
    return 0;
}





