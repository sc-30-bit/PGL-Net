#include "inference.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>

#define RET_OK nullptr

char* PGLNetOpenCVDNN::CreateSession(DL_INIT_PARAM& params) {
    try {
        imgSize = params.imgSize;
        warmupRuns = params.warmupRuns;
        benchRuns = params.benchRuns;
        net = cv::dnn::readNetFromONNX(params.modelPath);
        net.setPreferableTarget(params.target);
        net.setPreferableBackend(params.backend);
        return RET_OK;
    } catch (const std::exception& e) {
        std::cerr << "[PGLNet-OpenCV-DNN] CreateSession failed: " << e.what() << "\n";
        return (char*)"[PGLNet-OpenCV-DNN] CreateSession failed.";
    }
}

char* PGLNetOpenCVDNN::PreProcess(const cv::Mat& inputBgr, cv::Mat& blob) {
    originalSize = inputBgr.size();
    cv::Mat rgb;
    cv::cvtColor(inputBgr, rgb, cv::COLOR_BGR2RGB);
    cv::resize(rgb, rgb, cv::Size(imgSize[1], imgSize[0]), 0, 0, cv::INTER_LINEAR);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);
    blob = cv::dnn::blobFromImage(rgb);
    return RET_OK;
}

char* PGLNetOpenCVDNN::PostProcess(const cv::Mat& out, const cv::Size& originalSize_, cv::Mat& outputBgr, bool resizeBack) {
    if (out.dims != 4 || out.size[0] != 1) {
        return (char*)"[PGLNet-OpenCV-DNN] Unexpected output shape.";
    }
    int c = out.size[1], h = out.size[2], w = out.size[3];
    int useC = std::min(c, 3);

    cv::Mat rgb(h, w, CV_32FC3, cv::Scalar(0, 0, 0));
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            cv::Vec3f v(0, 0, 0);
            for (int ch = 0; ch < useC; ++ch) {
                float p = out.ptr<float>(0, ch, y)[x];
                v[ch] = std::min(1.0f, std::max(0.0f, p));
            }
            rgb.at<cv::Vec3f>(y, x) = v;
        }
    }

    cv::Mat outU8;
    rgb.convertTo(outU8, CV_8UC3, 255.0);
    cv::cvtColor(outU8, outU8, cv::COLOR_RGB2BGR);
    if (resizeBack) {
        cv::resize(outU8, outputBgr, originalSize_, 0, 0, cv::INTER_LINEAR);
    } else {
        outputBgr = outU8;
    }
    return RET_OK;
}

char* PGLNetOpenCVDNN::RunSession(cv::Mat& inputBgr, cv::Mat& outputBgr, bool resizeBack) {
    cv::Mat blob;
    auto ret = PreProcess(inputBgr, blob);
    if (ret != RET_OK) return ret;

    for (int i = 0; i < warmupRuns; ++i) {
        net.setInput(blob);
        (void)net.forward();
    }

    std::vector<double> cost;
    cv::Mat out;
    for (int i = 0; i < benchRuns; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        net.setInput(blob);
        out = net.forward();
        auto t1 = std::chrono::high_resolution_clock::now();
        cost.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    ret = PostProcess(out, originalSize, outputBgr, resizeBack);
    if (ret != RET_OK) return ret;

    double avg = std::accumulate(cost.begin(), cost.end(), 0.0) / static_cast<double>(cost.size());
    auto [mn, mx] = std::minmax_element(cost.begin(), cost.end());
    std::cout << "[PGLNet-OpenCV-DNN] avg/min/max: " << avg << "/" << *mn << "/" << *mx << " ms, FPS: " << 1000.0 / avg << "\n";
    return RET_OK;
}
