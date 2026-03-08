#pragma once

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

struct DL_INIT_PARAM {
    std::string modelPath;
    std::vector<int> imgSize = {512, 512};  // H, W
    int warmupRuns = 5;
    int benchRuns = 20;
    int backend = cv::dnn::DNN_BACKEND_OPENCV;
    int target = cv::dnn::DNN_TARGET_CPU;
};

class PGLNetOpenCVDNN {
public:
    char* CreateSession(DL_INIT_PARAM& params);
    char* RunSession(cv::Mat& inputBgr, cv::Mat& outputBgr, bool resizeBack);

private:
    char* PreProcess(const cv::Mat& inputBgr, cv::Mat& blob);
    char* PostProcess(const cv::Mat& out, const cv::Size& originalSize, cv::Mat& outputBgr, bool resizeBack);

private:
    cv::dnn::Net net;
    std::vector<int> imgSize;
    int warmupRuns = 5;
    int benchRuns = 20;
    cv::Size originalSize;
};

