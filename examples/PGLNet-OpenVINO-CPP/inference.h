#pragma once

#include <string>
#include <cstdint>
#include <vector>

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

struct DL_INIT_PARAM {
    std::string modelPath;
    std::vector<int> imgSize = {512, 512};  // H, W
    std::string device = "CPU";
    int warmupRuns = 5;
    int benchRuns = 20;
};

class PGLNetOpenVINO {
public:
    char* CreateSession(DL_INIT_PARAM& params);
    char* RunSession(cv::Mat& inputBgr, cv::Mat& outputBgr, bool resizeBack);

private:
    char* PreProcess(const cv::Mat& inputBgr, std::vector<float>& blob);
    char* PostProcess(const ov::Tensor& outTensor, const cv::Size& originalSize, cv::Mat& outputBgr, bool resizeBack);

private:
    ov::Core core;
    ov::CompiledModel compiledModel;
    ov::InferRequest inferRequest;
    ov::Output<const ov::Node> inputPort;
    ov::Output<const ov::Node> outputPort;
    std::vector<int> imgSize;
    int warmupRuns = 5;
    int benchRuns = 20;
    cv::Size originalSize;
    ov::element::Type inputType = ov::element::f32;
    ov::element::Type outputType = ov::element::f32;
};
