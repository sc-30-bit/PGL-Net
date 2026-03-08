#pragma once

#include <string>
#include <cstdint>
#include <vector>

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

struct DL_INIT_PARAM {
    std::string modelPath;
    std::vector<int> imgSize = {512, 512};  // H, W
    bool cudaEnable = false;
    int intraOpNumThreads = 1;
    int warmupRuns = 5;
    int benchRuns = 20;
};

class PGLNet {
public:
    PGLNet();
    ~PGLNet();

    char* CreateSession(DL_INIT_PARAM& params);
    char* RunSession(cv::Mat& inputBgr, cv::Mat& outputBgr, bool resizeBack);
    char* WarmUpSession();

private:
    char* PreProcess(const cv::Mat& inputBgr, std::vector<float>& blob);
    char* PostProcess(const std::vector<int64_t>& outShape, const float* outData, const cv::Size& originalSize, cv::Mat& outputBgr, bool resizeBack);

private:
    Ort::Env env;
    Ort::Session* session;
    Ort::RunOptions options;
    std::vector<const char*> inputNodeNames;
    std::vector<const char*> outputNodeNames;

    std::vector<int> imgSize;
    bool cudaEnable;
    int warmupRuns;
    int benchRuns;
    cv::Size originalSize;
    ONNXTensorElementDataType inputTensorType = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    ONNXTensorElementDataType outputTensorType = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
};
