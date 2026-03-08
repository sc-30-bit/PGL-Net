#pragma once

#include <string>
#include <cstdint>
#include <vector>

#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>

struct DL_INIT_PARAM {
    std::string modelPath;
    std::vector<int> imgSize = {512, 512};  // H, W
    int warmupRuns = 5;
    int benchRuns = 20;
};

class PGLNetTensorRT {
public:
    PGLNetTensorRT();
    ~PGLNetTensorRT();

    char* CreateSession(DL_INIT_PARAM& params);
    char* RunSession(cv::Mat& inputBgr, cv::Mat& outputBgr, bool resizeBack);

private:
    char* PreProcess(const cv::Mat& inputBgr, std::vector<float>& blob);
    char* PostProcess(const std::vector<float>& outData, cv::Mat& outputBgr, bool resizeBack);
    bool isInputBinding(int index) const;
    int64_t volume(const nvinfer1::Dims& d) const;

private:
    class Logger : public nvinfer1::ILogger {
    public:
        void log(Severity severity, const char* msg) noexcept override;
    } logger;

    nvinfer1::IRuntime* runtime;
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;
    cudaStream_t stream;

    int inputIndex;
    int outputIndex;
    nvinfer1::Dims inputDims;
    nvinfer1::Dims outputDims;
    std::vector<void*> deviceBindings;

    std::vector<int> imgSize;
    int warmupRuns;
    int benchRuns;
    cv::Size originalSize;
    nvinfer1::DataType inputType = nvinfer1::DataType::kFLOAT;
    nvinfer1::DataType outputType = nvinfer1::DataType::kFLOAT;
};
