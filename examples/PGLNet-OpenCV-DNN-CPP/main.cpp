#include <iostream>
#include <string>
#include <algorithm>

#include <opencv2/opencv.hpp>

#include "inference.h"

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cout << "Usage: pglnet_opencv_dnn <model.onnx> <input.jpg> <output.jpg> [runs=20] [resize_back=0|1] [device=cpu|gpu|gpu_fp16]\n";
        return 1;
    }

    const std::string modelPath = argv[1];
    const std::string inputPath = argv[2];
    const std::string outputPath = argv[3];
    const int runs = argc > 4 ? std::max(1, std::stoi(argv[4])) : 20;
    const bool resizeBack = argc > 5 ? std::stoi(argv[5]) != 0 : false;
    const std::string device = argc > 6 ? argv[6] : "cpu";

    cv::Mat input = cv::imread(inputPath);
    if (input.empty()) {
        std::cerr << "Failed to read input image: " << inputPath << "\n";
        return 1;
    }

    DL_INIT_PARAM params;
    params.modelPath = modelPath;
    params.benchRuns = runs;
    if (device == "gpu") {
        auto cudaTargets = cv::dnn::getAvailableTargets(cv::dnn::DNN_BACKEND_CUDA);
        if (cudaTargets.empty()) {
            std::cerr << "[PGLNet-OpenCV-DNN] CUDA backend is not available in this OpenCV build/runtime.\n";
            return 1;
        }
        params.backend = cv::dnn::DNN_BACKEND_CUDA;
        if (std::find(cudaTargets.begin(), cudaTargets.end(), cv::dnn::DNN_TARGET_CUDA) != cudaTargets.end()) {
            params.target = cv::dnn::DNN_TARGET_CUDA;
        } else if (std::find(cudaTargets.begin(), cudaTargets.end(), cv::dnn::DNN_TARGET_CUDA_FP16) != cudaTargets.end()) {
            params.target = cv::dnn::DNN_TARGET_CUDA_FP16;
        } else {
            std::cerr << "[PGLNet-OpenCV-DNN] No compatible CUDA target found for DNN backend.\n";
            return 1;
        }
    } else if (device == "gpu_fp16") {
        auto cudaTargets = cv::dnn::getAvailableTargets(cv::dnn::DNN_BACKEND_CUDA);
        if (std::find(cudaTargets.begin(), cudaTargets.end(), cv::dnn::DNN_TARGET_CUDA_FP16) == cudaTargets.end()) {
            std::cerr << "[PGLNet-OpenCV-DNN] CUDA FP16 target is not available.\n";
            return 1;
        }
        params.backend = cv::dnn::DNN_BACKEND_CUDA;
        params.target = cv::dnn::DNN_TARGET_CUDA_FP16;
    } else {
        params.backend = cv::dnn::DNN_BACKEND_OPENCV;
        params.target = cv::dnn::DNN_TARGET_CPU;
    }
    std::cout << "[PGLNet-OpenCV-DNN] device mode: " << device << "\n";

    PGLNetOpenCVDNN net;
    char* ret = net.CreateSession(params);
    if (ret != nullptr) {
        std::cerr << ret << "\n";
        return 1;
    }

    cv::Mat output;
    ret = net.RunSession(input, output, resizeBack);
    if (ret != nullptr) {
        std::cerr << ret << "\n";
        return 1;
    }

    cv::imwrite(outputPath, output);
    cv::Mat showInput = input;
    if (showInput.size() != output.size()) cv::resize(showInput, showInput, output.size());
    cv::Mat compare;
    cv::hconcat(showInput, output, compare);
    std::string comparePath = outputPath;
    auto dot = comparePath.find_last_of('.');
    if (dot == std::string::npos) dot = comparePath.size();
    comparePath.insert(dot, "_compare");
    cv::imwrite(comparePath, compare);

    std::cout << "Saved: " << outputPath << "\n";
    std::cout << "Saved: " << comparePath << "\n";
    return 0;
}
