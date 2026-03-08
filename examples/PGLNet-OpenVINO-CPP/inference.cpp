#include "inference.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <numeric>

#define RET_OK nullptr

char* PGLNetOpenVINO::CreateSession(DL_INIT_PARAM& params) {
    try {
        imgSize = params.imgSize;
        warmupRuns = params.warmupRuns;
        benchRuns = params.benchRuns;

        auto model = core.read_model(params.modelPath);
        inputPort = model->input(0);
        outputPort = model->output(0);
        inputType = inputPort.get_element_type();
        outputType = outputPort.get_element_type();
        compiledModel = core.compile_model(model, params.device);
        inferRequest = compiledModel.create_infer_request();
        std::cout << "[PGLNet-OpenVINO] input dtype: " << inputType << ", output dtype: " << outputType << "\n";
        return RET_OK;
    } catch (const std::exception& e) {
        std::cerr << "[PGLNet-OpenVINO] CreateSession failed: " << e.what() << "\n";
        return (char*)"[PGLNet-OpenVINO] CreateSession failed.";
    }
}

char* PGLNetOpenVINO::PreProcess(const cv::Mat& inputBgr, std::vector<float>& blob) {
    originalSize = inputBgr.size();
    cv::Mat rgb;
    cv::cvtColor(inputBgr, rgb, cv::COLOR_BGR2RGB);
    cv::resize(rgb, rgb, cv::Size(imgSize[1], imgSize[0]), 0, 0, cv::INTER_LINEAR);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);
    std::vector<cv::Mat> chw(3);
    cv::split(rgb, chw);
    int h = imgSize[0], w = imgSize[1];
    blob.resize(static_cast<size_t>(3 * h * w));
    size_t plane = static_cast<size_t>(h * w);
    std::memcpy(blob.data() + plane * 0, chw[0].data, plane * sizeof(float));
    std::memcpy(blob.data() + plane * 1, chw[1].data, plane * sizeof(float));
    std::memcpy(blob.data() + plane * 2, chw[2].data, plane * sizeof(float));
    return RET_OK;
}

char* PGLNetOpenVINO::PostProcess(const ov::Tensor& outTensor, const cv::Size& originalSize_, cv::Mat& outputBgr, bool resizeBack) {
    auto shape = outTensor.get_shape();
    if (shape.size() != 4 || shape[0] != 1) {
        return (char*)"[PGLNet-OpenVINO] Unexpected output shape.";
    }
    int c = static_cast<int>(shape[1]);
    int h = static_cast<int>(shape[2]);
    int w = static_cast<int>(shape[3]);
    int useC = std::min(c, 3);

    std::vector<float> outFp32(static_cast<size_t>(c * h * w));
    if (outputType == ov::element::f16) {
        const ov::float16* out16 = outTensor.data<const ov::float16>();
        for (size_t i = 0; i < outFp32.size(); ++i) outFp32[i] = static_cast<float>(out16[i]);
    } else if (outputType == ov::element::f32) {
        const float* out32 = outTensor.data<const float>();
        std::memcpy(outFp32.data(), out32, outFp32.size() * sizeof(float));
    } else {
        return (char*)"[PGLNet-OpenVINO] Unsupported output dtype.";
    }

    const float* out = outFp32.data();
    cv::Mat rgb(h, w, CV_32FC3, cv::Scalar(0, 0, 0));
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            cv::Vec3f v(0, 0, 0);
            for (int ch = 0; ch < useC; ++ch) {
                float p = out[ch * h * w + y * w + x];
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

char* PGLNetOpenVINO::RunSession(cv::Mat& inputBgr, cv::Mat& outputBgr, bool resizeBack) {
    std::vector<float> blob;
    auto ret = PreProcess(inputBgr, blob);
    if (ret != RET_OK) return ret;

    ov::Shape inShape = {1, 3, static_cast<size_t>(imgSize[0]), static_cast<size_t>(imgSize[1])};
    ov::Tensor inputTensor(inputType, inShape);
    if (inputType == ov::element::f16) {
        ov::float16* dst = inputTensor.data<ov::float16>();
        for (size_t i = 0; i < blob.size(); ++i) dst[i] = static_cast<ov::float16>(blob[i]);
    } else if (inputType == ov::element::f32) {
        float* dst = inputTensor.data<float>();
        std::memcpy(dst, blob.data(), blob.size() * sizeof(float));
    } else {
        return (char*)"[PGLNet-OpenVINO] Unsupported input dtype.";
    }
    inferRequest.set_input_tensor(inputTensor);

    for (int i = 0; i < warmupRuns; ++i) {
        inferRequest.infer();
    }

    std::vector<double> cost;
    for (int i = 0; i < benchRuns; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        inferRequest.infer();
        auto t1 = std::chrono::high_resolution_clock::now();
        cost.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    ov::Tensor outTensor = inferRequest.get_output_tensor(0);
    ret = PostProcess(outTensor, originalSize, outputBgr, resizeBack);
    if (ret != RET_OK) return ret;

    double avg = std::accumulate(cost.begin(), cost.end(), 0.0) / static_cast<double>(cost.size());
    auto [mn, mx] = std::minmax_element(cost.begin(), cost.end());
    std::cout << "[PGLNet-OpenVINO] avg/min/max: " << avg << "/" << *mn << "/" << *mx << " ms, FPS: " << 1000.0 / avg << "\n";
    return RET_OK;
}
