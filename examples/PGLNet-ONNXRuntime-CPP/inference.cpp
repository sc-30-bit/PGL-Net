#include "inference.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <numeric>
#include <cstdint>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

#ifdef USE_CUDA
#endif

#define RET_OK nullptr

static uint16_t float_to_half_bits(float f) {
    uint32_t x = *reinterpret_cast<uint32_t*>(&f);
    uint16_t sign = static_cast<uint16_t>((x >> 16) & 0x8000);
    uint32_t mantissa = x & 0x7fffff;
    int exp = static_cast<int>((x >> 23) & 0xff) - 127 + 15;
    if (exp <= 0) {
        if (exp < -10) return sign;
        mantissa = (mantissa | 0x800000) >> (1 - exp);
        return static_cast<uint16_t>(sign | ((mantissa + 0x1000) >> 13));
    }
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7c00);
    return static_cast<uint16_t>(sign | (exp << 10) | ((mantissa + 0x1000) >> 13));
}

static float half_bits_to_float(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) {
            out = sign;
        } else {
            exp = 127 - 15 + 1;
            while ((mant & 0x400) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x3ff;
            out = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7f800000 | (mant << 13);
    } else {
        out = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f = *reinterpret_cast<float*>(&out);
    return f;
}

PGLNet::PGLNet() : env(ORT_LOGGING_LEVEL_WARNING, "PGLNet"), session(nullptr), cudaEnable(false), warmupRuns(5), benchRuns(20) {}

PGLNet::~PGLNet() {
    delete session;
    for (auto* p : inputNodeNames) delete[] p;
    for (auto* p : outputNodeNames) delete[] p;
}

char* PGLNet::CreateSession(DL_INIT_PARAM& params) {
    try {
        imgSize = params.imgSize;
        cudaEnable = params.cudaEnable;
        warmupRuns = params.warmupRuns;
        benchRuns = params.benchRuns;

        Ort::SessionOptions so;
        so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        so.SetIntraOpNumThreads(params.intraOpNumThreads);
        auto providers = Ort::GetAvailableProviders();
        std::cout << "[PGLNet-ONNXRuntime] available providers:";
        for (const auto& p : providers) std::cout << " " << p;
        std::cout << "\n";
#ifdef USE_CUDA
        if (params.cudaEnable) {
            bool hasCuda = std::find(providers.begin(), providers.end(), "CUDAExecutionProvider") != providers.end();
            if (hasCuda) {
                try {
                    OrtCUDAProviderOptions cudaOpt;
                    cudaOpt.device_id = 0;
                    so.AppendExecutionProvider_CUDA(cudaOpt);
                    std::cout << "[PGLNet-ONNXRuntime] CUDA EP enabled (device 0)\n";
                } catch (const std::exception& e) {
                    std::cerr << "[PGLNet-ONNXRuntime] Failed to enable CUDA EP, fallback to CPU: " << e.what() << "\n";
                }
            } else {
                std::cerr << "[PGLNet-ONNXRuntime] CUDAExecutionProvider is not available, fallback to CPU.\n";
            }
        }
#endif

#ifdef _WIN32
        int n = MultiByteToWideChar(CP_UTF8, 0, params.modelPath.c_str(), static_cast<int>(params.modelPath.size()), nullptr, 0);
        std::wstring wpath(n, L'\0');
        MultiByteToWideChar(CP_UTF8, 0, params.modelPath.c_str(), static_cast<int>(params.modelPath.size()), &wpath[0], n);
        session = new Ort::Session(env, wpath.c_str(), so);
#else
        session = new Ort::Session(env, params.modelPath.c_str(), so);
#endif

        Ort::AllocatorWithDefaultOptions allocator;
        size_t inCount = session->GetInputCount();
        for (size_t i = 0; i < inCount; ++i) {
            auto name = session->GetInputNameAllocated(i, allocator);
            char* p = new char[128];
            std::strncpy(p, name.get(), 127);
            p[127] = '\0';
            inputNodeNames.push_back(p);
        }

        size_t outCount = session->GetOutputCount();
        for (size_t i = 0; i < outCount; ++i) {
            auto name = session->GetOutputNameAllocated(i, allocator);
            char* p = new char[128];
            std::strncpy(p, name.get(), 127);
            p[127] = '\0';
            outputNodeNames.push_back(p);
        }
        options = Ort::RunOptions{nullptr};

        auto inInfo = session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
        auto outInfo = session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo();
        inputTensorType = inInfo.GetElementType();
        outputTensorType = outInfo.GetElementType();
        std::cout << "[PGLNet-ONNXRuntime] input dtype: " << static_cast<int>(inputTensorType)
                  << ", output dtype: " << static_cast<int>(outputTensorType) << "\n";

        return WarmUpSession();
    } catch (const std::exception& e) {
        std::cerr << "[PGLNet] CreateSession failed: " << e.what() << "\n";
        return (char*)"[PGLNet] CreateSession failed.";
    }
}

char* PGLNet::PreProcess(const cv::Mat& inputBgr, std::vector<float>& blob) {
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

char* PGLNet::PostProcess(const std::vector<int64_t>& outShape, const float* outData, const cv::Size& originalSize_, cv::Mat& outputBgr, bool resizeBack) {
    if (outShape.size() != 4 || outShape[0] != 1) {
        return (char*)"[PGLNet] Unexpected output shape.";
    }
    int c = static_cast<int>(outShape[1]);
    int h = static_cast<int>(outShape[2]);
    int w = static_cast<int>(outShape[3]);
    if (c < 1) return (char*)"[PGLNet] Invalid output channels.";

    cv::Mat out(h, w, CV_32FC3, cv::Scalar(0, 0, 0));
    int useC = std::min(c, 3);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            cv::Vec3f v(0, 0, 0);
            for (int ch = 0; ch < useC; ++ch) {
                float p = outData[ch * h * w + y * w + x];
                v[ch] = std::min(1.0f, std::max(0.0f, p));
            }
            out.at<cv::Vec3f>(y, x) = v;
        }
    }

    cv::Mat outU8;
    out.convertTo(outU8, CV_8UC3, 255.0);
    cv::cvtColor(outU8, outU8, cv::COLOR_RGB2BGR);
    if (resizeBack) {
        cv::resize(outU8, outputBgr, originalSize_, 0, 0, cv::INTER_LINEAR);
    } else {
        outputBgr = outU8;
    }
    return RET_OK;
}

char* PGLNet::WarmUpSession() {
    cv::Mat dummy(imgSize[0], imgSize[1], CV_8UC3, cv::Scalar(114, 114, 114));
    cv::Mat out;
    for (int i = 0; i < warmupRuns; ++i) {
        auto ret = RunSession(dummy, out, false);
        if (ret != RET_OK) return ret;
    }
    return RET_OK;
}

char* PGLNet::RunSession(cv::Mat& inputBgr, cv::Mat& outputBgr, bool resizeBack) {
    if (!session) return (char*)"[PGLNet] Session is null.";
    std::vector<float> blob;
    auto ret = PreProcess(inputBgr, blob);
    if (ret != RET_OK) return ret;

    std::vector<int64_t> inputShape = {1, 3, imgSize[0], imgSize[1]};
    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor{nullptr};
    std::vector<Ort::Float16_t> blobFp16;
    if (inputTensorType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        blobFp16.resize(blob.size());
        for (size_t i = 0; i < blob.size(); ++i) blobFp16[i] = Ort::Float16_t(blob[i]);
        inputTensor = Ort::Value::CreateTensor<Ort::Float16_t>(memInfo, blobFp16.data(), blobFp16.size(), inputShape.data(), inputShape.size());
    } else if (inputTensorType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        inputTensor = Ort::Value::CreateTensor<float>(memInfo, blob.data(), blob.size(), inputShape.data(), inputShape.size());
    } else {
        return (char*)"[PGLNet] Unsupported input dtype.";
    }

    std::vector<double> cost;
    std::vector<Ort::Value> outputs;
    for (int i = 0; i < benchRuns; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        outputs = session->Run(options, inputNodeNames.data(), &inputTensor, 1, outputNodeNames.data(), outputNodeNames.size());
        auto t1 = std::chrono::high_resolution_clock::now();
        cost.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    auto info = outputs[0].GetTensorTypeAndShapeInfo();
    auto outShape = info.GetShape();
    std::vector<float> outFp32;
    if (outputTensorType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        const Ort::Float16_t* outDataFp16 = outputs[0].GetTensorData<Ort::Float16_t>();
        size_t total = 1;
        for (auto d : outShape) total *= static_cast<size_t>(d);
        outFp32.resize(total);
        for (size_t i = 0; i < total; ++i) outFp32[i] = static_cast<float>(outDataFp16[i]);
        ret = PostProcess(outShape, outFp32.data(), originalSize, outputBgr, resizeBack);
    } else if (outputTensorType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        const float* outData = outputs[0].GetTensorData<float>();
        ret = PostProcess(outShape, outData, originalSize, outputBgr, resizeBack);
    } else {
        return (char*)"[PGLNet] Unsupported output dtype.";
    }
    if (ret != RET_OK) return ret;

    double avg = std::accumulate(cost.begin(), cost.end(), 0.0) / static_cast<double>(cost.size());
    auto [mn, mx] = std::minmax_element(cost.begin(), cost.end());
    std::cout << "[PGLNet-ONNXRuntime] avg/min/max: " << avg << "/" << *mn << "/" << *mx << " ms, FPS: " << 1000.0 / avg << "\n";
    return RET_OK;
}
