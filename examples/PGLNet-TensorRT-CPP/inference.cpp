#include "inference.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>

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
        if (mant == 0) out = sign;
        else {
            exp = 127 - 15 + 1;
            while ((mant & 0x400) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x3ff;
            out = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) out = sign | 0x7f800000 | (mant << 13);
    else out = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    float f = *reinterpret_cast<float*>(&out);
    return f;
}

static size_t trt_dtype_size(nvinfer1::DataType t) {
    switch (t) {
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kINT8: return 1;
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kFLOAT:
        default: return 4;
    }
}

void PGLNetTensorRT::Logger::log(Severity severity, const char* msg) noexcept {
    if (severity <= Severity::kWARNING) std::cout << "[TensorRT] " << msg << "\n";
}

PGLNetTensorRT::PGLNetTensorRT()
    : runtime(nullptr), engine(nullptr), context(nullptr), stream(nullptr),
      inputIndex(-1), outputIndex(-1), warmupRuns(5), benchRuns(20) {}

PGLNetTensorRT::~PGLNetTensorRT() {
    for (auto* p : deviceBindings) {
        if (p) cudaFree(p);
    }
    if (stream) cudaStreamDestroy(stream);
    if (context) context->destroy();
    if (engine) engine->destroy();
    if (runtime) runtime->destroy();
}

bool PGLNetTensorRT::isInputBinding(int index) const {
#if NV_TENSORRT_MAJOR >= 10
    const char* name = engine->getIOTensorName(index);
    return engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
#else
    return engine->bindingIsInput(index);
#endif
}

int64_t PGLNetTensorRT::volume(const nvinfer1::Dims& d) const {
    int64_t v = 1;
    for (int i = 0; i < d.nbDims; ++i) v *= d.d[i];
    return v;
}

char* PGLNetTensorRT::CreateSession(DL_INIT_PARAM& params) {
    try {
        imgSize = params.imgSize;
        warmupRuns = params.warmupRuns;
        benchRuns = params.benchRuns;

        std::ifstream ifs(params.modelPath, std::ios::binary);
        if (!ifs) return (char*)"[PGLNet-TensorRT] Failed to open engine.";
        ifs.seekg(0, std::ios::end);
        size_t size = static_cast<size_t>(ifs.tellg());
        ifs.seekg(0, std::ios::beg);
        std::vector<char> data(size);
        ifs.read(data.data(), size);

        runtime = nvinfer1::createInferRuntime(logger);
        if (!runtime) return (char*)"[PGLNet-TensorRT] create runtime failed.";
        engine = runtime->deserializeCudaEngine(data.data(), size);
        if (!engine) return (char*)"[PGLNet-TensorRT] deserialize engine failed.";
        context = engine->createExecutionContext();
        if (!context) return (char*)"[PGLNet-TensorRT] create context failed.";
        cudaStreamCreate(&stream);

#if NV_TENSORRT_MAJOR >= 10
        int nb = engine->getNbIOTensors();
        deviceBindings.resize(nb, nullptr);
        for (int i = 0; i < nb; ++i) {
            const char* name = engine->getIOTensorName(i);
            auto dims = context->getTensorShape(name);
            auto dtype = engine->getTensorDataType(name);
            if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
                inputIndex = i;
                inputDims = dims;
                inputType = dtype;
            } else {
                outputIndex = i;
                outputDims = dims;
                outputType = dtype;
            }
            size_t bytes = static_cast<size_t>(volume(dims)) * trt_dtype_size(dtype);
            cudaMalloc(&deviceBindings[i], bytes);
        }
#else
        int nb = engine->getNbBindings();
        deviceBindings.resize(nb, nullptr);
        for (int i = 0; i < nb; ++i) {
            auto dims = context->getBindingDimensions(i);
            auto dtype = engine->getBindingDataType(i);
            if (isInputBinding(i)) {
                inputIndex = i;
                inputDims = dims;
                inputType = dtype;
            } else {
                outputIndex = i;
                outputDims = dims;
                outputType = dtype;
            }
            size_t bytes = static_cast<size_t>(volume(dims)) * trt_dtype_size(dtype);
            cudaMalloc(&deviceBindings[i], bytes);
        }
#endif
        std::cout << "[PGLNet-TensorRT] input dtype: " << static_cast<int>(inputType)
                  << ", output dtype: " << static_cast<int>(outputType) << "\n";
        return RET_OK;
    } catch (const std::exception& e) {
        std::cerr << "[PGLNet-TensorRT] CreateSession failed: " << e.what() << "\n";
        return (char*)"[PGLNet-TensorRT] CreateSession failed.";
    }
}

char* PGLNetTensorRT::PreProcess(const cv::Mat& inputBgr, std::vector<float>& blob) {
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

char* PGLNetTensorRT::PostProcess(const std::vector<float>& outData, cv::Mat& outputBgr, bool resizeBack) {
    int c = outputDims.d[1], h = outputDims.d[2], w = outputDims.d[3];
    int useC = std::min(c, 3);
    cv::Mat rgb(h, w, CV_32FC3, cv::Scalar(0, 0, 0));
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            cv::Vec3f v(0, 0, 0);
            for (int ch = 0; ch < useC; ++ch) {
                float p = outData[ch * h * w + y * w + x];
                v[ch] = std::min(1.0f, std::max(0.0f, p));
            }
            rgb.at<cv::Vec3f>(y, x) = v;
        }
    }
    cv::Mat outU8;
    rgb.convertTo(outU8, CV_8UC3, 255.0);
    cv::cvtColor(outU8, outU8, cv::COLOR_RGB2BGR);
    if (resizeBack) cv::resize(outU8, outputBgr, originalSize);
    else outputBgr = outU8;
    return RET_OK;
}

char* PGLNetTensorRT::RunSession(cv::Mat& inputBgr, cv::Mat& outputBgr, bool resizeBack) {
    std::vector<float> inBlob;
    auto ret = PreProcess(inputBgr, inBlob);
    if (ret != RET_OK) return ret;

    size_t inElems = inBlob.size();
    size_t inBytes = inElems * trt_dtype_size(inputType);
    size_t outElems = static_cast<size_t>(volume(outputDims));
    size_t outBytes = outElems * trt_dtype_size(outputType);
    std::vector<uint16_t> inBlobFp16;
    const void* inHostPtr = inBlob.data();
    if (inputType == nvinfer1::DataType::kHALF) {
        inBlobFp16.resize(inElems);
        for (size_t i = 0; i < inElems; ++i) inBlobFp16[i] = float_to_half_bits(inBlob[i]);
        inHostPtr = inBlobFp16.data();
    } else if (inputType != nvinfer1::DataType::kFLOAT) {
        return (char*)"[PGLNet-TensorRT] Unsupported input dtype.";
    }

    std::vector<float> outBlobFp32(outElems);
    std::vector<uint16_t> outBlobFp16;
    void* outHostPtr = outBlobFp32.data();
    if (outputType == nvinfer1::DataType::kHALF) {
        outBlobFp16.resize(outElems);
        outHostPtr = outBlobFp16.data();
    } else if (outputType != nvinfer1::DataType::kFLOAT) {
        return (char*)"[PGLNet-TensorRT] Unsupported output dtype.";
    }

    for (int i = 0; i < warmupRuns; ++i) {
        cudaMemcpyAsync(deviceBindings[inputIndex], inHostPtr, inBytes, cudaMemcpyHostToDevice, stream);
#if NV_TENSORRT_MAJOR >= 10
        const char* inName = engine->getIOTensorName(inputIndex);
        const char* outName = engine->getIOTensorName(outputIndex);
        context->setTensorAddress(inName, deviceBindings[inputIndex]);
        context->setTensorAddress(outName, deviceBindings[outputIndex]);
        context->enqueueV3(stream);
#else
        context->enqueueV2(deviceBindings.data(), stream, nullptr);
#endif
    }
    cudaStreamSynchronize(stream);

    std::vector<double> cost;
    for (int i = 0; i < benchRuns; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        cudaMemcpyAsync(deviceBindings[inputIndex], inHostPtr, inBytes, cudaMemcpyHostToDevice, stream);
#if NV_TENSORRT_MAJOR >= 10
        const char* inName = engine->getIOTensorName(inputIndex);
        const char* outName = engine->getIOTensorName(outputIndex);
        context->setTensorAddress(inName, deviceBindings[inputIndex]);
        context->setTensorAddress(outName, deviceBindings[outputIndex]);
        context->enqueueV3(stream);
#else
        context->enqueueV2(deviceBindings.data(), stream, nullptr);
#endif
        cudaMemcpyAsync(outHostPtr, deviceBindings[outputIndex], outBytes, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        auto t1 = std::chrono::high_resolution_clock::now();
        cost.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    if (outputType == nvinfer1::DataType::kHALF) {
        for (size_t i = 0; i < outElems; ++i) outBlobFp32[i] = half_bits_to_float(outBlobFp16[i]);
    }

    ret = PostProcess(outBlobFp32, outputBgr, resizeBack);
    if (ret != RET_OK) return ret;

    double avg = std::accumulate(cost.begin(), cost.end(), 0.0) / static_cast<double>(cost.size());
    auto [mn, mx] = std::minmax_element(cost.begin(), cost.end());
    std::cout << "[PGLNet-TensorRT] avg/min/max: " << avg << "/" << *mn << "/" << *mx << " ms, FPS: " << 1000.0 / avg << "\n";
    return RET_OK;
}
