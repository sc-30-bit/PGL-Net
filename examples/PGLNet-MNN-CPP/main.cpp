// PGLNet MNN C++ inference example

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <MNN/ImageProcess.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Module.hpp>
#include <cv/cv.hpp>
#include <opencv2/opencv.hpp>

using namespace MNN;
using namespace MNN::CV;
using namespace MNN::Express;

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

class PGLNetMNN {
public:
    bool loadModel(const std::string& modelPath, int forwardType = MNN_FORWARD_CPU, int precision = 0, int thread = 4) {
        MNN::ScheduleConfig sConfig;
        sConfig.type = static_cast<MNNForwardType>(forwardType);
        sConfig.numThread = thread;
        BackendConfig bConfig;
        bConfig.precision = static_cast<BackendConfig::PrecisionMode>(precision);
        sConfig.backendConfig = &bConfig;

        runtimeManager = std::shared_ptr<Executor::RuntimeManager>(Executor::RuntimeManager::createRuntimeManager(sConfig));
        if (runtimeManager == nullptr) {
            MNN_ERROR("Empty RuntimeManager\n");
            return false;
        }
        runtimeManager->setCache(".cachefile");

        net = std::shared_ptr<Module>(Module::load(std::vector<std::string>{}, std::vector<std::string>{}, modelPath.c_str(), runtimeManager));
        if (net == nullptr) {
            MNN_ERROR("Failed to load model: %s\n", modelPath.c_str());
            return false;
        }
        return true;
    }

    VARP preprocess(const std::string& imagePath, float& scale, int& origH, int& origW) {
        cv::Mat inputBgr = cv::imread(imagePath, cv::IMREAD_COLOR);
        if (inputBgr.empty()) {
            std::cerr << "Failed to read image: " << imagePath << "\n";
            return nullptr;
        }
        origH = inputBgr.rows;
        origW = inputBgr.cols;
        scale = 1.0f;

        cv::Mat rgb;
        cv::cvtColor(inputBgr, rgb, cv::COLOR_BGR2RGB);
        cv::Mat resized;
        cv::resize(rgb, resized, cv::Size(512, 512), 0, 0, cv::INTER_LINEAR);
        cv::Mat resizedF32;
        resized.convertTo(resizedF32, CV_32FC3, 1.0 / 255.0);

        std::vector<float> chw(3 * 512 * 512);
        for (int y = 0; y < 512; ++y) {
            const cv::Vec3f* row = resizedF32.ptr<cv::Vec3f>(y);
            for (int x = 0; x < 512; ++x) {
                const cv::Vec3f& p = row[x];
                const int idx = y * 512 + x;
                chw[idx] = p[0];                  // R
                chw[512 * 512 + idx] = p[1];      // G
                chw[2 * 512 * 512 + idx] = p[2];  // B
            }
        }

        auto input = _Const(chw.data(), {1, 3, 512, 512}, NCHW);
        input = _Convert(input, NC4HW4);
        return input;
    }

    VARP run(VARP input, int runs, std::vector<double>& timesMs) {
        VARP out;
        timesMs.clear();
        auto inInfo = input->getInfo();
        if (inInfo) {
            std::cout << "[PGLNet-MNN] input dtype bits: " << inInfo->type.bits << "\n";
        }
        for (int i = 0; i < runs; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto outputs = net->onForward({input});
            out = outputs[0];
            auto info = out->getInfo();
            if (info && info->type.bits == 16) out->readMap<uint16_t>();  // force compute
            else out->readMap<float>();  // force compute
            auto t1 = std::chrono::high_resolution_clock::now();
            timesMs.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        return out;
    }

    bool postprocess(VARP output, const std::string& inputPath, const std::string& outputPath, int origH, int origW, bool resizeBack) {
        output = _Convert(output, NCHW);
        output = _Squeeze(output);  // [C,H,W]
        auto info = output->getInfo();
        if (!info || info->dim.size() != 3) {
            std::cerr << "Unexpected output dims.\n";
            return false;
        }

        int c = info->dim[0];
        int h = info->dim[1];
        int w = info->dim[2];
        int useC = std::min(c, 3);
        std::vector<float> outFp32(static_cast<size_t>(c * h * w));
        if (info->type.bits == 16) {
            const uint16_t* ptr16 = output->readMap<uint16_t>();
            for (size_t i = 0; i < outFp32.size(); ++i) outFp32[i] = half_bits_to_float(ptr16[i]);
            std::cout << "[PGLNet-MNN] output dtype: FP16\n";
        } else {
            const float* ptr32 = output->readMap<float>();
            std::memcpy(outFp32.data(), ptr32, outFp32.size() * sizeof(float));
            std::cout << "[PGLNet-MNN] output dtype: FP32\n";
        }
        const float* ptr = outFp32.data();

        cv::Mat rgb(h, w, CV_32FC3, cv::Scalar(0, 0, 0));
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                cv::Vec3f v(0, 0, 0);
                for (int ch = 0; ch < useC; ++ch) {
                    float p = ptr[ch * h * w + y * w + x];
                    v[ch] = std::min(1.0f, std::max(0.0f, p));
                }
                rgb.at<cv::Vec3f>(y, x) = v;
            }
        }

        cv::Mat outU8;
        rgb.convertTo(outU8, CV_8UC3, 255.0);
        cv::cvtColor(outU8, outU8, cv::COLOR_RGB2BGR);
        if (resizeBack) {
            cv::resize(outU8, outU8, cv::Size(origW, origH), 0, 0, cv::INTER_LINEAR);
        }
        cv::imwrite(outputPath, outU8);

        cv::Mat inputBgr = cv::imread(inputPath);
        if (!inputBgr.empty()) {
            if (inputBgr.size() != outU8.size()) {
                cv::resize(inputBgr, inputBgr, outU8.size());
            }
            cv::Mat compare;
            cv::hconcat(inputBgr, outU8, compare);
            std::string comparePath = outputPath;
            auto dot = comparePath.find_last_of('.');
            if (dot == std::string::npos) dot = comparePath.size();
            comparePath.insert(dot, "_compare");
            cv::imwrite(comparePath, compare);
            std::cout << "Saved: " << comparePath << "\n";
        }
        return true;
    }

    void updateCache() {
        if (runtimeManager) runtimeManager->updateCache();
    }

private:
    std::shared_ptr<Module> net;
    std::shared_ptr<Executor::RuntimeManager> runtimeManager;
};

int main(int argc, const char* argv[]) {
    if (argc < 4) {
        std::cout << "Usage: ./pglnet_mnn model.mnn input.jpg output.jpg [forwardType] [precision] [thread] [runs] [resize_back]\n";
        return 0;
    }

    const std::string modelPath = argv[1];
    const std::string inputPath = argv[2];
    const std::string outputPath = argv[3];

    int forwardType = (argc >= 5) ? std::atoi(argv[4]) : MNN_FORWARD_AUTO;
    int precision = (argc >= 6) ? std::atoi(argv[5]) : 0;
    int thread = (argc >= 7) ? std::atoi(argv[6]) : 4;
    int runs = (argc >= 8) ? std::max(1, std::atoi(argv[7])) : 20;
    bool resizeBack = (argc >= 9) ? (std::atoi(argv[8]) != 0) : false;

    PGLNetMNN infer;
    if (!infer.loadModel(modelPath, forwardType, precision, thread)) return 1;

    float scale = 1.0f;
    int origH = 0, origW = 0;
    VARP input = infer.preprocess(inputPath, scale, origH, origW);
    if (input == nullptr) return 1;

    std::vector<double> times;
    VARP output = infer.run(input, runs, times);

    if (!infer.postprocess(output, inputPath, outputPath, origH, origW, resizeBack)) return 1;

    double avg = std::accumulate(times.begin(), times.end(), 0.0) / static_cast<double>(times.size());
    auto [mn, mx] = std::minmax_element(times.begin(), times.end());
    std::cout << "[PGLNet-MNN] avg/min/max: " << avg << "/" << *mn << "/" << *mx << " ms, FPS: " << 1000.0 / avg << "\n";
    std::cout << "Saved: " << outputPath << "\n";

    infer.updateCache();
    return 0;
}
