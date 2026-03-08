# PGLNet Examples

This directory contains Python inference examples for PGLNet on different deployment backends:

- `PGLNet-ONNXRuntime-Python`
- `PGLNet-OpenVINO-Python`
- `PGLNet-TensorRT-Python`
- `PGLNet-OpenCV-DNN-Python`
- `PGLNet-ONNXRuntime-CPP`
- `PGLNet-OpenVINO-CPP`
- `PGLNet-TensorRT-CPP`
- `PGLNet-OpenCV-DNN-CPP`
- `PGLNet-MNN-CPP`
- `PGLNet-Tiled-Inference-Python` (SAHI-like sliding window)

Each example supports:

- Single-image inference
- Auto preprocessing (`BGR -> RGB`, normalize, `NCHW`)
- Output image reconstruction and save
- Simple latency and FPS statistics
