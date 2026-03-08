# PGLNet TensorRT CPP

Supports automatic FP16/FP32 input-output compatibility based on TensorRT binding dtype.

## Build

```bash
cmake -S . -B build -DTensorRT_ROOT=/path/to/TensorRT
cmake --build build --config Release
```

## Run

```bash
./build/pglnet_tensorrt ../../rrshid_pglnet_s.engine ../../1.jpg ./output_tensorrt_cpp.jpg 20 1
```
