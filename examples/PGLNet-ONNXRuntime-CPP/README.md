# PGLNet ONNXRuntime CPP

This example uses a modular C++ file layout:

- `main.cpp`
- `inference.h`
- `inference.cpp`
- `CMakeLists.txt`

Supports automatic FP16/FP32 input-output compatibility based on model I/O tensor dtype.

## Build

```bash
cmake -S . -B build -DONNXRUNTIME_ROOT=/path/to/onnxruntime
cmake --build build --config Release
```

Windows (ORT 1.17 GPU package):

```bash
cmake -S . -B build -A x64 ^
  -DOpenCV_DIR="H:/libraries/Opencv_MSVC/opencv/build" ^
  -DONNXRUNTIME_ROOT="H:/libraries/onnxruntime-win-x64-gpu-1.17.0/onnxruntime-win-x64-gpu-1.17.0" ^
  -DUSE_CUDA=ON
cmake --build build --config Release
```

## Run

```bash
./build/pglnet_onnxruntime ../../rrshid_pglnet_s.onnx ../../1.jpg ./output_ort_cpp.jpg 20 1
```

Arguments:

1. model path (`.onnx`)
2. input image path
3. output image path
4. benchmark runs (optional, default `20`)
5. resize back to original size (`0`/`1`, optional, default `0`)
