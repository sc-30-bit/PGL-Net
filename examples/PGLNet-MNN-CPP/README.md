# PGLNet MNN CPP

This example uses a standalone C++ inference structure for MNN deployment.

Includes automatic FP16/FP32 output compatibility (dtype-aware postprocess path).

## Build

```bash
cmake -S . -B build -DMNN_ROOT=/path/to/MNN
cmake --build build --config Release
```

Windows (using local MNN 3.4.1 package paths):

```bash
cmake -S . -B build -A x64 ^
  -DOpenCV_DIR="H:/libraries/Opencv_MSVC/opencv/build" ^
  -DMNN_INCLUDE_DIR="H:/libraries/mnn_3.4.1_windows_x64_cpu_opencl/mnn_3.4.1_windows_x64_cpu_opencl/include" ^
  -DMNN_LIB_DIR="H:/libraries/mnn_3.4.1_windows_x64_cpu_opencl/mnn_3.4.1_windows_x64_cpu_opencl/lib/x64/Release/Dynamic/MD" ^
  -DMNN_DLL_PATH="H:/libraries/mnn_3.4.1_windows_x64_cpu_opencl/mnn_3.4.1_windows_x64_cpu_opencl/lib/x64/Release/Dynamic/MD/MNN.dll"
cmake --build build --config Release
```

## Run

```bash
./build/pglnet_mnn ../../rrshid_pglnet_s.mnn ../../1.jpg ./output_mnn_cpp.jpg 0 0 4 20 1
```

## ONNX -> MNN conversion (adapted to current export)

`export_pglnet_onnx.py` exports graph IO names as `input` / `output`, which is compatible with conversion logs showing:
`inputTensors : [ input ]`, `outputTensors: [ output ]`.

```bash
mnnconvert -f ONNX --modelFile ../../rw2ah_pglnet_t.onnx --MNNModel ../../rw2ah_pglnet_t.mnn --bizCode MNN
```

Note: if `mnnconvert` prompts `try 'pip install -U aliyun-log-python-sdk'`, this is a tooling dependency hint and conversion can still complete successfully.

Arguments:

1. `model.mnn`
2. input image path
3. output image path
4. `forwardType` (optional, default `0` CPU)
5. `precision` (optional, default `0`)
6. `thread` (optional, default `4`)
7. `runs` (optional, default `20`)
8. `resize_back` (optional, `0/1`, default `0`)
