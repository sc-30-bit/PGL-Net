# PGLNet OpenVINO CPP

Supports automatic FP16/FP32 input-output compatibility based on OpenVINO model I/O dtype.

## Build

```bash
cmake -S . -B build
cmake --build build --config Release
```

## Run

```bash
./build/pglnet_openvino ../../rrshid_pglnet_s.xml ../../1.jpg ./output_openvino_cpp.jpg 20 1
```
