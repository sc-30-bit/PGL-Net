# PGLNet OpenCV-DNN CPP

## Build

```bash
cmake -S . -B build
cmake --build build --config Release
```

## Run

```bash
./build/pglnet_opencv_dnn ../../rrshid_pglnet_s.onnx ../../1.jpg ./output_opencvdnn_cpp.jpg 20 1
```

