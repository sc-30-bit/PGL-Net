# PGLNet OpenCV-DNN (Python)

## Install

```bash
pip install -r requirements.txt
```

## Run (CPU)

```bash
python main.py ^
  --model ..\..\rrshid_pglnet_s.onnx ^
  --input ..\..\1.jpg ^
  --output .\output_dnn.jpg ^
  --backend default ^
  --target cpu ^
  --resize-back
```

## Run (CUDA)

```bash
python main.py ^
  --model ..\..\rrshid_pglnet_s.onnx ^
  --input ..\..\1.jpg ^
  --output .\output_dnn_cuda.jpg ^
  --backend cuda ^
  --target cuda_fp16
```

