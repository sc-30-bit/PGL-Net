# PGLNet ONNXRuntime (Python)

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py ^
  --model ..\..\rrshid_pglnet_s.onnx ^
  --input ..\..\1.jpg ^
  --output .\output_ort.jpg ^
  --provider auto ^
  --resize-back
```

`--provider` options:

- `auto`: prefer CUDA, fallback CPU
- `cpu`: CPU only
- `cuda`: CUDA + CPU fallback

