# PGLNet TensorRT (Python)

## Install

```bash
pip install -r requirements.txt
```

You also need a TensorRT runtime environment with CUDA and a compatible `.engine` file.

## Run

```bash
python main.py ^
  --model ..\..\rrshid_pglnet_s.engine ^
  --input ..\..\1.jpg ^
  --output .\output_trt.jpg ^
  --resize-back
```

