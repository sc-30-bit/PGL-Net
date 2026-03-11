# PGLNet Tiled Inference (Python)

This example demonstrates tiled inference for large images.

This example adds sliding-window inference for PGLNet ONNX models:

- `tile`: patch size
- `tile_overlap`: overlap between patches
- fusion by `output += patch`, `count_map += 1`, final `output / count_map`

## Install

```bash
pip install -r requirements.txt
```

## Image

```bash
python main.py ^
  --model ..\..\rrshid_pglnet_s.onnx ^
  --source ..\..\1.jpg ^
  --output .\output_tiled.jpg ^
  --tile 512 ^
  --tile-overlap 32 ^
  --provider auto
```

## Video

```bash
python main.py ^
  --model ..\..\rrshid_pglnet_s.onnx ^
  --source .\input.mp4 ^
  --output .\output_tiled.mp4 ^
  --tile 512 ^
  --tile-overlap 32 ^
  --provider auto ^
  --view-img
```
