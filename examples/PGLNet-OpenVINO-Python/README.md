# PGLNet OpenVINO (Python)

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py ^
  --model ..\..\rrshid_pglnet_s.xml ^
  --input ..\..\1.jpg ^
  --output .\output_ov.jpg ^
  --device CPU ^
  --resize-back
```

Common options:

- `--device`: `CPU` / `GPU` / `AUTO`
- `--threads`: set CPU thread count

