# PGL-Net

> [Paper Title Placeholder]

## Overview

PGL-Net is the official implementation for our paper.

- arXiv: `[Coming Soon]()`
- Datasets (Google Drive): `[Coming Soon]()`
- Pretrained Weights (Google Drive): `[Coming Soon]()`

## Abstract

> [Abstract will be added here.]

## Insights and Results

### Insight Figure

![Insight](figs/insight.png)

### Quantitative Results (Tables from Paper)

#### RRSHID Results

![RRSHID Results](figs/rrshid_results.png)

#### RW2AH Results

![RW2AH Results](figs/rw2ah_results.png)

#### RTTS Results

![RTTS Results](figs/rtts_results.png)

#### NTIRE Results

![NTIRE Results](figs/ntire_results.png)

## Environment Setup

```bash
conda create -n pglnet python=3.9
conda activate pglnet

conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Training

```bash
python main.py --config (config_path)
python main.py --config ./configs/RRSHID/pglnet_t.json
```

Available configs include:

- `configs/RRSHID/pglnet_t.json`
- `configs/RRSHID/pglnet_s.json`
- `configs/RUDB/pglnet_t.json`
- `configs/RUDB/pglnet_s.json`
- `configs/RW2AH/pglnet_t.json`
- `configs/RW2AH/pglnet_s.json`

## Testing

```bash
python test.py --weight (weight_path) --model_type (model_type) (--tile 1024 if RUDB) --test_dir (test_dir) --gt_dir (gt_dir)
python test.py --weight rrshid_pglnet_t.pk --model_type pglnet_t
```

## Model Overhead (Params / MACs)

```bash
python ./tools/overhead.py
```

## Latency

```bash
python ./tools/latency.py --shapes 1x3x512x512
```

## Citation

```bibtex
@article{your_paper_2026,
  title   = {PGL-Net},
  author  = {Anonymous},
  journal = {arXiv preprint arXiv:xxxx.xxxxx},
  year    = {2026}
}
```

## Acknowledgement

This README will be updated after paper release.
