# Efficient Real-World Dehazing via Physics-Inspired Global-Local Decoupling

<p align="left">
  <a href=""><img src="https://img.shields.io/badge/arXiv-Coming%20Soon-b31b1b?style=flat-square" alt="arXiv"></a>
  <a href=""><img src="https://img.shields.io/badge/Paper-Coming%20Soon-2ea44f?style=flat-square" alt="Paper"></a>
  <a href=""><img src="https://img.shields.io/badge/GoogleDrive-Datasets-e67e22?style=flat-square" alt="Datasets"></a>
  <a href=""><img src="https://img.shields.io/badge/GoogleDrive-Weights-f1c40f?style=flat-square" alt="Weights"></a>
</p>

## Overview

PGL-Net is the official implementation of **Efficient Real-World Dehazing via Physics-Inspired Global-Local Decoupling**.

- arXiv: `[Coming Soon]()`
- Paper: `[Coming Soon]()`
- Datasets (Google Drive): `[Coming Soon]()`
- Pretrained Weights (Google Drive): `[Coming Soon]()`

## Abstract

> [Abstract will be added here.]

## Network Architecture / Insight

![Insight](figs/insight.png)

## Quantitative Results

The following paper tables are provided as images. Click each section to expand.

<details>
<summary><strong>RRSHID Results</strong> (click to expand)</summary>
<br>

![RRSHID Results](figs/rrshid_results.png)

</details>

<details>
<summary><strong>RW2AH Results</strong> (click to expand)</summary>
<br>

![RW2AH Results](figs/rw2ah_results.png)

</details>

<details>
<summary><strong>RTTS Results</strong> (click to expand)</summary>
<br>

![RTTS Results](figs/rtts_results.png)

</details>

<details>
<summary><strong>NTIRE Results</strong> (click to expand)</summary>
<br>

![NTIRE Results](figs/ntire_results.png)

</details>

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
  title   = {Efficient Real-World Dehazing via Physics-Inspired Global-Local Decoupling},
  author  = {Anonymous},
  journal = {arXiv preprint arXiv:xxxx.xxxxx},
  year    = {2026}
}
```

## Acknowledgement

This README will be updated after paper release.
