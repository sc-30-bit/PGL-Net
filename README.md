# Efficient Real-World Dehazing via Physics-Inspired Global-Local Decoupling

<p align="left">
  <a href=""><img src="https://img.shields.io/badge/arXiv-Coming%20Soon-b31b1b?style=flat-square" alt="arXiv"></a>
  <a href=""><img src="https://img.shields.io/badge/Paper-Coming%20Soon-2ea44f?style=flat-square" alt="Paper"></a>
  <a href=""><img src="https://img.shields.io/badge/GoogleDrive-Datasets-e67e22?style=flat-square" alt="Datasets"></a>
  <a href=""><img src="https://img.shields.io/badge/GoogleDrive-Weights-f1c40f?style=flat-square" alt="Weights"></a>
</p>


> **Abstract:**
> Real-world single image dehazing is highly ill-posed due to spatially and spectrally varying scattering, while practical deployment demands lightweight and low-latency models. Existing approaches either rely on fragile physical inversion under simplified assumptions or adopt heavy blind architectures unsuitable for edge deployment. To overcome these limitations, we propose PGL-Net (Physics-Inspired Global-Local Decoupling Network), a lightweight framework that incorporates physical inductive biases via operator-level emulation, avoiding explicit parameter estimation. It decouples dehazing into global distribution rectification and local structural refinement. A Physics-Inspired Affine Fusion (PAF) module performs globally conditioned alignment across hierarchical skip connections to compensate for haze-induced bias, while a compact Degradation-Aware Modulation (DAM) block adaptively restores spatially and spectrally variant details through dynamic feature modulation. Extensive experiments on multiple real-world benchmarks demonstrate that PGL-Net achieves state-of-the-art restoration quality with significantly reduced complexity. Compared with the recent SOTA SGDN, the Tiny variant (PGL-Net-T) improves PSNR by up to +2.6 dB and consistently enhances downstream object detection accuracy, while achieving over a 10x reduction in inference latency.

## Insight

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

1. Create a new conda environment

```bash
conda create -n pglnet python=3.9
conda activate pglnet
```

2. Install dependencies

```bash
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Training

```bash
torchrun --nproc_per_node=* main.py --config (config_path) --use_ddp
```

Examples:

Single-GPU training (for example, RRSHID / PGL-Net-T)

```bash
torchrun --nproc_per_node=1 main.py --config ./configs/RRSHID/pglnet_t.json --use_ddp
```

Multi-GPU training  (for example, RESIDE-IN / PGL-Net-T, 2 GPUs)

```bash
torchrun --nproc_per_node=2 main.py --config ./configs/RESIDE-IN/pglnet_t.json --use_ddp
```

Note that we use mixed precision training and distributed data parallel by default.

Available configs include:

- `configs/RRSHID/pglnet_t.json`
- `configs/RRSHID/pglnet_s.json`
- `configs/RUDB/pglnet_t.json`
- `configs/RUDB/pglnet_s.json`
- `configs/RW2AH/pglnet_t.json`
- `configs/RW2AH/pglnet_s.json`
- `configs/RESIDE-IN/pglnet_t.json`
- `configs/RESIDE-IN/pglnet_s.json`
- `configs/RESIDE-OUT/pglnet_t.json`
- `configs/RESIDE-OUT/pglnet_s.json`

## Testing

```bash
python test.py --weight (weight_path) --model_type (model_type) (--tile 1024 if RUDB) --test_dir (test_dir) --gt_dir (gt_dir)
```
Examples:

Test on RRSHID (PGL-Net-T)

```bash
python test.py --weight rrshid_pglnet_t.pk --model_type pglnet_t --test_dir ./datasets/RRSHID/test/input --gt_dir ./datasets/RRSHID/test/gt
```

Test on RUDB (PGL-Net-T)

```bash
python test.py --weight rudb_pglnet_t.pk --model_type pglnet_t --test_dir ./datasets/RUDB/test/input --gt_dir ./datasets/RUDB/test/gt --tile 1024
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
