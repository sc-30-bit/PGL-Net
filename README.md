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

<p align="center">
  <img src="figs/insight.png" alt="Insight">
</p>

## Quantitative Results

The following paper tables are provided as images. Click each section to expand.

<details>
<summary><strong>RRSHID Results</strong> (click to expand)</summary>
<br>


<p align="center">
  <img src="figs/rrshid_results.png" alt="RRSHID Results">
</p>

</details>

<details>
<summary><strong>RW2AH Results</strong> (click to expand)</summary>
<br>


<p align="center">
  <img src="figs/rw2ah_results.png" alt="RW2AH Results">
</p>

</details>

<details>
<summary><strong>NTIRE Results</strong> (click to expand)</summary>
<br>


<p align="center">
  <img src="figs/ntire_results.png" alt="NTIRE Results">
</p>

</details>

<details>
<summary><strong>RTTS Results</strong> (click to expand)</summary>
<br>


<p align="center">
  <img src="figs/rtts_results.png" alt="RTTS Results">
</p>

</details>

## Deployment

FP16 latency is measured at `512 x 512`, using [TensorRT](https://developer.nvidia.com/tensorrt) on GPUs and [OpenVINO](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html) on CPU.

<table align="center">
  <thead>
    <tr>
      <th>Model</th>
      <th>Size (pixels)</th>
      <th>CPU OpenVINO (ms)</th>
      <th>T4 TensorRT10 (ms)</th>
      <th>RTX 3090 TensorRT10 (ms)</th>
      <th>Params (M)</th>
      <th>MACs (G)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Tiny</td>
      <td align="center">512</td>
      <td align="center">14.69</td>
      <td align="center">12.90</td>
      <td align="center">3.92</td>
      <td align="center">0.78</td>
      <td align="center">2.71</td>
    </tr>
    <tr>
      <td>Small</td>
      <td align="center">512</td>
      <td align="center">24.58</td>
      <td align="center">23.11</td>
      <td align="center">6.68</td>
      <td align="center">1.30</td>
      <td align="center">4.77</td>
    </tr>
  </tbody>
</table>


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

<details>
<summary><strong>Available configs</strong> (click to expand)</summary>
<br>

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

</details>

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
