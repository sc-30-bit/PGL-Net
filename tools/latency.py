#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from models import pglnet_t, pglnet_s, pglnet_b, pglnet_d


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def gpu_name():
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "CPU"

def percentile_ms(x_ms, p):
    return float(np.percentile(np.array(x_ms, dtype=np.float64), p))

def safe_sync(device: str):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def build_model(model_type: str) -> nn.Module:
    if model_type.startswith("pglnet_"):
        v = model_type.split("_")[-1]
        table = {
            "t": pglnet_t,
            "s": pglnet_s,
            "b": pglnet_b,
            "d": pglnet_d,
        }
        if v not in table:
            raise ValueError(f"Unsupported pglnet variant: {v}")
        return table[v]()
    raise ValueError(f"Unsupported model_type: {model_type}")

@torch.no_grad()
def bench_torch(net: nn.Module, device: str, shape, warmup=20, iters=200):
    b, c, h, w = shape
    x = torch.randn((b, c, h, w), device=device, dtype=torch.float32)

    for _ in range(warmup):
        _ = net(x)
    safe_sync(device)

    times_ms = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = net(x)
        safe_sync(device)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    return times_ms


def parse_shapes(s: str):
    shapes = []
    for item in s.split(","):
        item = item.strip().lower().replace(" ", "")
        b, c, h, w = item.split("x")
        shapes.append((int(b), int(c), int(h), int(w)))
    return shapes


def parse_model_list(s: str):
    items = []
    for x in s.split(","):
        x = x.strip()
        if x:
            items.append(x)
    return items


def default_all_models():
    return ["pglnet_t", "pglnet_s", "pglnet_b", "pglnet_d"]


def main():
    ap = argparse.ArgumentParser("PGLNet Latency benchmark: Torch CUDA FP32")

    ap.add_argument("--all_models", action="store_true",
                    help="benchmark all PGLNet models automatically")
    ap.add_argument("--models", default="",
                    help="comma-separated model list, e.g. pglnet_t,pglnet_s (overrides --all_models)")

    ap.add_argument("--shapes", default="1x3x512x512",
                    help="comma sep shapes, default: 1x3x512x512")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)

    ap.add_argument("--csv", default="./latency_results.csv")
    args = ap.parse_args()

    device = get_device()
    if device != "cuda":
        raise RuntimeError("CUDA is required for this benchmark")
    print(f"Device: {device} | GPU: {gpu_name()}")

    shapes = parse_shapes(args.shapes)

    if args.models.strip():
        model_list = parse_model_list(args.models)
    elif args.all_models:
        model_list = default_all_models()
    else:
        model_list = default_all_models()

    rows = []

    for model_type in model_list:
        print(f"\n===============================")
        print(f"Model: {model_type}")
        print(f"===============================")
        
        net = build_model(model_type).to(device).eval()

        for shape in shapes:
            # Torch CUDA FP32 bench
            t_times = bench_torch(
                net,
                device=device,
                shape=shape,
                warmup=args.warmup,
                iters=args.iters,
            )

            row = {
                "model_type": model_type,
                "backend": "torch_cuda_fp32",
                "gpu": gpu_name(),
                "shape": f"{shape[0]}x{shape[1]}x{shape[2]}x{shape[3]}",
                "mean_ms": float(np.mean(t_times)),
                "p50_ms": percentile_ms(t_times, 50),
                "p90_ms": percentile_ms(t_times, 90),
                "p99_ms": percentile_ms(t_times, 99),
                "fps_mean": float(1000.0 / np.mean(t_times)) if np.mean(t_times) > 0 else 0.0,
            }
            rows.append(row)
            print(f"[Torch CUDA FP32] {row['shape']} "
                  f"mean={row['mean_ms']:.2f}ms p50={row['p50_ms']:.2f} "
                  f"p90={row['p90_ms']:.2f} p99={row['p99_ms']:.2f} "
                  f"fps={row['fps_mean']:.2f}")

    # write CSV
    import csv
    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["model_type", "backend", "gpu", "shape",
                      "mean_ms", "p50_ms", "p90_ms", "p99_ms", "fps_mean"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"\nSaved -> {args.csv}")

if __name__ == "__main__":
    main()
#python test_latency/latency.py --all_models --shapes 1x3x512x512