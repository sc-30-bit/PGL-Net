#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np


def preprocess(image_path: str, width: int, height: int, input_dtype: np.dtype):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    orig_h, orig_w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    image = image_rgb.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))[None, ...]
    if input_dtype == np.float16:
        image = image.astype(np.float16)
    return image, image_bgr, (orig_h, orig_w)


def postprocess(output: np.ndarray, orig_size: tuple[int, int], resize_back: bool):
    if output.ndim == 4:
        output = output[0]
    if output.ndim == 3 and output.shape[0] in (1, 3):
        output = np.transpose(output, (1, 2, 0))
    if output.ndim == 2:
        output = output[:, :, None]
    if output.shape[2] == 1:
        output = np.repeat(output, 3, axis=2)

    output = output.astype(np.float32)
    output = np.clip(output, 0.0, 1.0)
    output = (output * 255.0).astype(np.uint8)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    if resize_back:
        output = cv2.resize(output, (orig_size[1], orig_size[0]), interpolation=cv2.INTER_LINEAR)
    return output


def main():
    parser = argparse.ArgumentParser(description="PGLNet OpenCV-DNN inference")
    parser.add_argument("--model", "-m", required=True, help="Path to .onnx model")
    parser.add_argument("--input", "-i", required=True, help="Path to input image")
    parser.add_argument("--output", "-o", default="output.jpg", help="Path to output image")
    parser.add_argument("--backend", choices=["default", "cuda"], default="default")
    parser.add_argument("--target", choices=["cpu", "cuda", "cuda_fp16"], default="cpu")
    parser.add_argument("--height", type=int, default=512, help="Input height")
    parser.add_argument("--width", type=int, default=512, help="Input width")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup runs")
    parser.add_argument("--runs", type=int, default=20, help="Benchmark runs")
    parser.add_argument("--resize-back", action="store_true", help="Resize output back to original size")
    args = parser.parse_args()

    net = cv2.dnn.readNetFromONNX(args.model)
    if args.backend == "cuda":
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    else:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

    if args.target == "cuda":
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        input_dtype = np.float32
    elif args.target == "cuda_fp16":
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        input_dtype = np.float16
    else:
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        input_dtype = np.float32

    input_tensor, orig_img, orig_size = preprocess(args.input, args.width, args.height, input_dtype)
    net.setInput(input_tensor)

    for _ in range(args.warmup):
        _ = net.forward()

    times = []
    output = None
    for _ in range(args.runs):
        t0 = time.perf_counter()
        output = net.forward()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    out_img = postprocess(output, orig_size, args.resize_back)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    cv2.imwrite(args.output, out_img)

    if orig_img.shape[:2] != out_img.shape[:2]:
        show_orig = cv2.resize(orig_img, (out_img.shape[1], out_img.shape[0]), interpolation=cv2.INTER_LINEAR)
    else:
        show_orig = orig_img
    compare = np.hstack((show_orig, out_img))
    compare_path = str(Path(args.output).with_stem(Path(args.output).stem + "_compare"))
    cv2.imwrite(compare_path, compare)

    arr = np.array(times, dtype=np.float64)
    print(f"Latency avg/min/max: {arr.mean():.3f}/{arr.min():.3f}/{arr.max():.3f} ms")
    print(f"FPS: {1000.0 / arr.mean():.2f}")
    print(f"Saved: {args.output}")
    print(f"Saved: {compare_path}")


if __name__ == "__main__":
    main()

