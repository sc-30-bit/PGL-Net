#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import time

import cv2
import numpy as np
import onnxruntime as ort


def pad_to_multiple(x: np.ndarray, div: int = 8) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = x.shape[-2:]
    pad_h = (div - h % div) % div
    pad_w = (div - w % div) % div
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0)
    x = np.pad(x, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
    return x, (pad_h, pad_w)


def preprocess_frame(frame_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    nchw = np.transpose(rgb, (2, 0, 1))[None, ...]
    return nchw


def postprocess_nchw(pred: np.ndarray) -> np.ndarray:
    if pred.ndim == 4:
        pred = pred[0]
    if pred.shape[0] == 1:
        pred = np.repeat(pred, 3, axis=0)
    hwc = np.transpose(np.clip(pred, 0.0, 1.0), (1, 2, 0))
    bgr = cv2.cvtColor((hwc * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return bgr


class TiledONNXRunner:
    def __init__(self, model_path: str, provider: str = "auto"):
        providers = ort.get_available_providers()
        if provider == "cuda":
            if "CUDAExecutionProvider" not in providers:
                raise RuntimeError("CUDAExecutionProvider is not available.")
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif provider == "cpu":
            providers = ["CPUExecutionProvider"]
        else:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in providers else ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name
        self.providers = providers

    def infer_once(self, x: np.ndarray) -> np.ndarray:
        return self.sess.run([self.output_name], {self.input_name: x})[0]

    def process_single_image(self, x: np.ndarray, tile: int | None, tile_overlap: int) -> np.ndarray:
        b, c, h, w = x.shape
        if tile is None or (h <= tile and w <= tile):
            x_pad, (pad_h, pad_w) = pad_to_multiple(x, div=8)
            pred = self.infer_once(x_pad)
            if pad_h > 0 or pad_w > 0:
                pred = pred[..., :h, :w]
            return pred

        stride = tile - tile_overlap
        if stride <= 0:
            raise ValueError("tile must be greater than tile_overlap")

        h_idx_list = list(range(0, max(1, h - tile), stride)) + [max(0, h - tile)]
        w_idx_list = list(range(0, max(1, w - tile), stride)) + [max(0, w - tile)]

        output = np.zeros((b, c, h, w), dtype=np.float32)
        count_map = np.zeros((b, c, h, w), dtype=np.float32)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                h_start = max(0, h_idx)
                w_start = max(0, w_idx)
                h_end = min(h, h_idx + tile)
                w_end = min(w, w_idx + tile)
                in_patch = x[..., h_start:h_end, w_start:w_end]
                in_patch, (pad_h, pad_w) = pad_to_multiple(in_patch, div=8)
                out_patch = self.infer_once(in_patch)
                if pad_h > 0 or pad_w > 0:
                    out_patch = out_patch[..., : h_end - h_start, : w_end - w_start]
                output[..., h_start:h_end, w_start:w_end] += out_patch
                count_map[..., h_start:h_end, w_start:w_end] += 1.0
        return output / np.maximum(count_map, 1e-6)


def run_image(runner: TiledONNXRunner, input_path: str, output_path: str, tile: int | None, tile_overlap: int):
    frame = cv2.imread(input_path)
    if frame is None:
        raise FileNotFoundError(input_path)
    x = preprocess_frame(frame)
    t0 = time.perf_counter()
    pred = runner.process_single_image(x, tile, tile_overlap)
    t1 = time.perf_counter()
    out = postprocess_nchw(pred)
    cv2.imwrite(output_path, out)
    compare = np.hstack((frame if frame.shape == out.shape else cv2.resize(frame, (out.shape[1], out.shape[0])), out))
    compare_path = str(Path(output_path).with_stem(Path(output_path).stem + "_compare"))
    cv2.imwrite(compare_path, compare)
    print(f"Providers: {runner.providers}")
    print(f"Inference: {(t1 - t0) * 1000.0:.2f} ms")
    print(f"Saved: {output_path}")
    print(f"Saved: {compare_path}")


def run_video(runner: TiledONNXRunner, input_path: str, output_path: str, tile: int | None, tile_overlap: int, view: bool):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    times = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        x = preprocess_frame(frame)
        t0 = time.perf_counter()
        pred = runner.process_single_image(x, tile, tile_overlap)
        t1 = time.perf_counter()
        out = postprocess_nchw(pred)
        if out.shape[:2] != (h, w):
            out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
        writer.write(out)
        times.append((t1 - t0) * 1000.0)
        if view:
            cv2.imshow("PGLNet Tiled Inference", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    if times:
        arr = np.array(times)
        print(f"Latency avg/min/max: {arr.mean():.2f}/{arr.min():.2f}/{arr.max():.2f} ms")
        print(f"FPS: {1000.0 / arr.mean():.2f}")
    print(f"Saved: {output_path}")


def parse_opt():
    p = argparse.ArgumentParser(description="PGLNet tiled inference (SAHI-like sliding window)")
    p.add_argument("--model", required=True, help="Path to PGLNet ONNX model")
    p.add_argument("--source", required=True, help="Image or video path")
    p.add_argument("--output", required=True, help="Output image or video path")
    p.add_argument("--provider", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--tile", type=int, default=None, help="Tile size, e.g. 512. None means whole image inference")
    p.add_argument("--tile-overlap", type=int, default=32, help="Tile overlap size")
    p.add_argument("--view-img", action="store_true", help="Show video inference window")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_opt()
    runner = TiledONNXRunner(args.model, args.provider)
    suffix = Path(args.source).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
        run_image(runner, args.source, args.output, args.tile, args.tile_overlap)
    else:
        run_video(runner, args.source, args.output, args.tile, args.tile_overlap, args.view_img)

