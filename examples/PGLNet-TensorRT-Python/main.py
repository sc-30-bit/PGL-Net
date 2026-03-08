#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt


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


def postprocess(output: np.ndarray, out_shape: tuple[int, ...], orig_size: tuple[int, int], resize_back: bool):
    output = output.reshape(out_shape)
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


def trt_to_numpy_dtype(dtype):
    if dtype == trt.float16:
        return np.float16
    if dtype == trt.float32:
        return np.float32
    if dtype == trt.int32:
        return np.int32
    if dtype == trt.int8:
        return np.int8
    return np.float32


class TRTInfer:
    def __init__(self, engine_path: str):
        logger = trt.Logger(trt.Logger.INFO)
        with open(engine_path, "rb") as f:
            engine_data = f.read()
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        if self.engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine.")
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self.input = None
        self.output = None
        self.bindings = []

        if hasattr(self.engine, "num_io_tensors"):
            self._init_new_api()
        else:
            self._init_old_api()

    def _init_new_api(self):
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = trt_to_numpy_dtype(self.engine.get_tensor_dtype(name))
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT

            if any(d < 0 for d in shape):
                raise RuntimeError(f"Dynamic shape is not configured for tensor {name}: {shape}")
            size = int(trt.volume(shape))
            device_mem = cuda.mem_alloc(size * np.dtype(dtype).itemsize)
            binding = {"name": name, "shape": shape, "dtype": dtype, "size": size, "device_mem": device_mem}
            self.bindings.append(int(device_mem))
            if is_input:
                self.input = binding
            else:
                self.output = binding

    def _init_old_api(self):
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            shape = tuple(self.context.get_binding_shape(i))
            dtype = trt_to_numpy_dtype(self.engine.get_binding_dtype(i))
            is_input = self.engine.binding_is_input(i)

            if any(d < 0 for d in shape):
                raise RuntimeError(f"Dynamic shape is not configured for binding {name}: {shape}")
            size = int(trt.volume(shape))
            device_mem = cuda.mem_alloc(size * np.dtype(dtype).itemsize)
            binding = {"name": name, "shape": shape, "dtype": dtype, "size": size, "device_mem": device_mem}
            self.bindings.append(int(device_mem))
            if is_input:
                self.input = binding
            else:
                self.output = binding

    def infer(self, input_tensor: np.ndarray):
        inp = input_tensor.astype(self.input["dtype"]).ravel()
        out = np.empty(self.output["size"], dtype=self.output["dtype"])

        cuda.memcpy_htod_async(self.input["device_mem"], inp, self.stream)
        if hasattr(self.context, "execute_async_v3"):
            self.context.set_tensor_address(self.input["name"], int(self.input["device_mem"]))
            self.context.set_tensor_address(self.output["name"], int(self.output["device_mem"]))
            self.context.execute_async_v3(stream_handle=self.stream.handle)
        else:
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(out, self.output["device_mem"], self.stream)
        self.stream.synchronize()
        return out


def main():
    parser = argparse.ArgumentParser(description="PGLNet TensorRT inference")
    parser.add_argument("--model", "-m", required=True, help="Path to .engine")
    parser.add_argument("--input", "-i", required=True, help="Path to input image")
    parser.add_argument("--output", "-o", default="output.jpg", help="Path to output image")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup runs")
    parser.add_argument("--runs", type=int, default=20, help="Benchmark runs")
    parser.add_argument("--resize-back", action="store_true", help="Resize output back to original size")
    args = parser.parse_args()

    infer = TRTInfer(args.model)
    in_shape = infer.input["shape"]  # NCHW
    width, height = int(in_shape[3]), int(in_shape[2])
    input_tensor, orig_img, orig_size = preprocess(args.input, width, height, infer.input["dtype"])

    print(f"Input: {infer.input['name']} {infer.input['shape']} {infer.input['dtype']}")
    print(f"Output: {infer.output['name']} {infer.output['shape']} {infer.output['dtype']}")
    print(f"Tensor: {input_tensor.shape} {input_tensor.dtype}")

    for _ in range(args.warmup):
        _ = infer.infer(input_tensor)

    times = []
    out = None
    for _ in range(args.runs):
        t0 = time.perf_counter()
        out = infer.infer(input_tensor)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    out_img = postprocess(out, infer.output["shape"], orig_size, args.resize_back)
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

