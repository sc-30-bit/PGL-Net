import os
import sys

# Add parent directory to path to import models
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import glob
import re
import torch
import torch.nn as nn

from models.PGLNet import pglnet_t, pglnet_s, pglnet_b, pglnet_d, DSConv, DAMBlock, GateLayer

def fold_bn_into_following_conv(conv, bn):
    """
    Fuse BatchNorm2d + Conv2d when execution order is BN -> Conv.
    This is required for PGLNet blocks where BN is applied BEFORE convs.
    """
    if not isinstance(conv, nn.Conv2d) or not isinstance(bn, nn.BatchNorm2d):
        raise TypeError("fold_bn_into_following_conv expects (Conv2d, BatchNorm2d)")
    if conv.in_channels != bn.num_features:
        raise ValueError(
            f"BN features ({bn.num_features}) must match Conv in_channels ({conv.in_channels})"
        )

    with torch.no_grad():
        w = conv.weight
        if conv.bias is None:
            b = torch.zeros(conv.out_channels, device=w.device, dtype=w.dtype)
        else:
            b = conv.bias

        # BN(x) = a * x + c
        a = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        c = bn.bias - bn.running_mean * a
        a = a.to(dtype=w.dtype, device=w.device)
        c = c.to(dtype=w.dtype, device=w.device)

        g = conv.groups
        in_per_group = conv.in_channels // g
        out_per_group = conv.out_channels // g

        w_group = w.view(g, out_per_group, in_per_group, w.shape[2], w.shape[3])
        w_orig = w_group.clone()

        # Weight scaling for input channels by 'a'
        a_group = a.view(g, in_per_group)
        w_fused = w_group * a_group[:, None, :, None, None]

        # Bias shift by convolution over constant term 'c'
        c_group = c.view(g, in_per_group)
        w_sum = w_orig.view(g, out_per_group, in_per_group, -1).sum(dim=-1)
        bias_shift = (w_sum * c_group[:, None, :]).sum(dim=-1)

        b_group = b.view(g, out_per_group)
        b_fused = (b_group + bias_shift).reshape(-1)

        conv.weight.copy_(w_fused.view_as(w))
        conv.bias = nn.Parameter(b_fused)


def fuse_bn_pglnet_structured(model):
    """
    PGLNet-aware BN fusion.

    Why not recursive Conv->BN fusion:
    - Several modules execute BN BEFORE conv in forward(), even if declaration order differs.
    - Blind Conv->BN fusion can fuse the wrong pair and change numerics drastically.
    """
    for m in model.modules():
        # DSConv forward: x = bn(x) -> dw(x) -> pw(x)
        if isinstance(m, DSConv) and isinstance(m.bn, nn.BatchNorm2d):
            fold_bn_into_following_conv(m.dw, m.bn)
            m.bn = nn.Identity()

        # DAMBlock forward: x = norm(x) -> GateLayer(x)
        # GateLayer uses x in two branches: Wv[0](x), Wg[0](x)
        elif isinstance(m, DAMBlock) and isinstance(m.norm, nn.BatchNorm2d) and isinstance(m.conv, GateLayer):
            wv0 = m.conv.Wv[0] if len(m.conv.Wv) > 0 else None
            wg0 = m.conv.Wg[0] if len(m.conv.Wg) > 0 else None
            if isinstance(wv0, nn.Conv2d) and isinstance(wg0, nn.Conv2d):
                fold_bn_into_following_conv(wv0, m.norm)
                fold_bn_into_following_conv(wg0, m.norm)
                m.norm = nn.Identity()


def build_model(name):
    return {
        "pglnet_t": pglnet_t,
        "pglnet_s": pglnet_s,
        "pglnet_b": pglnet_b,
        "pglnet_d": pglnet_d,
    }[name]()


def load_weights(model, weight_path):
    """Load model weights from checkpoint file."""
    if not os.path.exists(weight_path):
        print(f"Warning: Weight file not found: {weight_path}")
        print("Proceeding with random initialization.")
        return model
    
    checkpoint = torch.load(weight_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present (from DataParallel)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    print(f"Loaded weights from: {weight_path}")
    return model


def _is_weight_file(path):
    return os.path.isfile(path) and path.lower().endswith((".pk", ".pth", ".pt"))


def parse_weight_file(path):
    """Parse one weight file path into (model_name, weight_path, output_name)."""
    filename = os.path.basename(path)
    match = re.search(r"(pglnet_[tsbd])", filename)
    if not match:
        return None
    model_name = match.group(1)
    output_name = os.path.splitext(filename)[0]
    return model_name, path, output_name

def find_weight_files(weight_dir):
    """Find all available weight files in the weight directory.
    
    Returns a list of tuples: [(model_name, weight_path, output_name), ...]
    e.g., [("pglnet_s", ".../rrshid_pglnet_s.pk", "rrshid_pglnet_s"), ...]
    """
    weight_files = []
    if not os.path.exists(weight_dir):
        return weight_files
    
    for filename in os.listdir(weight_dir):
        weight_path = os.path.join(weight_dir, filename)
        if _is_weight_file(weight_path):
            parsed = parse_weight_file(weight_path)
            if parsed is not None:
                weight_files.append(parsed)
    return weight_files


def resolve_weight_files(weights_arg, default_weight_dir):
    """
    Resolve --weights into tuples [(model_name, weight_path, output_name), ...].
    Supports file, directory, glob pattern, and comma/semicolon-separated list.
    """
    if not weights_arg:
        return find_weight_files(default_weight_dir)

    entries = [p.strip() for p in re.split(r"[;,]", weights_arg) if p.strip()]
    candidates = []
    for entry in entries:
        if os.path.isdir(entry):
            candidates.extend(glob.glob(os.path.join(entry, "*.pk")))
            candidates.extend(glob.glob(os.path.join(entry, "*.pth")))
            candidates.extend(glob.glob(os.path.join(entry, "*.pt")))
        elif _is_weight_file(entry):
            candidates.append(entry)
        else:
            matches = glob.glob(entry)
            if matches:
                candidates.extend(matches)

    seen = set()
    ordered = []
    for p in candidates:
        ap = os.path.abspath(p)
        if ap in seen or not _is_weight_file(ap):
            continue
        seen.add(ap)
        ordered.append(ap)

    weight_files = []
    for p in ordered:
        parsed = parse_weight_file(p)
        if parsed is None:
            print(f"Skip weight file (cannot infer model variant): {p}")
            continue
        weight_files.append(parsed)
    return weight_files

def export_model(model_name, weight_path, output_path, shape, opset, fp16, onnxsim, bn_fold=False):
    """Export a single model to ONNX."""
    b, c, h, w = map(int, shape.split("x"))
    device = "cuda" if fp16 else "cpu"

    print(f"\n{'='*60}")
    print(f"Exporting {model_name}")
    print(f"Weight: {weight_path}")
    print(f"Output: {output_path}")
    print(f"Shape: {shape}, FP16: {fp16}, ONNXSim: {onnxsim}")
    print(f"{'='*60}")

    model = build_model(model_name).eval()

    if weight_path and os.path.exists(weight_path):
        model = load_weights(model, weight_path)
    else:
        print(f"Warning: No weight file found, using random initialization")

    if bn_fold:
        fuse_bn_pglnet_structured(model)

    if fp16:
        model = model.half()

    model.to(device)

    class Wrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            out = self.m(x)
            return out[-1] if isinstance(out, (list, tuple)) else out

    wrapper = Wrapper(model)

    x = torch.randn(
        b, c, h, w,
        device=device,
        dtype=torch.float16 if fp16 else torch.float32
    )

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            x,
            output_path,
            opset_version=opset,
            input_names=["input"],
            output_names=["output"],
            do_constant_folding=True,
            dynamic_axes=None,
        )

    if onnxsim:
        import onnx
        from onnxsim import simplify
        model_onnx = onnx.load(output_path)
        model_simp, check = simplify(model_onnx)
        assert check
        onnx.save(model_simp, output_path)

    report = {
        "model": model_name,
        "shape": shape,
        "bn_folded": bool(bn_fold),
        "fp16": fp16,
        "onnxsim": onnxsim,
        "opset": opset,
        "weights_loaded": weight_path is not None and os.path.exists(weight_path),
        "weight_path": weight_path if weight_path and os.path.exists(weight_path) else None,
    }
    with open(output_path + ".report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"=== EXPORT DONE: {model_name} ===")
    print(json.dumps(report, indent=2))
    print(f"ONNX: {output_path}")
    
    return report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    choices=["pglnet_t", "pglnet_s", "pglnet_b", "pglnet_d"],
                    help="Model to export. If not specified, exports all available models with weights")
    ap.add_argument("--weights", default=None,
                    help="Weights source: file/dir/glob/list (comma or semicolon separated).")
    ap.add_argument("--onnx", default=None,
                    help="Output ONNX path. If not specified, uses default naming in output/ folder")
    ap.add_argument("--shape", default="1x3x512x512")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--onnxsim", action="store_true")
    ap.add_argument("--bn_fold", action="store_true",
                    help="Enable Conv+BN folding. Disabled by default for PGLNet safety.")
    ap.add_argument("--batch", action="store_true",
                    help="Batch export all available models with weights")
    args = ap.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    weight_dir = os.path.join(os.path.dirname(script_dir), "PGLNet_Weights")
    output_dir = os.path.join(script_dir, "onnx_output")
    os.makedirs(output_dir, exist_ok=True)
    
    weight_files = resolve_weight_files(args.weights, weight_dir)
    print(f"Found weight files: {weight_files}")

    if args.batch:
        # Batch export all models with available weights
        if not weight_files:
            source_hint = args.weights if args.weights else weight_dir
            print(f"No valid weight files found from: {source_hint}")
            return
        
        # In batch mode, --onnx is treated as an output directory.
        # This works even when the directory does not exist yet.
        batch_out_dir = output_dir
        if args.onnx:
            batch_out_dir = args.onnx
            # Prevent accidental file path usage in batch mode.
            if str(batch_out_dir).lower().endswith(".onnx"):
                raise ValueError("--batch mode expects --onnx to be a directory, not a .onnx file path")
        os.makedirs(batch_out_dir, exist_ok=True)
        
        for model_name, weight_path, output_name in weight_files:
            # Determine output path - use output_name (e.g., "rrshid_pglnet_s")
            output_path = os.path.join(batch_out_dir, f"{output_name}.onnx")
            
            try:
                export_model(
                    model_name=model_name,
                    weight_path=weight_path,
                    output_path=output_path,
                    shape=args.shape,
                    opset=args.opset,
                    fp16=args.fp16,
                    onnxsim=args.onnxsim,
                    bn_fold=args.bn_fold
                )
            except Exception as e:
                print(f"Error exporting {model_name} ({output_name}): {e}")
                import traceback
                traceback.print_exc()
    
    elif args.model:
        # Single model export
        if args.weights:
            matches = [(m, w, o) for (m, w, o) in weight_files if m == args.model]
            if not matches:
                raise ValueError(f"No weight file matches --model {args.model} from --weights {args.weights}")
            _, weight_path, output_name = matches[0]
        else:
            # Auto-detect weight path - find first matching model
            weight_path = None
            output_name = args.model
            for m_name, w_path, o_name in weight_files:
                if m_name == args.model:
                    weight_path = w_path
                    output_name = o_name
                    break
        
        # Determine output path
        if args.onnx:
            output_path = args.onnx
        else:
            output_path = os.path.join(output_dir, f"{output_name}.onnx")
        
        export_model(
            model_name=args.model,
            weight_path=weight_path,
            output_path=output_path,
            shape=args.shape,
            opset=args.opset,
            fp16=args.fp16,
            onnxsim=args.onnxsim,
            bn_fold=args.bn_fold
        )
    
    else:
        print("Please specify --model or use --batch to export all available models")
        available_models = list(set([m for m, _, _ in weight_files]))
        print(f"Available models with weights: {available_models}")

if __name__ == "__main__":
    main()
