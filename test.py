import torch
import os
import sys
import argparse
import numpy as np
import cv2
from PIL import Image
import torch.nn as nn
import torchvision.transforms as tfs
import warnings
import time
import importlib

from models import pglnet_t, pglnet_s, pglnet_b, pglnet_d
from basicsr.metrics.niqe import calculate_niqe
from utils.metrics import psnr, ssim, calculate_lpips, _get_lpips_model

warnings.filterwarnings('ignore')

DEFAULT_TEST_DIR = "/home/klay/papersToReproduce/subset_RW2AH/test/input"
DEFAULT_GT_DIR = "/home/klay/papersToReproduce/subset_RW2AH/test/gt"

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def extract_model_type(weight_filename):
    base_name = os.path.basename(weight_filename)
    base_name = os.path.splitext(base_name)[0]

    parts = base_name.split('_')

    if len(parts) < 2:
        raise ValueError(f"Cannot parse weight filename format: {base_name}")

    for i, part in enumerate(parts):
        if part == 'pglnet' and i + 1 < len(parts):
            variant = parts[i + 1]
            if variant in ['t', 's', 'b', 'd']:
                return f'pglnet_{variant}'

    return 'pglnet_t'

def load_model(weights_path, device, model_type=None):
    if model_type is None:
        model_type = extract_model_type(weights_path)
    else:
        print(f"Using explicitly specified model type: {model_type}")
    print(f"Initializing model: {model_type}...")
    
    if model_type.startswith('pglnet_'):
        variant = model_type.split('_')[-1]
        if variant == 't':
            net = pglnet_t()
        elif variant == 's':
            net = pglnet_s()
        elif variant == 'b':
            net = pglnet_b()
        elif variant == 'd':
            net = pglnet_d()
        else:
            raise ValueError(f"Unsupported pglnet variant: {variant}")
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    print(f'Loading weights file: {weights_path}')
    try:
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        
        state_dict = None
        if isinstance(checkpoint, dict):
            for key in ['params', 'params_ema', 'model', 'net', 'state_dict']:
                if key in checkpoint:
                    print(f'Detected key "{key}", attempting to load...')
                    state_dict = checkpoint[key]
                    break
            
            if state_dict is None and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        if state_dict is None:
            raise ValueError("Unable to parse weight file structure")

        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        try:
            net.load_state_dict(new_state_dict, strict=True)
            print("Weights loaded successfully (Strict mode)")
        except Exception as e_strict:
            print(f"Strict loading failed ({str(e_strict)}), attempting non-strict loading...")
            net.load_state_dict(new_state_dict, strict=False)
            print("Weights loaded successfully (Non-Strict mode)")

    except Exception as e:
        print(f"Standard loading failed: {e}")
        print("Attempting to load after wrapping with DataParallel...")
        net = nn.DataParallel(net)
        net.load_state_dict(checkpoint['params'] if 'params' in checkpoint else checkpoint)

    net.to(device)
    net.eval()
    return net

def evaluate_metrics(net, test_dir, gt_dir, device, save_results=False, save_dir=None,
                    val_only_psnr=False, tile=None, tile_overlap=32, lpips_eval=True):
    metrics = {'psnr': [], 'ssim': [] if not val_only_psnr else None, 'niqe': [], 
               'lpips': [] if (lpips_eval and not val_only_psnr) else None}
    
    metrics_per_image = []
    
    test_images = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    test_images.sort()
    total_images = len(test_images)
    
    if total_images == 0:
        print("Error: No images found in test directory")
        return {}

    lpips_model = None
    try:
        if lpips_eval and not val_only_psnr:
            lpips_model = _get_lpips_model(device=device)
    except Exception as e:
        print(f"LPIPS initialization failed: {e}, skipping LPIPS calculation")

    if save_results and save_dir:
        os.makedirs(os.path.join(save_dir, 'input'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'output'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'gt'), exist_ok=True)
    
    def process_single_image(inputs, net, tile, tile_overlap):
        b, c, h, w = inputs.shape
        
        if tile is None or (h <= tile and w <= tile):
            pad_h = (32 - h % 32) % 32
            pad_w = (32 - w % 32) % 32  
            if pad_h > 0 or pad_w > 0:
                inputs = torch.nn.functional.pad(inputs, (0, pad_w, 0, pad_h), 'reflect')
        
            pred = net(inputs)

            if pad_h > 0 or pad_w > 0:
                pred = pred[..., :h, :w]

            return pred

        else:
            stride = tile - tile_overlap
            h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
            w_idx_list = list(range(0, w - tile, stride)) + [w - tile]

            output = torch.zeros((b, c, h, w), device=inputs.device)
            count_map = torch.zeros((b, c, h, w), device=inputs.device)

            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    h_start = max(0, h_idx)
                    w_start = max(0, w_idx)
                    h_end = min(h, h_idx + tile)
                    w_end = min(w, w_idx + tile)

                    in_patch = inputs[..., h_start:h_end, w_start:w_end]

                    out_patch = net(in_patch)

                    output[..., h_start:h_end, w_start:w_end] += out_patch
                    count_map[..., h_start:h_end, w_start:w_end] += 1.0

            return output / count_map

    print(f"Starting evaluation on {total_images} images...")

    with torch.no_grad():
        for i, im_name in enumerate(test_images):
            img_path = os.path.join(test_dir, im_name)
            haze_img = Image.open(img_path).convert('RGB')
            haze_tensor = tfs.ToTensor()(haze_img).unsqueeze(0).to(device)

            gt_tensor = None
            gt_img = None
            if gt_dir:
                base, ext = os.path.splitext(im_name)
                candidates = [
                    im_name,
                    base.replace('_input', '') + ext,
                    base.replace('_input', '_gt') + ext,
                    base + '_gt' + ext,
                    base + ext
                ]
                found = False
                for cand in candidates:
                    gt_path = os.path.join(gt_dir, cand)
                    if os.path.exists(gt_path):
                        gt_img = Image.open(gt_path).convert('RGB')
                        gt_tensor = tfs.ToTensor()(gt_img).unsqueeze(0).to(device)
                        if cand != im_name:
                            print(f"Matched GT file: {cand} <- {im_name}")
                        found = True
                        break
                if not found:
                    print(f"Warning: No corresponding GT image found for: {im_name}")

            output = process_single_image(haze_tensor, net, tile, tile_overlap)

            output_clamped = output.clamp(0, 1)

            pred_np = (output_clamped.squeeze().cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)

            curr_psnr = 0
            if gt_tensor is not None:
                curr_psnr = psnr(output_clamped, gt_tensor)
                metrics['psnr'].append(curr_psnr)

            curr_ssim = 0
            curr_lpips = 0
            if not val_only_psnr and gt_tensor is not None:
                curr_ssim = ssim(output_clamped, gt_tensor, window_size=11).item()
                metrics['ssim'].append(curr_ssim)

                if lpips_eval and lpips_model is not None:
                    curr_lpips = calculate_lpips(output_clamped, gt_tensor, device=device, tile=tile, tile_overlap=tile_overlap)
                    metrics['lpips'].append(curr_lpips)

            curr_niqe = 0
            try:
                pred_bgr = cv2.cvtColor(pred_np, cv2.COLOR_RGB2BGR)
                curr_niqe = calculate_niqe(pred_bgr, crop_border=0, input_order='HWC', convert_to='y')
                metrics['niqe'].append(curr_niqe)
            except Exception:
                pass 
            
            metrics_per_image.append({
                'image_name': im_name,
                'psnr': curr_psnr,
                'ssim': curr_ssim,
                'niqe': curr_niqe,
                'lpips': curr_lpips
            })

            if val_only_psnr:
                print(f"[{i+1}/{total_images}] {im_name} -> PSNR: {curr_psnr:.2f}")
            else:
                print(f"[{i+1}/{total_images}] {im_name} -> "
                      f"PSNR: {curr_psnr:.2f} | SSIM: {curr_ssim:.4f} | "
                      f"NIQE: {curr_niqe:.4f} | LPIPS: {curr_lpips:.4f}")

            if save_results and save_dir:
                cv2.imwrite(os.path.join(save_dir, 'output', im_name), cv2.cvtColor(pred_np, cv2.COLOR_RGB2BGR))
                input_np = np.array(haze_img)
                cv2.imwrite(os.path.join(save_dir, 'input', im_name), cv2.cvtColor(input_np, cv2.COLOR_RGB2BGR))
                if gt_img:
                    gt_np = np.array(gt_img)
                    cv2.imwrite(os.path.join(save_dir, 'gt', im_name), cv2.cvtColor(gt_np, cv2.COLOR_RGB2BGR))

    avg_psnr = np.mean(metrics['psnr']) if metrics['psnr'] else 0
    avg_ssim = np.mean(metrics['ssim']) if metrics['ssim'] else 0
    avg_niqe = np.mean(metrics['niqe']) if metrics['niqe'] else 0
    avg_lpips = np.mean(metrics['lpips']) if metrics['lpips'] else 0

    print("\n" + "="*40)
    print("           Final Results           ")
    print("="*40)
    print(f"Average PSNR  : {avg_psnr:.4f} dB")
    print(f"Average SSIM  : {avg_ssim:.4f}")
    print(f"Average NIQE  : {avg_niqe:.4f}")
    print(f"Average LPIPS : {avg_lpips:.4f}")
    print("="*40)

    if save_results and save_dir:
        import csv
        metrics_path = os.path.join(save_dir, 'metrics_per_image.csv')
        if metrics_per_image:
            with open(metrics_path, 'w', newline='', encoding='utf-8') as f:
                fieldnames = metrics_per_image[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(metrics_per_image)
            print(f"Per-image metrics saved to: {metrics_path}")
    
    return {
        'psnr': avg_psnr, 'ssim': avg_ssim,
        'niqe': avg_niqe, 'lpips': avg_lpips,
        'per_image_metrics': metrics_per_image
    }

def main():
    parser = argparse.ArgumentParser(description='PGLNet Model Evaluation Script')
    
    parser.add_argument('--weight', type=str, required=True, help='Path to weight file (.pth / .pk / .pt)')
    parser.add_argument('--test_dir', type=str, default=DEFAULT_TEST_DIR, 
                        help=f'Test images directory (default: {DEFAULT_TEST_DIR})')
    parser.add_argument('--gt_dir', type=str, default=DEFAULT_GT_DIR, 
                        help=f'GT images directory (default: {DEFAULT_GT_DIR})')
    parser.add_argument('--save_results', action='store_true', help='Save inference result images')
    parser.add_argument('--save_dir', type=str, default=None, help='Root directory for saving results')
    parser.add_argument('--val_only_psnr', action='store_true', help='Calculate only PSNR metric')
    parser.add_argument('--tile', type=int, default=None, help='Tile size for processing large images')
    parser.add_argument('--tile_overlap', type=int, default=32, help='Overlap size between tiles')
    parser.add_argument('--no_lpips', action='store_true', help='Skip LPIPS calculation')
    parser.add_argument('--model_type', type=str, default='none', help='Explicitly specify model type (e.g., pglnet_t, pglnet_s, etc.). If not specified, inferred from weight filename.')

    args = parser.parse_args()

    print(f"Script configuration:")
    print(f"  - Weight: {args.weight}")
    print(f"  - Input directory: {args.test_dir}")
    print(f"  - GT directory: {args.gt_dir}")
    print(f"  - PSNR only: {args.val_only_psnr}")
    print(f"  - Tile size: {args.tile}")
    print(f"  - Tile overlap: {args.tile_overlap}")
    print(f"  - Calculate LPIPS: {not args.no_lpips}")
    print(f"  - Explicit model type: {args.model_type if args.model_type is not None else 'Auto inference'}")
    
    device = get_device()
    
    if not os.path.isfile(args.weight):
        print(f"Error: Weight file {args.weight} does not exist")
        return
    
    weight_file = args.weight
    
    print(f"\n" + "="*60)
    print(f"Starting evaluation of weight file: {os.path.basename(weight_file)}")
    print("="*60)

    try:
        net = load_model(weight_file, device, args.model_type)
    except Exception as e:
        print(f"Failed to load weight file {weight_file}: {e}")
        return

    if args.save_results:
        full_filename = os.path.basename(weight_file)
        model_name = full_filename
        if args.save_dir is not None:
            save_path = os.path.join(args.save_dir, "evaluation_results", model_name)
        else:
            save_path = f"./evaluation_results/{model_name}"
        print(f"  - Results will be saved to: {save_path}")
    else:
        save_path = None

    results = evaluate_metrics(
        net=net,
        test_dir=args.test_dir,
        gt_dir=args.gt_dir,
        device=device,
        save_results=args.save_results,
        save_dir=save_path,
        val_only_psnr=args.val_only_psnr,
        tile=args.tile,
        tile_overlap=args.tile_overlap,
        lpips_eval=not args.no_lpips
    )

    if args.save_results and save_path:
        final_metrics_path = os.path.join(save_path, 'model_final_metrics.txt')
        with open(final_metrics_path, 'w', encoding='utf-8') as f:
            f.write("Model Final Performance\n")
            f.write("="*20 + "\n")
            f.write(f"Average PSNR  : {results['psnr']:.4f} dB\n")
            f.write(f"Average SSIM  : {results['ssim']:.4f}\n")
            f.write(f"Average NIQE  : {results['niqe']:.4f}\n")
            f.write(f"Average LPIPS : {results['lpips']:.4f}\n")
        print(f"Model final average metrics saved to: {final_metrics_path}")

if __name__ == '__main__':
    main()