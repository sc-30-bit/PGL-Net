from math import exp
import math
import numpy as np

import torch
from torchvision.transforms import ToPILImage
import torch.nn.functional as F
from torch.autograd import Variable
import lpips  
from pytorch_msssim import SSIM 

def ssim(img1, img2, window_size=11, size_average=True):
    img1 = torch.clamp(img1, min=0, max=1)
    img2 = torch.clamp(img2, min=0, max=1)
    (_, channel, _, _) = img1.size()
    ssim_calculator = SSIM(data_range=1.0, win_size=window_size, size_average=size_average, channel=channel)
    with torch.no_grad():
        ssim_value = ssim_calculator(img1, img2)
    if not size_average:
        ssim_value = ssim_value.mean(1).mean(1).mean(1)
    
    return ssim_value
    
def psnr(pred, gt):
    pred = pred.clamp(0, 1)
    gt = gt.clamp(0, 1)
    imdff = pred - gt
    
    mse = torch.mean(imdff ** 2)
    
    rmse = torch.sqrt(mse)
    
    if rmse == 0:
        return 100.0

    psnr_val = 20 * torch.log10(1.0 / rmse)
    
    return psnr_val.item()


_LPIPS_MODEL = None
_LPIPS_NET = None
_LPIPS_DEVICE = None

def _get_lpips_model(net_type='vgg', device=None):
    global _LPIPS_MODEL, _LPIPS_NET, _LPIPS_DEVICE
    needs_new = (_LPIPS_MODEL is None) or (net_type != _LPIPS_NET) or (device != _LPIPS_DEVICE)
    if needs_new:
        model = lpips.LPIPS(net=net_type)
        if device is not None:
            model = model.to(device)
        model.eval()
        _LPIPS_MODEL = model
        _LPIPS_NET = net_type
        _LPIPS_DEVICE = device
    return _LPIPS_MODEL

def calculate_lpips(pred, gt, net_type='vgg', device=None, tile=None, tile_overlap=32):
    pred = torch.clamp(pred, min=0, max=1)
    gt = torch.clamp(gt, min=0, max=1)

    model = _get_lpips_model(net_type, device)

    with torch.no_grad():
        b, c, h, w = pred.shape
        
        if tile is None or (h <= tile and w <= tile):
            dist = model(pred, gt)
            return dist.mean().item()
        
        else:
            stride = tile - tile_overlap
            
            h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
            w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
            
            lpips_values = []
            
            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    h_start = max(0, h_idx)
                    w_start = max(0, w_idx)
                    h_end = min(h, h_idx + tile)
                    w_end = min(w, w_idx + tile)
                    
                    pred_patch = pred[..., h_start:h_end, w_start:w_end]
                    gt_patch = gt[..., h_start:h_end, w_start:w_end]
                    
                    patch_dist = model(pred_patch, gt_patch)
                    lpips_values.append(patch_dist.mean().item())

            return sum(lpips_values) / len(lpips_values)

if __name__ == "__main__":
    pass