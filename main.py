import torch,os,sys,torchvision,argparse
import torchvision.transforms as tfs
from metrics import psnr,ssim,calculate_lpips
from models import *
import time,math
import numpy as np
from torch.backends import cudnn
from torch import optim
import torch,warnings
from torch import nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
warnings.filterwarnings('ignore')
from option import opt,model_name,log_dir
from data_utils import *
from losses import *
from lr_scheduler import *
from utils import *

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
cudnn.deterministic = True
cudnn.benchmark = False

setup_console_logger(log_dir, model_name)
print('log_dir :',log_dir)
print('model_name:',model_name)

import csv
from datetime import datetime

LOG_CSV = os.path.join(log_dir, 'csv', 'model_saves.csv')

def log_model_save_csv(step, max_psnr, max_ssim, best_lpips=None, log_file=LOG_CSV):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    exists = os.path.exists(log_file)
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(['timestamp', 'step', 'max_psnr', 'max_ssim', 'best_lpips'])
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            int(step),
            float(max_psnr),
            float(max_ssim),
            '' if best_lpips is None else float(best_lpips)
        ])


models_={
    'pglnet_t': pglnet_t(),
    'pglnet_s': pglnet_s(),
    'pglnet_b': pglnet_b(),
    'pglnet_d': pglnet_d(),
}
loaders_={
	'rw2ah_train':RW2AH_train_loader,
	'rw2ah_test':RW2AH_test_loader,
    'rudb_train':RUDB_train_loader,
	'rudb_test':RUDB_test_loader,
    'rrshid_train':RRSHID_train_loader,
	'rrshid_test':RRSHID_test_loader,
}
start_time=time.time()
T=opt.steps	

def lr_schedule_cosdecay(t,T,init_lr=opt.lr):
	lr=0.5*(1+math.cos(t*math.pi/T))*init_lr
	return lr

def train(net,loader_train,loader_test,optim,criterion, scaler, scheduler=None):
    losses=[]
    start_step=0
    max_ssim=0
    max_psnr=0
    min_lpips = float('inf') if opt.lpips_eval else None 
    ssims=[]
    psnrs=[]
    lpips_list = [] if opt.lpips_eval else None
    if opt.resume and os.path.exists(opt.model_dir):
        print(f'Resume from {opt.model_dir}')
        ckp=torch.load(opt.model_dir,map_location=opt.device,weights_only=False)
        net.load_state_dict(ckp['model'])
        start_step=ckp['step']
        max_ssim=ckp['max_ssim']
        max_psnr=ckp['max_psnr']
        psnrs=ckp['psnrs']
        ssims=ckp['ssims']
        if opt.lpips_eval:
            min_lpips = ckp['min_lpips']
            lpips_list = ckp['lpips_list']
        if scheduler is not None and 'scheduler_state_dict' in ckp:
            scheduler.last_epoch = start_step - 1
            print(f'The scheduler state has been restored, current last_epoch: {scheduler.last_epoch}')
        print(f'start_step:{start_step} start training ---')
    else :
        print('Train from scratch *** ')
    
    writer = SummaryWriter(log_dir=log_dir, comment=model_name)
    train_iter = iter(loader_train)
    
    for step in range(start_step+1,opt.steps+1):
        net.train()
        lr=opt.lr
        if not opt.no_lr_sche:
            if scheduler is not None:
                scheduler.step()
                lr = scheduler.get_last_lr()[0]
            else:
                lr=lr_schedule_cosdecay(step,T)
                for param_group in optim.param_groups:
                    param_group["lr"] = lr  
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(loader_train)
            x, y = next(train_iter)

        x = x.to(opt.device)  
        y = y.to(opt.device)  

        optim.zero_grad()
        if opt.amp:
            with torch.cuda.amp.autocast():
                out = net(x)

                loss_config = criterion[-1]

                l1_loss_val = criterion[0](out, y).mean()
                loss = l1_loss_val

                if loss_config['use_fft']:
                    fft_loss_val = criterion[1](out, y)
                    loss += loss_config['fft_weight'] * fft_loss_val
            
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print(f"Warning: NaN/Inf detected in loss at step {step}, skipping gradient update")
                continue
            
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            out = net(x)

            loss_config = criterion[-1]

            l1_loss_val = criterion[0](out, y).mean()
            loss = l1_loss_val

            loss.backward()
            optim.step()

        losses.append(loss.item())
        print(f'\rtrain loss : {loss.item():.5f}| step :{step}/{opt.steps}|lr :{lr :.7f} |time_used :{(time.time()-start_time)/60 :.1f}',end='',flush=True)

        writer.add_scalar('data/loss',loss,step)

        if step % opt.eval_step ==0 and step > 0:
            with torch.no_grad():
                if opt.lpips_eval:
                    val_ssim, val_psnr, val_lpips = test(net, loader_test, max_psnr, max_ssim, step)
                    if opt.val_only_psnr:
                        print(f'\nstep :{step} |val_psnr :{val_psnr:.4f}')
                    else:
                        print(f'\nstep :{step} |val_psnr :{val_psnr:.4f}|val_ssim:{val_ssim:.4f}|val_lpips:{val_lpips:.4f}')
                else:
                    val_ssim, val_psnr = test(net, loader_test, max_psnr, max_ssim, step)
                    if opt.val_only_psnr:
                        print(f'\nstep :{step} |val_psnr :{val_psnr:.4f}')
                    else:
                        print(f'\nstep :{step} |val_psnr :{val_psnr:.4f}|val_ssim:{val_ssim:.4f}')
            
            psnr_normalizer.update(val_psnr)
            if opt.val_only_psnr:
                psnr_norm = psnr_normalizer.norm(float(val_psnr))
                score = psnr_norm  
            else:
                score = compute_score_normalized(
                    val_psnr,
                    val_ssim,
                    val_lpips if opt.lpips_eval else None,
                    use_lpips=bool(opt.lpips_eval),
                    psnr_normer=psnr_normalizer
                )

            candidate = {
            'step': step,
            'max_psnr': max_psnr,
            'max_ssim': max_ssim,
            'min_lpips': min_lpips if opt.lpips_eval else None,
            'psnr': val_psnr,
            'ssim': val_ssim,
            'psnrs': psnrs,
            'ssims': ssims,
            'lpips_list': lpips_list if opt.lpips_eval else None,
            'lpips': val_lpips if (opt.lpips_eval and not opt.val_only_psnr) else None, 
            'losses': losses,
            'model': net.state_dict(),
            'score': score,
            'save_dir': opt.model_dir,
            'val_only_psnr': opt.val_only_psnr  
        }
            
            entered = update_frontier(candidate, bool(opt.lpips_eval), frontier_root=opt.frontier_root, model_name=opt.frontier_model_name)
            if entered:
                if opt.val_only_psnr:
                    print(f"Frontier updated with step {step}: PSNR={val_psnr:.4f}, SSIM=NA, LPIPS=NA, score={score:.6f}")
                else:
                    print(f"Frontier updated with step {step}: PSNR={val_psnr:.4f}, SSIM={val_ssim:.4f}, LPIPS={val_lpips if opt.lpips_eval else 'NA'}, score={score:.6f}")
            else:
                print(f"Step {step}: not entering frontier")
                
            writer.add_scalar('data/ssim',val_ssim,step)
            writer.add_scalar('data/psnr',val_psnr,step)
            if opt.lpips_eval:
                writer.add_scalar('data/lpips', val_lpips, step)
            scalar_dict = {'ssim': val_ssim, 'psnr': val_psnr}
            if opt.lpips_eval:
                scalar_dict['lpips'] = val_lpips
            writer.add_scalars('group', scalar_dict, step)

            ssims.append(val_ssim)
            psnrs.append(val_psnr)
            if opt.lpips_eval:
                lpips_list.append(val_lpips)
                if opt.val_only_psnr:
                    if val_psnr > max_psnr:
                        max_psnr = max(max_psnr, val_psnr)
                        max_ssim = 0
                        min_lpips = val_lpips
                        checkpoint = {
                                    'step':step,
                                    'max_psnr':max_psnr,
                                    'max_ssim':max_ssim,
                                    'min_lpips':min_lpips,
                                    'ssims':ssims,
                                    'psnrs':psnrs,
                                    'lpips_list':lpips_list,
                                    'losses':losses,
                                    'model':net.state_dict(),
                            }
                        if scheduler is not None:
                            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
                        torch.save(checkpoint, opt.model_dir)
                        print(f'\n model saved at step :{step}| max_psnr:{max_psnr:.4f} |min_lpips:{min_lpips:.4f}')
                        log_model_save_csv(step, max_psnr, max_ssim, best_lpips=min_lpips)
                else:
                    if val_ssim > max_ssim and val_psnr > max_psnr:
                        max_ssim=max(max_ssim,val_ssim)
                        max_psnr=max(max_psnr,val_psnr)
                        min_lpips =val_lpips
                        checkpoint = {
                                    'step':step,
                                    'max_psnr':max_psnr,
                                    'max_ssim':max_ssim,
                                    'min_lpips':min_lpips,
                                    'ssims':ssims,
                                    'psnrs':psnrs,
                                    'lpips_list':lpips_list,
                                    'losses':losses,
                                    'model':net.state_dict(),
                            }
                        if scheduler is not None:
                            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
                        torch.save(checkpoint, opt.model_dir)
                        print(f'\n model saved at step :{step}| max_psnr:{max_psnr:.4f}|max_ssim:{max_ssim:.4f}|min_lpips:{min_lpips:.4f}')
                        log_model_save_csv(step, max_psnr, max_ssim, best_lpips=min_lpips)
            else:
                if opt.val_only_psnr:
                    if val_psnr > max_psnr:
                        max_psnr = max(max_psnr, val_psnr)
                        max_ssim = 0
                        checkpoint = {
                                    'step':step,
                                    'max_psnr':max_psnr,
                                    'max_ssim':max_ssim,
                                    'ssims':ssims,
                                    'psnrs':psnrs,
                                    'losses':losses,
                                    'model':net.state_dict(),
                        }
                        if scheduler is not None:
                            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
                        torch.save(checkpoint, opt.model_dir)
                        #save max log
                        log_model_save_csv(step, max_psnr, max_ssim)
                        print(f'\n model saved at step :{step}| max_psnr:{max_psnr:.4f} ')
                else:
                    if val_ssim > max_ssim and val_psnr > max_psnr:
                        max_ssim=max(max_ssim,val_ssim)
                        max_psnr=max(max_psnr,val_psnr)
                        checkpoint = {
                                    'step':step,
                                    'max_psnr':max_psnr,
                                    'max_ssim':max_ssim,
                                    'ssims':ssims,
                                    'psnrs':psnrs,
                                    'losses':losses,
                                    'model':net.state_dict(),
                        }
                        if scheduler is not None:
                            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
                        torch.save(checkpoint, opt.model_dir)
                        log_model_save_csv(step, max_psnr, max_ssim)
                        print(f'\n model saved at step :{step}| max_psnr:{max_psnr:.4f}|max_ssim:{max_ssim:.4f}')

            if opt.lpips_eval:
                if opt.val_only_psnr:
                    print(f'\nstep :{step} |max_psnr :{max_psnr:.4f} |min_lpips:{min_lpips:.4f}')
                else:
                    print(f'\nstep :{step} |max_psnr :{max_psnr:.4f}|max_ssim:{max_ssim:.4f}|min_lpips:{min_lpips:.4f}')
            else:
                if opt.val_only_psnr:
                    print(f'\nstep :{step} |max_psnr :{max_psnr:.4f} ')
                else:
                    print(f'\nstep :{step} |max_psnr :{max_psnr:.4f}|max_ssim:{max_ssim:.4f}')

    np.save(f'./numpy_files/{model_name}_{opt.steps}_losses.npy',losses)
    if not opt.val_only_psnr:
        np.save(f'./numpy_files/{model_name}_{opt.steps}_ssims.npy',ssims)
    np.save(f'./numpy_files/{model_name}_{opt.steps}_psnrs.npy',psnrs)
    if opt.lpips_eval:
        np.save(f'./numpy_files/{model_name}_{opt.steps}_lpips.npy', lpips_list)
    writer.close()

def test(net, loader_test, max_psnr, max_ssim, step):
    net.eval()
    ssims = [] if not opt.val_only_psnr else None
    psnrs = []
    lpips_values = [] if (opt.lpips_eval and not opt.val_only_psnr) else None 
    
    def process_single_image(inputs, net, opt):
        b, c, h, w = inputs.shape
        
        if opt.tile is None or (h <= opt.tile and w <= opt.tile):
            with torch.no_grad():
                pad_h = (8 - h % 8) % 8
                pad_w = (8 - w % 8) % 8
                if pad_h > 0 or pad_w > 0:
                    inputs = torch.nn.functional.pad(inputs, (0, pad_w, 0, pad_h), 'reflect')
                    
                pred = net(inputs)

                # Unpad
                if pad_h > 0 or pad_w > 0:
                    pred = pred[..., :h, :w]
                    
                return pred

        else:
            tile = opt.tile
            tile_overlap = opt.tile_overlap if hasattr(opt, 'tile_overlap') else 32
            stride = tile - tile_overlap

            h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
            w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
            
            output = torch.zeros((b, c, h, w), device=inputs.device)
            count_map = torch.zeros((b, c, h, w), device=inputs.device)
            
            with torch.no_grad():
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
    
    for i, batch in enumerate(loader_test):
        inputs = batch[0].to(opt.device)
        targets = batch[1].to(opt.device)
        
        pred = process_single_image(inputs, net, opt)
        
        psnr1 = psnr(pred, targets)
        psnrs.append(psnr1)
        if not opt.val_only_psnr:
            ssim1 = ssim(pred, targets).item()
            ssims.append(ssim1)
            if opt.lpips_eval:
                lpips_val = calculate_lpips(pred, targets, net_type=opt.lpips_net, device=opt.device)
                lpips_values.append(lpips_val)
    
    avg_psnr = np.mean(psnrs)
    if opt.val_only_psnr:
        if opt.lpips_eval:
            return 0, avg_psnr, 0  
        else:
            return 0, avg_psnr  
    elif opt.lpips_eval:
        return np.mean(ssims), avg_psnr, np.mean(lpips_values)
    else:
        return np.mean(ssims), avg_psnr


def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    loader_train = loaders_[opt.trainset]
    loader_test = loaders_[opt.testset]
    net = models_[opt.net]
    net = net.to(opt.device)
    
    total_params = count_parameters(net)
    print(f"\nTotal model parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    if opt.amp:
        scaler = torch.cuda.amp.GradScaler()
        print('Mixed precision training enabled')
    else:
        scaler = None
        print('Mixed precision training disabled')
    
    if opt.device == 'cuda':
        if torch.cuda.device_count() > 1:
            print(f"Detected {torch.cuda.device_count()} GPUs, enabling DataParallel acceleration...")
            net = torch.nn.DataParallel(net)
        else:
            print("Detected single GPU, skipping DataParallel wrapper to avoid NCCL/CUDA errors...")
        
    criterion = []

    loss_config = {
        'use_l1': True,
        'use_fft': opt.loss_type == 'l1_fft',
        'fft_weight': opt.fft_loss_weight if hasattr(opt, 'fft_loss_weight') else 0.1,
    }

    criterion.append(nn.L1Loss(reduction='none').to(opt.device))
    criterion.append(FFTLoss(loss_weight=1.0, reduction='mean').to(opt.device))
    criterion.append(loss_config)

    model_params = filter(lambda x: x.requires_grad, net.parameters())
    all_params = list(model_params)
    
    if opt.optimizer == 'adam':
        optimizer = optim.Adam(params=all_params, lr=opt.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=opt.weight_decay)
    elif opt.optimizer == 'adamw':
        optimizer = optim.AdamW(params=all_params, lr=opt.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=opt.weight_decay)
    
    scheduler = None
    if not opt.no_lr_sche:
        if opt.scheduler_type == 'cosrestart':
            periods = list(map(int, opt.restart_periods.split(',')))
            restart_weights = list(map(float, opt.restart_weights.split(',')))
            scheduler = CosineAnnealingRestartLR(
                optimizer, 
                periods=periods, 
                restart_weights=restart_weights, 
                eta_min=opt.eta_min
            )
            print(f'Using CosineAnnealingRestartLR scheduler, periods={periods}, restart_weights={restart_weights}, eta_min={opt.eta_min}')
        else:
            print('Using cosine annealing learning rate scheduler')
    
    optimizer.zero_grad()
    train(net, loader_train, loader_test, optimizer, criterion, scaler, scheduler)
    
    final_frontier_deploy(use_lpips=bool(opt.lpips_eval), top_k=5, frontier_root=opt.frontier_root, model_name=opt.frontier_model_name)

    close_console_logger()