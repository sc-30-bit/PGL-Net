import torch,os,sys,torchvision,argparse
import torchvision.transforms as tfs
import time,math
import numpy as np
from torch.backends import cudnn
from torch import optim
import torch,warnings
from torch import nn
import torchvision.utils as vutils
import json
warnings.filterwarnings('ignore')

def str2bool(v):
	if isinstance(v, bool):
		return v
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

parser=argparse.ArgumentParser()
parser.add_argument('--config', type=str, default=None, help='Path to JSON config file')
parser.add_argument('--steps',type=int,default=100000)
parser.add_argument('--device',type=str,default='Automatic detection')
parser.add_argument('--resume',type=str2bool,default=True)
parser.add_argument('--eval_step',type=int,default=5000)
parser.add_argument('--lr', default=0.0004, type=float, help='learning rate')
parser.add_argument('--model_dir',type=str,default='./trained_models/')
parser.add_argument('--trainset',type=str,default='rw2ah_train',choices=['rw2ah_train','merged_train','rrshid_train'])
parser.add_argument('--testset',type=str,default='rw2ah_test',choices=['rw2ah_test','merged_test','rrshid_test'])
parser.add_argument('--net', type=str, default='pglnet_t', choices=['pglnet_t', 'pglnet_s', 'pglnet_b', 'pglnet_d'], help='net type (pglnet_t, pglnet_s, pglnet_b, pglnet_d)')
parser.add_argument('--bs',type=int,default=16,help='batch size')
parser.add_argument('--crop',action='store_true')
parser.add_argument('--crop_size',type=int,default=256,help='Takes effect when using --crop ')
parser.add_argument('--no_lr_sche',action='store_true',help='no lr cos schedule')
parser.add_argument('--loss_type', type=str, default='l1', choices=['l1', 'l1_fft'], help='loss type: l1 (L1 only) or l1_fft (L1 + simple FFT)')
parser.add_argument('--fft_loss_weight', type=float, default=0.1, help='FFT loss weight when using l1_fft')
parser.add_argument('--amp',type=str2bool,default=False, help='enable automatic mixed precision training')
parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw'], help='optimizer type: adam or adamw')
parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay for optimizer')
parser.add_argument('--exp_num', type=int, default=1, help='experiment number to distinguish different experiments with same model configuration')
parser.add_argument('--scheduler_type', type=str, default='cosrestart', choices=['cosdecay', 'cosrestart'], help='learning rate scheduler type')
parser.add_argument('--restart_periods', type=str, default='120000', help='periods for CosineAnnealingRestartLR, separated by commas')
parser.add_argument('--restart_weights', type=str, default='1', help='restart weights for CosineAnnealingRestartLR, separated by commas')
parser.add_argument('--eta_min', type=float, default=4e-8, help='minimum learning rate for scheduler')
parser.add_argument('--tile', type=int, default=None, help='Tile size for large images, None for no tile (test as a whole)')
parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
parser.add_argument('--window_size', type=int, default=32, help='Window size for model, used for tile processing')
parser.add_argument('--val_only_psnr', type=str2bool, default=True, help='Only calculate PSNR during validation to save training time')
parser.add_argument('--lpips_eval', type=str2bool, default=False, help='evaluate using LPIPS')
parser.add_argument('--lpips_net', type=str, default='vgg', choices=['alex', 'vgg', 'squeeze'], help='LPIPS network type')

opt=parser.parse_args()

# Load config from JSON file if provided
if opt.config is not None and os.path.exists(opt.config):
    print(f"Loading configuration from: {opt.config}")
    with open(opt.config, 'r') as f:
        config_dict = json.load(f)
    
    # Override command-line arguments with JSON config
    for key, value in config_dict.items():
        if hasattr(opt, key):
            setattr(opt, key, value)
elif opt.config is not None:
    print(f"Warning: Config file not found: {opt.config}")

opt.device='cuda' if torch.cuda.is_available() else 'cpu'
model_name=opt.trainset+'_'+opt.net.split('.')[0]+'_'+'_exp'+str(opt.exp_num)
opt.model_dir=opt.model_dir+model_name+'.pk'
log_dir='logs/'+model_name

opt.frontier_root = './frontier_saves/'
opt.frontier_model_name = model_name  
frontier_model_base = os.path.join(opt.frontier_root, opt.frontier_model_name)
opt.frontier_weights_dir = os.path.join(frontier_model_base, 'weights')    
opt.frontier_archives_dir = os.path.join(frontier_model_base, 'archives')  
os.makedirs(opt.frontier_weights_dir, exist_ok=True)
os.makedirs(opt.frontier_archives_dir, exist_ok=True)

print(opt)
print('model_dir:',opt.model_dir)

if not os.path.exists(opt.frontier_root):
    os.makedirs(opt.frontier_root)
if not os.path.exists('trained_models'):
	os.mkdir('trained_models')
if not os.path.exists('numpy_files'):
	os.mkdir('numpy_files')
if not os.path.exists('logs'):
	os.mkdir('logs')
if not os.path.exists('samples'):
	os.mkdir('samples')
if not os.path.exists(f"samples/{model_name}"):
	os.mkdir(f'samples/{model_name}')
if not os.path.exists(log_dir):
	os.mkdir(log_dir)
