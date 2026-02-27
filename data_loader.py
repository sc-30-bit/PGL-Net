import torch.utils.data as data
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
import os,sys
sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch
import random
from PIL import Image
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
from option import opt
# Set random seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

local_rank = int(os.environ.get("LOCAL_RANK", 0))

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(seed) # Use the previously defined seed (42)

try:
    import torch.backends.cudnn as cudnn
    cudnn.deterministic = True
    cudnn.benchmark = False
except:
    pass

numworkers=opt.num_workers
BS=opt.bs
if local_rank == 0:
    print("num_workers(total):",numworkers)
    print("batchsize(total):",BS)
crop_size='whole_img'
if opt.crop:
    crop_size=opt.crop_size
    if local_rank == 0:
        print('crop size',crop_size)

def tensorShow(tensors,titles=None):
        '''
        t:BCWH
        '''
        fig=plt.figure()
        for tensor,tit,i in zip(tensors,titles,range(len(tensors))):
            img = make_grid(tensor)
            npimg = img.numpy()
            ax = fig.add_subplot(321+i)  # Changed from 211+i to 321+i
            ax.imshow(np.transpose(npimg, (1, 2, 0)))
            ax.set_title(tit)
        plt.show()

class PairedLoader(data.Dataset):
    def __init__(self, path, localrank, train=True, size='whole_img'):
        super(PairedLoader, self).__init__()
        self.size = size
        self.train = train

        # Select different subdirectories based on train/test mode
        if train:
            self.input_dir = os.path.join(path, 'train', 'input')
            self.gt_dir = os.path.join(path, 'train', 'gt')
        else:
            self.input_dir = os.path.join(path, 'test', 'input')
            self.gt_dir = os.path.join(path, 'test', 'gt')

        # Load all input image paths
        self.input_imgs = [os.path.join(self.input_dir, img) for img in os.listdir(self.input_dir)]
        if localrank == 0:
            print(f"Found {len(self.input_imgs)} input images in {self.input_dir}")

    def __getitem__(self, index):
        # Load hazy image
        haze = Image.open(self.input_imgs[index]).convert('RGB')

        # Get corresponding gt filename from input filename
        img_name = os.path.basename(self.input_imgs[index])
        # Replace input with gt to get clear image filename, handle different extensions
        base_name, ext = os.path.splitext(img_name)
        if '_input' in base_name:
            gt_name = base_name.replace('_input', '_gt') + ext
        else:
            # Handle case without _input suffix
            gt_name = img_name

        # Load corresponding clear image
        gt = Image.open(os.path.join(self.gt_dir, gt_name)).convert('RGB')

        # Ensure both images have the same size
        if haze.size != gt.size:
            gt = gt.resize(haze.size)

        # Crop operation
        if not isinstance(self.size, str):
            i, j, h, w = tfs.RandomCrop.get_params(haze, output_size=(self.size, self.size))
            haze = FF.crop(haze, i, j, h, w)
            gt = FF.crop(gt, i, j, h, w)

        # Data augmentation
        if self.train:
            haze, gt = self.augData(haze, gt)
        else:
            haze, gt = self.augTest(haze, gt)

        return haze, gt

    def augData(self, data, target):
        # Data augmentation for training
        if self.train:
            if random.random() > 0.5:
                data = FF.hflip(data)
                target = FF.hflip(target)
                
            rand_rot = random.randint(0, 3)
            if rand_rot > 0:
                data = FF.rotate(data, 90 * rand_rot)
                target = FF.rotate(target, 90 * rand_rot)

        # Convert to tensor without normalization
        data = tfs.ToTensor()(data)
        target = tfs.ToTensor()(target)
        return data, target

    def augTest(self, data, target):
        # No data augmentation for testing, only convert to tensor without normalization
        data = tfs.ToTensor()(data)
        target = tfs.ToTensor()(target)
        return data, target

    def __len__(self):
        return len(self.input_imgs)

import os
pwd=os.getcwd()
if local_rank == 0:
    print(pwd)
    
def get_dataloader(dataset_name, is_train, opt, localrank):
    BS = opt.bs
    numworkers = opt.num_workers
    crop_size = opt.crop_size if opt.crop else 'whole_img'

    train_kwargs = {
        'batch_size': BS, 'shuffle': True, 'num_workers': numworkers,
        'pin_memory': True, 'prefetch_factor': 2, 'persistent_workers': True,
        'worker_init_fn': seed_worker, 'generator': g
    }
    test_kwargs = {
        'batch_size': 1, 'shuffle': False, 'num_workers': 4, 'pin_memory': True
    }
    
    kwargs = train_kwargs if is_train else test_kwargs
    size = crop_size if is_train else 'whole_img'

    if 'rw2ah' in dataset_name:
        path = "/workspace/Datasets/RW2AH"
        dataset = PairedLoader(path, localrank, train=is_train, size=size)
    elif 'rudb' in dataset_name:
        path = "/workspace/Datasets/MergedDataset"
        dataset = PairedLoader(path, localrank, train=is_train, size=size)
    elif 'rrshid' in dataset_name:
        path = "/workspace/Datasets/RRSHID-noVal"
        dataset = PairedLoader(path, localrank, train=is_train, size=size)
        
    elif 'its' in dataset_name:
        path = "/workspace/RESIDE-IN"
        dataset = PairedLoader(path, localrank, train=is_train, size=size)
    elif 'ots' in dataset_name:
        path = "/workspace/RESIDE-OUT"
        dataset = PairedLoader(path, localrank, train=is_train, size=size)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return DataLoader(dataset=dataset, **kwargs)