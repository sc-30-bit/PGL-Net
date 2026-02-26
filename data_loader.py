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
print(numworkers)
print(BS)
crop_size='whole_img'
if opt.crop:
    crop_size=opt.crop_size

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

class RESIDE_Dataset(data.Dataset):
    def __init__(self,path,train,size=crop_size,format='.png'):
        super(RESIDE_Dataset,self).__init__()
        self.size=size
        print('crop size',size)
        self.train=train
        self.format=format
        self.haze_imgs_dir=os.listdir(os.path.join(path,'hazy'))
        self.haze_imgs=[os.path.join(path,'hazy',img) for img in self.haze_imgs_dir]
        self.clear_dir=os.path.join(path,'clear')
        print(f"Found {len(self.haze_imgs)} hazy images in {path}/input")
    def __getitem__(self, index):
        haze=Image.open(self.haze_imgs[index])
        if isinstance(self.size,int):
            while haze.size[0]<self.size or haze.size[1]<self.size :
                index = random.randint(0, len(self.haze_imgs) - 1)
                haze=Image.open(self.haze_imgs[index])
        img=self.haze_imgs[index]
        id=img.split('/')[-1].split('_')[0]
        clear_name=id+self.format
        clear=Image.open(os.path.join(self.clear_dir,clear_name))
        clear=tfs.CenterCrop(haze.size[::-1])(clear)
        if not isinstance(self.size,str):
            i,j,h,w=tfs.RandomCrop.get_params(haze,output_size=(self.size,self.size))
            haze=FF.crop(haze,i,j,h,w)
            clear=FF.crop(clear,i,j,h,w)
        haze,clear=self.augData(haze.convert("RGB") ,clear.convert("RGB") )
        return haze,clear
    def augData(self,data,target):
        if self.train:
            rand_hor=random.randint(0,1)
            rand_rot=random.randint(0,3)
            data=tfs.RandomHorizontalFlip(rand_hor)(data)
            target=tfs.RandomHorizontalFlip(rand_hor)(target)
            if rand_rot:
                data=FF.rotate(data,90*rand_rot)
                target=FF.rotate(target,90*rand_rot)
        data=tfs.ToTensor()(data)
        #data=tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])(data)
        target=tfs.ToTensor()(target)
        return  data ,target
    def __len__(self):
        return len(self.haze_imgs)

class RealWorld_Dataset(data.Dataset):
    def __init__(self, path, train=True, size='whole_img', format='.png'):
        super(RealWorld_Dataset, self).__init__()
        self.size = size
        self.train = train
        self.format = format

        # Select different subdirectories based on train/test mode
        if train:
            self.input_dir = os.path.join(path, 'train', 'input')
            self.gt_dir = os.path.join(path, 'train', 'gt')
        else:
            self.input_dir = os.path.join(path, 'test', 'input')
            self.gt_dir = os.path.join(path, 'test', 'gt')

        # Load all input image paths
        self.input_imgs = [os.path.join(self.input_dir, img) for img in os.listdir(self.input_dir)]

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
            rand_hor = random.randint(0, 1)
            rand_rot = random.randint(0, 3)
            # Horizontal flip
            data = tfs.RandomHorizontalFlip(rand_hor)(data)
            target = tfs.RandomHorizontalFlip(rand_hor)(target)
            # Rotation
            if rand_rot:
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
print(pwd)

rw2ah_path="/home/klay/papersToReproduce/subset_RW2AH" #path to your 'data' folder

RW2AH_train_loader=DataLoader(dataset=RealWorld_Dataset(rw2ah_path,train=True,size=crop_size),batch_size=BS,shuffle=True,num_workers=numworkers,        # <--- Key! Enable 8 processes for parallel image loading
    pin_memory=True,      # <--- Key! Pinned memory for faster transfer
    prefetch_factor=2,    # <--- Optional, each worker prefetches 2 batches
    persistent_workers=True, # <--- Optional, avoid destroying and recreating processes after each epoch
    worker_init_fn=seed_worker,
    generator=g,
    )
RW2AH_test_loader=DataLoader(dataset=RealWorld_Dataset(rw2ah_path,train=False,size='whole_img'),batch_size=1,shuffle=False,num_workers=4,        # Test set can also use a few workers
    pin_memory=True)

# Add support path for MergedDataset
rudb_path="/home/klay/papersToReproduce/MergedDataset_cropped" #path to your 'MergedDataset' folder
RUDB_train_loader=DataLoader(dataset=RealWorld_Dataset(rudb_path,train=True,size=crop_size),batch_size=BS,shuffle=True,num_workers=numworkers,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True,
    worker_init_fn=seed_worker,
    generator=g,
    )
RUDB_test_loader=DataLoader(dataset=RealWorld_Dataset(rudb_path,train=False,size='whole_img'),batch_size=1,shuffle=False,num_workers=4,
    pin_memory=True)

rrshid_path="/home/klay/papersToReproduce/RRSHID-noVal" #path to your 'RRSHID' folder
RRSHID_train_loader=DataLoader(dataset=RealWorld_Dataset(rrshid_path,train=True,size=crop_size),batch_size=BS,shuffle=True,num_workers=numworkers,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True,
    worker_init_fn=seed_worker,
    generator=g,
    )
RRSHID_test_loader=DataLoader(dataset=RealWorld_Dataset(rrshid_path,train=False,size='whole_img'),batch_size=1,shuffle=False,num_workers=4,
    pin_memory=True)

# synthetic data
path='/home/zhilin007/VS/FFA-Net/data'#path to your 'data' folder

ITS_train_loader=DataLoader(dataset=RESIDE_Dataset(path+'/RESIDE/ITS',train=True,size=crop_size),batch_size=BS,shuffle=True,num_workers=numworkers,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True,
    worker_init_fn=seed_worker,
    generator=g,
    )
ITS_test_loader=DataLoader(dataset=RESIDE_Dataset(path+'/RESIDE/SOTS/indoor',train=False,size='whole img'),batch_size=1,shuffle=False,num_workers=4,
    pin_memory=True)

OTS_train_loader=DataLoader(dataset=RESIDE_Dataset(path+'/RESIDE/OTS',train=True,format='.jpg'),batch_size=BS,shuffle=True,num_workers=numworkers,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True,
    worker_init_fn=seed_worker,
    generator=g,
    )
OTS_test_loader=DataLoader(dataset=RESIDE_Dataset(path+'/RESIDE/SOTS/outdoor',train=False,size='whole img',format='.png'),batch_size=1,shuffle=False,num_workers=4,
    pin_memory=True)
