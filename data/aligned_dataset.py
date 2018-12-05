import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import numpy as np

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)

        self.AB_paths = sorted(make_dataset(self.dir_AB))

        #assert(opt.resize_or_crop == 'resize_and_crop')

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        # AB = AB.resize(
        #     (self.opt.loadSizeX * 2, self.opt.loadSizeY), Image.BICUBIC)
        # AARON: change to keep original aspect ratio, use square crop afterwards
        width, height = AB.size
        aspect_ratio = width / height
        AB = AB.resize(
            (int(aspect_ratio * self.opt.loadSizeY), self.opt.loadSizeY),
            Image.BICUBIC)

        w_total = AB.width
        w = int(w_total / 2)
        h = AB.height

        # DAVID[1]: Let's randomly sample h/w dimension here
        patch_size = int(np.random.normal(self.opt.fineSize, self.opt.fineSizeSigma))
        patch_size = min(min(w,h)-1, patch_size)                # upper bound
        patch_size = max(int(self.opt.fineSize/2), patch_size)  # lower bound

        # DAVID[1]: Now, we can get the location and crop the image to to it
        w_offset = random.randint(0, max(0, w - patch_size - 1))
        h_offset = random.randint(0, max(0, h - patch_size - 1))
        
        # DAVID[1]: Finally, crop and resize to original input dimension
        A = AB.crop((w_offset, h_offset, 
                     w_offset+patch_size, h_offset+patch_size))
        A = A.resize((self.opt.fineSize, self.opt.fineSize), Image.BICUBIC)

        B = AB.crop((w+w_offset, h_offset, 
                     w+w_offset+patch_size, h_offset+patch_size))
        B = B.resize((self.opt.fineSize, self.opt.fineSize), Image.BICUBIC)
        
        # PIL -> Tensor
        A = self.transform(A)
        B = self.transform(B)

        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)

        return {'A': A, 'B': B,
                'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'AlignedDataset'
