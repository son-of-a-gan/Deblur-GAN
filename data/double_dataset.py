import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import torch
from data.base_data_loader import BaseDataLoader


class DoubleDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.gt_dir = os.path.join(opt.in_gt)
        self.gt_paths = make_dataset(self.gt_dir)
        self.gt_paths = sorted(self.gt_paths)

        self.fake_dir = os.path.join(opt.in_fake)
        self.fake_paths = make_dataset(self.fake_dir)
        self.fake_paths = sorted(self.fake_paths)

        self.size_ref_dir = os.path.join(opt.size_reference)
        self.size_ref_paths = make_dataset(self.size_ref_dir)
        self.size_ref_paths = sorted(self.size_ref_paths)

        transform_list = [transforms.ToTensor()]
        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        gt_path = self.gt_paths[index]
        gt_img = Image.open(gt_path).convert('L')

        fake_path = self.fake_paths[index]
        fake_img = Image.open(fake_path).convert('L')

        size_ref_path = self.size_ref_paths[index]
        size_ref_img = Image.open(size_ref_path).convert('L')

        gt_img = gt_img.resize(size_ref_img.size,
                               resample=Image.BICUBIC)
        fake_img = fake_img.resize(size_ref_img.size,
                                   resample=Image.BICUBIC)

        gt_img = self.transform(gt_img)
        fake_img = self.transform(fake_img)
        return {'gt': gt_img, 'fake': fake_img}

    def __len__(self):
        return len(self.gt_paths)

    def name(self):
        return 'DoubleDataset'


class DoubleDatasetPaths(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.gt_dir = os.path.join(opt.in_gt)
        self.gt_paths = make_dataset(self.gt_dir)
        self.gt_paths = sorted(self.gt_paths)

        self.fake_dir = os.path.join(opt.in_fake)
        self.fake_paths = make_dataset(self.fake_dir)
        self.fake_paths = sorted(self.fake_paths)

        self.size_ref_dir = os.path.join(opt.size_reference)
        self.size_ref_paths = make_dataset(self.size_ref_dir)
        self.size_ref_paths = sorted(self.size_ref_paths)

    def __getitem__(self, index):
        gt_path = self.gt_paths[index]
        fake_path = self.fake_paths[index]
        size_ref_path = self.size_ref_paths[index]

        return {'gt_path': gt_path, 'fake_path': fake_path, 'size_path': size_ref_path}

    def __len__(self):
        return len(self.gt_paths)

    def name(self):
        return 'DoubleDatasetPath'


class DoubleDatasetLoader(BaseDataLoader):
    def name(self):
        return 'DoubleDatasetLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = DoubleDataset()
        self.dataset.initialize(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False)

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return len(self.dataset)


class DoubleDatasetLoaderPath(BaseDataLoader):
    def name(self):
        return 'DoubleDatasetLoaderPath'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = DoubleDatasetPaths()
        self.dataset.initialize(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False)

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return len(self.dataset)
