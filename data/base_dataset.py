import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms


class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass


def get_transform(opt):
    transform_list = []
    if opt.resize_or_crop == 'resize_and_crop':
        osize = [opt.loadSizeX, opt.loadSizeY]
        transform_list.append(transforms.Scale(osize, Image.BICUBIC))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'crop':
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'scale_width':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.fineSize)))
    elif opt.resize_or_crop == 'scale_width_and_crop':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.loadSizeX)))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'scale_long_edge_x':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_long_edge(img, opt.loadSizeX)))
    elif opt.resize_or_crop == 'scale_short_edge_x':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_short_edge(img, opt.loadSizeX)))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.RandomHorizontalFlip())

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def __scale_width(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), Image.BICUBIC)


def __scale_long_edge(img, target_length):
    ow, oh = img.size
    if ow >= oh:
        if ow == target_length or ow <= target_length:
            return img
        else:
            ratio = oh / ow
            return img.resize((target_length, int(ratio * target_length)),
                              Image.BICUBIC)

    else:
        if oh == target_length or oh <= target_length:
            return img
        else:
            ratio = ow / oh
            return img.resize((int(ratio * target_length), target_length),
                              Image.BICUBIC)


def __scale_short_edge(img, target_length):
    # also handle multiples of 8
    ow, oh = img.size
    if oh <= ow:
        if oh <= target_length:
            new_w, new_h = get_8multiples(ow, oh)
            return img.resize((new_w, new_h), Image.BICUBIC)
        else:
            ratio = ow / oh
            new_w, new_h = get_8multiples(
                int(ratio * target_length), target_length)
            return img.resize((new_w, new_h), Image.BICUBIC)
    else:
        if ow <= target_length:
            new_w, new_h = get_8multiples(ow, oh)
            return img.resize((new_w, new_h), Image.BICUBIC)
        else:
            ratio = oh / ow
            new_w, new_h = get_8multiples(
                target_length, int(ratio * target_length))
            return img.resize((new_w, new_h), Image.BICUBIC)


def get_8multiples(ow, oh):
    new_w = int((ow / 8) + 1) * 8
    new_h = int((oh / 8) + 1) * 8
    return new_w, new_h
