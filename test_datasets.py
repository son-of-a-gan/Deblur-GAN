import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms


if __name__ == '__main__':
    width_list = np.random.randint(256, 1024, 1000)
    height_list = np.random.randint(256, 1024, 1000)
    small_size = 360
    crop_size = 256

    transform_list = [transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5),
                                           (0.5, 0.5, 0.5))]
    transform = transforms.Compose(transform_list)

    for i in range(1000):
        curr_fake_img = Image.new('RGB', (width_list[i], height_list[i]))

        # data handling algorithm here
        width, height = curr_fake_img.size
        if width / 2 > height:
            aspect_ratio = width / height
            curr_fake_img = curr_fake_img.resize(
                (int(aspect_ratio * small_size), small_size),
                Image.BICUBIC)
        else:
            aspect_ratio = height / width
            curr_fake_img = curr_fake_img.resize(
                (small_size * 2, int(aspect_ratio * small_size * 2)),
                Image.BICUBIC)

        curr_fake_img = transform(curr_fake_img)

        w_total = curr_fake_img.size(2)
        w = int(w_total / 2)
        h = curr_fake_img.size(1)
        w_offset = np.random.randint(0, max(0, w - crop_size - 1))
        h_offset = np.random.randint(0, max(0, h - crop_size - 1))

        A = curr_fake_img[:, h_offset:h_offset + crop_size,
                          w_offset:w_offset + crop_size]
        B = curr_fake_img[:, h_offset:h_offset + crop_size,
                          w + w_offset:w + w_offset + crop_size]

        # checking the final data input is good
        if A.shape[0] != 3 and A.shape[1] != crop_size and A.shape[2] != crop_size \
                and B.shape[0] != 3 and B.shape[1] != crop_size and B.shape[2] != crop_size:
            print("A: {},   B: {}".format(A.shape, B.shape))

    print("Test: <DATA> Data handling algorithm test is done.")
