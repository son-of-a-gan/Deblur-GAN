import time
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import ntpath
import torch
import torch.nn as nn
import argparse

from data.data_loader import CreateDataLoader
from data.double_dataset import DoubleDatasetLoader
from models.models import create_model
from util.util import save_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scoring script for finding deblur score.")
    parser.add_argument("--in-gt", type=str, default="folder_gt",
                        help="Path to folder containing sharp ground truth images.")
    parser.add_argument("--in-fake", type=str, default="folder_fake",
                        help="Path to folder containing deblurred images.")
    parser.add_argument("--size-reference", type=str,
                        default="folder_size_reference")
    parser.add_argument("--dataset-mode", type=str, default="double",
                        help="Type of dataset used.")

    # Load configuration
    args = parser.parse_args()

    # GT data loading
    data_loader = DoubleDatasetLoader()
    data_loader.initialize(args)
    dataset = data_loader.load_data()

    # loss function
    loss = nn.MSELoss()

    # things that accumulate
    iters = 0
    dataset_loss = 0

    with torch.no_grad():
        for i, data in enumerate(tqdm(dataset)):
            gt = data['gt']
            fake = data['fake']
            zero_imaginary_part = torch.zeros(gt.shape)

            # gt image fft
            gt_signal = torch.cat((gt, zero_imaginary_part), dim=1).squeeze()
            gt_signal = gt_signal.permute(1, 2, 0)
            gt_fft = torch.fft(gt_signal, 2)

            # fake image fft
            fake_signal = torch.cat(
                (fake, zero_imaginary_part), dim=1).squeeze()
            fake_signal = fake_signal.permute(1, 2, 0)
            fake_fft = torch.fft(fake_signal, 2)

            # find the L2 diff in FFT
            output = loss(fake_fft, gt_fft)

            # keep count
            dataset_loss += output
            iters += 1

    # find the dataset loss
    print("<TEST> Dataset: {}, Average Dataset Loss: {}".format(
        args.in_fake, dataset_loss / iters))
