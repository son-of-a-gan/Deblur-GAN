import time
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import ntpath
import torch

from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.util import save_image


if __name__ == "__main__":
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip

    # experiment results setup
    experiment_name = opt.name
    experiment_path = os.path.join(opt.results_dir, experiment_name)
    output_path = os.path.join(experiment_path, "deblurred")
    if not os.path.exists(opt.results_dir):
        os.mkdir(opt.results_dir)
    if not os.path.exists(experiment_path):
        os.mkdir(experiment_path)
        os.mkdir(output_path)
    else:
        if not os.path.exists(output_path):
            os.mkdir(output_path)

    # data and model loading
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)

    with torch.no_grad():
        for i, data in enumerate(tqdm(dataset)):
            if i >= opt.how_many:
                break

            # inference and collect visualization
            model.set_input(data)
            model.test()
            visuals = model.get_current_visuals()

            # getting the image name, works for batchsize 1
            image_filename = ntpath.basename(data['A_paths'][0])

            # getting the data for saving
            image_numpy = visuals['fake_B']

            # save images
            image_path = os.path.join(output_path, image_filename)
            save_image(image_numpy, image_path)
