#!/bin/sh
python test.py --dataroot datasets/mapillary_test --model test --dataset_mode single --learn_residual --resize_or_crop nothing --checkpoints_dir pretrained_checkpoints --results_dir pretrained_results
