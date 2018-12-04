'''
Here's the scene.  We have a dataset of combined blurry and sharp images.  
This is after running combine_A_and_B.py

We now want to split that dataset into train and test by some given percentage.

Let's load the data, randomly pick train and test proportionally, 
then move them to a train and test folder within the current folder.

Now let's git 'er done.
'''

import os
import argparse
import random
import shutil


parser = argparse.ArgumentParser('split combined dataset')
parser.add_argument('--fold_AB', dest='fold_AB', help='What is the folder for your combined images', type=str, default='~/')
parser.add_argument('--pTrain', dest='pTrain', help='What proportion of your images do you want to be training images?', type=float, default = 0.8)
args = parser.parse_args()

fold_in = args.fold_AB
imgs = os.listdir(fold_in)
numImgs = len(imgs)

#Randomly shuffle the images before splitting
imgs_shuffled = random.shuffle(imgs)

#Find the split index of train vs test data
split = int(args.pTrain*numImgs)

#Divide the images into train and test
trainImgs = imgs[:split]
testImgs = imgs[split:]

#Move train images to train image folder
trainDest = fold_in + 'train'
os.mkdir(trainDest)

for i in trainImgs:
  shutil.move(fold_in+i, trainDest)

#Move test images to test image folder
testDest = fold_in + 'test'
os.mkdir(testDest)

for i in testImgs:
  shutil.move(fold_in+i, testDest)
