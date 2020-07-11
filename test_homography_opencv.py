#!/usr/bin/env python
# encoding: utf-8
'''
@author: Pu Li
@contact: pli5270@sdsu.edu
@file: test_homography_opencv.py
@time: 10/21/19 9:11 PM
@desc: Homography estimation using ORB/SIFT+RANSAC (feature point based algorithm), ECC (direct pixel based algorithm).
'''

from src import util as util
import os
import numpy as np
import src.datasets.image_helpers as img_helper
import scipy.io as sio
import time
import src.logger as loggerFactory
from src.opencv_baseline.homography import identity_func, sift_ransac_func, orb_ransac_func, ecc_fuc
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, required=True, help='the path to processed testing data')
parser.add_argument('--ann_path', type=str, required=True, help='the path to annotation data')
parser.add_argument('--patch_select', type=str, default='center', help='one of {random, center, bottom_left, bottom_right, top_left, top_right}')
parser.add_argument('--prefix', type=str, default='clip', help='prefix of the testing data file names')
parser.add_argument('--method', type=str, required=True, help='one of {Identity, SIFT+RANSAC, ORB2+RANSAC, ORB3+RANSAC, ECC}')
parser.add_argument('--scale', type=float, required=True, help='image scale')
parser.add_argument('--output_dir', type=str, default='test_results', help='base output directory')

config = parser.parse_args()
data_base_path = config.data_path
output_dir = config.output_dir

subdirs = util.globx(data_base_path, [config.prefix + '*'])
subdirs, idx = util.sort_file(subdirs, config.prefix +'(\d+)')
sub_dirnames = [os.path.basename(subdir) for subdir in subdirs]
exp_path = os.getcwd()

exp_func_dict = {
    'Identity': identity_func(),
    'ORB2+RANSAC': orb_ransac_func(2),
}

flag = False
exp_name = config.method
method = exp_func_dict[config.method]

# Calculate the homography estimation
log_dir = os.path.join(output_dir, exp_name)
h_mat_dir = os.path.join(log_dir, 'h_mats')
util.checkDirs([log_dir, h_mat_dir])
timestamp = time.strftime("%Y%m%d_%H-%M-%S", time.localtime())
logFile = os.path.join(log_dir, 'test_' + exp_name + '_' + timestamp + '.log')
txt_logger = loggerFactory.getLogger(logFile)
scale = config.scale

for sub_dirname in sub_dirnames:
    data_group = sub_dirname
    patch_select = config.patch_select
    loss = 'l1_loss'
    pretrained_model = ''

    data_path = os.path.join(data_base_path, data_group)
    I_dir = os.path.join(data_path, str(scale))  # Large image
    I_prime_dir = os.path.join(data_path, str(scale))  # Large image
    full_img = False
    img_only = False

    add_item = '_fullimg' if full_img else ''
    n_frame = 2

    # Metadata file path.
    pts1_file = os.path.join(data_path, 'pts1_%s_%s_%s%s.txt' % (patch_select, str(scale), str(n_frame), add_item))
    filenames_file = os.path.join(data_path,
                                  'filenames_%s_%s_%s%s.txt' % (patch_select, str(scale), str(n_frame), add_item))
    ground_truth_file = os.path.join(data_path,
                                     'gt_%s_%s_%s%s.txt' % (patch_select, str(scale), str(n_frame), add_item))

    img_loader = img_helper.opencv_gray_loader
    imgs = util.read_array_file(filenames_file, str)
    H_mat = []

    for img_pair in imgs:
        start_time = time.time()
        paths = [os.path.join(I_dir, path) for path in img_pair]
        imgs = [img_loader(path) for path in paths]

        H = method(imgs[1], imgs[0], logger=txt_logger)
        if H is None:
            H = np.eye(3)
        H_mat.append(img_helper.scale_homography_mat(H, scale))
        end_time = time.time()
        txt_logger.info('\t'.join(img_pair) + '\ttime: ' + str(end_time - start_time) + ' s')

    H_mat_np = np.stack(H_mat, axis=0)
    h_mat_file = h_mat_dir + '/h_matrices_' + data_group + '_' + str(scale) + '.mat'
    sio.savemat(h_mat_file, {'H_mat': H_mat_np})

# Evaluate the homography estimation with MACE.
eval_output_dir = os.path.join(output_dir, exp_name, 'MACE')
util.eval_homography(h_mat_dir, scale, config.ann_path, eval_output_dir, prefix=config.prefix)