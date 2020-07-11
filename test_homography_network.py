"""
Testing code.

Output folder:
(1) test_results: Testing the input patch list for each file.
(2) test_results2: Test several randomly chosen patches in input frame pairs.
"""

import src.util as util
from src import logger as loggerFactory
from src.models.homographyNet import *
from src.datasets.homography_dataset import HomographyDataset

from torch.utils.data.dataloader import DataLoader
from src.globalVariables import dtype
from src import losses
from src.datasets import image_helpers as img_helper
import subprocess
import argparse
import os
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, required=True, help='the path to processed testing data')
parser.add_argument('--model_type', type=str, required=True, help='one of {CNN, LSTM}')
parser.add_argument('--model_file', type=str, required=True, help='path to the trained model file')
parser.add_argument('--patch_select', type=str, default='center', help='one of {random, center, bottom_left, bottom_right, top_left, top_right}')
parser.add_argument('--ann_path', type=str, required=True, help='the path to annotation data')
parser.add_argument('--prefix', type=str, default='clip', help='prefix of the testing data file names')
parser.add_argument('--scale', type=float, required=True, help='image scale')
parser.add_argument('--patch_size', type=int, default=128, help='image scale')
parser.add_argument('--output_dir', type=str, default='test_results', help='base output directory')

config = parser.parse_args()
# Read the trained model parameteres.
net_arch_dict = {
    'CNN': HomographyNet,
    'LSTM': HomographyNet_LSTM,
}
net = net_arch_dict[config.model_type](config).type(dtype)
net.loss = losses.l1_loss
iterations, epoch = util.load_model(net, config.model_file)


# Testing paramters
exp_name = os.path.basename(os.path.dirname(config.model_file))
log_dir = os.path.join(config.output_dir, exp_name)
h_mat_dir = os.path.join(log_dir, 'h_mats')
util.checkDirs([log_dir, h_mat_dir])
config.log_dir = log_dir

# Configure logger
timestamp = time.strftime("%Y%m%d_%H-%M-%S", time.localtime())
logFile = os.path.join(log_dir, 'test_' + exp_name + '_' + timestamp + '.log')
txt_logger = loggerFactory.getLogger(logFile)
config.txt_logger = txt_logger



data_base_path = config.data_path
subdirs = util.globx(data_base_path, ['*'])
sub_dirnames = [os.path.basename(subdir) for subdir in subdirs]
patch_select = config.patch_select
scale = config.scale
batch_size = 1

loss = 'l1_loss'
## Parameters

for sub_dirname in sub_dirnames:
    data_group = sub_dirname
    data_path = os.path.join(data_base_path, data_group)
    I_dir = os.path.join(data_path, str(scale))

    add_item = ''
    n_frame = 2

    # meta file
    pts1_file = os.path.join(data_path, 'pts1_%s_%s_%s%s.txt' % (patch_select, str(scale), str(n_frame), add_item))
    filenames_file = os.path.join(data_path,
                                  'filenames_%s_%s_%s%s.txt' % (patch_select, str(scale), str(n_frame), add_item))
    ground_truth_file = os.path.join(data_path,
                                     'gt_%s_%s_%s%s.txt' % (patch_select, str(scale), str(n_frame), add_item))


    # directory for image output
    images = util.globx(I_dir, ['*jpg'])
    image = cv2.imread(images[0])
    image_size = image.shape
    height = image_size[0]
    width = image_size[1]

    config.h_mat_file = h_mat_dir + '/h_matrices_' + data_group + '_' + str(scale) + '.mat'
    config.data_path = data_path
    config.I_dir = I_dir
    config.pts1_file = pts1_file
    config.gt_file = ground_truth_file
    config.filenames_file = filenames_file
    config.img_w = width
    config.img_h = height


    # log configuration and network
    txt_logger.info("Start program")
    cfg_log = 'Configuration: \n'
    for key in config.__dict__:
        cfg_log += ("%s: %s\n" % (key, config.__dict__[key]))
    codes = subprocess.getoutput('cat ' + __file__)
    txt_logger.info(codes)
    txt_logger.info(cfg_log)

    img_loader = img_helper.opencv_gray_loader
    transforms = [img_helper.Expand_dim(0), img_helper.ImageMoneOneNorm()]

    # Dataset for using patches defined in files.
    dataset = HomographyDataset(config, 'train', return_paths=True, transforms=transforms, loader=img_loader)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle=False, num_workers=1, worker_init_fn=util.worker_init_fn)
    test_samples = len(dataset)
    config.epoch_size = test_samples / batch_size

    # txt_logger.info('Loading last model ...')
    net.config = config
    txt_logger.info(str(net))

    # Start Testing
    txt_logger.info("Start testing %s ..." % data_group)
    s_time = time.time()
    net.eval()

    loss_list, loss_name, test_time = net.test(dataloader, out_img=False, out_html=False)
    txt_logger.info('Finished! eclapsed time: %s s' % (str(test_time)))
    for i in range(len(loss_list)):
        txt_logger.info('%s : %s' % (loss_name[i], str(loss_list[i])))

# Evaluate the homography estimation with MACE.
eval_output_dir = os.path.join(log_dir, 'MACE')
util.eval_homography(h_mat_dir, scale, config.ann_path, eval_output_dir, prefix=config.prefix)