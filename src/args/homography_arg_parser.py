import argparse
import os
import cv2
import numpy as np
import glob

def str2bool(s):
    return s.lower() == 'true'

class HomographyArgParser():
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--mode', type=str, default='test', help='Train or test', choices=['train', 'test'])
        parser.add_argument('--loss_type', type=str, default='l1_loss', help='Loss type',
                            choices=['h_loss', 'rec_loss', 'ssim_loss', 'l1_loss', 'l1_smooth_loss', 'ncc_loss'])
        parser.add_argument('--use_batch_norm', type=str2bool, default='False', help='Use batch_norm?')
        parser.add_argument('--leftright_consistent_weight', type=float, default=0,
                            help='UUse left right consistent in loss function? Set a small weight for loss(I2_to_I1 - I1)')
        parser.add_argument('--augment_list', nargs='+', default=['normalize'], help='List of augmentations')
        parser.add_argument('--do_augment', type=float, default=-1,
                            help='Possibility of augmenting image: color shift, brightness shift...')
        parser.add_argument('--num_gpus', type=int, default=1, help='Number of splits')

        parser.add_argument('--log_dir', type=str, default='./logs', help='The log path')
        parser.add_argument('--results_dir', type=str, default='./test_result', help='Store visualization for report')
        parser.add_argument('--model_dir', type=str, default='./model', help='The models path')

        parser.add_argument('--data_path', type=str, default='./train_data', help='The raw data path.')
        parser.add_argument('--I_dir', type=str, default='./train_data/I', help='The training image path')
        parser.add_argument('--I_prime_dir', type=str, default='./train_data/I_prime', help='The training image path')
        parser.add_argument('--pts1_file', type=str, default='./train_data/pts1.txt',
                            help='The training path to 4 corners on the first image - training dataset')
        parser.add_argument('--test_pts1_file', type=str, default='./train_data/test_pts1.txt',
                            help='The test path to 4 corners on the first image - test dataset')
        parser.add_argument('--gt_file', type=str, default='./train_data/gt.txt', help='The training ground truth file')
        parser.add_argument('--test_gt_file', type=str, default='./train_data/test_gt.txt',
                            help='The test ground truth file')
        parser.add_argument('--filenames_file', type=str, default='./train_data/train.txt',
                            help='File that contains all names of files, for training')
        parser.add_argument('--test_filenames_file', type=str, default='./train_data/test.txt',
                            help='File that contains all names of files for evaluation')

        parser.add_argument('--visual', type=str2bool, default='false', help='Visualize obtained images to debug')
        parser.add_argument('--save_visual', type=str2bool, default='True', help='Save visual images for report')

        parser.add_argument('--img_w', type=int, default=320)
        parser.add_argument('--img_h', type=int, default=180)
        parser.add_argument('--patch_size', type=int, default=128)
        parser.add_argument('--batch_size', type=int, default=1)
        parser.add_argument('--max_epoches', type=int, default=150)
        parser.add_argument('--lr', type=float, default=1e-4, help='Max learning rate')
        parser.add_argument('--min_lr', type=float, default=.9e-4, help='Min learning rate')
        parser.add_argument('--shuffle', type=bool, default=False,
                            help='Whether to do data shuffle in dataloader or not')
        parser.add_argument('--workers', type=int, default=16)
        parser.add_argument('--step_size', type=int, default=50000)
        parser.add_argument('--weight_decay', type=float, default=1e-5)
        parser.add_argument('--beta1', type=float, default=0.9)
        parser.add_argument('--beta2', type=float, default=0.999)
        parser.add_argument('--gamma', type=float, default=0.1)


        parser.add_argument('--exp_name', type=str, default='freeway_chan')
        parser.add_argument('--model_name', type=str, default='HCNN')
        parser.add_argument('--scale', type=float, default=0.25)
        parser.add_argument('--patch_select', type=str, default='random')
        parser.add_argument('--resume', type=str2bool, default='False',
                            help='True: restore the existing model. False: retrain')
        parser.add_argument('--retrain', type=str2bool, default='False',
                            help='True: restore the existing model, use max learning rate')

        parser.add_argument('--n_save_model', type=int, default=5)
        parser.add_argument('--n_train_log', type=float, default=100)
        parser.add_argument('--niter_save_model', type=float, default=5000)
        parser.add_argument('--nepoch_save_model', type=float, default=5)
        parser.add_argument('--niter_save_last', type=float, default=1000)
        parser.add_argument('--pretrained_model', type=str, default=None)
        parser.add_argument('--n_test_log', type=str, default=None)

        self.parser = parser

    def parse(self):
        return self.parser.parse_args()