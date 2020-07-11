import os
import numpy as np
import cv2
import glob
from src import util as my_utils

def select_patch(img_size, patch_size, mode='random'):
  if (not type(patch_size) is list) and (not type(patch_size) is tuple):
      patch_size = [patch_size, patch_size]
  if mode == 'random':
    x = np.random.randint(0, img_size[1] - patch_size[1] + 1)
    y = np.random.randint(0, img_size[0] - patch_size[0] + 1)
  elif mode == 'top_left':
    x = 0
    y = 0
  elif mode == 'bottom_left':
    x = 0
    y = img_size[0] - patch_size[0] - 1
  elif mode == 'top_right':
    x = img_size[1] - patch_size[1] - 1
    y = 0
  elif mode == 'bottom_right':
    x = img_size[1] - patch_size[1] - 1
    y = img_size[0] - patch_size[0] - 1
  elif mode == 'center':
    x = int((img_size[1] - patch_size[1]) / 2.0)
    y = int((img_size[0] - patch_size[0]) / 2.0)
  else:
    print('Patch Selection mode not implemented')
    exit(-1)

  top_left_point = (x, y)
  top_right_point = (patch_size[1] + x, y)
  bottom_right_point = (patch_size[1] + x, patch_size[0] + y)
  bottom_left_point = (x, patch_size[0] + y)

  four_points = [top_left_point, top_right_point, bottom_right_point, bottom_left_point]
  perturbed_four_points = []
  gt = []
  for point in four_points:
    perturbed_four_points.append((point[0], point[1]))
    gt.append((0, 0))

  return four_points, perturbed_four_points, gt


def check_and_rm(file):
  if os.path.exists(file):
    os.remove(file)

def check_and_mkdir(dir):
  if not os.path.exists(dir):
    os.makedirs(dir)


def sort_frames(imgs):
    imgnames = [os.path.basename(img) for img in imgs]
    frame_num = []
    for imgname in imgnames:
        temp = imgname.split('frame')
        temp = temp[1]
        temp = temp.split('.')
        temp = temp[0]
        temp = temp.split('_')
        temp = temp[0]
        frame_n = int(temp)
        frame_num.append(frame_n)

    sort_idx = np.argsort(frame_num)
    imgs = [imgs[i] for i in list(sort_idx)]
    frame_num = [frame_num[i] for i in list(sort_idx)]
    return imgs, frame_num


def write_to_records(image_names, nums, PATCH_SIZE, patch_select, scaled_image_size, TEST_PTS1_FILE, TEST_GROUND_TRUTH_FILE, TEST_FILENAMES_FILE,
                     n_frames=2):
    f_pts1 = open(TEST_PTS1_FILE, 'ab')
    f_gt = open(TEST_GROUND_TRUTH_FILE, 'ab')
    f_file_list = open(TEST_FILENAMES_FILE, 'a')

    for i in range(len(image_names)):
        if i < n_frames - 1:
            continue
        state = []
        for j in range(n_frames - 1):
            if nums[i - j] != nums[i - j - 1] + 1:
                state.append(True)
            else:
                state.append(False)
        if any(state):
            continue

        four_points, perturbed_four_points, gt = select_patch(scaled_image_size, PATCH_SIZE, patch_select)
        gt = np.array(gt).flatten().astype(np.float32)
        four_points = np.array(four_points).flatten().astype(np.float32)
        perturbed_four_points = np.array(perturbed_four_points).flatten().astype(np.float32)

        np.savetxt(f_gt, [gt], delimiter=' ')
        np.savetxt(f_pts1, [four_points], delimiter=' ')
        for j in range(n_frames):
            f_file_list.write('%s ' % (image_names[i - n_frames + 1 + j]))
        f_file_list.write('\n')

    f_pts1.close()
    f_gt.close()
    f_file_list.close()

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, required=True, help='the path to processed testing data')
parser.add_argument('--patch_select', type=str, default='center', help='one of {random, center, bottom_left, bottom_right, top_left, top_right}')
parser.add_argument('--scale', type=float, required=True, help='scale to resize the frame')
parser.add_argument('--height', type=int, required=True, help='frame height')
parser.add_argument('--width', type=int, required=True, help='frame width')
parser.add_argument('--patch_size', type=int, required=True, help='patch size')
parser.add_argument('--full_img', type=bool, default=False, help='if use entire image rather than the patch')
parser.add_argument('--regenerate', type=str2bool, default=True, help='if regenerate the image')
parser.add_argument('--output_dir', type=str, default='test_homography', help='base output directory')


config = parser.parse_args()
patch_select = config.patch_select
scale = config.scale
output_size = (int(config.height * scale), int(config.width * scale))
regenerate_image = config.regenerate
PATCH_SIZE = 128
full_img = config.full_img
if full_img:
    PATCH_SIZE = output_size
n_frame = 2

RAW_DATA_PATH = config.data_path
OUTPUT_DATA_PATH = config.output_dir

add_item = '_fullimg' if full_img else ''


subdirs = glob.glob(RAW_DATA_PATH + '/*')
for subdir in subdirs:
    DATA_PATH = os.path.join(OUTPUT_DATA_PATH, os.path.basename(subdir))
    I_DIR = DATA_PATH + '/' + str(scale)  # Large image
    my_utils.checkDir(DATA_PATH)

    PTS1_FILE = os.path.join(DATA_PATH, 'pts1_' + patch_select + '_' + str(scale) + '_' + str(
        n_frame) + add_item + '.txt')
    FILENAMES_FILE = os.path.join(DATA_PATH, 'filenames_' + patch_select + '_' + str(scale) + '_' + str(
        n_frame) + add_item + '.txt')
    GROUND_TRUTH_FILE = os.path.join(DATA_PATH, 'gt_' + patch_select + '_' + str(scale) + '_' + str(
        n_frame) + add_item + '.txt')

    check_and_rm(PTS1_FILE)
    check_and_rm(FILENAMES_FILE)
    check_and_rm(GROUND_TRUTH_FILE)

    img_dir = subdir
    images = my_utils.globx(img_dir, ['*jpg', '*png'])
    images, frame_num = sort_frames(images)
    image_names = [os.path.basename(image) for image in images]
    n_img = len(images)

    image = cv2.imread(images[0])
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_size = gray_image.shape
    scaled_image_size = output_size

    # Size of synthetic image and the pertubation range (RH0)
    HEIGHT = scaled_image_size[0] #
    WIDTH = scaled_image_size[1]

    # check the existence of images
    if (not os.path.exists(I_DIR)) or (len(glob.glob(I_DIR + '/*jpg')) != n_img):
        check_and_mkdir(I_DIR)

    if regenerate_image:
        for i in range(len(images)):
          img = images[i]
          img_name = os.path.basename(img)
          print(img_name)
          image = cv2.imread(img)
          image_resize = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
          cv2.imwrite(I_DIR + '/' + img_name, image_resize)

    write_to_records(image_names, frame_num, PATCH_SIZE, patch_select, scaled_image_size, PTS1_FILE, GROUND_TRUTH_FILE, FILENAMES_FILE,
                     n_frame)