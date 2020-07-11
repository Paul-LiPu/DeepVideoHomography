import os
import numpy as np
import cv2
import glob
import src.util as my_utils

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

def cp_file(origin, target, add_item=''):
    input_data = my_utils.read_array_file(origin, type=str)
    input_data = list(input_data)
    output_data = input_data
    with open(target, 'a') as output:
        for line in output_data:
            line = [add_item + item for item in line]
            output.write(' '.join(line) + '\n')

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
                     n_frames=2, n_patch=1, patch_method=None):
    assert patch_method is None or len(patch_method) == n_patch
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

        pt1_list = []
        gt_list = []
        for j in range(n_patch):
            if patch_method is None:
                four_points, perturbed_four_points, gt = select_patch(scaled_image_size, PATCH_SIZE, patch_select)
            else:
                four_points, perturbed_four_points, gt = select_patch(scaled_image_size, PATCH_SIZE, patch_method[j])
            gt = np.array(gt).flatten().astype(np.float32)
            four_points = np.array(four_points).flatten().astype(np.float32)
            pt1_list.append(four_points)
            gt_list.append(gt)

        pt1 = np.concatenate(pt1_list, axis=0)
        gt = np.concatenate(gt_list, axis=0)
        np.savetxt(f_gt, [gt], delimiter=' ')
        np.savetxt(f_pts1, [pt1], delimiter=' ')

        for j in range(n_frames):
            f_file_list.write('%s ' % (image_names[i - n_frames + 1 + j]))
        f_file_list.write('\n')

        pt1_list = []
        gt_list = []
        for j in range(n_patch):
            if patch_method is None:
                four_points, perturbed_four_points, gt = select_patch(scaled_image_size, PATCH_SIZE, patch_select)
            else:
                four_points, perturbed_four_points, gt = select_patch(scaled_image_size, PATCH_SIZE, patch_method[j])
            gt = np.array(gt).flatten().astype(np.float32)
            four_points = np.array(four_points).flatten().astype(np.float32)
            pt1_list.append(four_points)
            gt_list.append(gt)

        pt1 = np.concatenate(pt1_list, axis=0)
        gt = np.concatenate(gt_list, axis=0)
        np.savetxt(f_gt, [gt], delimiter=' ')
        np.savetxt(f_pts1, [pt1], delimiter=' ')

        for j in range(n_frames):
            f_file_list.write('%s ' % (image_names[i - j]))
        f_file_list.write('\n')

    f_pts1.close()
    f_gt.close()
    f_file_list.close()


def write_to_records_test(image_names, nums, PATCH_SIZE, patch_select, scaled_image_size, TEST_PTS1_FILE, TEST_GROUND_TRUTH_FILE, TEST_FILENAMES_FILE,
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
parser.add_argument('--pattern', type=str, default='clip_(\d+)_.+', help='prefix of the filenames in the video files')
parser.add_argument('--scale', type=float, required=True, help='scale to resize the frame')
parser.add_argument('--split', type=float, required=True, help='percentage of frames used for generating data')
parser.add_argument('--height', type=int, required=True, help='frame height')
parser.add_argument('--width', type=int, required=True, help='frame width')
parser.add_argument('--patch_size', type=int, required=True, help='patch size')
parser.add_argument('--n_frame', type=int, required=True, help='number of frames in one training data window')
parser.add_argument('--n_patch', type=int, required=True, help='number of patches selected from one training data window')
parser.add_argument('--full_img', type=bool, default=False, help='if use entire image rather than the patch')
parser.add_argument('--regenerate', type=str2bool, default=True, help='if regenerate the image')
parser.add_argument('--output_dir', type=str, default='test_homography', help='base output directory')
config = parser.parse_args()

# Data Path.
input_data_path = config.data_path
output_data_path = config.output_dir
my_utils.checkDir(output_data_path)

# Important parameters.
split = config.split
patch_select = config.patch_select
scale = config.scale
output_size = (int(config.width * scale), int(config.height * scale))
PATCH_SIZE = config.patch_size

full_img = config.full_img
if full_img:
    PATCH_SIZE = output_size

n_frame = config.n_frame
n_patch = config.n_patch
patch_method = None

# Not necessary parameters.
regenerate_label = True
regenerate_images = config.regenerate

# Name of output files.
add_item = '_fullimg' if full_img else ''
add_item2 = '_%s' % str(n_patch) if n_patch > 1 else ''
split_item = '_split_%s' % str(split)

# Assign file names for training data and validation data.
PTS1_FILE = os.path.join(output_data_path, 'pts1_%s_%s_%s%s%s%s.txt' % (patch_select, str(scale), str(n_frame), add_item, add_item2, split_item))
FILENAMES_FILE = os.path.join(output_data_path, 'filenames_%s_%s_%s%s%s%s.txt' % (patch_select, str(scale), str(n_frame), add_item, add_item2, split_item))
GROUND_TRUTH_FILE= os.path.join(output_data_path, 'gt_%s_%s_%s%s%s%s.txt' % (patch_select, str(scale), str(n_frame), add_item, add_item2, split_item))
TEST_PTS1_FILE = os.path.join(output_data_path, 'test_pts1_%s_%s_%s%s%s%s.txt' % (patch_select, str(scale), str(n_frame), add_item, add_item2, split_item))
TEST_FILENAMES_FILE = os.path.join(output_data_path, 'test_filenames_%s_%s_%s%s%s%s.txt' % (patch_select, str(scale), str(n_frame), add_item, add_item2, split_item))
TEST_GROUND_TRUTH_FILE = os.path.join(output_data_path, 'test_gt_%s_%s_%s%s%s%s.txt' % (patch_select, str(scale), str(n_frame), add_item, add_item2, split_item))

# If regenerate label file is specified, rm previous existing files.
if regenerate_label:
    check_and_rm(PTS1_FILE)
    check_and_rm(FILENAMES_FILE)
    check_and_rm(GROUND_TRUTH_FILE)
    check_and_rm(TEST_PTS1_FILE)
    check_and_rm(TEST_FILENAMES_FILE)
    check_and_rm(TEST_GROUND_TRUTH_FILE)

subdirs = glob.glob(input_data_path + '/*')
# subdirs, idx = my_utils.sort_file(subdirs, config.prefix + '_(\d+)_.+')
subdirs, idx = my_utils.sort_file(subdirs, config.pattern)
print(idx)
for subdir in subdirs:
    img_dir = subdir
    img_group = os.path.basename(img_dir)
    images = my_utils.globx(img_dir, ['*jpg', '*png'])
    images, frame_num = sort_frames(images)
    image_names = [os.path.basename(image) for image in images]
    n_img = len(images)
    output_subdir = os.path.join(output_data_path, img_group)
    my_utils.checkDir(output_subdir)

    sub_PTS1_FILE = os.path.join(output_subdir, 'pts1_%s_%s_%s%s%s%s.txt' % (
    patch_select, str(scale), str(n_frame), add_item, add_item2, split_item))
    sub_FILENAMES_FILE = os.path.join(output_subdir, 'filenames_%s_%s_%s%s%s%s.txt' % (
    patch_select, str(scale), str(n_frame), add_item, add_item2, split_item))
    sub_GROUND_TRUTH_FILE = os.path.join(output_subdir, 'gt_%s_%s_%s%s%s%s.txt' % (
    patch_select, str(scale), str(n_frame), add_item, add_item2, split_item))
    sub_TEST_PTS1_FILE = os.path.join(output_subdir, 'test_pts1_%s_%s_%s%s%s%s.txt' % (
    patch_select, str(scale), str(n_frame), add_item, add_item2, split_item))
    sub_TEST_FILENAMES_FILE = os.path.join(output_subdir, 'test_filenames_%s_%s_%s%s%s%s.txt' % (
    patch_select, str(scale), str(n_frame), add_item, add_item2, split_item))
    sub_TEST_GROUND_TRUTH_FILE = os.path.join(output_subdir, 'test_gt_%s_%s_%s%s%s%s.txt' % (
    patch_select, str(scale), str(n_frame), add_item, add_item2, split_item))
    if regenerate_label:
        check_and_rm(sub_PTS1_FILE)
        check_and_rm(sub_FILENAMES_FILE)
        check_and_rm(sub_GROUND_TRUTH_FILE)
        check_and_rm(sub_TEST_PTS1_FILE)
        check_and_rm(sub_TEST_FILENAMES_FILE)
        check_and_rm(sub_TEST_GROUND_TRUTH_FILE)

    image = cv2.imread(images[0])
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_size = gray_image.shape
    # scaled_image_size = [int(scale * size) for size in image_size]
    scaled_image_size = output_size[::-1]

    # Size of synthetic image and the pertubation range (RH0)
    HEIGHT = scaled_image_size[1]
    WIDTH = scaled_image_size[0]

    output_img_dir = os.path.join(output_subdir, str(scale))
    my_utils.checkDir(output_img_dir)
    if regenerate_images:
        for i in range(len(images)):
          img = images[i]
          img_name = os.path.basename(img)
          print(img_name)
          image = cv2.imread(img)
          image_resize = cv2.resize(image, output_size, interpolation=cv2.INTER_LINEAR)
          cv2.imwrite(os.path.join(output_img_dir, img_name), image_resize)

    # generate data information files
    n_train = int(n_img * split)
    n_test = n_img - n_train

    train_imgs = image_names[0:n_train]
    train_num = frame_num[0:n_train]
    test_imgs = image_names[n_train:]
    test_num = frame_num[n_train:]

    write_to_records(train_imgs, train_num, PATCH_SIZE, patch_select, scaled_image_size, sub_PTS1_FILE, sub_GROUND_TRUTH_FILE,
                     sub_FILENAMES_FILE, n_frame, n_patch=n_patch, patch_method=patch_method)
    write_to_records_test(test_imgs, test_num, PATCH_SIZE, patch_select, scaled_image_size, sub_TEST_PTS1_FILE, sub_TEST_GROUND_TRUTH_FILE,
                     sub_TEST_FILENAMES_FILE, n_frame)

    relative_path = os.path.relpath(output_img_dir, output_data_path)
    cp_file(sub_PTS1_FILE, PTS1_FILE)
    cp_file(sub_GROUND_TRUTH_FILE, GROUND_TRUTH_FILE)
    cp_file(sub_FILENAMES_FILE, FILENAMES_FILE, relative_path + '/')
    cp_file(sub_TEST_PTS1_FILE, TEST_PTS1_FILE)
    cp_file(sub_TEST_GROUND_TRUTH_FILE, TEST_GROUND_TRUTH_FILE)
    cp_file(sub_TEST_FILENAMES_FILE, TEST_FILENAMES_FILE, relative_path + '/')

