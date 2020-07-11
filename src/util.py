import numpy as np
import os
import glob
import torch
import re

####################### File System ###############################
def readLines(file):
    """
    Read all lines in a file , and return the lines without linefeed using a list.
    :param file: path the file
    :return: list of strings, each string is a line in the file
    """
    with open(file) as f:
        data = f.readlines()
    data = [line.strip() for line in data]
    return data

def checkDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def checkDirs(dirs):
    for dir in dirs:
        checkDir(dir)

def globx(dir, patterns):
    result = []
    for pattern in patterns:
        subdirs = glob.glob(dir + '/' + pattern)
        result.extend(subdirs)
    result = list(set(result))
    list.sort(result)
    return result

def globxx(dir, extensions):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if any(fname.endswith(extension) for extension in extensions):
                path = os.path.join(root, fname)
                images.append(path)
    return images

def globxxx(dir, fragments):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if all(fragment in fname for fragment in fragments):
                path = os.path.join(root, fname)
                images.append(path)
    return images



####################### Image ######################################

def im_unit8(images, lb=0, ub=1):
    """
    convert numpy img from any range to [0, 255] and unit8
    :param images: input images
    :param lb: lower bound of input images pixel values
    :param ub: upper bound of input images pixel values
    :return: an unit8 image which is ready for
    """
    if images.dtype == np.uint8:
        return images
    images = np.clip((images - lb) * 1.0 / (ub - lb) * 255, 0, 255).astype('uint8')
    return images

def im_float32(images, lb=0, ub=255):
    """
    convert numpy img from any range to [0, 1] and float
    :param images: input images
    :param lb: lower bound of input images pixel values
    :param ub: upper bound of input images pixel values
    :return: an unit8 image which is ready for
    """
    images = images.astype('float32')
    images = (images - lb) * 1.0 / (ub - lb)
    images = images.astype('float32')
    return images

def im_float32_symm(images, lb=0, ub=255):
    """
    convert numpy img from any range to [-1, 1] and float
    :param images: input images
    :param lb: lower bound of input images pixel values
    :param ub: upper bound of input images pixel values
    :return: an unit8 image which is ready for
    """
    images = images.astype('float32')
    images = (images - (lb + ub) / 2.0) * 2.0 / (ub - lb)
    images = images.astype('float32')
    return images

def im_min_max_norm(img):
    min_v = np.min(img)
    max_v = np.max(img)
    if max_v == min_v:
        img[...] = 1
    else:
        img = (img - min_v) / (max_v - min_v)
    return img





####################### PyTorch ######################################
def worker_init_fn(worker_id):
    np.random.seed((torch.initial_seed() + worker_id) % (2 ** 32))


def find_last_model(model_dir, model_name):
    models = globx(model_dir, [model_name + '*.pth'])
    latest_model = max(models, key=os.path.getctime)
    return latest_model


def get_model_name(model_name, iter, itername='iter'):
    return model_name + '-' + itername + '-' + str(iter) + '.pth'

def resume_model(model, model_dir, model_name):
    last_model = find_last_model(model_dir, model_name)
    niter, nepoch = load_model(model, last_model)
    return last_model, niter, nepoch

def load_part_of_model(model, model_dir, model_name):
    last_model = find_last_model(model_dir, model_name)
    dict_in_file = torch.load(last_model)
    model = load_part_model(model, dict_in_file['model'])
    return last_model, dict_in_file['n_iter'], dict_in_file['n_epoch']

def load_part_model(model, pretrained_dict):
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    return model

def load_part_weights(model, weights_file):
    model_dict = model.state_dict()
    dict_in_file = torch.load(weights_file)
    pretrained_dict = dict_in_file['model']
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    return dict_in_file['n_iter'], dict_in_file['n_epoch']

def save_model(model, iter, epoch):
    return {'model': model.state_dict(), 'n_iter':iter, 'n_epoch': epoch}

def load_model(model, weights_file):
    dict_in_file = torch.load(weights_file)
    model.load_state_dict(dict_in_file['model'])
    return dict_in_file['n_iter'], dict_in_file['n_epoch']

def check_num_model(model_dir, model_name, num):
    models = globx(model_dir, [model_name + '*.pth'])
    return len(models) >= num

def rm_suffix(filename):
    temp = filename.split('.')
    result = '.'.join(temp[:-1])
    return result

def read_array_file(file, type):
    data = readLines(file)
    data = [x.split() for x in data]
    data = np.array(data).astype(type)
    return data

def get_column(data, col, type):
    data = [type(item) for item in data[:, col]]
    return np.squeeze(np.asarray(data))


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

    return imgs


def sort_file(files, pattern, type=int):
    filenames = [os.path.basename(file) for file in files]
    nums = []
    matcher = re.compile(pattern)
    for filename in filenames:
        results = matcher.findall(filename)
        num = type(results[0])
        nums.append(num)

    sort_idx = np.argsort(nums)
    files = [files[i] for i in list(sort_idx)]
    return files, sort_idx


############ eval ######################
from collections import defaultdict as ddict
import scipy.io as sio
def eval_homography(homography, scale, pt_ann_dir, output_dir, prefix='clip'):
    """
    Evaluate homography estimation result with MACE
    :param homography: str for the homography mat file path, or numpy array of the estimated homography
    :param pt_ann_dir: points correspondence annotation directory
    :param output_dir: output directory
    :param prefix: prefix of the annotations file.
    :return:
    """
    # Get all annotation file path
    pt_anns = globx(pt_ann_dir, ['*pairs'])
    pt_anns, sort_idx = sort_file(pt_anns, prefix + '(\d+).pairs')

    # Create output_dir if it does not exist
    checkDir(output_dir)

    # create output summary file
    output_summary_file = os.path.join(output_dir, 'summary.txt')
    output_summary_h = open(output_summary_file, 'w')

    pair_count = 0
    rmse_sum_total = 0
    for pt_ann in pt_anns: # Loop over all annotation files.
        se_dict = ddict(lambda: [])
        rmse_dict = {}
        pt_ann_name = os.path.basename(pt_ann)
        data_group = rm_suffix(pt_ann_name)
        print('Dealing with %s' % (data_group))

        # Read the estimated homography.
        if isinstance(homography, str):
            h_mat_file = homography + '/h_matrices_' + data_group + '_' + str(scale) + '.mat'
            if not os.path.exists(h_mat_file):
                continue
            mat_data = sio.loadmat(h_mat_file)
            h_mats = mat_data['H_mat']
        else:
            h_mats = homography[data_group]

        # Read the annotation files.
        pt_ann_data = read_array_file(pt_ann, str)
        if 'frame' in pt_ann_data[0, 0]:
            pt_ann_data = pt_ann_data[1:, ]
        idx1 = get_column(pt_ann_data, 0, int)
        idx2 = get_column(pt_ann_data, 3, int)
        p1_x = get_column(pt_ann_data, 1, float)
        p1_y = get_column(pt_ann_data, 2, float)
        p2_x = get_column(pt_ann_data, 4, float)
        p2_y = get_column(pt_ann_data, 5, float)


        # calculate the distance between each correspondence points pair.
        for i in range(len(idx1)):
            f1 = idx1[i]
            f2 = idx2[i]
            mat_f1_f2 = np.eye(3)
            for j in range(f1, f2):
                mat_f1_f2 = np.matmul(mat_f1_f2, h_mats[j, :, :])
            v2 = np.reshape(np.asarray([p2_x[i], p2_y[i], 1]), (3, 1))
            v1 = np.matmul(mat_f1_f2, v2)
            v1 = v1 / v1[-1]
            v1 = np.squeeze(v1)

            code = '%s-%s' % (str(f1), str(f2))
            rse = np.sqrt((v1[0] - p1_x[i]) ** 2 + (v1[1] - p1_y[i]) ** 2)
            se_dict[code].append(rse)

        # calculate the average distance between each frame pair. .
        for k, v in se_dict.items():
            rmse_bet_pair = np.mean(v)
            rmse_dict[k] = rmse_bet_pair

        # sort the frame pairs keys with its first frame number.
        frame_pairs = list(rmse_dict.keys())
        frame_pair_nums = [pair.split('-') for pair in frame_pairs]
        frame_pair_nums = np.asarray(frame_pair_nums).astype(int)
        first_frames = frame_pair_nums[:, 0]
        sort_idx = np.argsort(first_frames)
        frame_pair_nums = frame_pair_nums[sort_idx,]
        frame_pairs = [frame_pairs[i] for i in list(sort_idx)]


        # Write individual testing result for this testing video file
        output_log_file = os.path.join(output_dir, data_group + '.txt')
        rmse_sum = 0
        with open(output_log_file, 'w') as f:
            for i in range(len(frame_pairs)):
                rmse_v = rmse_dict[frame_pairs[i]]
                record = '\t'.join([str(frame_pair_nums[i, 0]), str(frame_pair_nums[i, 1]), str(rmse_v)])
                f.write(record + '\n')
                rmse_sum += rmse_v

        # Write testing result for this testing video file
        n_pairs = len(frame_pairs)
        pair_count += n_pairs
        rmse_sum_total += rmse_sum
        if n_pairs == 0:
            record = '\t'.join([data_group, str(0), str(0)])
        else:
            record = '\t'.join([data_group, str(n_pairs), str(rmse_sum / n_pairs)])
        output_summary_h.write(record + '\n')

    if pair_count == 0:
        record = '\t'.join(['Total', str(0), str(0)])
    else:
        record = '\t'.join(['Total', str(pair_count), str(rmse_sum_total / pair_count)])
    output_summary_h.write(record + '\n')
    output_summary_h.close()