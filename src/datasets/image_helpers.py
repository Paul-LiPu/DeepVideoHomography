import cv2
from PIL import Image
import numpy as np
import torch
from matplotlib import pyplot as plt
import os
from abc import ABC, abstractmethod
from ..util import *
#
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]
#
#
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    return Image.open(path).convert('RGB')

def opencv_loader(path):
    img = cv2.imread(path)
    # img = img[..., ::-1]
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
def opencv_gray_loader(path):
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
def opencv_RGB_loader(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def opencv_BGR_loader(path):
    img = cv2.imread(path)
    return img

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def image_mone_one_norm(image):
    image = image.astype(np.float32)
    image = image / 255. * 2 - 1
    image = image[np.newaxis, ]
    return image

def image_zero_one_norm(image):
    image = image.astype(np.float32)
    image = image / 255.
    image = image[np.newaxis, ]
    return image

class ImgTransform(ABC):
    @abstractmethod
    def transform(self, img):
        pass

    @abstractmethod
    def reverse(self, img):
        pass

class ImageMoneOneNorm(ImgTransform):
    """
    transform image from uint8 to [-1, 1]
    """
    def transform(self, image):
        image = image.astype(np.float32)
        image = image / 255. * 2 - 1
        return image

    def reverse(self, image):
        image = np.clip((image + 1) / 2. * 255, 0, 255).astype(np.uint8)
        return image

class ImageZeroOneNorm(ImgTransform):
    """
    transform image from uint8 to [-1, 1]
    """
    def transform(self, image):
        image = image.astype(np.float32)
        image = image / 255.
        return image

    def reverse(self, image):
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
        return image


class VGGNorm(ImgTransform):
    """
    transform image from [0, 1] to VGG input
    """
    def __init__(self):
        """
        parameters got from https://github.com/pytorch/examples/blob/97304e232807082c2e7b54c597615dc0ad8f6173/imagenet/main.py#L197-L198
        """
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def transform(self, image):
        assert len(image.shape) == 3 and image.shape[0] == 3
        for i in range(3):
            image[i,] = (image[i,] - self.mean[i]) / self.std[i]
        return image.astype(np.float32)

    def reverse(self, image):
        ndim = len(image.shape)
        assert ndim > 2
        if ndim == 3:
            for i in range(3):
                image[i,] = image[i,] *  self.std[i] + self.mean[i]
        elif ndim == 4:
            for i in range(3):
                image[:, i, ] = image[:, i, ] *  self.std[i] + self.mean[i]
        else:
            raise NotImplementedError()
        return image


class ChannelTranpose(ImgTransform):
    """
    transform image from [0, 1] to VGG input
    """

    def __init__(self, trans_dim=(2, 0, 1)):
        self.trans_dim = trans_dim
        self.reverse_trans_dim = list(np.argsort(trans_dim))

    def transform(self, image):
        assert len(image.shape) == len(self.trans_dim)
        return np.transpose(image, self.trans_dim)

    def reverse(self, image):
        ndim = len(image.shape)
        img_dim = len(self.reverse_trans_dim)
        assert ndim >= img_dim
        if ndim == img_dim:
            image = np.transpose(image, self.reverse_trans_dim)
        elif ndim == img_dim + 1:
            reverse_trans_dim = [0]
            reverse_trans_dim.extend([dim + 1 for dim in self.reverse_trans_dim])
            image = np.transpose(image, reverse_trans_dim)
        else:
            raise NotImplementedError()
        return image


class BGR2RGB(ImgTransform):
    """
    transform image from BGR to RGB
    """

    def transform(self, image):
        return image[..., ::-1]

    def reverse(self, image):
        return image[..., ::-1]


class Expand_dim(ImgTransform):
    """
    transform image from [0, 1] to VGG input
    """

    def __init__(self, dim=0):
        self.dim = dim

    def transform(self, image):
        return np.expand_dims(image, axis=self.dim)

    def reverse(self, image):
        ndim = len(image.shape)
        assert ndim > 2
        if ndim == 3:
            image = np.squeeze(image, self.dim)
        elif ndim == 4:
            image = np.squeeze(image, self.dim + 1)
        else:
            raise NotImplementedError()
        return image


def draw_correspondences_img(img1, img2, corr1, corr2, pred_corr2):
  """ Save pair of images with their correspondences into a single image. Used for report"""
  # Draw prediction
  # if len(img1.shape) > 2:
  #     img1 = np.squeeze(img1)
  #     img2 = np.squeeze(img2)
  # if len(img1.shape) > 2:
  #     img1 = np.transpose(img1, (1, 2, 0))
  #     img2 = np.transpose(img2, (1, 2, 0))
  copy_img2 = img2.copy()
  copy_img1 = img1.copy()

  cv2.polylines(copy_img2, np.int32([pred_corr2]), 1, (5, 225, 225),3)

  point_color = (0,255,255)
  line_color_set = [(255,102,255), (51,153,255), (102,255,255), (255,255,0), (102, 102, 244), (150, 202, 178), (153,240,142), (102,0,51), (51,51,0) ]
  # Draw 4 points (ground truth)
  full_stack_images = draw_matches(copy_img1, corr1, copy_img2 , corr2, '/tmp/tmp.jpg', color_set = line_color_set, show=False)
  # Save image
  return full_stack_images
#
#
def draw_matches(img1, kp1, img2, kp2, output_img_file=None, color_set=None, show=True):
    """Draws lines between matching keypoints of two images without matches.
    This is a replacement for cv2.drawMatches
    Places the images side by side in a new image and draws circles
    around each keypoint, with line segments connecting matching pairs.
    You can tweak the r, thickness, and figsize values as needed.
    Args:
        img1: An openCV image ndarray in a grayscale or color format.
        kp1: A list of cv2.KeyPoint objects for img1.
        img2: An openCV image ndarray of the same format and with the same
        element type as img1.
        kp2: A list of cv2.KeyPoint objects for img2.
        color_set: The colors of the circles and connecting lines drawn on the images.
        A 3-tuple for color images, a scalar for grayscale images.  If None, these
        values are randomly generated.
    """
    # We're drawing them side by side.  Get dimensions accordingly.
    # Handle both color and grayscale images.
    if len(img1.shape) == 3:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], img1.shape[2])
    elif len(img1.shape) == 2:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1])
    new_img = np.zeros(new_shape, type(img1.flat[0]))
    # Place images onto the new image.
    new_img[0:img1.shape[0],0:img1.shape[1]] = img1
    new_img[0:img2.shape[0],img1.shape[1]:img1.shape[1]+img2.shape[1]] = img2

    # Draw lines between points

    kp2_on_stack_image = (kp2 + np.array([img1.shape[1], 0])).astype(np.int32)

    kp1 = kp1.astype(np.int32)
    # kp2_on_stack_image[0:4,0:2]
    line_color1 = (2, 10, 240)
    line_color2 = (2, 10, 240)
    # We want to make connections between points to make a square grid so first count the number of rows in the square grid.
    grid_num_rows = int(np.sqrt(kp1.shape[0]))

    if output_img_file is not None and grid_num_rows >= 3:
        for i in range(grid_num_rows):
            # cv2.line(new_img, tuple(kp1[i*grid_num_rows]), tuple(kp1[i*grid_num_rows + (grid_num_rows-1)]), line_color1, 1,  LINE_AA)
            cv2.line(new_img, tuple(kp1[i]), tuple(kp1[i + (grid_num_rows-1)*grid_num_rows]), line_color1, 1,  cv2.LINE_AA)
            cv2.line(new_img, tuple(kp2_on_stack_image[i*grid_num_rows]), tuple(kp2_on_stack_image[i*grid_num_rows + (grid_num_rows-1)]), line_color2, 1,  cv2.LINE_AA)
            cv2.line(new_img, tuple(kp2_on_stack_image[i]), tuple(kp2_on_stack_image[i + (grid_num_rows-1)*grid_num_rows]), line_color2, 1,  cv2.LINE_AA)

    if output_img_file is not None and grid_num_rows == 2:
        cv2.polylines(new_img, np.int32([kp2_on_stack_image]), 1, line_color2, 3)
        cv2.polylines(new_img, np.int32([kp1]), 1, line_color1, 3)
    # Draw lines between matches.  Make sure to offset kp coords in second image appropriately.
    r = 7
    thickness = 1

    for i in range(len(kp1)):
        key1 = kp1[i]
        key2 = kp2[i]
        # Generate random color for RGB/BGR and grayscale images as needed.
        try:
            c  = color_set[i]
        except:
            c = np.random.randint(0,256,3) if len(img1.shape) == 3 else np.random.randint(0,256)
        # So the keypoint locs are stored as a tuple of floats.  cv2.line(), like most other things,
        # wants locs as a tuple of ints.
        end1 = tuple(np.round(key1).astype(int))
        end2 = tuple(np.round(key2).astype(int) + np.array([img1.shape[1], 0]))
        cv2.line(new_img, end1, end2, c, thickness,  cv2.LINE_AA)
        cv2.circle(new_img, end1, r, c, thickness,  cv2.LINE_AA)
        cv2.circle(new_img, end2, r, c, thickness,  cv2.LINE_AA)
    # pdb.set_trace()
    if show:
        plt.figure(figsize=(15,15))
        if len(img1.shape) == 3:
            plt.imshow(new_img)
        else:
            plt.imshow(new_img)
        plt.axis('off')
        plt.show()
    if output_img_file is not None:
        cv2.imwrite(output_img_file, new_img)

    return new_img


def scale_mat(scale):
    """
    Calculate the matrix help to modify homography matrix to original scale
    :param scale:
    :return: H = SCALE_MAT * H * SCALE_MAT_INV
    """
    SCALE_MAT = np.array([
    [1 / scale, 0, 0],
    [0, 1 / scale, 0],
    [0, 0, 1]], dtype=np.float64)
    SCALE_MAT_INV = np.array([
    [scale, 0, 0],
    [0, scale, 0],
    [0, 0, 1]], dtype=np.float64)

    return SCALE_MAT, SCALE_MAT_INV
#
#
def scale_homography_mat(H, scale):
    """
    map homography matrix of original size image from homography matrix calculate from scaled image.
    :param H: homography matrix calculate using image resize to scale of original size
    :param scale: scaling factor for images when get H
    :return: homography matrix for original size image
    """
    SCALE_MAT, SCALE_MAT_INV = scale_mat(scale)
    return np.matmul(np.matmul(SCALE_MAT, H), SCALE_MAT_INV)

