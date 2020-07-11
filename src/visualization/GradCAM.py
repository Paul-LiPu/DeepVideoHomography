#!/usr/bin/env python
# encoding: utf-8
'''
@author: Pu Li
@contact: pli5270@sdsu.edu
@file: GradCAM.py
@time: 10/3/19 4:44 PM
@desc: Grad-CAM visualization. Mainly borrowed from https://github.com/jacobgil/pytorch-grad-cam
Calculate the attention similar map in Grad-CAM or guided-backpropagation way.
'''
import torch
from torch.autograd import Variable
from torch.autograd import Function
from torch.utils import model_zoo
from torchvision.models import vgg as models
from torchvision import utils
from torch.nn import functional as F
import torch.nn as nn
import cv2
import numpy as np
import argparse


class FeatureExtractor():
    """
    Class for extracting activations and
    registering gradients from target intermediate layers
    """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def get_gradients(self):
        return self.gradients

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class GradCam:
    """
    Class for doing Grad-CAM.
    reference: Selvaraju, Ramprasaath R., et al. "Grad-cam: Visual explanations from deep networks via gradient-based
    localization." Proceedings of the IEEE International Conference on Computer Vision. 2017.
    """
    def __init__(self, model, target_layer_names, feature_module=None):
        """
        :param model: Pytorch Module, must have attribute features (Pytorch module to get features of input images),
        and function get_loss_from_features (Use the output of features module to get loss).
        :param target_layer_names: Feature layer name for calculating Grad-CAM.
        """
        self.model = model
        self.model.eval()
        if feature_module is None:
            feature_module = model.features
        self.extractor = FeatureExtractor(feature_module, target_layer_names)
        self.n_target_layers = len(target_layer_names)

    def __call__(self, input, normalize=True):
        """
        :param input: 4-D tensor, [B, C, H, W]
        :return: Grad-CAM images at all layers defined by target_layer_names.
        """
        input_size = input.size()

        # Forward and backward propagation.
        features, output = self.extractor(input)
        loss = self.model.get_loss_from_features(output)
        loss.backward(retain_graph=True)
        self.model.zero_grad()

        cams = []
        for i in range(self.n_target_layers):  # Loop over all target layers.
            # Extract features map and gradients on target layers.
            grads_val = self.extractor.get_gradients()[i]
            grads_val = grads_val.view((grads_val.size(0), grads_val.size(1), -1))
            target = features[i]
            target_size = target.size() # store previous target layer blob size
            target = target.view((target.size(0), target.size(1), -1))

            # Calculate weights by gradients.
            weights = grads_val.mean(dim=-1, keepdim=True)

            # multiplication of weights and features map.
            weights = weights.expand_as(target)
            cam = weights * target

            # Sum over all channels.
            cam = cam.view(target_size)
            cam = cam.sum(dim=1, keepdim=True)

            # Remove value smaller than 0
            cam = F.relu(cam)

            # Normalize to[0, 1]
            if normalize:
                cam_size = cam.size()
                cam = cam.view(cam_size[0], cam_size[1], -1)
                cam = cam - cam.min(dim=-1, keepdim=True)[0]
                cam = cam / cam.max(dim=-1, keepdim=True)[0]
                cam = cam.view(cam_size)

            # Resize feature map to size of input.
            cam = F.interpolate(cam, size=input_size[2:], mode='bilinear')
            cams.append(cam)
        return cams


class GuidedBackpropReLU_F(Function):
    def forward(self, input):
        """
            Guided Back Propagation ReLU Function.
            Reference: Springenberg, Jost Tobias, et al. "Striving for simplicity: The all
            convolutional net." arXiv preprint arXiv:1412.6806 (2014).
            """
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input),
                                   torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output,
                                                 positive_mask_1), positive_mask_2)
        return grad_input


class GuidedBackpropReLUModel:
    """
    Wrapper class of original model.
    Replace the ReLU layer with GuidedBackPropReLU in features module.
    """
    def __init__(self, model, feature_module=None):
        self.model = model
        self.model.eval()

        if feature_module is None:
            self.feature_module = model.features
        else:
            self.feature_module = feature_module
        # replace ReLU with GuidedBackpropReLU
        for idx, module in self.feature_module._modules.items():
            if module.__class__.__name__ == 'ReLU':
                self.feature_module._modules[idx] = GuidedBackpropReLU()

    def __call__(self, input):
        """
        :param input: 4-D tensor as input
        :return: Gradients on the input by certain loss defined in model.get_loss_from_features.
        """
        features = self.feature_module(input)
        loss = self.model.get_loss_from_features(features)
        loss.backward(retain_graph=True)
        output = input.grad.clone()
        self.model.zero_grad()
        input.grad.zero_()
        return output


class GuidedBackpropReLU(nn.Module):
    """
        Guided Back Propagation ReLU layer.
        Reference: Springenberg, Jost Tobias, et al. "Striving for simplicity: The all
        convolutional net." arXiv preprint arXiv:1412.6806 (2014).
        """
    def __init__(self):
        super(GuidedBackpropReLU, self).__init__()
        self.relu = GuidedBackpropReLU_F()

    def forward(self, input):
        return self.relu(input)


def show_cam_on_image(img, mask):
    if img.ndim == 2:
        img = img[..., np.newaxis]
    if img.dtype is np.dtype(np.uint8):
        img = img.astype(np.float32) / 255.
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def heatmap(img):
    img = img - np.min(img)
    img = img / np.max(img)
    heatmap = cv2.applyColorMap(np.uint8(255 * img), cv2.COLORMAP_JET)
    return heatmap


if __name__ == '__main__':
    """
        Test of the implementation of Grad-CAM and Guided-Backpropagation
        Usage: python grad_cam.py <path_to_image>
        Workflow:
        1. Loads an image with opencv.
        2. Preprocesses it for VGG19 and converts to a pytorch variable.
        3. Makes a forward pass to find the category index with the highest score,
        and computes intermediate activations.
        Makes the visualization. """
    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--use-cuda', action='store_true', default=True,
                            help='Use NVIDIA GPU acceleration')
        parser.add_argument('--image-path', type=str, default='./both.png',
                            help='Input image path')
        args = parser.parse_args()
        args.use_cuda = args.use_cuda and torch.cuda.is_available()
        # print(torch.cuda.is_available())
        if args.use_cuda:
            print("Using GPU for acceleration")
        else:
            print("Using CPU for computation")

        return args


    def preprocess_image(img):
        means = [0.485, 0.456, 0.406]
        stds = [0.229, 0.224, 0.225]

        preprocessed_img = img.copy()[:, :, ::-1]
        for i in range(3):
            preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
            preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
        preprocessed_img = \
            np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
        preprocessed_img = preprocessed_img[np.newaxis, ]

        # Warning: Input with requires_grad=True must initialize in GPU.
        # Otherwise, the gradients of input image will store in CPU.
        preprocessed_img = torch.tensor(preprocessed_img, device=0, requires_grad=True)
        return preprocessed_img


    class cam_vgg19(models.VGG):
        def __init__(self, pretrained=False, **kwargs):
            if pretrained:
                kwargs['init_weights'] = False
            super(cam_vgg19, self).__init__(models.make_layers(models.cfg['E']), **kwargs)
            if pretrained:
                self.load_state_dict(model_zoo.load_url(models.model_urls['vgg19']))

        def get_loss_from_features(self, features, index=None):
            output = features.view(features.size(0), -1)
            output = self.classifier(output)
            if index == None:
                index = torch.argmax(output, dim=-1)

            one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
            one_hot[0][index] = 1
            one_hot = torch.tensor(one_hot, device=0, requires_grad=True)
            one_hot = torch.sum(one_hot * output)
            return one_hot


    args = get_args()

    # Can work with any model, but it assumes that the model has a
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.
    model = cam_vgg19(pretrained=True).cuda()
    grad_cam = GradCam(model=model, target_layer_names=["35"])

    img = cv2.imread(args.image_path, 1)
    img = np.float32(cv2.resize(img, (224, 224))) / 255
    input = preprocess_image(img)

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.
    # masks = grad_cam(input.cuda())
    masks = grad_cam(input)
    mask = masks[0]
    mask = mask.detach().cpu().numpy()
    mask = mask.squeeze()
    img_with_mask = show_cam_on_image(img, mask)
    cv2.imwrite('cam2.jpg', img_with_mask)

    gb_model = GuidedBackpropReLUModel(model=model)
    gb = gb_model(input)
    save_gb = gb[0, :, :, :]
    utils.save_image(save_gb.cpu(), 'gb2.jpg')

    temp = masks[0]
    cam_gb = masks[0] * gb
    save_cam_gb = cam_gb[0, :, :, :]
    utils.save_image(save_cam_gb.cpu(), 'cam_gb2.jpg')
