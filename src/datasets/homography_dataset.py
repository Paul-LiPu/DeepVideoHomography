"""
Dataset class for homography estimation.
"""


from .base_dataset import BaseDataset
from .file_helper import *
from .image_helpers import *
from ..util import *

class HomographyDataset(BaseDataset):
    def __init__(self, opt, phase, transforms=None, return_paths=False,
                 loader=opencv_gray_loader, img_only=False, samplePatch=None, n_patch=1, patch_size=128):
        super(HomographyDataset, self).__init__()
        if phase == 'train':
            imgs = read_array_file(opt.filenames_file, str)
            # if samplePatch is None:
            pts1 = read_array_file(opt.pts1_file, np.float32)
            gt = None # Could be the corresponding feature points location
            if not opt.gt_file is None:
                gt = read_array_file(opt.gt_file, np.float32)
            # else:
            #     pts1 = None
            #     gt = None
        else:
            imgs = read_array_file(opt.test_filenames_file, str)
            # if samplePatch is None:
            pts1 = read_array_file(opt.test_pts1_file, np.float32)
            gt = None
            if not opt.gt_file is None:
                gt = read_array_file(opt.test_gt_file, np.float32)
            # else:
            #     pts1 = None
            #     gt = None
        self.opt = opt
        self.I_dir = opt.I_dir
        # self.I_prime_dir = opt.I_prime_dir
        self.filenames_file = opt.filenames_file
        self.pts1_file = opt.pts1_file
        self.gt_file = opt.gt_file
        self.patch_size = opt.patch_size

        self.imgs = imgs
        # self.pts1 = pts1.astype(int)
        self.pts1 = pts1
        self.gt = gt

        self.transforms = transforms
        self.loader = loader
        self.return_paths = return_paths
        self.img_only = img_only
        self.samplePatch = samplePatch
        self.n_patch = n_patch
        self.patch_size = patch_size

    def __getitem__(self, item):
        paths = self.imgs[item, :]
        paths = [os.path.join(self.I_dir, path) for path in paths]
        imgs = [self.loader(path) for path in paths]

        # image transformation
        if self.transforms is not None:
            for transformer in self.transforms:
                imgs = [transformer.transform(img) for img in imgs]

        # return images and bounding boxes of whole image.
        if self.img_only:
            c, h, w = imgs[0].shape
            pt1 = np.asarray([0, 0, w, 0, w, h, 0, h]).astype(np.float32)
            if self.return_paths:
                return imgs, pt1, paths
            else:
                return imgs, pt1

        # choose bounding boxes vertex points.
        if not self.samplePatch is None:
            c, h, w = imgs[0].shape
            pt1 = select_patch([h, w], self.patch_size, self.n_patch, mode=self.samplePatch)
        else:
            pt1 = self.pts1[item, ]
            pt1 = np.reshape(pt1, (-1, 8))

        # extract patches from image.
        patches = []
        for i in range(pt1.shape[0]):
            for img in imgs:
                patches.append(img[:, int(pt1[i, 1]):int(pt1[i, 5]), int(pt1[i, 0]):int(pt1[i, 4])])
        pt1 = np.reshape(pt1, (-1))

        # choose return variable.
        if self.return_paths:
            if self.gt is None:
                return imgs, patches, pt1, paths
            else:
                gt = self.gt[item,]
                return imgs, patches, pt1, gt, paths
        else:
            if self.gt is None:
                return imgs, patches, pt1
            else:
                gt = self.gt[item,]
                return imgs, patches, pt1, gt

    def __len__(self):
        return len(self.imgs)

    def name(self):
        return 'HomographyDataset'

    def get_patches(self, imgs, pts):
        patches = [imgs[i:i+1, ][:, :, int(pts.data[i, 1]):int(pts.data[i, 5]), int(pts.data[i, 0]):int(pts.data[i, 4])] for i in range(imgs.size(0))]
        patches = torch.cat(patches, 0)
        return patches

    def get_pts_indices(self, pts1, width, height):
        lux = int(pts1[0])
        luy = int(pts1[1])
        brx = int(pts1[4])
        bry = int(pts1[5])


from torch.nn import functional as F

class GradCAM_Patch_Loader(object):
    def __init__(self, gradcam_net, dataloader, patch_size=128):
        super(GradCAM_Patch_Loader).__init__()
        self.gradcam_net = gradcam_net
        self.dataloader = dataloader
        self.patch_size = patch_size
        self.pool = torch.nn.AvgPool2d(kernel_size=patch_size, stride=1)
        self.div_size = 2 ** self.gradcam_net.num_downs
        self.patchsize = patch_size

    def __iter__(self):
        for i_batch, sample_batched in enumerate(self.dataloader):
            images = sample_batched[0]
            with torch.no_grad():
                input = torch.cat(images, dim=1).cuda()
                input_h = input.size(2)
                input_w = input.size(3)
                # input = F.pad(input, [0, self.div_size - 1 - (input_w - 1) % self.div_size,
                #                       0, self.div_size - 1 - (input_h - 1) % self.div_size], mode='constant', value=0)
                input = input[:, :, 0:input_h // self.div_size * self.div_size, 0:input_w // self.div_size * self.div_size]
                output = self.gradcam_net(input)
                output = self.gradcam_net.normalize(output)
                score = self.pool(output)
                score_size = score.size()
                score = score.view(score_size[0] * score_size[1], -1)
                max_score_idx = torch.argmax(score, dim=1, keepdim=True)
                idx1 = max_score_idx // score_size[-1]
                idx2 = max_score_idx % score_size[-1]
                score = score.view(score_size)
                temp1 = torch.max(score)
                temp2 = score[0, 0, idx1, idx2]
                lu_x = idx2
                lu_y = idx1
                ru_x = lu_x + self.patchsize
                ru_y = lu_y
                rl_x = lu_x + self.patchsize
                rl_y = lu_y + self.patchsize
                ll_x = lu_x
                ll_y = lu_y + self.patchsize
                pt = torch.cat((lu_x, lu_y, ru_x, ru_y, rl_x, rl_y, ll_x, ll_y), 1)
                patches = [self.gradcam_net.homography_net.get_patches(imgs=image, pts=pt) for image in images]

            sample_batched_new = []
            sample_batched_new.append(sample_batched[0])
            sample_batched_new.append(patches)
            sample_batched_new.append(pt)
            sample_batched_new.append(sample_batched[-1])
            yield sample_batched_new

    def __len__(self):
        return len(self.dataloader)
