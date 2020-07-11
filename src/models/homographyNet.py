from __future__ import absolute_import
from __future__ import division

import time
from ..datasets.image_helpers import *
import os
from .vgg import *
import kornia

from .. import func as func
from .unet import *
from ..losses import *
from ..visualization import GradCAM

class HomographyNet(nn.Module):
    def __init__(self, config, mode=0):
        """ Homography matrix estimation network

            Args:
                mode (int): If 0, network return the homography matrix
                            If 1, network return the transformed images
            """
        super(HomographyNet, self).__init__()
        self.backbone = vgg_homography()
        self.dropout = nn.Dropout2d(p=0.5, inplace=False)
        self.regressor = nn.Sequential(
            nn.Linear(128 * 16 * 16, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 8),
        )
        self.mode = mode
        self.src_images = None
        self.dst_imgs = None
        self.src_points = None
        self.dst_points = None
        self.src_patches = None
        self.dst_patches = None
        self.offset = None
        self.M = None
        self.warped_dst_imgs = None
        self.warped_dst_patches = None
        self.loss = None
        self.config = None
        self.loss_list = None
        self.image_feature = None


    def forward(self, src, dst, src_points, scale=1.):
        self.src_images = src
        self.dst_imgs = dst
        self.src_points = src_points
        self.offset = self.offset_estimation(src, dst)
        self.cal_homography_from_offset(self.offset, scale)
        return self.M

    def cal_homography_from_offset(self, offset, scale=1.):
        self.dst_points = self.src_points + scale * offset
        self.dst_points = self.dst_points.view(-1, 4, 2)
        self.src_points = self.src_points.view(-1, 4, 2)
        self.M = self.DLT(self.src_points, self.dst_points)
        self.dst_points = self.dst_points.view(-1, 8)
        self.src_points = self.src_points.view(-1, 8)
        return self.M

    def offset_estimation(self, src, dst, x=None):
        if x is None:
            x = torch.cat([src, dst], 1)
        x = self.backbone(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        self.image_feature = x
        x = self.regressor(x)
        return x

    def build_supervised_loss(self, x, y):
        return nn.MSELoss(x, y)

    def build_unsupervised_loss(self, x):
        return nn.MSELoss(x[:, :1, :, :], x[:, 1:, :, :])

    def DLT(self, src, dst):
        """ Modified from kornia
            Calculates a perspective transform from four pairs of the corresponding
            points.

            The function calculates the matrix of a perspective transform so that:

            .. math ::

                \begin{bmatrix}
                t_{i}x_{i}^{'} \\
                t_{i}y_{i}^{'} \\
                t_{i} \\
                \end{bmatrix}
                =
                \textbf{map_matrix} \cdot
                \begin{bmatrix}
                x_{i} \\
                y_{i} \\
                1 \\
                \end{bmatrix}

            where

            .. math ::
                dst(i) = (x_{i}^{'},y_{i}^{'}), src(i) = (x_{i}, y_{i}), i = 0,1,2,3

            Args:
                src (Tensor): coordinates of quadrangle vertices in the source image.
                dst (Tensor): coordinates of the corresponding quadrangle vertices in
                    the destination image.

            Returns:
                Tensor: the perspective transformation.

            Shape:
                - Input: :math:`(B, 4, 2)` and :math:`(B, 4, 2)`
                - Output: :math:`(B, 3, 3)`
            """
        if not torch.is_tensor(src):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(src)))
        if not torch.is_tensor(dst):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(dst)))
        if not src.shape[-2:] == (4, 2):
            raise ValueError("Inputs must be a Bx4x2 tensor. Got {}"
                             .format(src.shape))
        if not src.shape == dst.shape:
            raise ValueError("Inputs must have the same shape. Got {}"
                             .format(dst.shape))
        if not (src.shape[0] == dst.shape[0]):
            raise ValueError("Inputs must have same batch size dimension. Got {}"
                             .format(src.shape, dst.shape))

        def ax(p, q):
            ones = torch.ones_like(p)[..., 0:1]
            zeros = torch.zeros_like(p)[..., 0:1]
            return torch.cat(
                [p[:, 0:1], p[:, 1:2], ones, zeros, zeros, zeros,
                 -p[:, 0:1] * q[:, 0:1], -p[:, 1:2] * q[:, 0:1]
                 ], dim=1)

        def ay(p, q):
            ones = torch.ones_like(p)[..., 0:1]
            zeros = torch.zeros_like(p)[..., 0:1]
            return torch.cat(
                [zeros, zeros, zeros, p[:, 0:1], p[:, 1:2], ones,
                 -p[:, 0:1] * q[:, 1:2], -p[:, 1:2] * q[:, 1:2]], dim=1)

        # we build matrix A by using only 4 point correspondence. The linear
        # system is solved with the least square method, so here
        # we could even pass more correspondence
        p = []
        p.append(ax(src[:, 0], dst[:, 0]))
        p.append(ay(src[:, 0], dst[:, 0]))

        p.append(ax(src[:, 1], dst[:, 1]))
        p.append(ay(src[:, 1], dst[:, 1]))

        p.append(ax(src[:, 2], dst[:, 2]))
        p.append(ay(src[:, 2], dst[:, 2]))

        p.append(ax(src[:, 3], dst[:, 3]))
        p.append(ay(src[:, 3], dst[:, 3]))

        # A is Bx8x8
        A = torch.stack(p, dim=1)

        # b is a Bx8x1
        b = torch.stack([
            dst[:, 0:1, 0], dst[:, 0:1, 1],
            dst[:, 1:2, 0], dst[:, 1:2, 1],
            dst[:, 2:3, 0], dst[:, 2:3, 1],
            dst[:, 3:4, 0], dst[:, 3:4, 1],
        ], dim=1)

        # solve the system Ax = b
        # A_LU = torch.btrifact(A)
        # X = torch.btrisolve(b, *A_LU)
        X, LU = torch.gesv(b, A)

        # X, LU = torch.solve(b, A) # This is supported only by Pytorch >

        # create variable to return
        batch_size = src.shape[0]
        M = torch.ones(batch_size, 9, device=src.device, dtype=src.dtype)
        M[..., :8] = torch.squeeze(X, dim=-1)
        return M.view(-1, 3, 3)  # Bx3x3

    def stn(self, src_imgs, M):
        img_warp = kornia.warp_perspective(src_imgs, M, dsize=(src_imgs.size(2), src_imgs.size(3)))
        return img_warp

    def load_weights(self, model_file):
        self.load_state_dict(torch.load(model_file)['model'])

    def init_weights(self):
        self.apply(func.weights_init_He_normal)

    def get_patches(self, imgs, pts):
        # temp = imgs[0:1, ]
        # pts = pts.astype(torch.cuda.IntegerTensor)
        patches = [
            imgs[i:i + 1, ][:, :, int(pts.data[i, 1]):int(pts.data[i, 5]), int(pts.data[i, 0]):int(pts.data[i, 4])] for
            i in range(int(imgs.size(0)))]
        patches = torch.cat(patches, 0)
        return patches

    def tensor_to_GPU(self, datas):
        return_data = []
        for data in datas:
            data = Variable(data).type(dtype)
            return_data.append(data)
        return tuple(return_data)

    def prepare_input(self, sample_batched):
        img1 = sample_batched[0][0]
        img2 = sample_batched[0][1]
        patch1 = sample_batched[1][0]
        patch2 = sample_batched[1][1]
        pt1 = sample_batched[2]

        if not img1.is_cuda:
            img1, img2, patch1, patch2, pt1 = self.tensor_to_GPU(datas=(img1, img2, patch1, patch2, pt1))
        self.src_images = img1
        self.dst_imgs = img2
        self.src_points = pt1
        self.src_patches = patch1
        self.dst_patches = patch2

        return img1, img2, patch1, patch2, pt1

    def prepare_input_img(self, sample_batched):
        img1 = sample_batched[0][0]
        img2 = sample_batched[0][1]
        pt1 = sample_batched[1]

        img1, img2, pt1 = self.tensor_to_GPU(datas=(img1, img2, pt1))
        return img1, img2, pt1


    def warp_img(self, img, pt, M):
        warped_img2 = self.stn(img, M)
        warped_patch2 = self.get_patches(warped_img2, pt)
        return warped_img2, warped_patch2

    def run(self, sample_batched, scale=1.):
        img1, img2, patch1, patch2, pt1 = self.prepare_input(sample_batched)
        M = self(patch1, patch2, pt1, scale)
        self.M = M
        warped_img2, warped_patch2 = self.warp_img(img2, pt1, M)
        pt2 = self.dst_points
        return img1, img2, patch1, patch2, warped_img2, warped_patch2, pt1, pt2

    def tensor2numpy(self, datas):
        result = []
        for data in datas:
            result.append(data.detach().cpu().numpy())
        return tuple(result)

    def test(self, dataloader, iterations=0, out_img=True, out_html=True):
        s_time = time.time()
        self.eval()
        whole_img_loss = []
        patch_loss = []
        H_mat = []
        count = 0

        start_time = time.time()
        for i_batch, sample_batched in enumerate(dataloader):
            img1, img2, patch1, patch2, warped_img2, warped_patch2, pt1, pt2 = self.run(sample_batched)
            whole_img_loss.append(float(self.loss(img1, warped_img2)))
            patch_loss.append(float(self.loss(patch1, warped_patch2)))

            if self.mode == 1:
                if i_batch == 0:
                    img1, img2, patch1, patch2, warped_img2, warped_patch2, pt1, pt2 = self.tensor2numpy((img1, img2, patch1, patch2, warped_img2, warped_patch2, pt1, pt2))
                    pts1 = pt1.reshape(-1, 4, 2)
                    pred_pts2 = pt2.reshape(-1, 4, 2)

                    if out_img:
                        for transformer in dataloader.dataset.transforms[::-1]:
                            img1 = transformer.reverse(img1)
                            img2 = transformer.reverse(img2)
                        corr_img = draw_correspondences_img(img1[0], img2[0], pts1[0], pts1[0], pred_pts2[0])
                        save_img_name = 'test-iter-%s.jpg' % (str(iterations))
                        self.config.img_logger.log(corr_img, save_img_name)
                    if out_html:
                        html_img_dir = os.path.join('.', os.path.basename(self.config.results_dir))
                        self.config.html_logger.log([os.path.join(html_img_dir, save_img_name)], ['correspondences picture'], iterations)
            else:
                img1, img2, patch1, patch2, warped_img2, warped_patch2, pt1, pt2, M = self.tensor2numpy(
                    (img1, img2, patch1, patch2, warped_img2, warped_patch2, pt1, pt2, self.M))

                if out_img:
                    for transformer in dataloader.dataset.transforms[::-1]:
                        img1 = transformer.reverse(img1)
                        img2 = transformer.reverse(img2)
                    pts1 = pt1.reshape(-1, 4, 2)
                    pred_pts2 = pt2.reshape(-1, 4, 2)
                    for i in range(pt1.shape[0]):
                        corr_img = draw_correspondences_img(img1[i], img2[i], pts1[i], pts1[i], pred_pts2[i])
                        save_img_name = 'test-figure-%s.jpg' % (str(count))
                        self.config.img_logger.log(corr_img, save_img_name)
                        if out_html:
                            html_img_dir = os.path.join('.', os.path.basename(self.config.results_dir))
                            self.config.html_logger.log([os.path.join(html_img_dir, save_img_name)],
                                                    ['correspondences picture'], count)
                for i in range(pt1.shape[0]):
                    output_line = 'Finished for figure: '
                    for path_list in sample_batched[-1]:
                        output_line += '\t' + os.path.basename(path_list[i])
                    self.config.txt_logger.info(output_line)
                    count += 1
                end_time = time.time()
                self.config.txt_logger.info("Test time for the batch: %s s" % (str(end_time - start_time)))
                start_time = time.time()
                H_mat.append(scale_homography_mat(M, self.config.scale))

        self.train()
        e_time = time.time()

        if self.mode == 0:
            H_mat_np = np.concatenate(H_mat, axis=0)
            if hasattr(self.config, 'h_mat_file'):
                h_mat_file = self.config.h_mat_file
            else:
                h_mat_file = self.config.log_dir + '/h_matrices_' + self.config.exp_name + '_' + self.config.loss_type + '_' + self.config.patch_select + '_' + str(self.config.scale) + '.mat'
            sio.savemat(h_mat_file, {'H_mat': H_mat_np})
            if out_html:
                self.config.html_logger.flush('Test image')
        else:
            if out_html:
                self.config.html_logger.flush('Iterations')
        return [np.mean(whole_img_loss), np.mean(patch_loss)], ['whole_img_loss', 'patch_loss'], e_time - s_time

    def parameters_train(self):
        return self.parameters()

    def get_loss(self, sample_batched):
        img1, img2, patch1, patch2, warped_img2, warped_patch2, pt1, pt2 = self.run(sample_batched)
        whole_img_loss = self.loss(img1, warped_img2)
        patch_loss = self.loss(patch1, warped_patch2)

        return [patch_loss, whole_img_loss], ['patch_loss', 'whole_img_loss']


class HomographyNet_TSS(HomographyNet):
    def __init__(self, config, mode=0, lamda1=(1, 1), lamda2=(1, 1),lamda3=(1, 0.5, 0.5), scales=[2]):
        """
        HomographyNet with temporal, spatial, scale regularization.
        :param config:
        :param mode:
        :param lamda1: coefficient on temporal homography regularization.
        :param lamda2: coefficient on spatial homography regularization.
        :param lamda3: coefficient on scale homography regularization.
        :param scale: scale for scale regularization
        """
        super(HomographyNet_TSS, self).__init__(config, mode=mode)
        self.lamda1 = lamda1
        self.lamda2 = lamda2
        self.lamda3 = lamda3
        self.scales = scales
        if not scales is None:
            self.upsamples = [nn.Upsample(scale_factor=scale, mode='bilinear') for scale in scales]

    def get_loss(self, sample_batched):
        n_img = len(sample_batched[0])
        n_patches = len(sample_batched[1])
        if any(x > 0 for x in self.lamda1):
            assert n_img > 2

        assert n_patches % n_img == 0

        n_patch_per_img = n_patches // n_img
        if any(x > 0 for x in self.lamda2):
            assert n_patch_per_img > 1

        whole_img_loss_list = []
        patch_img_loss_list = []

        whole_img_loss = torch.tensor(0).type(dtype)
        patch_img_loss = torch.tensor(0).type(dtype)

        M_list = []
        patch1_list = []
        img2_list = []
        pt1_list = []
        warped_img2_list = []

        # Calculate Homography for each adjacent frame pairs.
        for j in range(n_patch_per_img):
            M_list.append([])
            patch1_list.append([])
            img2_list.append([])
            pt1_list.append([])
            warped_img2_list.append([])
            for i in range(n_img - 1):
                sample = [[sample_batched[0][i], sample_batched[0][i + 1]],
                          [sample_batched[1][j*n_img + i], sample_batched[1][j*n_img + i + 1]],
                          sample_batched[2][..., j*8:(j+1)*8]]
                if len(sample_batched) > 3:
                    sample.extend(sample_batched[3:])
                img1, img2, patch1, patch2, warped_img2, warped_patch2, pt1, pt2 = self.run(sample)
                whole_img_loss_list.append(C_Loss(img1, warped_img2))
                patch_img_loss_list.append(C_Loss(patch1, warped_patch2))
                whole_img_loss += whole_img_loss_list[-1]
                patch_img_loss += patch_img_loss_list[-1]
                M_list[-1].append(self.M)
                patch1_list[-1].append(patch1)
                img2_list[-1].append(img2)
                pt1_list[-1].append(pt1)
                warped_img2_list[-1].append(warped_img2)

        whole_img_loss /= n_patch_per_img * (n_img - 1)
        patch_img_loss /= n_patch_per_img * (n_img - 1)

        # Temporal regularization.
        m_loss1 = torch.tensor(0).type(dtype)
        patch_img_loss_T = torch.tensor(0).type(dtype)
        patch0 = None
        M0 = None
        if any(x > 0 for x in self.lamda1):
            for j in range(n_patch_per_img):
                for i in range(n_img - 1):
                    if i < n_img - 2:
                        m_loss1 += l1_loss(M_list[j][i], M_list[j][i + 1])
                    if i == 0:
                        patch0 = patch1_list[j][i]
                        M0 = M_list[j][i]
                    else:
                        M0 = torch.matmul(M0, M_list[j][i])
                        warped_img2_0 = self.stn(img2_list[j][i], M0)
                        warped_patch2_0 = self.get_patches(warped_img2_0, pt1_list[j][i])
                        patch_img_loss_T += C_Loss(patch0, warped_patch2_0)
            m_loss1 /= n_patch_per_img * (n_img - 2)
            patch_img_loss_T /= n_patch_per_img * (n_img - 2)

        # Spatial regularization.
        whole_img_loss_Sp = torch.tensor(0).type(dtype)
        m_loss2 = torch.tensor(0).type(dtype)
        if any(x > 0 for x in self.lamda2):
            patch_h = sample_batched[1][0].size(2)
            patch_w = sample_batched[1][0].size(3)
            img_h = sample_batched[0][0].size(2)
            img_w = sample_batched[0][0].size(3)
            for i in range(n_img - 1):
                for j in range(n_patch_per_img):
                    for k in range(j+1, n_patch_per_img):
                        m_loss2 += l1_loss(M_list[j][i], M_list[k][i])
                        whole_img_loss_Sp += C_Loss(warped_img2_list[j][i], warped_img2_list[k][i])
            m_loss2 /= n_patch_per_img * (n_patch_per_img - 1) / 2 * (n_img - 1)
            whole_img_loss_Sp /= n_patch_per_img * (n_patch_per_img - 1) / 2 * (n_img - 1) * \
                                 (img_h * img_w / (patch_h * patch_w))

        # Scale regularization
        patch_img_loss_Sc = torch.tensor(0).type(dtype)
        whole_img_loss_Sc = torch.tensor(0).type(dtype)
        m_loss3 = torch.tensor(0).type(dtype)
        if any(x > 0 for x in self.lamda3):
            patch_h = sample_batched[1][0].size(2)
            patch_w = sample_batched[1][0].size(3)
            img_h = sample_batched[0][0].size(2)
            img_w = sample_batched[0][0].size(3)
            scale_M_list = []
            scale_warped_img2_list = []
            for s in range(len(self.scales)):
                scale_M_list.append([])
                scale_warped_img2_list.append([])
                scale = self.scales[s]
                stride_h = patch_h // scale
                stride_w = patch_w // scale
                # for k in range(n_patch_per_img):
                k = 0
                scale_M_list[-1].append([])
                scale_warped_img2_list[-1].append([])
                for i in range(n_img - 1):
                    idx_x = np.random.randint(0, patch_h - stride_h)
                    idx_y = np.random.randint(0, patch_w - stride_w)
                    sub_sample = [[sample_batched[0][i], sample_batched[0][i + 1]]]
                    patch1 = sample_batched[1][k*n_img + i][..., idx_x:idx_x+stride_h, idx_y:idx_y+stride_w]
                    patch1 = self.upsamples[s](patch1)
                    patch2 = sample_batched[1][k*n_img + i + 1][..., idx_x:idx_x+stride_h, idx_y:idx_y+stride_w]
                    patch2 = self.upsamples[s](patch2)
                    pt1 = sample_batched[2][..., k*8:(k+1)*8].clone()
                    pt1[..., 0] += idx_y
                    pt1[..., 1] += idx_x
                    pt1[..., 4] = pt1[..., 0] + stride_w
                    pt1[..., 5] = pt1[..., 1] + stride_h

                    pt1[..., 2] = pt1[..., 4]
                    pt1[..., 3] = pt1[..., 1]
                    pt1[..., 6] = pt1[..., 0]
                    pt1[..., 7] = pt1[..., 5]

                    sub_sample.append([patch1, patch2])
                    sub_sample.append(pt1)
                    img1, img2, patch1, patch2, warped_img2, warped_patch2, pt1, pt2 = self.run(sub_sample, scale=1./self.scales[s])
                    warped_patch2 = self.upsamples[s](warped_patch2)
                    patch_img_loss_Sc += C_Loss(patch1, warped_patch2)
                    scale_M_list[-1][-1].append(self.M)
                    scale_warped_img2_list[-1][-1].append(warped_img2)

            for i in range(n_img - 1):
                # for j in range(n_patch_per_img):
                j = 0
                for s in range(len(self.scales)):
                    m_loss3 += l1_loss(M_list[j][i], scale_M_list[s][j][i])
                    whole_img_loss_Sc += C_Loss(warped_img2_list[j][i], scale_warped_img2_list[s][j][i])
            m_loss3 /= (n_img - 1) * (n_patch_per_img) * len(self.scales)
            # patch_img_loss_Sc /= (n_img - 1) * (n_patch_per_img) * len(self.scales)
            # whole_img_loss_Sc /= (n_img - 1) * (n_patch_per_img) * len(self.scales)* \
            #                      (img_h * img_w / (patch_h * patch_w))
            patch_img_loss_Sc /= (n_img - 1) * len(self.scales)
            whole_img_loss_Sc /= (n_img - 1) * len(self.scales)* \
                                 (img_h * img_w / (patch_h * patch_w))

        total_loss = patch_img_loss + self.lamda1[0] * m_loss1 + self.lamda2[0] * m_loss2 + self.lamda3[0] * m_loss3 + \
                     self.lamda1[1] * patch_img_loss_T + self.lamda2[1] * whole_img_loss_Sp + \
                     self.lamda3[1] * patch_img_loss_Sc + self.lamda3[2] * whole_img_loss_Sc
        return [total_loss, patch_img_loss, whole_img_loss, m_loss1, m_loss2, m_loss3, patch_img_loss_T,
                whole_img_loss_Sp, patch_img_loss_Sc, whole_img_loss_Sc], \
               ['total_loss', 'patch_img_loss', 'whole_img_loss', 'M_loss_T', 'M_loss_S', 'M_loss_Sp',
                'patch_img_loss_T', 'whole_img_loss_Sp', 'patch_img_loss_Sc', 'whole_img_loss_Sc']



class HomographyNet_LSTM(HomographyNet):
    def __init__(self, config, mode=0, lstm_layers=1, lstm_bidirect=False, lstm_dropout=0.5, lstm_features=1024, lamda1=0, lamda2=0):
        super(HomographyNet_LSTM, self).__init__(config, mode)
        self.lstm_layers = lstm_layers
        self.lstm_bidirect = lstm_bidirect
        self.lstm_dropout = lstm_dropout
        self.lstm_features = lstm_features

        self.lstm = nn.LSTM(input_size=1024, hidden_size=lstm_features, num_layers=lstm_layers, bias=True, batch_first=False, dropout=lstm_dropout, bidirectional=lstm_bidirect)
        self.regressor = nn.Sequential(
            nn.Linear(128 * 16 * 16, 1024),
        )
        self.lstm_regressor = nn.Sequential(
            nn.Dropout(),
            nn.Linear(lstm_features, 8),
        )

        self.image_feature = None
        self.images = None
        self.patches = None
        self.lamda1 = lamda1
        self.lamda2 = lamda2

    def forward(self, src, dst, src_points, scale=1.):
        self.src_images = src
        self.dst_imgs = dst
        self.src_points = src_points
        self.offset = self.offset_estimation(src, dst)
        self.cal_homography_from_offset(self.offset, scale)
        return self.M


    def prepare_input(self, sample_batched):
        imgs = sample_batched[0]
        patches = sample_batched[1]
        pt1 = sample_batched[2]

        imgs = self.tensor_to_GPU(datas=imgs)
        patches = self.tensor_to_GPU(datas=patches)

        pt1 = self.tensor_to_GPU(datas=[pt1])
        pt1 = pt1[0]
        self.images = imgs
        self.patches = patches
        self.src_points = pt1

        return imgs, patches, pt1

    def get_features(self, src, dst):
        x = None
        with torch.no_grad():
            x = torch.cat([src, dst], 1)
            x = self.backbone(x)
            x = x.view(x.size(0), -1)
            x = self.regressor(x)
            self.image_feature = x
        return x

    def get_homography(self, features, scale=1.):
        self.offset = self.lstm_regressor(features)
        M = self.cal_homography_from_offset(self.offset, scale)
        return M

    def run(self, sample_batched, scale=1., h0=None, c0=None):
        imgs, patches, pt1 = self.prepare_input(sample_batched)
        n_data = len(patches)
        features = []
        for i in range(n_data - 1):
            features.append(self.get_features(patches[i], patches[i+1]))
        features = torch.stack(features, dim=0)
        features.detach()
        if h0 is None or c0 is None:
            output, (h_n, c_n) = self.lstm(features)
        else:
            output, (h_n, c_n) = self.lstm(features, (h0, c0))
        M_list = []
        warped_imgs = []
        warped_patches = []
        pts2 = []
        for i in range(n_data - 1):
            M = self.get_homography(output[i, :, :])
            warped_img2, warped_patch2 = self.warp_img(imgs[i + 1], pt1, M)
            warped_imgs.append(warped_img2)
            warped_patches.append(warped_patch2)
            M_list.append(M)
            pts2.append(self.dst_points)
        self.M = M_list
        self.dst_points = pts2
        return imgs, patches, warped_imgs, warped_patches, pt1, pts2, M_list, h_n, c_n

    def parameters_train(self):
        # return self.lstm.parameters() + self.lstm_regressor.parameters()
        return self.parameters()

    def get_loss(self, sample_batched):
        n_img = len(sample_batched[0])
        assert n_img > 2
        whole_img_loss_list = []
        patch_img_loss_list = []

        whole_img_loss = torch.tensor(0).type(dtype)
        patch_img_loss = torch.tensor(0).type(dtype)
        patch_img_loss2 = torch.tensor(0).type(dtype)
        m_loss = torch.tensor(0).type(dtype)


        imgs, patches, warped_imgs, warped_patches, pt1, pts2, M_list, h_n, c_n = self.run(sample_batched)
        for i in range(n_img - 1):
            whole_img_loss_list.append(C_Loss(imgs[i], warped_imgs[i]))
            patch_img_loss_list.append(C_Loss(patches[i], warped_patches[i]))
            whole_img_loss += whole_img_loss_list[-1]
            patch_img_loss += patch_img_loss_list[-1]

        whole_img_loss /= n_img - 1
        patch_img_loss /= n_img - 1

        if self.lamda1 != 0:
            patch0 = None
            for i in range(n_img - 1):
                if i == 0:
                    patch0 = patches[i]
                    M0 = M_list[i]
                else:
                    M0 = torch.matmul(M0, M_list[i])
                    warped_img2_0 = self.stn(imgs[i + 1], M0)
                    warped_patch2_0 = self.get_patches(warped_img2_0, pt1)
                    patch_img_loss2 += C_Loss(patch0, warped_patch2_0)
            patch_img_loss2 /= n_img - 2

        if self.lamda2 != 0:
            for i in range(n_img - 2):
                m_loss += l1_loss(M_list[i], M_list[i + 1])
            m_loss /= n_img - 2

        total_loss = patch_img_loss + patch_img_loss2 * self.lamda1 + m_loss * self.lamda2
        return [total_loss, patch_img_loss, patch_img_loss2, whole_img_loss, m_loss], ['total_loss',
                                                                                       'mean_patch_img_loss',
                                                                                       'mean_patch_img_loss2',
                                                                                       'mean_whole_img_loss', 'M_loss']
    def test(self, dataloader, iterations=0, out_img=True, out_html=True):
        s_time = time.time()
        self.eval()
        whole_img_loss = []
        patch_loss = []
        H_mat = []
        count = 0

        start_time = time.time()
        h0 = None
        c0 = None
        for i_batch, sample_batched in enumerate(dataloader):
            imgs, patches, warped_imgs, warped_patches, pt1, pts2, M_list, h_n, c_n = self.run(sample_batched, h0=h0, c0=c0)
            h0 = h_n.detach()
            c0 = c_n.detach()
            for i in range(len(imgs)-1):
                whole_img_loss.append(float(self.loss(imgs[i], warped_imgs[i])))
                patch_loss.append(float(self.loss(patches[i], warped_patches[i])))

            if self.mode == 1:
                if i_batch == 0:
                    if out_img:
                        imgs = self.tensor2numpy(imgs)
                        # patches = self.tensor2numpy(patches)
                        # warped_imgs = self.tensor2numpy(warped_imgs)
                        # warped_patches = self.tensor2numpy(warped_patches)

                        pt1 = self.tensor2numpy([pt1])
                        pt1 = pt1[0]
                        pts1 = pt1.reshape(-1, 4, 2)

                        pts2 = self.tensor2numpy(pts2)
                        pred_pts2 = pts2[0].reshape(-1, 4, 2)
                        img1 = imgs[0]
                        img2 = imgs[1]

                        for transformer in dataloader.dataset.transforms[::-1]:
                            img1 = transformer.reverse(img1)
                            img2 = transformer.reverse(img2)
                        corr_img = draw_correspondences_img(img1[0], img2[0], pts1[0], pts1[0], pred_pts2[0])
                        save_img_name = 'test-iter-%s.jpg' % (str(iterations))
                        self.config.img_logger.log(corr_img, save_img_name)
                    if out_html:
                        html_img_dir = os.path.join('.', os.path.basename(self.config.results_dir))
                        self.config.html_logger.log([os.path.join(html_img_dir, save_img_name)], ['correspondences picture'], iterations)
            else:
                imgs = self.tensor2numpy(imgs)
                # patches = self.tensor2numpy(patches)
                # warped_imgs = self.tensor2numpy(warped_imgs)
                # warped_patches = self.tensor2numpy(warped_patches)
                M_list = self.tensor2numpy(M_list)

                pt1 = self.tensor2numpy([pt1])
                pt1 = pt1[0]
                pts1 = pt1.reshape(-1, 4, 2)

                pts2 = self.tensor2numpy(pts2)
                pred_pts2 = pts2[0].reshape(-1, 4, 2)

                img1 = imgs[0]
                img2 = imgs[1]
                if out_img:
                    for transformer in dataloader.dataset.transforms[::-1]:
                        img1 = transformer.reverse(img1)
                        img2 = transformer.reverse(img2)

                    for i in range(pt1.shape[0]):
                        corr_img = draw_correspondences_img(img1[i], img2[i], pts1[i], pts1[i], pred_pts2[i])
                        save_img_name = 'test-figure-%s.jpg' % (str(count))
                        self.config.img_logger.log(corr_img, save_img_name)
                        if out_html:
                            html_img_dir = os.path.join('.', os.path.basename(self.config.results_dir))
                            self.config.html_logger.log([os.path.join(html_img_dir, save_img_name)],
                                                    ['correspondences picture'], count)
                for i in range(pt1.shape[0]):
                    output_line = 'Finished for figure: '
                    for path_list in sample_batched[-1]:
                        output_line += '\t' + os.path.basename(path_list[i])
                    self.config.txt_logger.info(output_line)
                    count += 1
                end_time = time.time()
                self.config.txt_logger.info("Test time for the batch: %s s" % (str(end_time - start_time)))
                start_time = time.time()
                for M in M_list:
                    H_mat.append(scale_homography_mat(M, self.config.scale))

        self.train()
        e_time = time.time()

        if self.mode == 0:
            H_mat_np = np.concatenate(H_mat, axis=0)
            if hasattr(self.config, 'h_mat_file'):
                h_mat_file = self.config.h_mat_file
            else:
                h_mat_file = self.config.log_dir + '/h_matrices_' + self.config.exp_name + '_' + self.config.loss_type + '_' + self.config.patch_select + '_' + str(self.config.scale) + '.mat'
            sio.savemat(h_mat_file, {'H_mat': H_mat_np})
            if out_html:
                self.config.html_logger.flush('Test image')
        else:
            if out_html:
                self.config.html_logger.flush('Iterations')
        return [np.mean(whole_img_loss), np.mean(patch_loss)], ['whole_img_loss', 'patch_loss'], e_time - s_time



class HomographyNet_LSTM_TSSReg(HomographyNet_LSTM):
    def __init__(self, config, mode=0, lstm_layers=1, lstm_bidirect=False, lstm_dropout=0.5, lstm_features=1024, lamda1=(0, 0),lamda2=0, lamda3=0, scales=(2)):
        super(HomographyNet_LSTM_TSSReg, self).__init__(config, mode, lstm_layers=lstm_layers, lstm_bidirect=lstm_bidirect,lstm_dropout=lstm_dropout, lstm_features=lstm_features, lamda2=lamda2)
        self.lamda3 = lamda3
        self.lamda1 = lamda1
        self.scales = scales

        if not scales is None:
            self.upsamples = [nn.Upsample(scale_factor=scale, mode='bilinear') for scale in scales]

    def get_loss(self, sample_batched):
        n_img = len(sample_batched[0])
        n_patches = len(sample_batched[1])
        if any(x > 0 for x in self.lamda1):
            assert n_img > 2

        assert n_patches % n_img == 0

        n_patch_per_img = n_patches // n_img
        # if any(x > 0 for x in self.lamda2):
        if self.lamda2 > 0:
            assert n_patch_per_img > 1

        whole_img_loss_list = []
        patch_img_loss_list = []

        whole_img_loss = torch.tensor(0).type(dtype)
        patch_img_loss = torch.tensor(0).type(dtype)

        M_list_all = []
        patch1_list = []
        img2_list = []
        pt1_list = []
        warped_img2_list = []

        # Run network on the batch
        imgs = None
        for j in range(n_patch_per_img):
            sample = [sample_batched[0],
                      sample_batched[1][j*n_img:(j+1)*n_img],
                      sample_batched[2][..., j*8:(j+1)*8]]
            if len(sample_batched) > 3:
                sample.extend(sample_batched[3:])
            imgs, patches, warped_imgs, warped_patches, pt1, pts2, M_list, h_n, c_n = self.run(sample)

            for i in range(n_img - 1):
                whole_img_loss_list.append(C_Loss(imgs[i], warped_imgs[i]))
                patch_img_loss_list.append(C_Loss(patches[i], warped_patches[i]))
                whole_img_loss += whole_img_loss_list[-1]
                patch_img_loss += patch_img_loss_list[-1]

            M_list_all.append(M_list)
            pt1_list.append(pt1)

        whole_img_loss /= (n_img - 1) * n_patch_per_img
        patch_img_loss /= (n_img - 1) * n_patch_per_img

        # Temporal regularization.
        m_loss1 = torch.tensor(0).type(dtype)
        patch_img_loss_T = torch.tensor(0).type(dtype)
        if any(x > 0 for x in self.lamda1):
            patch0 = None
            for j in range(n_patch_per_img):
                for i in range(n_img - 1):
                    if i == 0:
                        patch0 = patches[i]
                        M0 = M_list[j][i]
                    else:
                        M0 = torch.matmul(M0, M_list_all[j][i])
                        warped_img2_0 = self.stn(imgs[i + 1], M0)
                        warped_patch2_0 = self.get_patches(warped_img2_0, pt1_list[j])
                        patch_img_loss_T += C_Loss(patch0, warped_patch2_0)
                        m_loss1 += l1_loss(M_list_all[j][i - 1], M_list_all[j][i])
            m_loss1 /= n_patch_per_img * (n_img - 2)
            patch_img_loss_T /= n_patch_per_img * (n_img - 2)



        # Spatial regularization.
        m_loss2 = torch.tensor(0).type(dtype)
        if self.lamda2 > 0:
            patch_h = sample_batched[1][0].size(2)
            patch_w = sample_batched[1][0].size(3)
            img_h = sample_batched[0][0].size(2)
            img_w = sample_batched[0][0].size(3)
            for i in range(n_img - 1):
                for j in range(n_patch_per_img):
                    for k in range(j+1, n_patch_per_img):
                        m_loss2 += l1_loss(M_list_all[j][i], M_list_all[k][i])
            m_loss2 /= n_patch_per_img * (n_patch_per_img - 1) / 2 * (n_img - 1)

        # Scale regularization
        m_loss3 = torch.tensor(0).type(dtype)
        if self.lamda3 > 0:
            patch_h = sample_batched[1][0].size(2)
            patch_w = sample_batched[1][0].size(3)
            scale_M_list = []
            scale_warped_img2_list = []
            for s in range(len(self.scales)):
                scale_warped_img2_list.append([])
                scale = self.scales[s]
                stride_h = patch_h // scale
                stride_w = patch_w // scale
                # for k in range(n_patch_per_img):
                k = 0
                scale_warped_img2_list[-1].append([])

                idx_x = np.random.randint(0, patch_h - stride_h)
                idx_y = np.random.randint(0, patch_w - stride_w)
                sub_sample = [sample_batched[0]]
                sub_patches = []
                for i in range(n_img):
                    patch1 = sample_batched[1][k*n_img + i][..., idx_x:idx_x+stride_h, idx_y:idx_y+stride_w]
                    patch1 = self.upsamples[s](patch1)
                    sub_patches.append(patch1)
                pt1 = sample_batched[2][..., k*8:(k+1)*8].clone()
                pt1[..., 0] += idx_y
                pt1[..., 1] += idx_x
                pt1[..., 4] = pt1[..., 0] + stride_w
                pt1[..., 5] = pt1[..., 1] + stride_h

                pt1[..., 2] = pt1[..., 4]
                pt1[..., 3] = pt1[..., 1]
                pt1[..., 6] = pt1[..., 0]
                pt1[..., 7] = pt1[..., 5]

                sub_sample.append(sub_patches)
                sub_sample.append(pt1)
                imgs, patches, warped_imgs, warped_patches, pt1, pts2, M_list, h_n, c_n = self.run(sub_sample, scale=1./self.scales[s])
                scale_M_list.append(M_list)

            for i in range(n_img - 1):
                # for j in range(n_patch_per_img):
                j = 0
                for s in range(len(self.scales)):
                    m_loss3 += l1_loss(M_list_all[j][i], scale_M_list[s][i])
            m_loss3 /= (n_img - 1) * (n_patch_per_img) * len(self.scales)


        total_loss = patch_img_loss + self.lamda1[0] * m_loss1 + self.lamda2 * m_loss2 + self.lamda3 * m_loss3 + \
                     self.lamda1[1] * patch_img_loss_T
        return [total_loss, patch_img_loss, whole_img_loss, m_loss1, m_loss2, m_loss3, patch_img_loss_T], \
               ['total_loss', 'patch_img_loss', 'whole_img_loss', 'M_loss_T', 'M_loss_S', 'M_loss_Sp',
                'patch_img_loss_T']



class Homography_GradCAM(Unet):
    def __init__(self, config, mode, input_nc=2, output_nc=1, num_downs=5, ngf=64, norm_layer=torch.nn.BatchNorm2d, use_dropout=False):
        super(Homography_GradCAM, self).__init__(input_nc=input_nc, output_nc=output_nc, num_downs=num_downs, ngf=ngf, norm_layer=norm_layer, use_dropout=use_dropout)
        self.homography_net = None
        self.config = config
        self.mode = mode
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.num_downs = num_downs

    def train_parameters(self):
        return self.parameters()

    def set_net(self, net):
        self.homography_net = net
        self.homography_net.eval()

    def test(self, dataloader, iterations=0, out_img=True, out_html=True):
        s_time = time.time()
        self.eval()
        loss = []
        count = 0
        html_img_dir = os.path.join('.', os.path.basename(self.config.results_dir))
        start_time = time.time()
        for i_batch, sample_batched in enumerate(dataloader):
            output, grad_cam_masks, patch1, patch2, gb, gb_grad_cams = self.run(sample_batched)
            loss.append(float(self.loss(output, grad_cam_masks[0])))
            output = self.normalize(output)
            for i in range(len(grad_cam_masks)):
                grad_cam_masks[i] = self.normalize(grad_cam_masks[i])

            if self.mode == 1:
                if i_batch == 0:
                    patch1, patch2, output, gt = self.homography_net.tensor2numpy((patch1, patch2, output, grad_cam_masks[0]))
                    temp = gt - output
                    print('Diff: ' + str(np.max(np.abs((temp)))))
                    save_img_names = []
                    save_img_labels = []
                    if out_img:
                        for transformer in dataloader.dataset.transforms[::-1]:
                            patch1 = transformer.reverse(patch1)
                            patch2 = transformer.reverse(patch2)

                        grad_cam_on_img = GradCAM.show_cam_on_image(patch1[0], output[0, 0])
                        save_img_names.append('test-GradCAM-iter-%s.jpg' % (str(iterations)))
                        save_img_labels.append('output Grad-CAM')
                        self.config.img_logger.log(grad_cam_on_img, save_img_names[-1])

                        grad_cam_on_img = GradCAM.show_cam_on_image(patch1[0], np.zeros_like(output[0, 0]))
                        save_img_names.append('test-GradCAM-zero-iter-%s.jpg' % (str(iterations)))
                        save_img_labels.append('zero Grad-CAM')
                        self.config.img_logger.log(grad_cam_on_img, save_img_names[-1])

                        grad_cam_on_img = GradCAM.show_cam_on_image(patch1[0], gt[0, 0])
                        save_img_names.append('test-GradCAM-GT-%s.jpg' % (str(iterations)))
                        save_img_labels.append('GT Grad-CAM')
                        self.config.img_logger.log(grad_cam_on_img, save_img_names[-1])

                        save_img_names.append('test-patch1_iter-%s.jpg' % (str(iterations)))
                        save_img_labels.append('patch 1')
                        self.config.img_logger.log(patch1[0], save_img_names[-1])

                        save_img_names.append('test-patch2_iter-%s.jpg' % (str(iterations)))
                        save_img_labels.append('patch 2')
                        self.config.img_logger.log(patch2[0], save_img_names[-1])

                        save_img_names.append('test-output_iter-%s.jpg' % (str(iterations)))
                        save_img_labels.append('output')
                        self.config.img_logger.log(output[0, 0], save_img_names[-1])

                        save_img_names.append('test-gt_iter-%s.jpg' % (str(iterations)))
                        save_img_labels.append('ground truth')
                        self.config.img_logger.log(gt[0, 0], save_img_names[-1])
                    if out_html:
                        self.config.html_logger.log([os.path.join(html_img_dir, save_img_name) for save_img_name in save_img_names],
                                                    save_img_labels, iterations)
            else:
                patch1, patch2, output, gt = self.homography_net.tensor2numpy(
                    (patch1, patch2, output, grad_cam_masks[0]))
                if out_img:
                    for transformer in dataloader.dataset.transforms[::-1]:
                        patch1 = transformer.reverse(patch1)
                        patch2 = transformer.reverse(patch2)

                    for i in range(patch1.shape[0]):
                        save_img_names = []
                        save_img_labels = []
                        grad_cam_on_img = GradCAM.show_cam_on_image(patch1[i], output[i, 0])
                        save_img_names.append('test-GradCAM-%s.jpg' % (str(count)))
                        save_img_labels.append('output Grad-CAM')
                        self.config.img_logger.log(grad_cam_on_img, save_img_names[-1])
                        grad_cam_on_img = GradCAM.show_cam_on_image(patch1[i], gt[i, 0])
                        save_img_names.append('test-GradCAM-GT-%s.jpg' % (str(count)))
                        save_img_labels.append('GT Grad-CAM')
                        self.config.img_logger.log(grad_cam_on_img, save_img_names[-1])
                        save_img_names.append('test-patch1-%s.jpg' % (str(count)))
                        save_img_labels.append('patch 1')
                        self.config.img_logger.log(patch1[i], save_img_names[-1])
                        save_img_names.append('test-patch2-%s.jpg' % (str(count)))
                        save_img_labels.append('patch 2')
                        self.config.img_logger.log(patch2[i], save_img_names[-1])
                        save_img_names.append('test-output-%s.jpg' % (str(count)))
                        save_img_labels.append('output')
                        self.config.img_logger.log(output[i, 0], save_img_names[-1])
                        save_img_names.append('test-gt-%s.jpg' % (str(count)))
                        save_img_labels.append('ground truth')
                        self.config.img_logger.log(gt[i, 0], save_img_names[-1])
                        if out_html:
                            self.config.html_logger.log(
                                [os.path.join(html_img_dir, save_img_name) for save_img_name in save_img_names],
                                save_img_labels, iterations)

                for i in range(patch1.shape[0]):
                    output_line = 'Finished for figure: '
                    for path_list in sample_batched[-1]:
                        output_line += '\t' + os.path.basename(path_list[i])
                    self.config.txt_logger.info(output_line)
                    count += 1
                end_time = time.time()
                self.config.txt_logger.info("Test time for the batch: %s s" % (str(end_time - start_time)))
                start_time = time.time()

        self.train()
        e_time = time.time()

        if self.mode == 0:
            self.config.html_logger.flush('Test image')
        else:
            self.config.html_logger.flush('Iterations')
        return [np.mean(loss)], ['C_loss'], e_time - s_time

    def run(self, sample_batched, guided_bp = False):
        grad_cam_masks, patch1, patch2, gb, gb_grad_cams = self.homography_net.run_grad_cam(sample_batched,
                                                                                            guided_bp=guided_bp,
                                                                                            normalize=False)
        for grad_cam_mask in grad_cam_masks:
            grad_cam_mask.detach()
        input = torch.cat((patch1, patch2), dim=1)
        output = self(input)
        # output_size = output.size()
        # output = output.view([output_size[0], -1])
        # output = output - torch.min(output, dim=1, keepdim=True)[0]
        # output = output / torch.max(output, dim =1 , keepdim=True)[0]
        # output = output.view(output_size)
        # output = self.relu(output)
        return output, grad_cam_masks, patch1, patch2, gb, gb_grad_cams

    def get_loss(self, sample_batched, guided_bp = False):
        output, grad_cam_masks, patch1, patch2, gb, gb_grad_cams = self.run(sample_batched)
        # output = self.sigmoid(self(input))
        # output1 = 1 - output
        # class_output = torch.cat((output, output1), dim = 1)
        # print(torch.max(output))
        # print(torch.min(output))
        # print(torch.max(grad_cam_masks[0]))
        # print(torch.min(grad_cam_masks[0]))
        loss = self.loss(output, grad_cam_masks[0])
        return [loss], ['C_loss']
        # return [self.loss(class_output, grad_cam_masks[0])], ['Cross-entropy']

    def normalize(self, x):
        output = x
        output_size = output.size()
        output = output.view([output_size[0], -1])
        output = output - torch.min(output, dim=1, keepdim=True)[0]
        output = output / torch.max(output, dim =1 , keepdim=True)[0]
        output = output.view(output_size)
        return output

class identity_layer(nn.Module):
    def __init__(self, nc):
        super(identity_layer, self).__init__()
        self.nc = nc

    def forward(self, input):
        return input