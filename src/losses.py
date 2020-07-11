import torch.nn as nn
import torch
from .globalVariables import dtype

bce_loss = nn.BCELoss().type(dtype)
mse_loss = nn.MSELoss().type(dtype)
l1_loss = nn.L1Loss()


class Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self, epsilon=1e-3):
        super(Charbonnier_loss, self).__init__()
        self.eps = epsilon ** 2
        # self.eps = Variable(torch.from_numpy(np.asarray([epsilon ** 2])))
        # self.eps = Variable(torch.ones())


    def forward(self, X, Y):
        batchsize = X.data.shape[0]
        diff = X - Y
        square_err = diff ** 2
        square_err_sum_list = torch.sum(square_err, dim=1)
        square_err_sum_list = torch.sum(square_err_sum_list, dim=1)
        square_err_sum_list = torch.sum(square_err_sum_list, dim=1)
        square_err_sum_list = square_err_sum_list + self.eps
        error = torch.sqrt(square_err_sum_list)
        loss = torch.sum(error) / batchsize
        return loss

def C_Loss(output, label):
    c_loss_func = Charbonnier_loss(epsilon=1e-3)
    return c_loss_func(output, label)