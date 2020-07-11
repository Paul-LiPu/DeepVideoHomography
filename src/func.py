import numpy as np
import torch.nn.init as init


def weights_init_constant(m, std):
    classname = m.__class__.__name__
#     print classname
    if classname.find('Conv') != -1:
        m.weight.data.normal_(mean = 0.0, std = std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1.)
        m.bias.data.fill_(0.)#zero_()
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, std)
        m.bias.data.zero_()


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        #std = np.sqrt(2./(m.kernel_size[0]*m.kernel_size[1]*m.out_channels))
        #m.weight.data.normal_(0.0, std)
        #m.bias.data.zero_()

        init.xavier_normal(m.weight.data)
        if m.bias is not None:
            init.constant(m.bias.data, 0.)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1.)
        m.bias.data.fill_(0.0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.001)
        m.bias.data.zero_()


def weights_init_msra(m):
    classname = m.__class__.__name__
#     print classname
    if classname.find('Conv') != -1:
        std = np.sqrt(2. / (m.kernel_size[0] * m.kernel_size[1] * m.in_channels))
        # init.kaiming_uniform(m.weight.data, mode='fan_in')
        m.weight.data.normal_(mean=0.0, std=std)
        m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        #print m.weight.data.numpy()
        m.weight.data.fill_(1.)
        #print m.weight.data.numpy()
        m.bias.data.fill_(0.)#zero_()
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.001)
        m.bias.data.zero_()

def weights_init_He_normal(m):
    classname = m.__class__.__name__
#     print classname
    if classname.find('Transpose') != -1:
        m.weight.data.normal_(0.0, 0.001)
        if not m.bias is None:
            m.bias.data.zero_()
    elif classname.find('Conv') != -1:
        # std = np.sqrt(2. / (m.kernel_size[0] * m.kernel_size[1] * m.out_channels))
        # m.weight.data.normal_(0.0, std)
        init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
        if not m.bias is None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1.)
        if not m.bias is None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.001)
        if not m.bias is None:
            m.bias.data.zero_()



