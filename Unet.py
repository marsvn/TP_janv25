import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict




# Custom Conv3d and ConvTranspose3d with standardization in initialization
class CustomConv3d(nn.Conv3d):
    def __init__(self, *args, **kwargs):
        super(CustomConv3d, self).__init__(*args, **kwargs)
        self.standardize_weight()

    def standardize_weight(self, gamma=None):
        weight_ = self.weight.view(self.weight.size(0), -1)
        weight_mean = weight_.mean(dim=1, keepdim=True)
        weight_var = weight_.var(dim=1, keepdim=True)
        weight_ = weight_ - weight_mean
        fan_in = torch.prod(torch.tensor(self.weight.size()[1:]))
        weight_ = weight_ / torch.sqrt(weight_var * fan_in) if gamma is None else weight_ * gamma / torch.sqrt(weight_var * fan_in)
        self.weight = torch.nn.Parameter(weight_.view(self.weight.size()))

class CustomConvTranspose3d(nn.ConvTranspose3d):
    def __init__(self, *args, **kwargs):
        super(CustomConvTranspose3d, self).__init__(*args, **kwargs)
        self.standardize_weight()

    def standardize_weight(self, gamma=None):
        weight_ = self.weight.view(self.weight.size(0), -1)
        weight_mean = weight_.mean(dim=1, keepdim=True)
        weight_var = weight_.var(dim=1, keepdim=True)
        weight_ = weight_ - weight_mean
        fan_in = torch.prod(torch.tensor(self.weight.size()[1:]))
        weight_ = weight_ / torch.sqrt(weight_var * fan_in) if gamma is None else weight_ * gamma / torch.sqrt(weight_var * fan_in)
        self.weight = torch.nn.Parameter(weight_.view(self.weight.size()))


def get_norm_layer(norm, act_mode, repeat=True):
    match norm :
        case 'nonorm.v1' :
            mode = 'C' +act_mode 
        case 'nonorm.v2' :
            mode = act_mode +'C'
        case 'batchnorm.v1' :
            mode = 'CB'+act_mode
        case 'instancenorm.v1' :
            mode = 'CI'+act_mode 
        case 'batchnorm.v2' :
            mode = 'B'+act_mode+ 'C'
        case 'instancenorm.v2' :
            mode = 'I'+act_mode+'C'
        case 'instancenormbf.v1' :
            mode = 'Ci'+act_mode
        case 'instancenormbf.v2' :
            mode = 'i'+act_mode+'C'
        case _:
            raise NotImplementedError('norm mode [{:s}] is not found'.format(norm))
    if repeat:
        mode = mode + mode
    return mode

def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)




# Call the appropriate function in your network code


# Then call apply_weight_standardization(net, gamma) before scripting the model


# def standardize_weight(m, gamma):
#     module_name = type(m).__name__
#     if module_name in ['Conv3d', 'ConvTranspose3d']:
#         weight_ = m.weight.data.view(m.weight.data.size(0), -1)
#         weight_mean = weight_.mean(dim=1, keepdim=True)
#         weight_var = weight_.var(dim=1, keepdim=True)

#         weight_ = weight_ - weight_mean

    
#         fan_in = np.prod(m.weight.data.shape[1:])
 
#         if gamma is None :
#             weight_ =weight_/((weight_var*fan_in)**0.5)
#         else :
#             weight_ =weight_*gamma/((weight_var*fan_in)**0.5)

#         m.weight.data = weight_.view(m.weight.data.size())

#         weight_ = m.weight.data.view(m.weight.data.size(0), -1)
#         weight_mean = weight_.mean(dim=1, keepdim=True)
#         weight_var = weight_.var(dim=1, keepdim=True)


# --------------------------------------------
# conv + subp (+ relu)
# --------------------------------------------
def upsample_pixelshuffle(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True, mode='2R', negative_slope=0.2, type_init='xavier'):
    assert len(mode)<4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    up1 = conv(in_channels, out_channels * (int(mode[0]) ** 2), kernel_size, stride, padding, bias, mode='C'+mode, negative_slope=negative_slope, type_init=type_init)
    return up1


# --------------------------------------------
# nearest_upsample + conv (+ R)
# --------------------------------------------
def upsample_upconv(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True, mode='2R', negative_slope=0.2, type_init='xavier'):
    assert len(mode)<4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR'
    if mode[0] == '2':
        uc = 'UC'
    elif mode[0] == '3':
        uc = 'uC'
    elif mode[0] == '4':
        uc = 'vC'
    mode = mode.replace(mode[0], uc)
    up1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode=mode, negative_slope=negative_slope, type_init=type_init)
    return up1


# --------------------------------------------
# convTranspose (+ relu)
# --------------------------------------------
def upsample_convtranspose(in_channels=64, out_channels=3, kernel_size=2, stride=2, padding=0, bias=True, mode='2R', negative_slope=0.2, type_init='xavier'):
    assert len(mode)<4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], 'T')
    up1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope, type_init=type_init)
    return up1


class PSA_i(nn.Module):
    def __init__(self, inplanes, kernel_size=1, stride=1):
        super(PSA_i, self).__init__()

        self.init = nn.InstanceNorm3d(1, affine=False)
        self.pre_act = nn.InstanceNorm3d(1, affine=False)

        self.inplanes = inplanes
        self.inter_planes = 16
        # self.planes = planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size - 1) // 2


        self.conv_q_left = CustomConv3d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                     bias=False)  # g
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv_v_left = CustomConv3d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                     bias=False)  # theta
        # self.softmax_left = nn.Softmax(dim=2)
        self.softmax_left = nn.Sigmoid()
        self.reset_parameters()

    def reset_parameters(self):
        kaiming_init(self.conv_q_left, mode='fan_in')
        kaiming_init(self.conv_v_left, mode='fan_in')

        self.conv_q_left.inited = True
        self.conv_v_left.inited = True


    def channel_pool(self, x):
        xinit = x
        x = self.init(x)
        # [N, IC, D, H, W]
        g_x = self.conv_q_left(x)

        batch, channel, depth, height, width = g_x.size()

        # [N, IC, 1, 1, 1]
        avg_x = self.avg_pool(g_x)

        batch, channel, avg_x_d, avg_x_h, avg_x_w = avg_x.size()

        # [N, 1, IC]
        avg_x = avg_x.view(batch, channel, avg_x_d * avg_x_h * avg_x_w).permute(0, 2, 1)

        # [N, IC, D*H*W]
        theta_x = self.conv_v_left(x).view(batch, self.inter_planes, depth * height * width)

        # [N, IC, D*H*W]
        theta_x = self.softmax_left(theta_x)

        # [N, 1, D*H*W]
        context = torch.matmul(avg_x, theta_x)

        # [N, 1, D, H, W]
        context = context.view(batch, 1, depth, height, width)

        context = self.pre_act(context)
        mask_sp = torch.sigmoid(context)

        out =  xinit * mask_sp #+ xinit
        return out, mask_sp

    def forward(self, x):
        # [N, C, H, W]

        # [N, C, H, W]
        context_channel, mask = self.channel_pool(x)

        # [N, C, H, W]
        # out = context_spatial + context_channel

        return context_channel, mask
    
def sequential(*args):
    """Advanced nn.Sequential.
    Args:
        nn.Sequential, nn.Module
    Returns:
        nn.Sequential
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)



def init_weights(m, type_init='xavier'):
    if type(m) == CustomConv3d or type(m) == CustomConvTranspose3d:
        match type_init:
            case 'xavier':  
                torch.nn.init.xavier_normal_(m.weight.data, gain=1.3)
                
            case 'kaiming': 
                torch.nn.init.kaiming_normal_(m.weight.data) 

            case 'zero': 
                m.weight.data.zero_()

            case _:
                raise NotImplementedError(f'init_weights: type_init={type_init} not implemented')           

        
        if m.bias is not None:
            m.bias.data.fill_(0)
    if type(m) == torch.nn.BatchNorm3d or type(m) == torch.nn.InstanceNorm3d:
        if m.weight is not None:

            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)

    if type(m) == nn.Linear:
        if type_init == 'zero':
            m.weight.data.zero_()
        else :
            torch.nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)
class Scaling(nn.Module):

    def __init__(self, scale: float):
        self.scale = scale
        super().__init__()
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.scale})"
    
    def forward(self, x):
        return x/self.scale



class PixelShuffle3d(nn.Module):
    '''
    This class is a 3d version of pixelshuffle.
    '''
    def __init__(self, upscale_factor):
        '''
        :param scale: upsample scale
        '''
        super().__init__()
        self.scale = upscale_factor

    def forward(self, input):
        batch_size, channels, in_depth, in_height, in_width = input.size()
        nOut = channels // self.scale ** 3

        out_depth = in_depth * self.scale
        out_height = in_height * self.scale
        out_width = in_width * self.scale

        input_view = input.contiguous().view(batch_size, nOut, self.scale, self.scale, self.scale, in_depth, in_height, in_width)

        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        return output.view(batch_size, nOut, out_depth, out_height, out_width)

# --------------------------------------------
# return nn.Sequantial of (Conv + BN + ReLU)
# --------------------------------------------
# def conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CBR', negative_slope=0.2, emb_dim=64, type_init='xavier'):
#     L = []
    
#     # Helper function to initialize weights
#     def initialize_layer(layer):
#         init_weights(layer, type_init=type_init)
#         return layer
    
#     for t in mode:
#         if t == 'C':
#             layer = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
#         elif t == 'c':
#             layer = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
#         elif t == 'S':
#             layer = nn.utils.spectral_norm(nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
#         elif t == 'T':
#             layer = nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
#         elif t == 'B':
#             layer = nn.BatchNorm3d(out_channels, momentum=0.9, eps=1e-04)
#         elif t == 'I':
#             layer = nn.InstanceNorm3d(out_channels, affine=True)
#         elif t == 'i':
#             layer = nn.InstanceNorm3d(out_channels, affine=False)
#         elif t == 'D':
#             layer = nn.Sequential(
#                 nn.SiLU(),
#                 nn.Linear(emb_dim, out_channels)
#             )
#         elif t == 'F':
#             layer = nn.Dropout3d(p=0.5, inplace=False)
#         elif t == 'G':
#             layer = nn.LayerNorm(out_channels, elementwise_affine=True)
#         elif t == 'p':
#             layer = nn.Linear(emb_dim, out_channels, bias=False)
#         elif t == 'R':
#             layer = nn.ReLU(inplace=False)
#         elif t == 'E':
#             layer = nn.ELU(inplace=False)
#         elif t == 'L':
#             layer = nn.LeakyReLU(negative_slope=negative_slope, inplace=False)
#         elif t == 's':
#             layer = nn.Softplus()
#         elif t == 'z':
#             layer = nn.Sigmoid()
#         elif t == '2':
#             layer = PixelShuffle3d(upscale_factor=2)
#         elif t == '3':
#             layer = PixelShuffle3d(upscale_factor=3)
#         elif t == '4':
#             layer = PixelShuffle3d(upscale_factor=4)
#         elif t == 'U':
#             layer = nn.Upsample(scale_factor=2)
#         elif t == 'u':
#             layer = nn.Upsample(scale_factor=3, mode='trilinear', align_corners=False)
#         elif t == 'v':
#             layer = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=False)
#         elif t == 'M':
#             layer = nn.MaxPool3d(kernel_size=kernel_size, stride=stride, padding=0)
#         elif t == 'A':
#             layer = nn.AvgPool3d(kernel_size=kernel_size, stride=stride, padding=0)
#         else:
#             raise NotImplementedError(f'Undefined type: {t}')
        
#         # Apply weight initialization directly after creating each layer
#         L.append(initialize_layer(layer))
#     print(sequential(*L))

#     return nn.Sequential(*L)

def conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CBR', negative_slope=0.2, emb_dim=64, type_init='xavier'):
    L = []
    for t in mode:
        if t == 'C':
            L.append(CustomConv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif t == 'c':
            L.append(nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif t == 'S':
            L.append(nn.utils.spectral_norm(CustomConv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)))
        elif t == 'T':
            L.append(CustomConvTranspose3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif t == 'B':
            L.append(nn.BatchNorm3d(out_channels, momentum=0.9, eps=1e-04))
            #L.append(nn.BatchNorm2d(out_channels))
        elif t == 'I':
            L.append(nn.InstanceNorm3d(out_channels, affine=True))
        elif t == 'i':
            L.append(nn.InstanceNorm3d(out_channels, affine=False))
        elif t == 'D':
            L.append(nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        ))
        elif t =='F':
            L.append(nn.Dropout3d(p=0.5, inplace=False))
        elif t == 'G':
            L.append(nn.LayerNorm(out_channels, elementwise_affine=True))
            
        elif t == 'p':
            L.append(nn.Linear(emb_dim,out_channels, bias = False))
        elif t == 'R':
        #     L.append(nn.ReLU(inplace=True))
        # elif t == 'r':
            L.append(nn.ReLU(inplace=False))
        # elif t == 'E':
        #     L.append(nn.ELU(inplace=True))
        elif t == 'E':
            L.append(nn.ELU(inplace=False))
        elif t == 'L':
        #     L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        # elif t == 'l':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=False))
        elif t == 's':
            L.append(nn.Softplus())
        elif t == 'z':
            L.append(nn.Sigmoid())
        elif t == '2':
            L.append(PixelShuffle3d(upscale_factor=2))
        elif t == '3':
            L.append(PixelShuffle3d(upscale_factor=3))
        elif t == '4':
            L.append(PixelShuffle3d(upscale_factor=4))
        elif t == 'U':
            L.append(nn.Upsample(scale_factor=2)) #, mode = 'trilinear', align_corners = False))
        elif t == 'u':
            L.append(nn.Upsample(scale_factor=3, mode = 'trilinear', align_corners = False))
        elif t == 'v':
            L.append(nn.Upsample(scale_factor=4, mode = 'trilinear', align_corners = False))
        elif t == 'M':
            L.append(nn.MaxPool3d(kernel_size=kernel_size, stride=stride, padding=0))
        elif t == 'A':
            L.append(nn.AvgPool3d(kernel_size=kernel_size, stride=stride, padding=0))
        else:
            raise NotImplementedError('Undefined type: '.format(t))
    return sequential(*L)

# --------------------------------------------
# strideconv (+ relu)
# --------------------------------------------
def downsample_strideconv(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0, bias=True, mode='2R', negative_slope=0.2, type_init='xavier'):
    assert len(mode)<4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], 'C')
    down1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope, type_init=type_init)
    return down1


# --------------------------------------------
# maxpooling + conv (+ relu)
# --------------------------------------------
def downsample_maxpool(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='2R', negative_slope=0.2, type_init='xavier'):
    assert len(mode)<4 and mode[0] in ['2', '3'], 'mode examples: 2, 2R, 2BR, 3, ..., 3BR.'
    kernel_size_pool = int(mode[0])
    stride_pool = int(mode[0])
    mode = mode.replace(mode[0], 'MC')
    pool = conv(kernel_size=kernel_size_pool, stride=stride_pool, mode=mode[0], negative_slope=negative_slope, type_init=type_init)
    pool_tail = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode=mode[1:], negative_slope=negative_slope, type_init=type_init)
    return sequential(pool, pool_tail)


# --------------------------------------------
# averagepooling + conv (+ relu)
# --------------------------------------------
def downsample_avgpool(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='2R', negative_slope=0.2, type_init='xavier'):
    assert len(mode)<4 and mode[0] in ['2', '3'], 'mode examples: 2, 2R, 2BR, 3, ..., 3BR.'
    kernel_size_pool = int(mode[0])
    stride_pool = int(mode[0])
    mode = mode.replace(mode[0], 'AC')
    pool = conv(kernel_size=kernel_size_pool, stride=stride_pool, mode=mode[0], negative_slope=negative_slope, type_init=type_init)
    pool_tail = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode=mode[1:], negative_slope=negative_slope, type_init=type_init)
    return sequential(pool, pool_tail)


# --------------------------------------------
# --------------------------------------------
class NFResBlock(nn.Module):
    def __init__(self, in_channels=64, 
                 out_channels=64, 
                 kernel_size=3, 
                 stride=1, 
                 padding=1, 
                 bias=True, 
                 mode='CRC', 
                 negative_slope=0.2, 
                 type_init='xavier', 
                 beta = 1.0,
                 alpha = 1.0, 
                 gain = 1.0
                 ):
        super(NFResBlock, self).__init__()

        self.alpha = alpha
        self.gain = gain
        self.beta = (1.0+alpha**2)**0.5

        self.residual_preact = nn.Sequential(
            Scaling(self.beta),
        )
        


        self.in_channels = in_channels
        self.out_channels = out_channels
        self.residual_branch = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope, type_init=type_init)

   

    def forward(self, x):
        residual = self.residual_preact(x)
        skip = residual
        
        # # Apply weight standardization before forward pass
        # self.standardize_weights()

        residual = self.residual_branch(residual)
        return self.alpha * residual + skip

class NFDownBlock(nn.Module):

    expansion = 4

    def __init__(self,
        downsample_mode='strideconv',
        nb=2, 
        wf=16,
        type_init= 'xavier', 
        prev_channels = 64,
        keep_bias = False,
        mode = 'CRC',
        alpha = 1.0, 
        downsample = True, 
        expected_std=1.0, 
        block_type = 'resblock',
        gain = 1.0


    ):
        super().__init__()
        self.block_type = block_type

    
        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))


        if block_type == 'NFblock': 
            self.branch = nn.Sequential(
                    *[NFResBlock(prev_channels, prev_channels, bias=keep_bias, mode=mode, type_init=type_init, 
                                beta = expected_std, alpha = alpha, gain=gain) for i in range(nb)]
                    )

        else:
            raise NotImplementedError('block type [{:s}] is not found'.format(block_type))
            



        self.downsample = downsample_block(prev_channels, wf, bias=keep_bias, mode='2', type_init=type_init) if downsample else nn.Identity()

    def forward(self, x):
        out = self.branch(x)
        return self.downsample(out) 


    

class NFUpBlock(nn.Module):


    def __init__(self,
        upsample_mode='strideconv',
        nb=2, 
        wf=16,
        type_init= 'xavier', 
        prev_channels = 64,
        keep_bias = False,
        mode = 'CRC',
        alpha = 1.0, 
        upsample = True, 
        expected_std=1.0,
        block_type = 'resblock',
        gain = 1.0
    ):
        super().__init__()

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.upsample = upsample_block(prev_channels, wf, bias=keep_bias, mode='2', type_init=type_init) if upsample else nn.Identity()
        self.block_type = block_type

        if self.block_type == 'NFblock':
            self.branch = nn.Sequential(
                    *[NFResBlock(in_channels= wf, out_channels =wf, bias=keep_bias, mode=mode, type_init=type_init, 
                                beta = (expected_std**2+(i+1)*alpha**2)**0.5, alpha = alpha, gain=gain) for i in range(nb)]
                    )
       
            
        else:
            raise NotImplementedError('block type [{:s}] is not found'.format(self.block_type))
        
      

    def forward(self, x):
        return self.branch(self.upsample(x))
        


class NFDRUNet(nn.Module):
    def __init__(self, in_nc=2, 
                out_nc=1, 
                nb=2, 
                depth = 2, 
                wf=16,
                act_mode='R', 
                downsample_mode='strideconv', 
                upsample_mode='convtranspose', 
                norm = 'nonorm', 
                type_init= 'xavier', 
                keep_bias = False, 
                alpha = 1.0,
                repeat = True, 
                bUseAtt = False
                ) :
        super(NFDRUNet, self).__init__() 

        self.down_path = nn.ModuleList()
        # if bExtendedConv :
        #     self.down_path.append(B.conv(in_nc, wf, bias=keep_bias, mode='CRCRCRC'))
        # else :
        self.down_path.append(conv(in_nc, wf, bias=keep_bias, mode='C'))
        prev_channels = wf
        self.down_scaling = nn.ModuleList()
        self.alpha = alpha
        self.nb = nb


        mode = get_norm_layer(norm, act_mode, repeat)        
        match act_mode:
            case 'R':  
                # with torch.no_grad():
                    # y = F.relu(torch.randn(1024, 256).cuda()) 
                    # gamma = y.var(dim=1, keepdim=True)
                gain = 1.7139588594436646 #1.2716004848480225 
                    #gamma.mean(dim=0, keepdim=True)

            case 'L':
                # with torch.no_grad():
                    # y = F.leaky_relu(torch.randn(1024, 256).cuda()) 
                    # gamma = y.var(dim=1, keepdim=True)
                gain =  1.70590341091156 #gamma.mean(dim=0, keepdim=True)
            case 'E':
                # with torch.no_grad():
                    # y = F.elu(torch.randn(1024, 256).cuda()) 
                    # gamma = y.var(dim=1, keepdim=True)
                gain = 1.2716004848480225 
                    #gamma.mean(dim=0, keepdim=True)

        # expected_std = ((1.0+(2*(depth)+1)*nb*(alpha**2))**0.5)
        expected_std = (((1.0+nb*(alpha**2))**(depth+1))*(1.0+nb*depth*(alpha**2)))**0.5
        self.init_scaling = Scaling(expected_std)

        list_expected_std_up = []
        list_expected_std_input = []

        self.bUseAtt = bUseAtt

        for i in range(depth-1):
                        
            self.down_path.append(NFDownBlock(downsample_mode=downsample_mode, prev_channels= prev_channels, wf = wf*(2 ** (i+1)), alpha = self.alpha, expected_std = expected_std,
                                               nb = nb, keep_bias=keep_bias, mode=mode, type_init=type_init, block_type = 'NFblock', gain = gain)
                                               )                         
            expected_std = (expected_std **2 +  nb*(alpha**2))**0.5
            list_expected_std_input.append(expected_std)
            list_expected_std_up.append((((1.0+nb*(alpha**2))**(i+1))*(1.0+nb*depth*(alpha**2)))**0.5)
            self.down_scaling.append(Scaling(list_expected_std_up[-1])) 
            prev_channels = wf*(2 ** (i+1))
            

        self.down_path.append(NFDownBlock(expected_std= expected_std, alpha = self.alpha, prev_channels= prev_channels, downsample=False, 
                                          nb = nb, keep_bias=keep_bias, 
                                          mode=mode, type_init=type_init, block_type = 'NFblock', gain=gain)
                                          )
        expected_std = (expected_std ** 2 + alpha**2) ** 0.5

        self.up_path = nn.ModuleList()
    
        for i in reversed(range(depth-1)):
            # beta = 1./ expected_std
            self.up_path.append(NFUpBlock(expected_std= expected_std, alpha = self.alpha, upsample_mode=upsample_mode, 
                                            prev_channels= prev_channels, nb=nb,
                                            wf=wf*(2 ** i), keep_bias=keep_bias, mode=mode,type_init=type_init, 
                                            block_type = 'NFblock' if (i > 0 or not bUseAtt) else 'NFblock-PSA', gain=gain
                                            ))
           


            prev_channels = wf*(2 ** i)
            expected_std = (expected_std **2 +  nb*(alpha**2))**0.5

        self.final = conv(wf, out_nc, bias=keep_bias, mode='C', type_init=type_init)
                   

    def forward(self, x):

       
        blocks = []
        xinit = x
        x_s = self.init_scaling(xinit)
        for i, down in enumerate(self.down_path):

            x_s = down(x_s)

            if i != len(self.down_path) - 1:
                blocks.append(x_s)

            # if scaling common to res and id branches swap 
            if i > 0 and i != len(self.down_path) - 1:
                x_s = self.down_scaling[i-1](x_s)
        
                
        
        for i, up in enumerate(self.up_path):
    

                        
           x_s = up(self.alpha*x_s+ blocks[-i - 1])
            
     
        x_s = self.final(x_s)   

        return F.relu(x_s)
  
        
        

class NFUNetResMultBranch2Decoder(nn.Module):
    def __init__(self, in_nc=2, 
                out_nc=1, 
                nb=2, 
                depth = 2, 
                wf=16,
                act_mode='R', 
                downsample_mode='strideconv', 
                upsample_mode='convtranspose', 
                norm = 'nonorm', 
                type_init= 'xavier', 
                alpha = 1.0,
                repeat = True, 
                bUse_N = True
                ) :
        super(NFUNetResMultBranch2Decoder, self).__init__() 

        self.down_path = nn.ModuleList()
        self.down_path.append(conv(in_nc, wf, bias=False, mode='C'))

        self.down_path2 = nn.ModuleList()
        self.down_path2.append(conv(in_nc, wf, bias=False, mode='C'))

        prev_channels = wf
        self.down_scaling = nn.ModuleList()
        self.alpha = alpha
        self.nb = nb


        mode = get_norm_layer(norm, act_mode, repeat)        
        match act_mode:
            case 'R':  
                gain = 1.7139588594436646 
            case 'L':
                gain =  1.70590341091156 
            case 'E':
                gain = 1.2716004848480225 
                   
        expected_std = (((1.0+nb*(alpha**2))**(depth+1))*(1.0+nb*depth*(alpha**2)))**0.5
        self.init_scaling = Scaling(expected_std)

        list_expected_std_up = []
        list_expected_std_input = []

        self.res_weights = nn.ParameterList([nn.Parameter(torch.FloatTensor([0.])) for i in range(depth-1)])



        for i in range(depth-1):
                        
            self.down_path.append(NFDownBlock(downsample_mode=downsample_mode, prev_channels= prev_channels, wf = wf*(2 ** (i+1)), alpha = self.alpha, expected_std = expected_std,
                                               nb = nb, keep_bias=False, mode=mode, type_init=type_init, block_type = 'NFblock', gain = gain)
                                               )
            self.down_path2.append(NFDownBlock(downsample_mode=downsample_mode, prev_channels= prev_channels, wf = wf*(2 ** (i+1)), alpha = self.alpha, expected_std = expected_std,
                                                  nb = nb, keep_bias=False, mode=mode, type_init=type_init, block_type = 'NFblock', gain = gain)
                                                  )
            expected_std = (expected_std **2 +  nb*(alpha**2))**0.5
            list_expected_std_input.append(expected_std)
            list_expected_std_up.append((((1.0+nb*(alpha**2))**(i+1))*(1.0+nb*depth*(alpha**2)))**0.5)
            self.down_scaling.append(Scaling(list_expected_std_up[-1])) 
            prev_channels = wf*(2 ** (i+1))
            

        self.down_path.append(NFDownBlock(expected_std= expected_std, alpha = self.alpha, prev_channels= prev_channels, downsample=False, 
                                          nb = nb, keep_bias=False, mode=mode, type_init=type_init, block_type = 'NFblock', gain=gain)
                                          )
        self.down_path2.append(NFDownBlock(expected_std= expected_std, alpha = self.alpha, prev_channels= prev_channels, downsample=False,
                                           nb=nb, keep_bias=False, mode=mode, type_init=type_init, block_type = 'NFblock', gain=gain)
        
                                  )
        self.fusion = conv(2*prev_channels,prev_channels, bias=False, mode='CE', type_init=type_init)
        if bUse_N :
            self.normalization = conv(out_channels=[2*prev_channels, int(104/(2 ** (depth-1))),  int(128/(2 ** (depth-1))),  int(128/(2 ** (depth-1)))], mode="G")
        else :
            self.normalization = nn.Identity()
        expected_std = (expected_std ** 2 + alpha**2) ** 0.5

        self.up_path = nn.ModuleList()
    
        for i in reversed(range(depth-1)):
            self.up_path.append(NFUpBlock(expected_std= expected_std, alpha = self.alpha, upsample_mode=upsample_mode, 
                                            prev_channels= prev_channels, nb=nb,
                                            wf=wf*(2 ** i), keep_bias=False, mode=mode,type_init=type_init, 
                                            block_type = 'NFblock', gain=gain))
           


            prev_channels = wf*(2 ** i)
            expected_std = (expected_std **2 +  nb*(alpha**2))**0.5

        self.final = conv(wf, out_nc, bias=False, mode='C', type_init=type_init)


    def forward(self, x, xcond):
        
       
        blocks = []
        blocks2 = []
        xinit = x
        x_s = self.init_scaling(xinit)
        xcond_s = self.init_scaling(xcond)

        # i = 0
        x_s = self.down_path[0](x_s)
        xcond_s = self.down_path2[0](xcond_s)
        # i = 1
        x_s = self.down_path[1](x_s)
        xcond_s = self.down_path2[1](xcond_s)
        blocks.append(x_s)
        blocks2.append(xcond_s)
        x_s = self.down_scaling[0](x_s)
        xcond_s = self.down_scaling[0](xcond_s)
        # i = 2
        x_s = self.down_path[2](x_s)
        xcond_s = self.down_path2[2](xcond_s)
        blocks.append(x_s)
        blocks2.append(xcond_s)
        x_s = self.down_scaling[1](x_s)
        xcond_s = self.down_scaling[1](xcond_s)
        # i = 2
        x_s = self.down_path[3](x_s)
        xcond_s = self.down_path2[3](xcond_s)


        x_s = self.normalization(torch.cat((x_s, xcond_s), 1))
        x_s = self.fusion(x_s)
            
        # unfold up path
        x_s = self.up_path[0](self.alpha*x_s+ blocks[ - 1] + self.res_weights[0]*blocks2[- 1])
        x_s = self.up_path[1](self.alpha*x_s+ blocks[ - 2] + self.res_weights[1]*blocks2[- 2])


        x_s = self.final(x_s)   
       
        return F.relu(x_s)  
   
class NFUNetRes2(nn.Module):
    def __init__(self, in_nc=2, 
                out_nc=1, 
                nb=2, 
                depth = 2, 
                wf=16,
                act_mode='R', 
                downsample_mode='strideconv', 
                upsample_mode='convtranspose', 
                norm = 'nonorm', 
                type_init= 'xavier', 
                keep_bias = False, 
                alpha = 1.0,
                repeat = True, 
                bUseAtt = False
                ) :
        super(NFUNetRes2, self).__init__() 

        self.down_path = nn.ModuleList()
        self.down_path.append(conv(in_nc, wf, bias=keep_bias, mode='C'))
        prev_channels = wf
        self.down_scaling = nn.ModuleList()
        self.alpha = alpha
        self.nb = nb


        mode = get_norm_layer(norm, act_mode, repeat)        
        match act_mode:
            case 'R':  
                gain = 1.7139588594436646 
            case 'L':
                gain =  1.70590341091156 
            case 'E':
                gain = 1.2716004848480225 

        expected_std = (((1.0+nb*(alpha**2))**(depth+1))*(1.0+nb*depth*(alpha**2)))**0.5
        self.init_scaling = Scaling(expected_std)

        list_expected_std_up = []
        list_expected_std_input = []

        self.bUseAtt = bUseAtt

        for i in range(depth-1):
                        
            self.down_path.append(NFDownBlock(downsample_mode=downsample_mode, prev_channels= prev_channels, wf = wf*(2 ** (i+1)), alpha = self.alpha, expected_std = expected_std,
                                               nb = nb, keep_bias=keep_bias, mode=mode, type_init=type_init, block_type = 'NFblock', gain = gain)
                                               )                         
            expected_std = (expected_std **2 +  nb*(alpha**2))**0.5
            list_expected_std_input.append(expected_std)
            list_expected_std_up.append((((1.0+nb*(alpha**2))**(i+1))*(1.0+nb*depth*(alpha**2)))**0.5)
            self.down_scaling.append(Scaling(list_expected_std_up[-1])) 
            prev_channels = wf*(2 ** (i+1))
            

        self.down_path.append(NFDownBlock(expected_std= expected_std, alpha = self.alpha, prev_channels= prev_channels, downsample=False, 
                                          nb = nb, keep_bias=keep_bias, 
                                          mode=mode, type_init=type_init, block_type = 'NFblock', gain=gain)
                                          )
        expected_std = (expected_std ** 2 + alpha**2) ** 0.5

        self.up_path = nn.ModuleList()
    
        for i in reversed(range(depth-1)):
            self.up_path.append(NFUpBlock(expected_std= expected_std, alpha = self.alpha, upsample_mode=upsample_mode, 
                                            prev_channels= prev_channels, nb=nb,
                                            wf=wf*(2 ** i), keep_bias=keep_bias, mode=mode,type_init=type_init, 
                                            block_type = 'NFblock' if (i > 0 or not bUseAtt) else 'NFblock-PSA', gain=gain
                                            ))
           


            prev_channels = wf*(2 ** i)
            expected_std = (expected_std **2 +  nb*(alpha**2))**0.5

        self.final = conv(wf, out_nc, bias=keep_bias, mode='C', type_init=type_init)

       
           

    def forward(self, x):
        
        blocks = []
        xinit = x
        x_s = self.init_scaling(xinit)

        # i = 0
        x_s = self.down_path[0](x_s)

        # i = 1
        x_s = self.down_path[1](x_s)
        blocks.append(x_s)
        x_s = self.down_scaling[0](x_s)
        # i = 2
        x_s = self.down_path[2](x_s)
        blocks.append(x_s)
        x_s = self.down_scaling[1](x_s)
        # i = 2
        x_s = self.down_path[3](x_s)

        # unfold up path
        x_s = self.up_path[0](self.alpha*x_s+ blocks[ - 1])
        x_s = self.up_path[1](self.alpha*x_s+ blocks[ - 2])

        x_s = self.final(x_s) 

        return F.relu(x_s)
        
    
class MyDeepModel(torch.nn.Module):
    def __init__(self):
        super(MyDeepModel, self).__init__()
        self.model = NFUNetRes2(in_nc=1, out_nc=1, nb=1, wf= 32, depth = 3, act_mode = 'E',
                             downsample_mode='strideconv', upsample_mode='upconv', 
                             norm = 'nonorm.v1', 
                             keep_bias=False, repeat=False, bUseAtt=False)
        self.model.eval()
        self.model.requires_grad_(False)

    def forward(self, x):
        return self.model(x)

class MyDeepModelMRI(torch.nn.Module):
    def __init__(self):
        super(MyDeepModelMRI, self).__init__()
        self.model = NFUNetResMultBranch2Decoder(in_nc=1, out_nc=1, nb=1, wf= 32, depth = 3, 
                             act_mode = 'E', upsample_mode='upconv',repeat=False,
                             downsample_mode='strideconv', 
                             norm = 'nonorm.v1',  bUse_N = False)    

        self.model.eval()
        self.model.requires_grad_(False)

    def forward(self, x, xmri):
        return self.model(x, xmri)
    

class MyDenoiser(torch.nn.Module):
    def __init__(self):
        super(MyDenoiser, self).__init__()
        self.base = MyDeepModel()
        

    def forward(self, x, mask):
        return self.base(x*mask)*mask



class MyDenoiserMRI(torch.nn.Module):
    def __init__(self):
        super(MyDenoiserMRI, self).__init__()
        self.base = MyDeepModelMRI()
        

    def forward(self, x, mask, xmri):
        return self.base(x*mask, xmri)*mask