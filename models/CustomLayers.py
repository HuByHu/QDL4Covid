from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from deepquantum import Circuit
from deepquantum.utils import dag, ptrace, encoding, expecval_ZI


class PixelNormLayer(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)


class Upscale2d(nn.Module):
    """
    扩展特征图大小为(MB, C, H*factor, W*factor)
    """
    @staticmethod
    def upscale2d(x, factor=4, gain=1):
        assert x.dim() == 4
        if gain != 1:
            x = x * gain
        if factor != 1:
            shape = x.shape
            x = x.view(shape[0], shape[1], shape[2], 1, shape[3], 1).expand(-1, -1, -1, factor, -1, factor)
            x = x.contiguous().view(shape[0], shape[1], factor * shape[2], factor * shape[3])
        return x

    def __init__(self, factor=4, gain=1):
        super().__init__()
        assert isinstance(factor, int) and factor >= 1
        self.gain = gain
        self.factor = factor

    def forward(self, x):
        return self.upscale2d(x, factor=self.factor, gain=self.gain)


class Downscale2d(nn.Module):
    """
    缩小特征图大小为(MB, C, H/factor, W/factor)
    """
    def __init__(self, factor=4, gain=1):
        super().__init__()
        assert isinstance(factor, int) and factor >= 1
        self.factor = factor
        self.gain = gain
        if factor == 4:
            # [0.5, 0.5]
            f = [np.sqrt(gain) / factor] * factor
            self.blur = BlurLayer(kernel=f, normalize=False, stride=factor)
        else:
            self.blur = None

    def forward(self, x):
        assert x.dim() == 4
        # 2x2, float32 => downscale using _blur2d().
        if self.blur is not None and x.dtype == torch.float32:
            return self.blur(x)

        # Apply gain.
        if self.gain != 1:
            x = x * self.gain

        # No-op => early exit.
        if self.factor == 1:
            return x

        # Large factor => downscale using tf.nn.avg_pool().
        # NOTE: Requires tf_config['graph_options.place_pruned_graph']=True to work.
        return F.avg_pool2d(x, self.factor)


class EqualizedLinear(nn.Module):
    """Linear layer with equalized learning rate and custom learning rate multiplier."""

    def __init__(self, input_size, output_size, gain=2 ** 0.5, use_wscale=False, lrmul=1, bias=True):
        super().__init__()
        he_std = gain * input_size ** (-0.5)  # He init
        # Equalized learning rate and custom learning rate multiplier.
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
        self.weight = torch.nn.Parameter(torch.randn(output_size, input_size) * init_std)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(output_size))
            self.b_mul = lrmul
        else:
            self.bias = None

    def forward(self, x):
        bias = self.bias
        if bias is not None:
            bias = bias * self.b_mul
        return F.linear(x, self.weight * self.w_mul, bias)


class EqualizedConv2d(nn.Module):
    """Conv layer with equalized learning rate and custom learning rate multiplier."""

    def __init__(self, input_channels, output_channels, kernel_size, stride=1, gain=2 ** 0.5, use_wscale=False,
                 lrmul=1, bias=True, intermediate=None, upscale=False, downscale=False):
        super().__init__()
        if upscale:
            self.upscale = Upscale2d()
        else:
            self.upscale = None
        if downscale:
            self.downscale = Downscale2d()
        else:
            self.downscale = None
        he_std = gain * (input_channels * kernel_size ** 2) ** (-0.5)  # He init
        self.kernel_size = kernel_size
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
        self.weight = torch.nn.Parameter(
            torch.randn(output_channels, input_channels, kernel_size, kernel_size) * init_std)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(output_channels))
            self.b_mul = lrmul
        else:
            self.bias = None
        self.intermediate = intermediate

    def forward(self, x):
        bias = self.bias
        if bias is not None:
            bias = bias * self.b_mul

        have_convolution = False
        if self.upscale is not None and min(x.shape[2:]) * 2 >= 128:
            # this is the fused upscale + conv from StyleGAN, sadly this seems incompatible with the non-fused way
            # this really needs to be cleaned up and go into the conv...
            w = self.weight * self.w_mul
            w = w.permute(1, 0, 2, 3)
            # probably applying a conv on w would be more efficient. also this quadruples the weight (average)?!
            w = F.pad(w, [1, 1, 1, 1])
            w = w[:, :, 1:, 1:] + w[:, :, :-1, 1:] + w[:, :, 1:, :-1] + w[:, :, :-1, :-1]
            # x = F.conv_transpose2d(x, w, stride=2, padding=(w.size(-1) - 1) // 2)
            x = F.conv_transpose2d(x, w, stride=4, padding=0)
            
            have_convolution = True
        elif self.upscale is not None:
            x = self.upscale(x)

        downscale = self.downscale
        intermediate = self.intermediate
        if downscale is not None and min(x.shape[2:]) * 2>= 128:
            w = self.weight * self.w_mul
            w = F.pad(w, [1, 1, 1, 1])
            # in contrast to upscale, this is a mean...
            # 卷积核被轻微平移了四次并对自身做了叠加然后取平均值，或许对提取特征有帮助，参数特殊初始化
            w = (w[:, :, 1:, 1:] + w[:, :, :-1, 1:] + w[:, :, 1:, :-1] + w[:, :, :-1, :-1]) * 0.25  # avg_pool?
            x = F.conv2d(x, w, stride=4, padding=(w.size(-1) - 1) // 2)
            have_convolution = True
            downscale = None
        elif downscale is not None:
            assert intermediate is None
            intermediate = downscale

        if not have_convolution and intermediate is None:
            return F.conv2d(x, self.weight * self.w_mul, bias, padding=self.kernel_size // 2)
        elif not have_convolution:
            x = F.conv2d(x, self.weight * self.w_mul, None, padding=self.kernel_size // 2)

        if intermediate is not None:
            x = intermediate(x)

        if bias is not None:
            x = x + bias.view(1, -1, 1, 1)
        return x


class NoiseLayer(nn.Module):
    """adds noise. noise is per pixel (constant over channels) with per-channel weight"""

    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(channels))
        self.noise = None

    def forward(self, x, noise=None):
        if noise is None and self.noise is None:
            noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        elif noise is None:
            # here is a little trick: if you get all the noise layers and set each
            # modules .noise attribute, you can have pre-defined noise.
            # Very useful for analysis
            noise = self.noise
        x = x + self.weight.view(1, -1, 1, 1) * noise
        return x


class StyleMod(nn.Module):
    def __init__(self, latent_size, channels, use_wscale):
        super(StyleMod, self).__init__()
        self.lin = EqualizedLinear(latent_size,
                                   channels * 2,
                                   gain=1.0, use_wscale=use_wscale)

    def forward(self, x, latent):
        style = self.lin(latent)  # style => [batch_size, n_channels*2]

        shape = [-1, 2, x.size(1)] + (x.dim() - 2) * [1]
        style = style.view(shape)  # [batch_size, 2, n_channels, ...]
        x = x * (style[:, 0] + 1.) + style[:, 1]
        return x


class LayerEpilogue(nn.Module):
    """Things to do at the end of each layer."""

    def __init__(self, channels, dlatent_size, use_wscale,
                 use_noise, use_pixel_norm, use_instance_norm, use_styles, activation_layer):
        super().__init__()

        layers = []
        if use_noise:
            layers.append(('noise', NoiseLayer(channels)))
        layers.append(('activation', activation_layer))
        if use_pixel_norm:
            layers.append(('pixel_norm', PixelNormLayer()))
        if use_instance_norm:
            layers.append(('instance_norm', nn.InstanceNorm2d(channels)))

        self.top_epi = nn.Sequential(OrderedDict(layers))

        if use_styles:
            self.style_mod = StyleMod(dlatent_size, channels, use_wscale=use_wscale)
        else:
            self.style_mod = None

    def forward(self, x, dlatents_in_slice=None):
        x = self.top_epi(x)
        if self.style_mod is not None:
            x = self.style_mod(x, dlatents_in_slice)
        else:
            assert dlatents_in_slice is None
        return x


class BlurLayer(nn.Module):
    """
    实现主要有两个部分
    第一个部分是对于卷积核的处理，包括维度的规范和归一化处理；
    第二个部分是模糊的实现，即用卷积核对 x 实行分组groups=x.size(1)卷积
    """
    def __init__(self, kernel=None, normalize=True, flip=False, stride=1):
        super(BlurLayer, self).__init__()
        if kernel is None:
            kernel = [1, 2, 1]
        kernel = torch.tensor(kernel, dtype=torch.float32)
        kernel = kernel[:, None] * kernel[None, :]
        kernel = kernel[None, None]
        if normalize:
            kernel = kernel / kernel.sum()
        if flip:
            kernel = torch.flip(kernel, [2, 3])
        self.register_buffer('kernel', kernel)
        self.stride = stride

    def forward(self, x):
        # expand kernel channels
        kernel = self.kernel.expand(x.size(1), -1, -1, -1)
        x = F.conv2d(
            x,
            kernel,
            stride=self.stride,
            padding=int((self.kernel.size(2) - 1) / 2),
            groups=x.size(1)
        )
        return x


class View(nn.Module):
    """
    返回x的大小为(x.size(0), *shape)
    """
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)


class StddevLayer(nn.Module):
    """
    在 4*4 分辨率的 block 中，构建了一个小批量标准偏差层，将特征图标准化处理，这样能让判别网络收敛得更快。
    输出特征图大小torch.Size([MB, 513, 4, 4])
    """
    def __init__(self, group_size=4, num_new_features=1):
        super().__init__()
        self.group_size = group_size
        self.num_new_features = num_new_features

    def forward(self, x):
        b, c, h, w = x.shape
        group_size = min(self.group_size, b)
        y = x.reshape([group_size, -1, self.num_new_features,
                       c // self.num_new_features, h, w])
        y = y - y.mean(0, keepdim=True)
        y = (y ** 2).mean(0, keepdim=True)
        y = (y + 1e-8) ** 0.5
        y = y.mean([3, 4, 5], keepdim=True).squeeze(3)  # don't keep the meaned-out channels
        y = y.expand(group_size, -1, -1, h, w).clone().reshape(b, self.num_new_features, h, w)
        z = torch.cat([x, y], dim=1)
        return z


class Truncation(nn.Module):
    def __init__(self, avg_latent, max_layer=8, threshold=0.7, beta=0.995):
        super().__init__()
        self.max_layer = max_layer
        self.threshold = threshold
        self.beta = beta
        self.register_buffer('avg_latent', avg_latent)

    def update(self, last_avg):
        self.avg_latent.copy_(self.beta * self.avg_latent + (1. - self.beta) * last_avg)

    def forward(self, x):
        assert x.dim() == 3
        interp = torch.lerp(self.avg_latent, x, self.threshold)
        do_trunc = (torch.arange(x.size(1)) < self.max_layer).view(1, -1, 1).to(x.device)
        return torch.where(do_trunc, interp, x)


def quchunk(x, n_qubits):
    """
    输入torch.tensor, 进行4*4分块划分，输出包含所有4*4分块的嵌套列表chunk_list
    """
    chunks = int(2 ** n_qubits / 4)
    a = torch.chunk(x, chunks=chunks, axis=0)
    chunk_list = []
    for i in range(len(a)):
        b = torch.chunk(a[i], chunks=chunks, axis=1)
        b_list = list(b)
        chunk_list.append(b_list)
    return chunk_list


def quconcat(chunk_list):
    """
    输入包含所有4*4分块的嵌套列表chunk_list, 进行拼接，输出torch.tensor
    """
    concat_list = []
    for i in range(len(chunk_list)):
        c = torch.cat(chunk_list[i], axis=1)
        concat_list.append(c)
    re_data = torch.cat(concat_list, axis=0)
    return re_data


class QuEqualizedConv0(nn.Module):
    """Quantum Conv layer with equalized learning rate and custom learning rate multiplier.
       放置5个量子门，也即有5个参数。
    """

    def __init__(self, n_qubits, gain=2 ** 0.5, use_wscale=True, lrmul=1):
        super().__init__()

        he_std = gain * 5 ** (-0.5)  # He init
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
        self.weight = nn.Parameter(nn.init.uniform_(torch.empty(5), a=0.0, b=2*np.pi) * init_std)
        
        self.n_qubits = n_qubits


    def qconv0(self):
        w = self.weight * self.w_mul
        cir = Circuit(self.n_qubits)
        for which_q in range(0, self.n_qubits, 2):
            cir.rx(which_q, w[0])
            cir.rx(which_q + 1, w[1])
            cir.rxx(which_q, which_q + 1, w[2])
            cir.rx(which_q, w[3])      
            cir.rx(which_q + 1, w[4])
        U = cir.get() 
        return U


    def forward(self, x):
        E_qconv0 = self.qconv0()
        qconv0_out = E_qconv0 @ x @ dag(E_qconv0)   
        return qconv0_out


class QuPool(nn.Module):
    """Quantum Pool layer.
       放置4个量子门，2个参数。
    """

    def __init__(self, n_qubits, gain=2 ** 0.5, use_wscale=True, lrmul=1):
        super().__init__()

        he_std = gain * 5 ** (-0.5)  # He init
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
        self.weight = nn.Parameter(nn.init.uniform_(torch.empty(2), a=0.0, b=2*np.pi) * init_std)
        
        self.n_qubits = n_qubits


    def qpool(self):
        w = self.weight * self.w_mul
        cir = Circuit(self.n_qubits)
        for which_q in range(0, self.n_qubits, 2):
            cir.rx(which_q, w[0])
            cir.rx(which_q + 1, w[1])
            cir.cnot(which_q, which_q + 1)
            cir.rx(which_q + 1, -w[1])
        U = cir.get()       
        return U


    def forward(self, x):
        E_qpool = self.qpool()
        qpool_out = E_qpool @ x @ dag(E_qpool)    
        return qpool_out


class QuBlur(nn.Module):
    """Quantum Blur layer.
       放置5个量子门，有5个参数。
    """

    def __init__(self, n_qubits, gain=2 ** 0.5, use_wscale=True, lrmul=1):
        super().__init__()

        he_std = gain * 5 ** (-0.5)  # He init
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
        self.weight = nn.Parameter(nn.init.uniform_(torch.empty(5), a=0.0, b=2*np.pi) * init_std)
        
        self.n_qubits = n_qubits


    def qblur(self):
        w = self.weight * self.w_mul
        cir = Circuit(2)
        cir.ry(0, w[0])
        cir.ry(1, w[1])
        cir.ryy(0, 1, w[2])
        cir.ry(0, w[3])      
        cir.ry(1, w[4])
        U = cir.get()      
        return U


    def forward(self, x):
        E_qblur = self.qblur()
        chunk_list = quchunk(x, self.n_qubits)
        blur_list = []
        for i in range(len(chunk_list)):
            blur_inner_list = []
            for j in range(len(chunk_list[i])):
                if chunk_list[i][j].norm() != 0:
                    blur_inner_out = E_qblur @ encoding(chunk_list[i][j]) @ dag(E_qblur)
                else:
                    blur_inner_out = chunk_list[i][j]
                blur_inner_list.append(blur_inner_out)
            blur_list.append(blur_inner_list)
        blur_out = quconcat(blur_list)
        blur_out = encoding(blur_out)  
        return blur_out


class QuEqualizedConvDown(nn.Module):
    """Quantum Conv layer with equalized learning rate and custom learning rate multiplier.
       放置5个量子门，有5个参数。
    """

    def __init__(self, n_qubits, gain=2 ** 0.5, use_wscale=True, lrmul=1):
        super().__init__()

        he_std = gain * 5 ** (-0.5)  # He init
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
        self.weight = nn.Parameter(nn.init.uniform_(torch.empty(5), a=0.0, b=2*np.pi) * init_std)
        
        self.n_qubits = n_qubits


    def qconv_down(self):
        w = self.weight * self.w_mul
        cir = Circuit(self.n_qubits)
        for which_q in range(0, self.n_qubits, 2):
            cir.rz(which_q, w[0])
            cir.rz(which_q + 1, w[1])
            cir.rzz(which_q, which_q + 1, w[2])
            cir.rz(which_q, w[3])      
            cir.rz(which_q + 1, w[4])
        U = cir.get()    
        return U


    def forward(self, x):
        E_qconv_down = self.qconv_down()
        qconv_down_out = E_qconv_down @ x @ dag(E_qconv_down)    
        return qconv_down_out


class QuPoolDown(nn.Module):
    """Quantum Pool layer.
       放置4个量子门，2个参数。
    """

    def __init__(self, n_qubits, gain=2 ** 0.5, use_wscale=True, lrmul=1):
        super().__init__()

        he_std = gain * 5 ** (-0.5)  # He init
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
        self.weight = nn.Parameter(nn.init.uniform_(torch.empty(2), a=0.0, b=2*np.pi) * init_std)
        
        self.n_qubits = n_qubits


    def qpool_down(self):
        w = self.weight * self.w_mul
        cir = Circuit(self.n_qubits)
        for which_q in range(0, self.n_qubits, 2):
            cir.rz(which_q, w[0])
            cir.rz(which_q + 1, w[1])
            cir.cnot(which_q, which_q + 1)
            cir.rz(which_q + 1, -w[1])
        U = cir.get()           
        return U


    def forward(self, x):
        E_qpool_down = self.qpool_down()
        qpool_down_out = E_qpool_down @ x @ dag(E_qpool_down)
        qpool_down_out_pt = ptrace(qpool_down_out, self.n_qubits - 2, 2)
        return qpool_down_out_pt


class QuEqualizedConvLast(nn.Module):
    """Quantum Conv layer with equalized learning rate and custom learning rate multiplier.
       放置5个量子门，有5个参数。
    """

    def __init__(self, gain=2 ** 0.5, use_wscale=True, lrmul=1):
        super().__init__()

        he_std = gain * 5 ** (-0.5)  # He init
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
        self.weight = nn.Parameter(nn.init.uniform_(torch.empty(5), a=0.0, b=2*np.pi) * init_std)


    def qconv_last(self):
        w = self.weight * self.w_mul
        cir = Circuit(2)
        cir.rx(0, w[0])
        cir.rx(1, w[1])
        cir.ryy(0, 1, w[2])
        cir.rz(0, w[3])        
        cir.rz(1, w[4])
        U = cir.get()        
        return U


    def forward(self, x):
        E_qconv_last = self.qconv_last()
        qconv_last_out = E_qconv_last @ x @ dag(E_qconv_last)
        return qconv_last_out


class QuPoolLast(nn.Module):
    """Quantum Pool layer.
       放置10个量子门，6个参数。
    """

    def __init__(self, gain=2 ** 0.5, use_wscale=True, lrmul=1):
        super().__init__()

        he_std = gain * 5 ** (-0.5)  # He init
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
        self.weight = nn.Parameter(nn.init.uniform_(torch.empty(6), a=0.0, b=2*np.pi) * init_std)


    def qpool_last(self):
        w = self.weight * self.w_mul
        cir = Circuit(2)
        cir.rx(0, w[0])
        cir.ry(0, w[1])
        cir.rz(0, w[2])
        cir.rx(1, w[3])
        cir.ry(1, w[4])
        cir.rz(1, w[5])
        cir.cnot(0, 1)
        cir.rz(1, -w[5])      
        cir.ry(1, -w[4])
        cir.rx(1, -w[3])
        U = cir.get()        
        return U


    def forward(self, x):
        E_qpool_last = self.qpool_last()
        qpool_last_out = E_qpool_last @ x @ dag(E_qpool_last)  
        return qpool_last_out


class QuDense(nn.Module):
    """Quantum dense layer.
       放置7个量子门，6个参数。
    """

    def __init__(self, gain=2 ** 0.5, use_wscale=True, lrmul=1):
        super().__init__()

        he_std = gain * 5 ** (-0.5)  # He init
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
        self.weight = nn.Parameter(nn.init.uniform_(torch.empty(6), a=0.0, b=2*np.pi) * init_std)


    def qdense(self):
        w = self.weight * self.w_mul
        cir = Circuit(2)
        cir.rx(0, w[0])
        cir.ry(0, w[1])
        cir.rz(0, w[2])
        cir.rx(1, w[3])
        cir.ry(1, w[4])
        cir.rz(1, w[5])
        cir.cnot(0, 1)
        U = cir.get()
        return U


    def forward(self, x):
        E_qdense = self.qdense()
        qdense_out = E_qdense @ x @ dag(E_qdense)
        qdense_out_measure = expecval_ZI(qdense_out, 2, 0)
        return (qdense_out_measure + 1) / 2

