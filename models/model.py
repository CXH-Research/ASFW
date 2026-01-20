import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
import numpy as np
from einops import rearrange
import kornia


def conv(X, W, s):
    x1_use = X[:,:,s,:,:]
    x1_out = torch.einsum('ncskj,dckj->nds',x1_use,W)
    return x1_out


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation = 1, bias = True, groups = 1):
        super(ConvLayer, self).__init__()
        reflection_padding = (kernel_size + (dilation - 1)*(kernel_size - 1))//2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, groups = groups, bias = bias, dilation = dilation)
        
        self.normalization = LayerNorm(out_channels)
        self.activation = nn.GELU()
          
    def forward(self, x): 
        out = self.conv2d(self.reflection_pad(x))
        if self.normalization is not None:
            out = self.normalization(out)
        if self.activation is not None:
            out = self.activation(out)
        
        return out
    

class DynConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation = 1, bias = True):
        super(DynConvLayer, self).__init__()
        reflection_padding = (kernel_size + (dilation - 1)*(kernel_size - 1))//2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias = bias, dilation = dilation)
        self.augment_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias = bias, dilation = dilation)
        self.light_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, groups=in_channels, bias = bias, dilation = dilation)
        self.dilation = dilation
        self.groups = in_channels

        self.normalization = None
        self.augment_normalization = None
          
        self.activation = nn.GELU()

    def forward(self, x, mask):

        mask_d = kornia.morphology.dilation(mask, torch.ones(self.dilation * 2 + 1, self.dilation * 2 + 1).to(x.device))
        mask_e = kornia.morphology.erosion(mask, torch.ones(self.dilation * 2 + 1, self.dilation * 2 + 1).to(x.device))

        if self.training:
            out_shadow = self.conv2d(self.reflection_pad(x))
            if self.normalization is not None:
                out_shadow = self.normalization(out_shadow)
            if self.activation is not None:
                out_shadow = self.activation(out_shadow)

            out_shadow = self.augment_conv2d(self.reflection_pad(out_shadow))
            if self.normalization is not None:
                out_shadow = self.normalization(out_shadow)
            if self.activation is not None:
                out_shadow = self.activation(out_shadow)

            # Unshadow region color mapping
            out_unshadow = self.light_conv2d(self.reflection_pad(x))
            if self.normalization is not None:
                out_unshadow = self.normalization(out_unshadow)
            if self.activation is not None:
                out_unshadow = self.activation(out_unshadow)

            for i, out_s in enumerate(out_shadow):
                if i == 0:
                    if mask[i].sum() <= 100 or mask[i].sum() > 0.99 * mask[i].numel():
                        out_shadow_mean = out_shadow[i].mean(dim=(1, 2)).unsqueeze(0)
                        out_unshadow_mean = out_unshadow[i].mean(dim=(1, 2)).unsqueeze(0)
                        continue
                    out_shadow_mean = out_shadow[i][(mask[i] - mask_e[i]).expand_as(x[i]) == 1].reshape(
                        (x.shape[1], -1)).mean(-1).unsqueeze(0)
                    out_unshadow_mean = out_unshadow[i][(mask_d[i] - mask[i]).expand_as(x[i]) == 1].reshape(
                        (x.shape[1], -1)).mean(-1).unsqueeze(0)
                else:
                    if mask[i].sum() <= 100 or mask[i].sum() > 0.99 * mask[i].numel():
                        out_shadow_mean = torch.cat((out_shadow_mean, out_shadow[i].mean(dim=(1, 2)).unsqueeze(0)),
                                                    dim=0)
                        out_unshadow_mean = torch.cat(
                            (out_unshadow_mean, out_unshadow[i].mean(dim=(1, 2)).unsqueeze(0)), dim=0)
                        continue
                    out_shadow_mean = torch.cat((out_shadow_mean,
                                                 out_shadow[i][(mask[i] - mask_e[i]).expand_as(x[i]) == 1].reshape(
                                                     (x.shape[1], -1)).mean(dim=-1).unsqueeze(0)), dim=0)
                    out_unshadow_mean = torch.cat((out_unshadow_mean,
                                                   out_unshadow[i][(mask_d[i] - mask[i]).expand_as(x[i]) == 1].reshape(
                                                       (x.shape[1], -1)).mean(-1).unsqueeze(0)), dim=0)
            return mask * (out_shadow + x) + (1 - mask) * (out_unshadow + x)

        else:
            #T1 = time.time()
            shape = x.shape
            Bx,Cx,Hx,Wx = x.shape
            dilation = self.dilation

            mask_d1 = kornia.morphology.dilation(mask, torch.ones(self.dilation * 2 + 1, self.dilation * 2 + 1).to(x.device))

            md = torch.flatten(mask_d1).bool()
            sd = torch.flatten(mask).bool()

            x_ori = x.clone()

            w_conv_1 = self.conv2d.weight
            FN, C, ksize1, ksize, = w_conv_1.shape
            x1 = self.reflection_pad(x_ori)
            x_k = F.unfold(x1, ksize, dilation=dilation, stride=1) #N*(Ckk)*(hw)
            x_k_k = x_k.reshape(x_k.shape[0],Cx,ksize,ksize,x_k.shape[2]).permute(0,1,4,2,3) #N*C*(hw)*k*k
            out_shadow_conv = conv(x_k_k,w_conv_1,md)+self.conv2d.bias.unsqueeze(0).unsqueeze(-1) #N*C*num(mask)

            if self.normalization is not None:
                out_shadow_conv = self.normalization(out_shadow_conv)
            if self.activation is not None:
                out_shadow_conv = self.activation(out_shadow_conv)


            x_ori = x_ori.reshape(x_ori.shape[0],x_ori.shape[1],-1)
            x_ori[:, :, md] = out_shadow_conv
            x_ori = x_ori.reshape(shape)

            x1 = self.reflection_pad(x_ori)
            x_k = F.unfold(x1, ksize, dilation=dilation, stride=1)
            x_k_k = x_k.reshape(x_k.shape[0], Cx, ksize, ksize, x_k.shape[2]).permute(0, 1, 4, 2, 3)
            w_conv_2 = self.augment_conv2d.weight
            out_shadow_conv = conv(x_k_k, w_conv_2, sd) + self.augment_conv2d.bias.unsqueeze(0).unsqueeze(-1)

            if self.normalization is not None:
                out_shadow_conv = self.normalization(out_shadow_conv)
            if self.activation is not None:
                out_shadow_conv = self.activation(out_shadow_conv)

            ns = ~sd
            x1 = self.reflection_pad(x)
            x_k = F.unfold(x1, ksize, dilation=dilation, stride=1)  # N*(Ckk)*(hw)
            x_k_k = x_k.reshape(x_k.shape[0], Cx, ksize, ksize, x_k.shape[2]).permute(0, 1, 4, 2, 3)  # N*C*(hw)*k*k
            x_k_k = x_k_k.reshape(x_k_k.shape[0], self.groups, -1, x_k_k.shape[2], x_k_k.shape[3],x_k_k.shape[4])  # N*g*(C/g)*(hw)*k*k
            w_conv_3 = self.light_conv2d.weight
            d, c_g, k, j = w_conv_3.shape
            w_conv_3_1 = w_conv_3.reshape(self.groups, -1, c_g, k, j)  # g*(d/g)*c_g*k*k
            x1_use = x_k_k[:, :, :, ns, :, :]
            x1_out = torch.einsum('ngcskj,gdckj->ngds', x1_use, w_conv_3_1)
            x1_out_1 = x1_out.reshape(x1_out.shape[0], d, x1_out.shape[3])
            out_unshadow_conv = x1_out_1 + self.light_conv2d.bias.unsqueeze(0).unsqueeze(-1)

            if self.normalization is not None:
                out_unshadow_conv = self.normalization(out_unshadow_conv)
            if self.activation is not None:
                out_unshadow_conv = self.activation(out_unshadow_conv)

            x_ori = x_ori.reshape(x_ori.shape[0],x_ori.shape[1],-1)
            x_ori[:, :, sd] = out_shadow_conv
            x_ori[:, :, ns] = out_unshadow_conv
            x_ori = x_ori.reshape(shape)

            x = x_ori + x
            #T2 = time.time()
            #print('runtime:%s ms' % ((T2 - T1) * 1000))

            return x
        

class Self_Attention(nn.Module):
    def __init__(self, channels, k):
      super(Self_Attention, self).__init__()
      self.channels = channels
      self.k = k
      
      self.linear1 = nn.Linear(channels, channels//k)
      self.linear2 = nn.Linear(channels//k, channels)
      self.global_pooling = nn.AdaptiveAvgPool2d((1,1))
      
      self.activation = nn.GELU()
      
    def attention(self, x):
      N, C, H, W = x.size()
      out = torch.flatten(self.global_pooling(x), 1)
      out = self.activation(self.linear1(out))
      out = torch.sigmoid(self.linear2(out)).view(N, C, 1, 1)
      
      return out.mul(x)
      
    def forward(self, x):
      return self.attention(x)
    

class Aggreation(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Aggreation, self).__init__()
        self.attention = Self_Attention(in_channels, k = 8)
        self.conv = ConvLayer(in_channels, out_channels, kernel_size=3, stride=1, dilation = 1)
      
    def forward(self, x):
        return self.conv(self.attention(x))
    

class AggBlock(nn.Module):
    def __init__(self, channels, ks1, ks2, dilation1, dilation2, mode=2):
        super(AggBlock, self).__init__()
        self.mode = mode
        self.block1 = DynConvLayer(in_channels=channels, out_channels=channels,
                                    kernel_size=ks1, stride=1, dilation=dilation1)
        self.block2 = DynConvLayer(in_channels=channels, out_channels=channels,
                                    kernel_size=ks2, stride=1, dilation=dilation2)

        self.aggreation0_rgb = Aggreation(
            in_channels=channels*self.mode, out_channels=channels)
      
    def forward(self, x, mask, prev_x=None):
        out_1 = self.block1(x, mask)
        out_2 = self.block2(out_1, mask)

        if self.mode == 2:
            out = self.aggreation0_rgb(torch.cat((out_1, out_2), dim = 1))
        elif self.mode == 3:
            out = self.aggreation0_rgb(torch.cat((x, out_1, out_2), dim = 1))
        else:
            out = self.aggreation0_rgb(torch.cat((x, out_1, out_2, prev_x), dim = 1))

        return out
    

def window_partition(x, window_size: int, h, w):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size(M)

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    pad_l = pad_t = 0
    pad_r = (window_size - w % window_size) % window_size
    pad_b = (window_size - h % window_size) % window_size
    x = F.pad(x, [pad_l, pad_r, pad_t, pad_b])
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size,
               W // window_size, window_size)
    windows = x.permute(0, 1, 2, 4, 3, 5).contiguous(
    ).view(-1, C, window_size, window_size)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size(M)
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    pad_l = pad_t = 0
    pad_r = (window_size - W % window_size) % window_size
    pad_b = (window_size - H % window_size) % window_size
    H = H+pad_b
    W = W+pad_r
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, -1, H // window_size, W //
                     window_size, window_size, window_size)
    x = x.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, -1, H, W)
    windows = F.pad(x, [pad_l, -pad_r, pad_t, -pad_b])
    return windows


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*3)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.relu(x1) * x2
        x = self.project_out(x)
        return x


class SWPSA(nn.Module):
    def __init__(self, dim, window_size, shift_size, bias):
        super(SWPSA, self).__init__()
        self.window_size = window_size
        self.shift_size = shift_size

        self.qkv_conv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1,groups=dim * 3, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=3, padding=1,bias=bias)
        self.project_out1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1,bias=bias)

        self.qkv_conv1 = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv1 = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1,groups=dim * 3, bias=bias)

    def window_partitions(self,x, window_size: int):
        """
        Args:
            x: (B, H, W, C)
            window_size (int): window size(M)

        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        return windows

    def create_mask(self, x):

        n,c,H,W = x.shape
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # [1, Hp, Wp, 1]
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = self.window_partitions(img_mask, self.window_size)  # [nW, Mh, Mw, 1]
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # [nW, Mh*Mw]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
        # [nW, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x):
        shortcut = x
        b, c, h, w = x.shape

        x = window_partition(x,self.window_size,h,w)

        qkv = self.qkv_dwconv(self.qkv_conv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b c h w -> b c (h w)')
        k = rearrange(k, 'b c h w -> b c (h w)')
        v = rearrange(v, 'b c h w -> b c (h w)')

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q.transpose(-2,-1) @ k)/self.window_size
        attn = attn.softmax(dim=-1)
        out = (v @ attn )
        out = rearrange(out, 'b c (h w) -> b c h w', h=int(self.window_size),
                        w=int(self.window_size))
        out = self.project_out(out)
        out = window_reverse(out,self.window_size,h,w)

        shift = torch.roll(out,shifts=(-self.shift_size,-self.shift_size),dims=(2,3))
        shift_window = window_partition(shift,self.window_size,h,w)
        qkv = self.qkv_dwconv1(self.qkv_conv1(shift_window))
        q, k,v  = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b c h w -> b c (h w)')
        k = rearrange(k, 'b c h w -> b c (h w)')
        v = rearrange(v, 'b c h w -> b c (h w)')

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q.transpose(-2,-1) @ k)/self.window_size
        mask = self.create_mask(shortcut)
        attn = attn.view(b,-1,self.window_size*self.window_size,self.window_size*self.window_size) + mask.unsqueeze(0)
        attn = attn.view(-1,self.window_size*self.window_size,self.window_size*self.window_size)
        attn = attn.softmax(dim=-1)

        out = (v @ attn)

        out = rearrange(out, 'b c (h w) -> b c h w', h=int(self.window_size),
                        w=int(self.window_size))

        out = self.project_out1(out)
        out = window_reverse(out,self.window_size,h,w)
        out = torch.roll(out,shifts=(self.shift_size,self.shift_size),dims=(2,3))

        return out


class SWPSATransformerBlock(nn.Module):
    def __init__(self, dim, window_size=8, shift_size=3, bias=False):
        super(SWPSATransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = SWPSA(dim, window_size, shift_size, bias)
        self.norm2 = LayerNorm(dim)
        self.ffn = FeedForward(dim,bias)

    def forward(self, x):
        y = self.attn(self.norm1(x))
        diffY = x.size()[2] - y.size()[2]
        diffX = x.size()[3] - y.size()[3]

        y = F.pad(y, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = x + y
        x = x + self.ffn(self.norm2(x))

        return x
    

class ResidualBlock(nn.Module):
    """残差块，用于细节增强"""
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            LayerNorm(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            LayerNorm(in_channels)
        )
    
    def forward(self, x):
        return x + self.conv(x)
    

class MSGNet(nn.Module):
    """多尺度梯度网络"""

    def __init__(self):
        super().__init__()
        self.down1 = nn.Conv2d(4, 32, 3, padding=1)  # 1/2
        self.down2 = nn.Conv2d(32, 64, 3, padding=1)  # 1/4

        self.res_blocks = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64)
        )

        self.up1 = nn.Conv2d(64, 32, 3, padding=1)
        self.up2 = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.down1(x))
        x = F.relu(self.down2(x))
        x = self.res_blocks(x)
        x = F.relu(self.up1(x))
        return torch.sigmoid(self.up2(x))


class CoarseGenerator(nn.Module):
    """带Swin-Transformer的U-Net生成器"""
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(4, 64, 3, 1, 1),  # 224 -> 112
            nn.GELU(),
            LayerNorm(64)
        )
        
        # D-DHAN
        self.block1 = AggBlock(64, ks1=1, ks2=3, dilation1=1,
                            dilation2=1, mode=2)
        self.block2 = AggBlock(64, ks1=3, ks2=3, dilation1=2,
                            dilation2=4, mode=3)
        self.block3 = AggBlock(64, ks1=3, ks2=3, dilation1=8,
                            dilation2=16, mode=3)
        self.block4 = AggBlock(64, ks1=3, ks2=3, dilation1=32,
                            dilation2=64, mode=4)
        
        # Decoder
        self.final = nn.Sequential(
            nn.Conv2d(64, 3, 3, padding=1),
            nn.GELU()
        )
    
    def forward(self, x, mask):
        # 输入拼接阴影掩膜
        x = torch.cat([x, mask], dim=1)
        
        # Encoder
        e = self.enc1(x)    # [B, 64, 112, 112]
        
        b0 = self.block1(e, mask)
        b1 = self.block2(b0, mask)
        b2 = self.block3(b1, mask)
        b3 = self.block4(b2, mask, b1)
        
        # Decoder
        return self.final(b3)
    

class ConditionNet(nn.Module):
    def __init__(self, in_ch=3, nf=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, nf, 7, 2, 1)
        self.conv2 = nn.Conv2d(nf, nf, 3, 2, 1)
        self.conv3 = nn.Conv2d(nf, nf, 3, 2, 1)
        self.act = nn.GELU()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        out = self.act(self.conv1(x))
        out = self.act(self.conv2(out))
        out = self.act(self.conv3(out))
        cond = self.avg_pool(out)
        return cond


class GFM(nn.Module):
    def __init__(self, cond_nf, in_nf, base_nf):
        super().__init__()
        self.mlp_scale = nn.Conv2d(cond_nf, base_nf, 1, 1, 0)
        self.mlp_shift = nn.Conv2d(cond_nf, base_nf, 1, 1, 0)
        self.conv = nn.Conv2d(in_nf, base_nf, 1, 1, 0)
        self.act = nn.GELU()

    def forward(self, x, cond):
        feat = self.conv(x)
        scale = self.mlp_scale(cond)
        shift = self.mlp_shift(cond)
        out = feat * scale + shift + feat
        out = self.act(out)
        return out


class IlluminationCorrector(nn.Module):
    def __init__(self, in_ch=4,
                 out_ch=3,
                 base_nf=64,
                 cond_nf=32):
        super().__init__()
        self.condnet = ConditionNet(in_ch, cond_nf)
        self.gfm1 = GFM(cond_nf, in_ch, base_nf)
        self.gfm2 = GFM(cond_nf, base_nf, base_nf)
        self.gfm3 = GFM(cond_nf, base_nf, out_ch)

    def forward(self, x, shadow_mask):
        cond = self.condnet(torch.cat([x, shadow_mask], dim=1))
        out = self.gfm1(torch.cat([x, shadow_mask], dim=1), cond)
        out = self.gfm2(out, cond)
        out = self.gfm3(out, cond)
        return out


class NILUT(nn.Module):
    # NILUT: neural implicit 3D LUT
    def __init__(self, in_ch=4, nf=256, n_layer=3, out_ch=3):
        super().__init__()
        self.in_ch = in_ch
        layers = list()
        layers.append(nn.Linear(in_ch, nf))
        layers.append(nn.GELU())
        for _ in range(n_layer):
            layers.append(nn.Linear(nf, nf))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(nf, out_ch))
        self.body = nn.Sequential(*layers)

    def forward(self, x, mask):
        # x size: [n, c, h, w]
        inp = torch.cat([x, mask], dim=1)
        n, c, h, w = x.size()

        inp = torch.permute(inp, (0, 2, 3, 1))
        inp = torch.reshape(inp, (n, h * w, self.in_ch))

        res = self.body(inp)
        
        res = torch.reshape(res, (n, h, w, c))
        res = torch.permute(res, (0, 3, 1, 2))

        out = x + res
        return out
    

class RefinementModule(nn.Module):
    def __init__(self):
        super().__init__()
        # 初始特征提取
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)

        # 修改下采样层，因为输入是224x224
        self.downsample = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 224 -> 112
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 112 -> 56
            nn.GELU(),
            LayerNorm(64)
        )

        # 第一个Swin block (56x56)
        self.norm1 = LayerNorm(64)  # 匹配 [B, 56, 56, 64] 的输入
        self.swin1 = SWPSATransformerBlock(64)

        # 下采样到28x28
        self.downsample2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 56 -> 28
            nn.GELU(),
            LayerNorm(128)
        )

        # 第二个Swin block (28x28)
        self.norm2 = LayerNorm(128)  # 匹配 [B, 28, 28, 128] 的输入
        self.swin2 = SWPSATransformerBlock(128)

        # 上采样回原始分辨率
        self.upsample = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # 28 -> 56
            nn.GELU(),
            LayerNorm(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 56 -> 112
            nn.GELU(),
            LayerNorm(64),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),  # 112 -> 224
            nn.GELU(),
            LayerNorm(32)
        )

        self.final_conv = nn.Conv2d(32, 3, kernel_size=3, padding=1)
        self.illum_corrector = IlluminationCorrector()
        # self.lut = NILUT()


    def forward(self, coarse_img, shadow_mask):

        # 初始特征提取
        x = self.conv1(coarse_img)

        # 下采样到56x56
        x = self.downsample(x)

        # 第一个Swin block
        b = self.norm1(x)
        b = self.swin1(b)

        # 下采样到28x28
        x = self.downsample2(b)  # [B, 128, 28, 28]

        # 第二个Swin block
        b = self.norm2(x)
        b = self.swin2(b)

        # 上采样和最终处理
        x = self.upsample(b)  # [B, 32, 224, 224]
        x = self.final_conv(x)  # [B, 3, 224, 224]

        # 光照校正和色彩增强
        corrected = self.illum_corrector(coarse_img, shadow_mask)
        # detail = self.lut(coarse_img, shadow_mask)

        return corrected + x


# class CSC_Block(nn.Module):
#     def __init__(self, dim) -> None:
#         super().__init__()

#         ker = 31
#         pad = ker // 2
#         self.in_conv = nn.Sequential(
#             nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1),
#             nn.GELU(),
#         )
#         self.out_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1)
#         # Horizontal Strip Convolution
#         self.dw_13 = nn.Conv2d(dim, dim, kernel_size=(1, ker), padding=(0, pad), stride=1, groups=dim)
#         # Vertical Strip Convolution
#         self.dw_31 = nn.Conv2d(dim, dim, kernel_size=(ker, 1), padding=(pad, 0), stride=1, groups=dim)
#         # Square Kernel Convolution
#         self.dw_33 = nn.Conv2d(dim, dim, kernel_size=ker, padding=pad, stride=1, groups=dim)
#         self.dw_11 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=dim)

#         self.act = nn.GELU()

#     def forward(self, x):
#         out = self.in_conv(x)
#         out = x + self.dw_13(out) + self.dw_31(out) + self.dw_33(out) + self.dw_11(out)
#         out = self.act(out)
#         return self.out_conv(out)
    

class Model(nn.Module):
    """完整的阴影去除网络"""
    def __init__(self):
        super().__init__()
        # 阴影掩码预测
        self.mask_predictor = MSGNet()
        
        # 粗糙生成器
        self.coarse = CoarseGenerator()
        # 精修模块
        self.refinement = RefinementModule()
    
    def forward(self, x, mask):
        
        # 预测阴影掩码
        shadow_mask = self.mask_predictor(torch.cat([x, mask], dim=1))
        
        # 粗糙去除
        coarse = self.coarse(x, shadow_mask)
        
        # 精修
        refined = self.refinement(coarse, shadow_mask)
        return refined
    

# class Ab1(nn.Module):
#     """完整的阴影去除网络"""
#     def __init__(self):
#         super().__init__()
#         # 阴影掩码预测
#         # self.mask_predictor = MSGNet()
        
#         # 粗糙生成器
#         self.coarse = CoarseGenerator()
#         # 精修模块
#         self.refinement = RefinementModule()
    
#     def forward(self, x, mask):
        
#         # 预测阴影掩码
#         # shadow_mask = self.mask_predictor(torch.cat([x, mask], dim=1))
        
#         # 粗糙去除
#         coarse = self.coarse(x, mask)
        
#         # 精修
#         refined = self.refinement(coarse, mask)
#         return refined
    

# class Ab2(nn.Module):
#     """完整的阴影去除网络"""
#     def __init__(self):
#         super().__init__()
#         # 阴影掩码预测
#         self.mask_predictor = MSGNet()
        
#         # 粗糙生成器
#         # self.coarse = CoarseGenerator()
#         # 精修模块
#         self.refinement = RefinementModule()
    
#     def forward(self, x, mask):
        
#         # 预测阴影掩码
#         shadow_mask = self.mask_predictor(torch.cat([x, mask], dim=1))
        
#         # 粗糙去除
#         # coarse = self.coarse(x, shadow_mask)
        
#         # 精修
#         refined = self.refinement(x, shadow_mask)
#         return refined
    

# class Ab3(nn.Module):
#     """完整的阴影去除网络"""
#     def __init__(self):
#         super().__init__()
#         # 阴影掩码预测
#         self.mask_predictor = MSGNet()
        
#         # 粗糙生成器
#         self.coarse = CoarseGenerator()
#         # 精修模块
#         # self.refinement = RefinementModule()
    
#     def forward(self, x, mask):
        
#         # 预测阴影掩码
#         shadow_mask = self.mask_predictor(torch.cat([x, mask], dim=1))
        
#         # 粗糙去除
#         refined = self.coarse(x, shadow_mask)
        
#         # 精修
#         # refined = self.refinement(coarse, shadow_mask)
#         return refined
    

# class Ab4(nn.Module):
#     """完整的阴影去除网络"""
#     def __init__(self):
#         super().__init__()
#         # 阴影掩码预测
#         # self.mask_predictor = MSGNet()
        
#         # 粗糙生成器
#         self.coarse = CoarseGenerator()
#         # 精修模块
#         # self.refinement = RefinementModule()
    
#     def forward(self, x, mask):
        
#         # 预测阴影掩码
#         # shadow_mask = self.mask_predictor(torch.cat([x, mask], dim=1))
        
#         # 粗糙去除
#         refined = self.coarse(x, mask)
        
#         # 精修
#         # refined = self.refinement(coarse, shadow_mask)
#         return refined
    

# class Ab5(nn.Module):
#     """完整的阴影去除网络"""
#     def __init__(self):
#         super().__init__()
#         # 阴影掩码预测
#         # self.mask_predictor = MSGNet()
        
#         # 粗糙生成器
#         # self.coarse = CoarseGenerator()
#         # 精修模块
#         self.refinement = RefinementModule()
    
#     def forward(self, x, mask):
        
#         # 预测阴影掩码
#         # shadow_mask = self.mask_predictor(torch.cat([x, mask], dim=1))
        
#         # 粗糙去除
#         # coarse = self.coarse(x, shadow_mask)
        
#         # 精修
#         refined = self.refinement(x, mask)
#         return refined
    

if __name__ == '__main__':
    model = Model().cuda().eval()
    x = torch.randn(1, 3, 256, 256).cuda()
    mask = torch.randn(1, 1, 256, 256).cuda()
    with torch.no_grad():
        out = model(x, mask)
        print(f"Output shape: {out.shape}")