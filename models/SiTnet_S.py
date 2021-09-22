"""
## Multi-Stage Progressive Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## https://arxiv.org/abs/2102.02808
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from pdb import set_trace as stx
from torchsummary import summary
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

##---------- Resizing Modules ----------
class DownSample(nn.Module):
    def __init__(self, in_channels,s_factor):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels+s_factor, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x

class UpSample(nn.Module):
    def __init__(self, in_channels,s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels+s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x

#########################################################################
def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels,out_channels,kernel_size,
        padding=(kernel_size//2),bias=bias,stride = stride)

#########################################################################
## Channel Attention Layer(CA)
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

#########################################################################
## Channel Attention Block(CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.cal   = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.cal(res)
        res += x
        return res

##########################################################################
## Supervised Attention Module
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1*x2
        x1 = x1+x
        return x1, img

##########################################################################
## Original Resolution Block (ORB)
class ORB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_cab):
        super(ORB, self).__init__()
        modules_body = []
        modules_body = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(num_cab)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

##########################################################################
class ORSNet(nn.Module):
    def __init__(self, n_feat, scale_orsnetfeats, kernel_size, reduction, act, bias, scale_unetfeats, num_cab):
        super(ORSNet, self).__init__()

        self.orb1 = ORB(n_feat+scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)
        self.orb2 = ORB(n_feat+scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)
        self.orb3 = ORB(n_feat+scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)

        self.up_enc1 = UpSample(n_feat, scale_unetfeats)
        self.up_dec1 = UpSample(n_feat, scale_unetfeats)

        self.up_enc2 = nn.Sequential(UpSample(n_feat+scale_unetfeats, scale_unetfeats), UpSample(n_feat, scale_unetfeats))
        self.up_dec2 = nn.Sequential(UpSample(n_feat+scale_unetfeats, scale_unetfeats), UpSample(n_feat, scale_unetfeats))

        self.conv_enc1 = nn.Conv2d(n_feat, n_feat+scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_enc2 = nn.Conv2d(n_feat, n_feat+scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_enc3 = nn.Conv2d(n_feat, n_feat+scale_orsnetfeats, kernel_size=1, bias=bias)

        self.conv_dec1 = nn.Conv2d(n_feat, n_feat+scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_dec2 = nn.Conv2d(n_feat, n_feat+scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_dec3 = nn.Conv2d(n_feat, n_feat+scale_orsnetfeats, kernel_size=1, bias=bias)

    def forward(self, x, encoder_outs, decoder_outs):
        x = self.orb1(x)
        # [2, 128, 256, 256]
        # [2, 96,  256, 256]
        x = x + self.conv_enc1(encoder_outs) + self.conv_dec1(decoder_outs)

        #x = self.orb2(x)

        #x = self.orb3(x)

        return x

#########################################################################
## Mlp
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

#########################################################################
## WindowPartition

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

#########################################################################
## WindowReverse
def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

#########################################################################
## WindowAttention (Supports both shifted and non-shifted)
class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads # 每一个head所计算的dim
        self.scale = qk_scale or head_dim ** -0.5 # 在Attention(Q,K,V)中的公式要除以根号下dk

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) #qkv是直接通过全连接层实现的，直接用一个全连接层得到的，和使用3个效果差不多
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim) #在head得到的结果拼接之后通过一个全连接层

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # [batch_size, num_patches , total_embed_dim]
        B_, N, C = x.shape
        # 将最后一个维度拆分成3个部分，对应了head进行拆分。
        # qkv():   -> [b, n, 3*c]
        # reshape: -> [b, n, 3,         num_heads, embed_dim_per_head]
        # permute: -> [3, b, num_heads, n       , embed_dim_per_head]
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        # 表示每个head对应的qkv进行相乘，对每个head进行操作
        # @ 是矩阵乘法，如果矩阵是多维的，那么矩阵乘法就针对最后两个维度进行操作
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, n]
        # @        : -> [batch_size, num_heads, n                 , n]
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


#########################################################################
## Swin Transformer Block

class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
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

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 256.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=256, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        #img_size -> truple(img_size,img_size) as (224,224)
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        #patched_resolution(128/4) = [32,32]
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]

        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        # the number of patches
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        #Swin tansformer 原版在这里添加了一个卷积层，前向传播的过程对输入的x应用了这个卷积层
        # self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size = patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        #这里的x是B，C，H，W 4维，经过flatten将数组变成B, C, H*W 3维，再通过transpose转换为 B，H*W，C 3维
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

    def flops(self):
        flops = 0
        return flops



#########################################################################
## BasicLayer
class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

#########################################################################
## Residual Swin Transformer Block(RSTB)
class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=256, patch_size=4, resi_connection='1conv'):
        super(RSTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         downsample=downsample,
                                         use_checkpoint=use_checkpoint)
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

    def forward(self, x, x_size):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        H, W = self.input_resolution
        flops += H * W * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops

class SiTNet_s(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=96, kernel_size=3, reduction=4, bias=False, cab=True,
                 embed_dim=96, depths=[6, 6, 6], num_heads=[6, 6, 6], window_size=8, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,norm_layer=nn.LayerNorm,
                 img_size=64, patch_size=1, resi_connection='1conv', drop_path_rate=0.1, use_checkpoint=False,
                 patch_norm=True, scale_orsnetfeats=32, scale_unetfeats=48, num_cab=8
                 ):
        super(SiTNet_s, self).__init__()

        act = nn.PReLU()
        self.mlp_ratio = mlp_ratio
        self.patch_norm = patch_norm
        num_out_ch = out_c
        #########################提取Shallow feature#############################
        # Stage:1
        #########################################################################
        if bool(cab):
            self.shallow_feat1 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias),CAB(n_feat, kernel_size, reduction, bias=bias,act=act))
            self.shallow_feat2 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias),CAB(n_feat, kernel_size, reduction, bias=bias,act=act))
            self.shallow_feat3 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias),CAB(n_feat, kernel_size, reduction, bias=bias,act=act))
        else:
            self.shallow_feat1 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias))
            self.shallow_feat2 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias))
            self.shallow_feat3 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias))

        #self.num_layer = len(depths)
        #self.stage1_rtsb =

        #########################提取Shallow feature#############################
        # Stage:1
        #########################################################################
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        # split image into non-overlapping patches

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.patch_embed3 = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.rstb1_1 = RSTB(dim=embed_dim,
                            input_resolution=(128, 128),
                            depth=depths[0],
                            num_heads=num_heads[0],
                            window_size=window_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dpr[sum(depths[:0]):sum(depths[:0 + 1])],  # no impact on SR results
                            norm_layer=norm_layer,
                            downsample=None,
                            use_checkpoint=use_checkpoint,
                            img_size=img_size,
                            patch_size=patch_size,
                            resi_connection=resi_connection
                            )
        self.rstb1_2 = RSTB(dim=embed_dim,
                            input_resolution=(128, 128),
                            depth=depths[0],
                            num_heads=num_heads[0],
                            window_size=window_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dpr[sum(depths[:0]):sum(depths[:0 + 1])],  # no impact on SR results
                            norm_layer=norm_layer,
                            downsample=None,
                            use_checkpoint=use_checkpoint,
                            img_size=img_size,
                            patch_size=patch_size,
                            resi_connection=resi_connection
                            )
        self.rstb1_3 = RSTB(dim=embed_dim,
                            input_resolution=(128, 128),
                            depth=depths[0],
                            num_heads=num_heads[0],
                            window_size=window_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dpr[sum(depths[:0]):sum(depths[:0 + 1])],  # no impact on SR results
                            norm_layer=norm_layer,
                            downsample=None,
                            use_checkpoint=use_checkpoint,
                            img_size=img_size,
                            patch_size=patch_size,
                            resi_connection=resi_connection
                            )


        self.rstb1_4 = RSTB(dim=embed_dim,
                         input_resolution=(128, 128),
                         depth=depths[0],
                         num_heads=num_heads[0],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:0]):sum(depths[:0 + 1])],  # no impact on SR results
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=patch_size,
                         resi_connection=resi_connection
                         )

        self.rstb2_1 = RSTB(dim=embed_dim,
                            input_resolution=(128, 256),
                            depth=depths[1],
                            num_heads=num_heads[1],
                            window_size=window_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dpr[sum(depths[:0]):sum(depths[:0 + 1])],  # no impact on SR results
                            norm_layer=norm_layer,
                            downsample=None,
                            use_checkpoint=use_checkpoint,
                            img_size=img_size,
                            patch_size=patch_size,
                            resi_connection=resi_connection
                            )

        self.rstb2_2 = RSTB(dim=embed_dim,
                         input_resolution=(128, 256),
                         depth=depths[1],
                         num_heads=num_heads[1],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:0]):sum(depths[:0 + 1])],  # no impact on SR results
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=patch_size,
                         resi_connection=resi_connection
                         )

        self.rstb3 = RSTB(dim=embed_dim,
                            input_resolution=(256, 256),
                            depth=depths[2],
                            num_heads=num_heads[2],
                            window_size=window_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dpr[sum(depths[:0]):sum(depths[:0 + 1])],  # no impact on SR results
                            norm_layer=norm_layer,
                            downsample=None,
                            use_checkpoint=use_checkpoint,
                            img_size=img_size,
                            patch_size=patch_size,
                            resi_connection=resi_connection
                            )

        self.num_features = embed_dim
        # 没有训练参数，可以直接使用
        self.norm = norm_layer(self.num_features)

        self.conv_last1_1 = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)
        self.conv_last1_2 = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)
        self.conv_last1_3 = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)
        self.conv_last1_4 = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.conv_last2_1 = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)
        self.conv_last2_2 = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.sam12 = SAM(n_feat, kernel_size=1, bias=bias)
        self.sam23 = SAM(n_feat, kernel_size=1, bias=bias)
        # self.sam3  = SAM(n_feat, kernel_size=1, bias=bias)

        self.concat12  = conv(n_feat*2, n_feat, kernel_size, bias=bias)
        self.concat23  = conv(n_feat*2, n_feat, kernel_size, bias=bias)
        self.tail      = conv(n_feat,   out_c,  kernel_size, bias=bias)

        #self.stage3_orsnet = ORSNet(n_feat, scale_orsnetfeats, kernel_size, reduction, act, bias, scale_unetfeats,
        #                            num_cab)

        # self.conv_last1_2 = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)
        # self.conv_last1_3 = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)
        # self.conv_last1_4 = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)
        #
        # self.conv_last2_1 = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)
        # self.conv_last2_2 = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

    def forward(self, x3_img):
        # 在第三阶段用到原始分辨率的图像得到图像的文本信息
        H = x3_img.size(2)
        W = x3_img.size(3)

        # 第二阶段分割为 Two Patches
        x2top_img = x3_img[:, :, 0:int(H / 2), :]
        x2bot_img = x3_img[:, :, int(H / 2):H, :]

        # 第一阶段分割为 four Patches
        x1ltop_img = x2top_img[:,:,:,0:int(W/2)]
        x1rtop_img = x2top_img[:,:,:,int(W/2):W]
        x1lbot_img = x2bot_img[:,:,:,0:int(W/2)]
        x1rbot_img = x2bot_img[:,:,:,int(W/2):W]

        ##################################################
        ##------------------Stage 1-----------------------
        ##################################################
        ## Compute Shallow Features
        # [2, 3, 128, 128]
        x1ltop = self.shallow_feat1(x1ltop_img)
        x1ltop_size = (x1ltop.shape[2], x1ltop.shape[3])
        # [2, 16382, 96]
        x1ltop = self.patch_embed(x1ltop)
        x1ltop = self.pos_drop(x1ltop)
        x1ltop = self.rstb1_1(x1ltop, x1ltop_size)
        # [2, 16384, 96]
        x1ltop = self.norm(x1ltop)  # B L C
        x1ltop = self.patch_unembed(x1ltop, x1ltop_size)
        # [2, 96, 128, 128]
        #x1ltop = self.conv_last1_1(x1ltop)

        x1rtop = self.shallow_feat1(x1rtop_img)
        x1rtop_size = (x1rtop.shape[2], x1rtop.shape[3])
        x1rtop = self.patch_embed(x1rtop)
        x1rtop = self.pos_drop(x1rtop)
        x1rtop = self.rstb1_2(x1rtop, x1rtop_size)
        x1rtop = self.norm(x1rtop)  # B L C
        x1rtop = self.patch_unembed(x1rtop, x1rtop_size)
        # x1rtop = self.conv_last1_2(x1rtop)

        x1lbot = self.shallow_feat1(x1lbot_img)
        x1lbot_size = (x1lbot.shape[2], x1lbot.shape[3])
        x1lbot = self.patch_embed(x1lbot)
        x1lbot = self.pos_drop(x1lbot)
        x1lbot = self.rstb1_2(x1lbot, x1lbot_size)
        x1lbot = self.norm(x1lbot)  # B L C
        x1lbot = self.patch_unembed(x1lbot, x1lbot_size)
        # [2, 3, 128, 128]
        # x1lbot = self.conv_last1_3(x1lbot)

        x1rbot = self.shallow_feat1(x1rbot_img)
        x1rbot_size = (x1rbot.shape[2], x1rbot.shape[3])
        x1rbot = self.patch_embed(x1rbot)
        x1rbot = self.pos_drop(x1rbot)
        x1rbot = self.rstb1_2(x1rbot, x1rbot_size)
        x1rbot = self.norm(x1rbot)  # B L C
        x1rbot = self.patch_unembed(x1rbot, x1rbot_size)
        # [2, 96, 128, 128]
        #x1rbot = self.conv_last1_4(x1rbot)

        ## top and bot concat
        res1_top = torch.cat([x1rtop, x1ltop], 3)
        res1_bot = torch.cat([x1rbot, x1lbot], 3)

        ## SAM modules
        # [2, 96,128, 256]
        x2top_samfeats, stage1_img_top = self.sam12(res1_top, x2top_img)
        x2bot_samfeats, stage1_img_bot = self.sam12(res1_bot, x2bot_img)

        ## Concat deep features
        #stage1_img_top = torch.cat([x1rtop, x1ltop], 2)
        #stage1_img_bot = torch.cat([x1rbot, x1lbot], 2)
        stage1_img     = torch.cat([stage1_img_top,stage1_img_bot], 2)
        ## Apply Supervised Attention Module (SAM)

        ##################################################
        ##------------------Stage 2-----------------------
        ##################################################
        ## Compute Shallow Features
        x2top  = self.shallow_feat2(x2top_img)
        x2bot = self.shallow_feat2(x2bot_img)

        # concat Stage1 feature and Stage2 feature
        # x2_top = [2, 96, 128, 256]

        x2top_cat = self.concat12(torch.cat([x2top, x2top_samfeats], 1))
        x2bot_cat = self.concat12(torch.cat([x2bot, x2bot_samfeats], 1))

        ## Process features of both patches with Transformer of Stage 2
        x2top_size = (x2top_cat.shape[2], x2top_cat.shape[3])
        # [2, 32768, 96]
        x2top_cat = self.patch_embed(x2top_cat)
        x2top_cat = self.pos_drop(x2top_cat)
        x2top_cat = self.rstb2_1(x2top_cat, x2top_size)
        x2top_cat = self.norm(x2top_cat)  # B L C
        x2top_cat = self.patch_unembed(x2top_cat, x2top_size)
        # x2top_cat = self.conv_last2_1(x2top_cat)


        x2bot_size = (x2bot_cat.shape[2], x2bot_cat.shape[3])
        x2bot_cat = self.patch_embed(x2bot_cat)
        x2bot_cat = self.pos_drop(x2bot_cat)
        x2bot_cat = self.rstb2_2(x2bot_cat, x2bot_size)
        x2bot_cat = self.norm(x2bot_cat)  # B L C
        x2bot_cat = self.patch_unembed(x2bot_cat, x2bot_size)
        # x2bot_cat = self.conv_last2_2(x2bot_cat)

        feat2 = torch.cat([x2top_cat, x2bot_cat], 2)
        #feat2 = [torch.cat((k,v), 1) for k,v in zip(x2top_cat, x2bot_cat)]

        res2 = torch.cat([x2top_cat, x2bot_cat], 2)

        x3_samfeats, stage2_img = self.sam23(res2, x3_img)

        ##################################################
        ##------------------Stage 3-----------------------
        ##################################################
        ## Compute Shallow Features
        x3      = self.shallow_feat3(x3_img)

        # [2, 128, 256, 256]
        x3_cat = self.concat23(torch.cat([x3, x3_samfeats], 1))
        # [2, 96, 256, 256]
        x3_size = (x3_cat.shape[2], x3_cat.shape[3])

        x3_cat = self.patch_embed3(x3_cat)
        x3_cat = self.pos_drop(x3_cat)
        x3_cat = self.rstb3(x3_cat, x3_size)
        x3_cat = self.norm(x3_cat)  # B L C
        x3_cat = self.patch_unembed(x3_cat, x3_size)


        # x2top_cat = self.concat12(torch.cat([x2top, x2top_samfeats], 1))
        # x3_cat = self.stage3_orsnet(x3_cat, feat2, res2)

        stage3_img = self.tail(x3_cat)

        return [stage3_img + x3_img ,stage2_img, stage1_img]

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiTNet().cuda()

    #model_wa = WindowAttention(dim=180, window_size=to_2tuple(7), num_heads=6).to(device)
    #net = net.to(torch.device("cpu"))
    summary(model, (3, 256, 256))
