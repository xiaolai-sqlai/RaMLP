from functools import partial
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers.helpers import to_2tuple
from einops import rearrange

class Downsampling(nn.Module):
    """
    Downsampling implemented by a layer of convolution.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, pre_norm=None, post_norm=None, pre_permute=False):
        super().__init__()
        self.pre_norm = pre_norm(in_channels, eps=1e-6) if pre_norm else nn.Identity()
        self.pre_permute = pre_permute
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.post_norm = post_norm(out_channels, eps=1e-6) if post_norm else nn.Identity()

    def forward(self, x):
        x = self.pre_norm(x)
        if self.pre_permute:
            # if take [B, H, W, C] as input, permute it to [B, C, H, W]
            x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1).contiguous() # [B, C, H, W] -> [B, H, W, C]
        x = self.post_norm(x)
        return x


class Scale(nn.Module):
    """
    Scale vector by element multiplications.
    """
    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        return x * self.scale

class SquaredReLU(nn.Module):
    """
        Squared ReLU: https://arxiv.org/abs/2109.08668
    """
    def __init__(self, inplace=False):
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        return torch.square(self.relu(x))


class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """
    def __init__(self, bias_value=0.0, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.bias = nn.Parameter(bias_value * torch.ones(1), requires_grad=True)

    def forward(self, x):
        return self.relu(x).pow(2) + self.bias


class SepConv(nn.Module):
    def __init__(self, dim, expansion_ratio=2, bias=False, kernel_size=8, head_dim=1):
        super().__init__()
        self.hd = head_dim
        self.ks = kernel_size
        self.scale = self.ks * self.ks * self.hd
        mid_dim = int(expansion_ratio * dim)

        self.pwconv1 = nn.Linear(dim, mid_dim, bias=bias)
        self.dwconv1 = nn.Conv2d(mid_dim, mid_dim, kernel_size=5, padding=2, groups=mid_dim, bias=bias)
        self.sfc = nn.Linear(self.ks * self.ks * self.hd, self.ks * self.ks * self.hd, bias=bias)
        self.bn1 = nn.BatchNorm2d(mid_dim)

        self.pwconv2 = nn.Linear(dim, mid_dim, bias=bias)
        self.bn2 = nn.BatchNorm2d(mid_dim)

        self.proj = nn.Linear(mid_dim, dim, bias=bias)

    def forward(self, inp):
        b, h, w, c = inp.shape
        hh = h//self.ks
        ww = w//self.ks

        x1 = self.pwconv1(inp)
        x1 = x1.permute(0, 3, 1, 2).contiguous()
        x1 = self.dwconv1(x1)
        x1 = rearrange(x1, 'b (c hd) (ss1 hh) (ss2 ww) -> b (hh ww) c (hd ss1 ss2)', ss1=self.ks, ss2=self.ks, hd=self.hd)
        x1 = self.sfc(x1 / self.scale)
        x1 = rearrange(x1, 'b (hh ww) c (hd ss1 ss2) -> b (c hd) (ss1 hh) (ss2 ww)', hh=hh, ww=ww, ss1=self.ks, hd=self.hd)
        x1 = self.bn1(x1)

        x2 = self.pwconv2(inp)
        x2 = x2.permute(0, 3, 1, 2).contiguous()
        x2 = self.bn2(x2)

        x = (x1 * x2).permute(0, 2, 3, 1).contiguous()

        x = self.proj(x)
        return x

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    Mostly copied from timm.
    """
    def __init__(self, dim, mlp_ratio=4, out_features=None, bias=False, **kwargs):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        mid_dim = int(mlp_ratio * in_features)

        self.fc1 = nn.Linear(in_features, mid_dim, bias=bias)
        self.act = StarReLU()
        self.fc2 = nn.Linear(mid_dim, out_features, bias=bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


'''class MlpHead(nn.Module):
    """ MLP classification head
    """
    def __init__(self, dim, num_classes=1000, mlp_ratio=4, bias=True):
        super().__init__()
        hidden_dim = int(mlp_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, num_classes, bias=bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x'''

class MlpHead(nn.Module):
    """ MLP classification head
    """
    def __init__(self, dim, num_classes=1000, mlp_ratio=4, act_layer=SquaredReLU, norm_layer=nn.LayerNorm, head_dropout=0., bias=True):
        super().__init__()
        hidden_features = int(mlp_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden_features, bias=bias)
        self.act = act_layer()
        self.norm = norm_layer(hidden_features)
        self.fc2 = nn.Linear(hidden_features, num_classes, bias=bias)
        self.head_dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.head_dropout(x)
        x = self.fc2(x)
        return x

class RaBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4, expansion_ratio=2, kernel_size=8, head_dim=1, drop_path=0., res_scale_init_value=None, scale_trainable=True):

        super().__init__()
        self.cpe = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)

        self.norm1 = nn.LayerNorm(dim, eps=1e-06)
        self.token_mixer = SepConv(dim=dim, expansion_ratio=expansion_ratio, kernel_size=kernel_size, head_dim=head_dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.res_scale1 = Scale(dim=dim, init_value=res_scale_init_value, trainable=scale_trainable) \
            if res_scale_init_value else nn.Identity()

        self.norm2 = nn.LayerNorm(dim, eps=1e-06)
        self.mlp = Mlp(dim=dim, mlp_ratio=mlp_ratio)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.res_scale2 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()

    def forward(self, x):
        x = x + self.cpe(x.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        x = self.res_scale1(x) + self.drop_path1(self.token_mixer(self.norm1(x)))
        x = self.res_scale2(x) + self.drop_path2(self.mlp(self.norm2(x)))
        return x

class RaMLP(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[2, 2, 6, 2],
                 dims=[64, 128, 256, 512],
                 mlp_ratio=[4, 4, 3, 3],
                 expansion_ratio=[3, 3, 2, 2],
                 kernel_size=[8, 4, 2, 1], 
                 head_dims=[1, 4, 16, 64],
                 drop_path_rate=0.,
                 res_scale_init_values=[None, None, 1.0, 1.0],
                 ):
        super().__init__()
        self.num_classes = num_classes

        if not isinstance(depths, (list, tuple)):
            depths = [depths] # it means the model has only one stage
        if not isinstance(dims, (list, tuple)):
            dims = [dims]

        num_stage = len(depths)

        self.down_0 = Downsampling(in_chans, dims[0], kernel_size=7, stride=4, padding=2, post_norm=nn.LayerNorm, pre_permute=False)
        self.down_1 = Downsampling(dims[0], dims[1], kernel_size=3, stride=2, padding=1, pre_norm=nn.LayerNorm, pre_permute=True)
        self.down_2 = Downsampling(dims[1], dims[2], kernel_size=3, stride=2, padding=1, pre_norm=nn.LayerNorm, pre_permute=True)
        self.down_3 = Downsampling(dims[2], dims[3], kernel_size=3, stride=2, padding=1, pre_norm=nn.LayerNorm, pre_permute=True)

        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        if not isinstance(res_scale_init_values, (list, tuple)):
            res_scale_init_values = [res_scale_init_values] * num_stage

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        cur = 0
        for i in range(num_stage):
            stage = nn.Sequential(
                *[RaBlock(dim=dims[i],
                mlp_ratio=mlp_ratio[i],
                expansion_ratio = expansion_ratio[i],
                kernel_size = kernel_size[i],
                head_dim = head_dims[i],
                drop_path=dp_rates[cur + j],
                res_scale_init_value=res_scale_init_values[i],
                ) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-06)
        self.head = MlpHead(dims[-1], num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'norm'}

    def forward_features(self, x):
        x = self.down_0(x)
        x = self.stages[0](x)

        x = self.down_1(x)
        x = self.stages[1](x)

        x = self.down_2(x)
        x = self.stages[2](x)

        x = self.down_3(x)
        x = self.stages[3](x)

        return self.norm(x.mean([1, 2])) # (B, H, W, C) -> (B, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

