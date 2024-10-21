import os
import timm
import torch
import numpy as np
import torch.nn as nn
from resnet.resnet import resnet50
from einops import rearrange
import warnings

warnings.filterwarnings("ignore")


#######################################################################################################################
#                                                      IMG PATH                                                       #
#######################################################################################################################

#######################################################################################################################
#                                                      Global                                                         #
#######################################################################################################################
class IMGGlobalBottleNeck(nn.Module):
    def __init__(self):
        super(IMGGlobalBottleNeck, self).__init__()
        self.norm = nn.BatchNorm2d(3)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((224, 224))
        self.conv = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """
        Forward pass for the ConvPoolEncoder module.

        Parameters:
        x (torch.Tensor): Input tensor with shape (batch_size, height, width, channels).
                          The spatial dimensions (height, width) can vary, but the
                          number of channels should be 3.

        Returns:
        torch.Tensor: Output tensor with shape (batch_size, 224, 224, 3),
                      where the spatial dimensions are fixed at 24x24 and the
                      number of channels is preserved as 3.
        """
        assert x.ndim == 4, f"Expected input to be a 4D tensor but got {x.ndim}D"
        x = self.norm(x)
        x = self.adaptive_pool(x)
        x = self.conv(x)
        return x


class UNIEncoder(nn.Module):
    def __init__(self,
                 model_local_dir='/mnt/public/luoling/9-23-BEGIN_FROM_HEAD/code_space/DSCAN/pretrain_model',
                 device='cuda',
                 img_size=224,
                 patch_size=16,
                 init_values=1e-5,
                 num_classes=0,
                 dynamic_img_size=True,
                 ):
        super(UNIEncoder, self).__init__()
        self.vit = timm.create_model(
            "vit_large_patch16_224",
            img_size=img_size,
            patch_size=patch_size,
            init_values=init_values,
            num_classes=num_classes,
            dynamic_img_size=dynamic_img_size
        )
        self.vit.load_state_dict(torch.load(os.path.join(model_local_dir, "pytorch_model.bin"), map_location=device),
                                 strict=True)

    def forward(self, global_feature):
        """
        Returns:
        torch.Tensor: Output tensor with shape (batch_size, 1024),
        """
        out = self.vit(global_feature)
        return out


class ConvUNI(nn.Module):
    def __init__(self):
        super(ConvUNI, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=2, stride=2)

    def forward(self, global_img):
        img_feature = global_img.unsqueeze(1)
        img_feature = self.conv1(img_feature)
        return img_feature


class GlobalImgPath(nn.Module):
    def __init__(self):
        super(GlobalImgPath, self).__init__()
        self.global_img = nn.Sequential(
            IMGGlobalBottleNeck(),
            UNIEncoder(),
            ConvUNI()
        )

    def forward(self, global_img_feature):
        """
        Returns:
        torch.Tensor: Output tensor with shape (batch_size, 8, 512),
        """
        out = self.global_img(global_img_feature)
        return out


#######################################################################################################################
#                                                      Local                                                          #
#######################################################################################################################

class ResEncoder(nn.Module):
    def __init__(self,
                 ):
        super(ResEncoder, self).__init__()
        self.resnet = resnet50()

    def forward(self, local_feature):
        """
        Returns:
        torch.Tensor: Output tensor with shape (batch_size, 512),
        """
        out = self.resnet(local_feature)
        return out


class ConvImgLocal(nn.Module):
    def __init__(self):
        super(ConvImgLocal, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=1)

    def forward(self, local_img):
        img_feature = local_img.unsqueeze(1)
        img_feature = self.conv1(img_feature)
        return img_feature


class LocalImgPath(nn.Module):
    def __init__(self):
        super(LocalImgPath, self).__init__()
        self.local_img = nn.Sequential(
            ResEncoder(),
            ConvImgLocal()
        )

    def forward(self, local_feature):
        """
        Returns:
        torch.Tensor: Output tensor with shape (batch_size, 8, 512),
        """
        out = self.local_img(local_feature)
        return out


#######################################################################################################################
#                                                     GENE PATH                                                       #
#######################################################################################################################
#######################################################################################################################
#                                                      Global                                                         #
#######################################################################################################################

# Input_size -> (batch_size, 4384)
class Flatten3DTo2D(nn.Module):
    def __init__(self, input_dim=4384, output_shape=(8, 548)):
        super(Flatten3DTo2D, self).__init__()
        self.output_shape = output_shape

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, output_shape[0] * output_shape[1])
        )

    def forward(self, x):
        x = self.mlp(x)

        x = x.view(-1, *self.output_shape)
        return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim=548, depth=6, heads=8, dim_head=64, mlp_dim=1024, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class MLPReshapeTo512(nn.Module):
    def __init__(self, input_dim=548, output_dim=512):
        super(MLPReshapeTo512, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size * seq_len, -1)
        x = self.mlp(x)

        x = x.view(batch_size, seq_len, -1)

        return x


class GlobalGenePath(nn.Module):
    def __init__(self):
        super(GlobalGenePath, self).__init__()
        self.global_gene = nn.Sequential(
            Flatten3DTo2D(),
            Transformer(),
            MLPReshapeTo512()
        )

    def forward(self, global_gene):
        return self.global_gene(global_gene)


#######################################################################################################################
#                                                      Local                                                          #
#######################################################################################################################

"""
SCGPT
--> (batch_size, 512) 
"""


class LocalGenePath(nn.Module):
    def __init__(self):
        super(LocalGenePath, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=1)

    def forward(self, local_gene):
        gene_feature = local_gene.unsqueeze(1)
        gene_feature = self.conv1(gene_feature)
        return gene_feature


#######################################################################################################################
#                                                      FUSION                                                         #
#######################################################################################################################

class CrossAttention(nn.Module):
    def __init__(self, dim_q=512, dim_kv=512, heads=8, dim_head=64, dropout=0.1):
        """
        Initializes the CrossAttention module for 3D input tensors.

        Parameters:
        dim_q (int): Dimension of the query (from x).
        dim_kv (int): Dimension of the key and value (from y).
        heads (int): Number of attention heads.
        dim_head (int): Dimension per attention head.
        dropout (float): Dropout rate.
        """
        super(CrossAttention, self).__init__()

        self.heads = heads
        self.scale = dim_head ** -0.5

        inner_dim_q = dim_head * heads
        inner_dim_kv = dim_head * heads

        self.to_q = nn.Linear(dim_q, inner_dim_q, bias=False)
        self.to_k = nn.Linear(dim_kv, inner_dim_kv, bias=False)
        self.to_v = nn.Linear(dim_kv, inner_dim_kv, bias=False)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim_q, dim_q),
            nn.Dropout(dropout)
        )

        # LayerNorm applied before attention
        self.norm_q = nn.LayerNorm(dim_q)
        self.norm_kv = nn.LayerNorm(dim_kv)

    def forward(self, x, y):
        """
        Forward pass for the CrossAttention module with 3D inputs.

        Parameters:
        x (torch.Tensor): Query tensor with shape (batch_size, seq_len, channels).
        y (torch.Tensor): Key/Value tensor with shape (batch_size, seq_len, channels).

        Returns:
        torch.Tensor: Output tensor with shape (batch_size, seq_len, channels).
        torch.Tensor: Attention heatmap tensor with shape (batch_size, heads, seq_len_q, seq_len_kv).
        """
        b, seq_len_q, c_q = x.shape
        _, seq_len_kv, c_kv = y.shape

        # Apply LayerNorm
        x = self.norm_q(x)
        y = self.norm_kv(y)

        q = self.to_q(x)
        k = self.to_k(y)
        v = self.to_v(y)

        q = q.view(b, seq_len_q, self.heads, -1).transpose(1, 2)
        k = k.view(b, seq_len_kv, self.heads, -1).transpose(1, 2)
        v = v.view(b, seq_len_kv, self.heads, -1).transpose(1, 2)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).reshape(b, seq_len_q, -1)
        out = self.to_out(out)

        return out, attn


#######################################################################################################################
#                                                LOCAL FUSION                                                         #
#######################################################################################################################
class LocalFusion(nn.Module):
    def __init__(self, channel=512):
        super(LocalFusion, self).__init__()
        self.cross_attn_x = CrossAttention()
        self.cross_attn_y = CrossAttention()

        self.ffn_one = nn.Sequential(
            nn.Linear(channel, channel // 8),
            nn.GELU(),
            nn.Linear(channel // 8, channel // 2),
        )
        self.ffn_two = nn.Sequential(
            nn.Linear(channel, channel // 8),
            nn.GELU(),
            nn.Linear(channel // 8, channel // 2),
        )
        self.ln_fuse = nn.LayerNorm(channel)

    def forward(self, img_spot, gene_spot):
        feature_one, _ = self.cross_attn_x(img_spot, gene_spot)
        feature_two, _ = self.cross_attn_y(gene_spot, img_spot)

        feature_one_down = self.ffn_one(feature_one)
        feature_two_down = self.ffn_two(feature_two)

        fuse_feat = torch.cat([feature_one_down, feature_two_down], dim=-1)
        out = self.ln_fuse(fuse_feat)
        return out


#######################################################################################################################
#                                                GLOBAL FUSION                                                        #
#######################################################################################################################
class GlobalFusion(nn.Module):
    def __init__(self, channel=512):
        super(GlobalFusion, self).__init__()
        self.cross_attn_x = CrossAttention()
        self.cross_attn_y = CrossAttention()

        self.ffn_one = nn.Sequential(
            nn.Linear(channel, channel // 8),
            nn.GELU(),
            nn.Linear(channel // 8, channel // 2),
        )
        self.ffn_two = nn.Sequential(
            nn.Linear(channel, channel // 8),
            nn.GELU(),
            nn.Linear(channel // 8, channel // 2),
        )
        self.ln_fuse = nn.LayerNorm(channel)

    def forward(self, img_global, gene_global):
        feature_one, _ = self.cross_attn_x(img_global, gene_global)
        feature_two, _ = self.cross_attn_y(gene_global, img_global)

        feature_one_down = self.ffn_one(feature_one)
        feature_two_down = self.ffn_two(feature_two)

        fuse_feat = torch.cat([feature_one_down, feature_two_down], dim=-1)
        out = self.ln_fuse(fuse_feat)
        return out


#######################################################################################################################
#                                                     DSCAN                                                           #
#######################################################################################################################
class DSCAN(nn.Module):
    def __init__(self):
        super(DSCAN, self).__init__()
        self.img_local = LocalImgPath()
        self.img_global = GlobalImgPath()
        self.gene_local = LocalGenePath()
        self.gene_global = GlobalGenePath()
        self.global_fusion = GlobalFusion()
        self.local_fusion = LocalFusion()

        self.img_local_mlp = nn.Sequential(
            nn.Linear(8*512, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
        )

        self.gene_local_mlp = nn.Sequential(
            nn.Linear(8*512, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
        )

        self.local_mlp = nn.Sequential(
            nn.Linear(8*512, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
        )

        self.global_mlp = nn.Sequential(
            nn.Linear(8*512, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
        )

    def forward(self, local_img, local_gene, global_img, global_gene):
        local_img_feature = self.img_local(local_img)
        local_gene_feature = self.gene_local(local_gene)

        global_img_feature = self.img_global(global_img)
        global_gene_feature = self.gene_global(global_gene)

        local_feature = self.local_fusion(local_img_feature, local_gene_feature)
        global_feature = self.global_fusion(global_img_feature, global_gene_feature)

        local_img_feature = self.img_local_mlp(local_img_feature.view(local_img_feature.size(0), -1))
        local_gene_feature = self.gene_local_mlp(local_gene_feature.view(local_gene_feature.size(0), -1))

        local_feature = self.local_mlp(local_feature.view(local_feature.size(0), -1))
        global_feature = self.global_mlp(global_feature.view(global_feature.size(0), -1))

        return local_img_feature, local_gene_feature, local_feature, global_feature
