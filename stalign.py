import os
import warnings
from resnet.resnet import resnet50
from einops import rearrange
import torch
import torch.nn as nn
import timm

# Suppress warnings
warnings.filterwarnings("ignore")


class IMGGlobalBottleNeck(nn.Module):
    """
    Image Global BottleNeck Module.

    This module normalizes the input image tensor, applies adaptive average pooling
    to a fixed size, and performs a convolution operation to preserve the number
    of channels.
    """
    def __init__(self):
        super(IMGGlobalBottleNeck, self).__init__()
        self.norm = nn.BatchNorm2d(3)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((224, 224))
        self.conv = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """
        Forward pass for the IMGGlobalBottleNeck module.

        Parameters:
            x (torch.Tensor): Input tensor with shape (batch_size, 3, height, width).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, 3, 224, 224).
        """
        assert x.ndim == 4, f"Expected input to be a 4D tensor but got {x.ndim}D"
        x = self.norm(x)
        x = self.adaptive_pool(x)
        x = self.conv(x)
        return x


class UNIEncoder(nn.Module):
    """
    Universal Image Transformer Encoder.

    This module initializes a Vision Transformer (ViT) model using the `timm` library,
    loads pre-trained weights, and processes global image features.
    """
    def __init__(self,
                 model_local_dir='/pretrain_model',
                 device='cuda',
                 img_size=224,
                 patch_size=16,
                 init_values=1e-5,
                 num_classes=0,
                 dynamic_img_size=True):
        super(UNIEncoder, self).__init__()
        self.vit = timm.create_model(
            "vit_large_patch16_224",
            img_size=img_size,
            patch_size=patch_size,
            init_values=init_values,
            num_classes=num_classes,
            dynamic_img_size=dynamic_img_size
        )
        self.vit.load_state_dict(
            torch.load(os.path.join(model_local_dir, "pytorch_model.bin"), map_location=device),
            strict=True
        )

    def forward(self, global_feature):
        """
        Forward pass for the UNIEncoder module.

        Parameters:
            global_feature (torch.Tensor): Input tensor with shape (batch_size, 3, 224, 224).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, 1024).
        """
        out = self.vit(global_feature)
        return out


class ConvUNI(nn.Module):
    """
    Convolutional Layer for Universal Image Features.

    This module applies a 1D convolution to the input global image features.
    """
    def __init__(self):
        super(ConvUNI, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=2, stride=2)

    def forward(self, global_img):
        """
        Forward pass for the ConvUNI module.

        Parameters:
            global_img (torch.Tensor): Input tensor with shape (batch_size, 1024).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, 8, 512).
        """
        img_feature = global_img.unsqueeze(1)
        img_feature = self.conv1(img_feature)
        return img_feature


class GlobalImgPath(nn.Module):
    """
    Global Image Processing Path.

    This module sequentially applies image bottleneck normalization, encoding,
    and convolution to process global image features.
    """
    def __init__(self):
        super(GlobalImgPath, self).__init__()
        self.global_img = nn.Sequential(
            IMGGlobalBottleNeck(),
            UNIEncoder(),
            ConvUNI()
        )

    def forward(self, global_img_feature):
        """
        Forward pass for the GlobalImgPath module.

        Parameters:
            global_img_feature (torch.Tensor): Input tensor with shape (batch_size, 3, height, width).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, 8, 512).
        """
        out = self.global_img(global_img_feature)
        return out


class ResEncoder(nn.Module):
    """
    ResNet-50 Encoder.

    This module initializes a ResNet-50 model for encoding local image features.
    """
    def __init__(self):
        super(ResEncoder, self).__init__()
        self.resnet = resnet50()

    def forward(self, local_feature):
        """
        Forward pass for the ResEncoder module.

        Parameters:
            local_feature (torch.Tensor): Input tensor with shape (batch_size, 3, height, width).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, 512).
        """
        out = self.resnet(local_feature)
        return out


class ConvImgLocal(nn.Module):
    """
    Convolutional Layer for Local Image Features.

    This module applies a 1D convolution to the input local image features.
    """
    def __init__(self):
        super(ConvImgLocal, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=1)

    def forward(self, local_img):
        """
        Forward pass for the ConvImgLocal module.

        Parameters:
            local_img (torch.Tensor): Input tensor with shape (batch_size, 512).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, 8, 512).
        """
        img_feature = local_img.unsqueeze(1)
        img_feature = self.conv1(img_feature)
        return img_feature


class LocalImgPath(nn.Module):
    """
    Local Image Processing Path.

    This module sequentially applies ResNet encoding and convolution to process
    local image features.
    """
    def __init__(self):
        super(LocalImgPath, self).__init__()
        self.local_img = nn.Sequential(
            ResEncoder(),
            ConvImgLocal()
        )

    def forward(self, local_feature):
        """
        Forward pass for the LocalImgPath module.

        Parameters:
            local_feature (torch.Tensor): Input tensor with shape (batch_size, 3, height, width).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, 8, 512).
        """
        out = self.local_img(local_feature)
        return out


class Flatten3DTo2D(nn.Module):
    """
    Flatten 3D Tensor to 2D Tensor.

    This module flattens a 3D tensor to a 2D tensor using a Multi-Layer Perceptron (MLP)
    and reshapes it to a specified output shape.
    """
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
        """
        Forward pass for the Flatten3DTo2D module.

        Parameters:
            x (torch.Tensor): Input tensor with shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, *output_shape).
        """
        x = self.mlp(x)
        x = x.view(-1, *self.output_shape)
        return x


class PreNorm(nn.Module):
    """
    Pre-Layer Normalization Module.

    This module applies layer normalization before passing the input to a specified function.
    """
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        """
        Forward pass for the PreNorm module.

        Parameters:
            x (torch.Tensor): Input tensor.
            **kwargs: Additional keyword arguments for the function `fn`.

        Returns:
            torch.Tensor: Output tensor after normalization and function application.
        """
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """
    Feed-Forward Network Module.

    This module implements a simple feed-forward neural network with GELU activation
    and dropout layers.
    """
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Forward pass for the FeedForward module.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after feed-forward processing.
        """
        return self.net(x)


class Attention(nn.Module):
    """
    Multi-Head Self-Attention Module.

    This module implements multi-head self-attention with scaling, softmax,
    and dropout.
    """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.1):
        super(Attention, self).__init__()
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
        """
        Forward pass for the Attention module.

        Parameters:
            x (torch.Tensor): Input tensor with shape (batch_size, seq_len, dim).

        Returns:
            torch.Tensor: Output tensor after attention.
        """
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads),
            qkv
        )

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    """
    Transformer Encoder Module.

    This module stacks multiple layers of multi-head self-attention and feed-forward networks.
    """
    def __init__(self, dim=548, depth=6, heads=8, dim_head=64, mlp_dim=1024, dropout=0.1):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]) for _ in range(depth)
        ])

    def forward(self, x):
        """
        Forward pass for the Transformer module.

        Parameters:
            x (torch.Tensor): Input tensor with shape (batch_size, seq_len, dim).

        Returns:
            torch.Tensor: Output tensor after transformer encoding.
        """
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class MLPReshapeTo512(nn.Module):
    """
    MLP for Reshaping to 512 Dimensions.

    This module applies an MLP to transform the input tensor to a 512-dimensional space
    and reshapes it accordingly.
    """
    def __init__(self, input_dim=548, output_dim=512):
        super(MLPReshapeTo512, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        """
        Forward pass for the MLPReshapeTo512 module.

        Parameters:
            x (torch.Tensor): Input tensor with shape (batch_size, seq_len, input_dim).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, seq_len, output_dim).
        """
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size * seq_len, -1)
        x = self.mlp(x)
        x = x.view(batch_size, seq_len, -1)
        return x


class GlobalGenePath(nn.Module):
    """
    Global Gene Processing Path.

    This module processes global gene features by flattening, applying a transformer,
    and reshaping to a 512-dimensional space.
    """
    def __init__(self):
        super(GlobalGenePath, self).__init__()
        self.global_gene = nn.Sequential(
            Flatten3DTo2D(),
            Transformer(),
            MLPReshapeTo512()
        )

    def forward(self, global_gene):
        """
        Forward pass for the GlobalGenePath module.

        Parameters:
            global_gene (torch.Tensor): Input tensor with shape (batch_size, 4384).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, 8, 548).
        """
        return self.global_gene(global_gene)


class LocalGenePath(nn.Module):
    """
    Local Gene Processing Path.

    This module applies a convolution to local gene features.
    """
    def __init__(self):
        super(LocalGenePath, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=1)

    def forward(self, local_gene):
        """
        Forward pass for the LocalGenePath module.

        Parameters:
            local_gene (torch.Tensor): Input tensor with shape (batch_size, 512).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, 8, 512).
        """
        gene_feature = local_gene.unsqueeze(1)
        gene_feature = self.conv1(gene_feature)
        return gene_feature


class CrossAttention(nn.Module):
    """
    Cross-Attention Module.

    This module implements cross-attention between two input tensors, allowing
    one to attend to the other.
    """
    def __init__(self, dim_q=512, dim_kv=512, heads=8, dim_head=64, dropout=0.1):
        """
        Initializes the CrossAttention module.

        Parameters:
            dim_q (int): Dimension of the query.
            dim_kv (int): Dimension of the key and value.
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

        self.norm_q = nn.LayerNorm(dim_q)
        self.norm_kv = nn.LayerNorm(dim_kv)

    def forward(self, x, y):
        """
        Forward pass for the CrossAttention module.

        Parameters:
            x (torch.Tensor): Query tensor with shape (batch_size, seq_len_q, dim_q).
            y (torch.Tensor): Key/Value tensor with shape (batch_size, seq_len_kv, dim_kv).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Output tensor with shape (batch_size, seq_len_q, dim_q).
                - Attention heatmap with shape (batch_size, heads, seq_len_q, seq_len_kv).
        """
        b, seq_len_q, c_q = x.shape
        _, seq_len_kv, c_kv = y.shape

        # Apply LayerNorm
        x = self.norm_q(x)
        y = self.norm_kv(y)

        q = self.to_q(x)
        k = self.to_k(y)
        v = self.to_v(y)

        q = q.view(b, seq_len_q, self.heads, -1).transpose(1, 2)  # (b, heads, seq_len_q, dim_head)
        k = k.view(b, seq_len_kv, self.heads, -1).transpose(1, 2)  # (b, heads, seq_len_kv, dim_head)
        v = v.view(b, seq_len_kv, self.heads, -1).transpose(1, 2)  # (b, heads, seq_len_kv, dim_head)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (b, heads, seq_len_q, seq_len_kv)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (b, heads, seq_len_q, dim_head)
        out = out.transpose(1, 2).reshape(b, seq_len_q, -1)  # (b, seq_len_q, heads * dim_head)
        out = self.to_out(out)

        return out, attn


class LocalFusion(nn.Module):
    """
    Local Feature Fusion Module.

    This module fuses local image and gene features using cross-attention and
    feed-forward networks.
    """
    def __init__(self, channel=512):
        super(LocalFusion, self).__init__()
        self.cross_attn_x = CrossAttention(dim_q=channel, dim_kv=channel)
        self.cross_attn_y = CrossAttention(dim_q=channel, dim_kv=channel)

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
        """
        Forward pass for the LocalFusion module.

        Parameters:
            img_spot (torch.Tensor): Image feature tensor with shape (batch_size, 8, 512).
            gene_spot (torch.Tensor): Gene feature tensor with shape (batch_size, 8, 512).

        Returns:
            torch.Tensor: Fused feature tensor with shape (batch_size, 512).
        """
        feature_one, _ = self.cross_attn_x(img_spot, gene_spot)
        feature_two, _ = self.cross_attn_y(gene_spot, img_spot)

        feature_one_down = self.ffn_one(feature_one)
        feature_two_down = self.ffn_two(feature_two)

        fuse_feat = torch.cat([feature_one_down, feature_two_down], dim=-1)
        out = self.ln_fuse(fuse_feat)
        return out


class GlobalFusion(nn.Module):
    """
    Global Feature Fusion Module.

    This module fuses global image and gene features using cross-attention and
    feed-forward networks.
    """
    def __init__(self, channel=512):
        super(GlobalFusion, self).__init__()
        self.cross_attn_x = CrossAttention(dim_q=channel, dim_kv=channel)
        self.cross_attn_y = CrossAttention(dim_q=channel, dim_kv=channel)

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
        """
        Forward pass for the GlobalFusion module.

        Parameters:
            img_global (torch.Tensor): Global image feature tensor with shape (batch_size, 8, 512).
            gene_global (torch.Tensor): Global gene feature tensor with shape (batch_size, 8, 512).

        Returns:
            torch.Tensor: Fused feature tensor with shape (batch_size, 512).
        """
        feature_one, _ = self.cross_attn_x(img_global, gene_global)
        feature_two, _ = self.cross_attn_y(gene_global, img_global)

        feature_one_down = self.ffn_one(feature_one)
        feature_two_down = self.ffn_two(feature_two)

        fuse_feat = torch.cat([feature_one_down, feature_two_down], dim=-1)
        out = self.ln_fuse(fuse_feat)
        return out


class STAlign(nn.Module):
    """
    STAlign Model.

    This module integrates image and gene processing paths, performs feature fusion,
    and applies MLPs to generate aligned features.
    """
    def __init__(self):
        super(STAlign, self).__init__()
        self.img_local = LocalImgPath()
        self.img_global = GlobalImgPath()
        self.gene_local = LocalGenePath()
        self.gene_global = GlobalGenePath()
        self.global_fusion = GlobalFusion()
        self.local_fusion = LocalFusion()

        self.img_local_mlp = nn.Sequential(
            nn.Linear(8 * 512, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
        )

        self.gene_local_mlp = nn.Sequential(
            nn.Linear(8 * 512, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
        )

        self.local_mlp = nn.Sequential(
            nn.Linear(8 * 512, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
        )

        self.global_mlp = nn.Sequential(
            nn.Linear(8 * 512, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
        )

    def forward(self, local_img, local_gene, global_img, global_gene):
        """
        Forward pass for the STAlign module.

        Parameters:
            local_img (torch.Tensor): Local image tensor with shape (batch_size, 3, height, width).
            local_gene (torch.Tensor): Local gene tensor with shape (batch_size, 512).
            global_img (torch.Tensor): Global image tensor with shape (batch_size, 3, height, width).
            global_gene (torch.Tensor): Global gene tensor with shape (batch_size, 4384).

        Returns:
            Tuple[torch.Tensor, ...]:
                - local_img_feature (torch.Tensor).
                - local_gene_feature (torch.Tensor).
                - global_img_feature (torch.Tensor).
                - global_gene_feature (torch.Tensor).
                - local_feature (torch.Tensor).
                - global_feature (torch.Tensor).
        """
        # Process local features
        local_img_feature = self.img_local(local_img)
        local_gene_feature = self.gene_local(local_gene)

        # Process global features
        global_img_feature = self.img_global(global_img)
        global_gene_feature = self.gene_global(global_gene)

        # Fuse local and global features
        local_feature = self.local_fusion(local_img_feature, local_gene_feature)
        global_feature = self.global_fusion(global_img_feature, global_gene_feature)

        # Apply MLPs
        local_img_feature = self.img_local_mlp(
            local_img_feature.view(local_img_feature.size(0), -1)
        )
        local_gene_feature = self.gene_local_mlp(
            local_gene_feature.view(local_gene_feature.size(0), -1)
        )

        local_feature = self.local_mlp(
            local_feature.view(local_feature.size(0), -1)
        )
        global_feature = self.global_mlp(
            global_feature.view(global_feature.size(0), -1)
        )

        return (
            local_img_feature,
            local_gene_feature,
            global_img_feature,
            global_gene_feature,
            local_feature,
            global_feature
        )
