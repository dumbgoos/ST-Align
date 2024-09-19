# @author: Ling LUO
# @date: 9.2.2024

import numpy as np
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

# ST Path
class ConvPoolEncoder(nn.Module):
    def __init__(self):
        """
        Initializes the ConvPoolEncoder module.

        This module takes an input tensor with variable spatial dimensions 
        and processes it through a batch normalization layer, then an adaptive
        pooling layer to standardize the spatial dimensions to 24x24, followed by 
        a convolution operation to refine the features.

        The input tensor is expected to have 3 channels, which is preserved 
        throughout the network.
        """
        super(ConvPoolEncoder, self).__init__()

        self.norm = nn.BatchNorm2d(3)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((24, 24))
        self.conv = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """
        Forward pass for the ConvPoolEncoder module.

        Parameters:
        x (torch.Tensor): Input tensor with shape (batch_size, height, width, channels).
                          The spatial dimensions (height, width) can vary, but the
                          number of channels should be 3.

        Returns:
        torch.Tensor: Output tensor with shape (batch_size, 24, 24, 3), 
                      where the spatial dimensions are fixed at 24x24 and the 
                      number of channels is preserved as 3.
        """
        assert x.ndim == 4, f"Expected input to be a 4D tensor but got {x.ndim}D"
        x = x.permute(0, 3, 1, 2)
        x = self.norm(x)
        x = self.adaptive_pool(x)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        
        return x
    

class STEncoder(nn.Module):
    def __init__(self, dim=24*24, heads=8, dim_head=64, dropout=0.5):
        """
        Initializes the STEncoder module.

        Parameters:
        dim (int): Input feature dimension, which now represents the combined spatial dimensions (height * width).
        heads (int): Number of attention heads.
        dim_head (int): Dimension per attention head.
        dropout (float): Dropout rate.
        """
        super(STEncoder, self).__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=True)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        """
        Forward pass for the STEncoder module.

        Parameters:
        x (torch.Tensor): Input tensor with shape (batch_size, 24, 24, 3).

        Returns:
        torch.Tensor: Output tensor with the same spatial dimensions as input but with `dim` as the channel dimension.
        torch.Tensor: Attention heatmap tensor with shape (batch_size, heads, num_tokens, num_tokens).
        """
        assert x.ndim == 4, f"Expected input to be a 4D tensor but got {x.ndim}D"  
        batch_size, height, width, channels = x.shape
        assert height == 24 and width == 24, "Input spatial dimensions must be 24x24"

        x = rearrange(x, 'b h w c -> b c (h w)')

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b c (h d) -> b h c d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h c d -> b c (h d)')
        out = self.to_out(out)

        out = rearrange(out, 'b c (h w) -> b h (w c)', h=height, w=width)
        
        return out, attn


class ConvUpsampler(nn.Module):
    def __init__(self):
        """
        Initializes the ConvUpsampler module.

        This module takes an input tensor of shape (64, 1024) and reshapes it into
        (64, 32, 32, 1), then uses transposed convolutions to upscale it to (64, 24, 24, 3).
        """
        super(ConvUpsampler, self).__init__()
        
        self.fc = nn.Linear(1024, 32 * 32 * 1)
        
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(1, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        """
        Forward pass for the ConvUpsampler module.

        Parameters:
        x (torch.Tensor): Input tensor with shape (batch_size, 1024).

        Returns:
        torch.Tensor: Output tensor reshaped to (batch_size, 24, 24*3).
        """
        x = self.fc(x)
        x = x.view(x.size(0), 1, 32, 32)
        x = self.upsample(x)
        x = x[:, :, :24, :24]
        x = x.permute(0, 2, 3, 1)
        x = x.contiguous().view(x.size(0), 24, 24 * 3)
        return x
    

class CrossAttention(nn.Module):
    def __init__(self, dim_q=24*3, dim_kv=24*3, heads=8, dim_head=64, dropout=0.6):
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


class STDPAFI(nn.Module):
    def __init__(self, channel=24*3):
        """
        Initializes the STDPAFI module.

        Parameters:
        channel (int): The number of channels in the input.
        """
        super(STDPAFI, self).__init__()
        self.conv_pool = ConvPoolEncoder()
        self.st_encoder = STEncoder()
        self.conv_upsampler = ConvUpsampler()

        self.corss_attention_x = CrossAttention()
        self.corss_attention_y = CrossAttention()

        self.ffn_one = nn.Sequential(
            nn.Linear(channel,channel//8),
            nn.GELU(),
            nn.Linear(channel//8,channel//2),
        )
        self.ffn_two = nn.Sequential(
            nn.Linear(channel,channel//8),
            nn.GELU(),
            nn.Linear(channel//8,channel//2),
        )

        self.ln_fuse = nn.LayerNorm(channel)

    def forward(self, trainabel_val, frozen_val):
        """
        Forward pass for the STDPAFI module.

        Parameters:
        trainabel_val (torch.Tensor): Input tensor that will be processed by the trainable encoder.
        frozen_val (torch.Tensor): Input tensor that will be processed by the frozen encoder.

        Returns:
        torch.Tensor: Fused feature tensor.
        """
        trainabel_x = self.conv_pool(trainabel_val)

        x, _ = self.st_encoder(trainabel_x)
        y = self.conv_upsampler(frozen_val)

        feature_one, _ = self.corss_attention_x(x, y)
        feature_two, _ = self.corss_attention_y(y, x)

        feature_one_feat_down = self.ffn_one(feature_one)
        feature_two_feat_down = self.ffn_two(feature_two)

        fuse_feat = torch.cat([feature_one_feat_down, feature_two_feat_down], dim=-1) 
        # fuse_feat = fuse_feat.permute(0, 2, 1)
        # compressed_fuse_feat = self.compress_conv(fuse_feat)
        # compressed_fuse_feat = compressed_fuse_feat.permute(0, 3, 2, 1)
        fuse_feat = self.ln_fuse(fuse_feat + feature_one + feature_two)

        return fuse_feat
    

# Pathway Path
class PathwayEncoder(nn.Module):
    def __init__(self, dim=2343, heads=8, dim_head=64, dropout=0.5):
        """
        Initializes the PathwayEncoder module.

        Parameters:
        dim (int): Input feature dimension (number of channels).
        heads (int): Number of attention heads.
        dim_head (int): Dimension per attention head.
        dropout (float): Dropout rate.
        """
        super(PathwayEncoder, self).__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=True)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        """
        Forward pass for the PathwayEncoder module.

        Parameters:
        x (torch.Tensor): Input tensor with shape (batch_size, sequence_length, feature_dimension).

        Returns:
        torch.Tensor: Output tensor with the same shape as input.
        torch.Tensor: Attention heatmap tensor with shape (batch_size, heads, sequence_length, sequence_length).
        """
        assert x.ndim == 3, f"Expected input to be a 3D tensor but got {x.ndim}D"

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out, attn


class MultiLayerProjection(nn.Module):
    def __init__(self, input_dim=768, intermediate_dim=1024, output_dim=1728):
        """
        Initializes the MultiLayerProjection module with multiple linear layers.

        Parameters:
        input_dim (int): The input feature dimension (e.g., 768).
        intermediate_dim (int): The feature dimension after the first linear layer (e.g., 1024).
        output_dim (int): The final output feature dimension after projection (e.g., 1728).
        """
        super(MultiLayerProjection, self).__init__()
        
        self.layer1 = nn.Linear(input_dim, intermediate_dim)
        self.activation1 = nn.ReLU()

        self.layer2 = nn.Linear(intermediate_dim, output_dim)
        self.activation2 = nn.ReLU()

    def forward(self, x):
        """
        Forward pass for the MultiLayerProjection module.

        Parameters:
        x (torch.Tensor): Input tensor with shape (batch_size, input_dim).

        Returns:
        torch.Tensor: Output tensor with shape (batch_size, 24, 72).
        """
        x = self.layer1(x)
        x = self.activation1(x)

        x = self.layer2(x)
        x = self.activation2(x)

        x = x.view(x.size(0), 24, 24, 3)
        x = x.view(x.size(0), 24, 24 * 3)

        return x


class ConvFeatureExtractor(nn.Module):
    def __init__(self):
        """
        Initializes the ConvFeatureExtractor module.

        This module applies a series of convolutional layers followed by linear layers
        to extract features from an input tensor of shape (batch_size, 25, 2343)
        and produces an output tensor of shape (batch_size, 3, 576).

        Parameters:
        None (all hyperparameters are hardcoded within the class).
        """
        super(ConvFeatureExtractor, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=25, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=8, out_channels=3, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)

        self.fc1 = nn.Linear(292, 512)
        self.fc2 = nn.Linear(512, 576)

    def forward(self, x):
        """
        Forward pass for the ConvFeatureExtractor module.

        Parameters:
        x (torch.Tensor): Input tensor with shape (batch_size, 25, 2343).

        Returns:
        torch.Tensor: Output tensor with shape (batch_size, 24, 72).
        """
        assert x.ndim == 3, f"Expected input to be a 3D tensor but got {x.ndim}D"  
        _, channels, sequence_length = x.shape
        assert channels == 25 and sequence_length == 2343, "Input tensor must have shape (batch_size, 25, 2343)"

        x = self.relu(self.conv1(x))
        x = self.pool(x)

        x = self.relu(self.conv2(x))
        x = self.pool(x)

        x = self.relu(self.conv3(x))
        x = self.pool(x)

        x = self.relu(self.fc1(x))

        x = self.fc2(x)

        x = x.view(x.size(0), 3, 24, 24)
        x = x.permute(0, 2, 1, 3).reshape(x.size(0), 24, 3 * 24)

        return x


class PathwayDPAFI(nn.Module):
    def __init__(self, channel=24*3):
        """
        Initializes the PathwayDPAFI module.

        Parameters:
        channel (int): The number of channels in the input.
        """
        super(PathwayDPAFI, self).__init__()
        self.pathway_encoder = PathwayEncoder()
        self.downstream_pathway = ConvFeatureExtractor()
        self.upstream_embedding = MultiLayerProjection()

        self.corss_attention_x = CrossAttention()
        self.corss_attention_y = CrossAttention()

        self.ffn_one = nn.Sequential(
            nn.Linear(channel,channel//8),
            nn.GELU(),
            nn.Linear(channel//8,channel//2),
        )
        self.ffn_two = nn.Sequential(
            nn.Linear(channel,channel//8),
            nn.GELU(),
            nn.Linear(channel//8,channel//2),
        )

        self.ln_fuse = nn.LayerNorm(channel)

    def forward(self, trainabel_val, frozen_val):
        """
        Forward pass for the STDPAFI module.

        Parameters:
        trainabel_val (torch.Tensor): Input tensor that will be processed by the trainable encoder.
        frozen_val (torch.Tensor): Input tensor that will be processed by the frozen encoder.

        Returns:
        torch.Tensor: Fused feature tensor.
        """
        x, _ = self.pathway_encoder(trainabel_val)
        x = self.downstream_pathway(x)
        y = self.upstream_embedding(frozen_val)

        feature_one, _ = self.corss_attention_x(x, y)
        feature_two, _ = self.corss_attention_y(y, x)

        feature_one_feat_down = self.ffn_one(feature_one)
        feature_two_feat_down = self.ffn_two(feature_two)

        fuse_feat = torch.cat([feature_one_feat_down, feature_two_feat_down], dim=-1) 
        fuse_feat = self.ln_fuse(fuse_feat + feature_one + feature_two)

        return fuse_feat


class Transformer(nn.Module):
    def __init__(self, d_model=24*3, nhead=8, dim_feedforward=1024, dropout=0.5, batch_first=True, num_layers=2):
        """
        Initializes the Transformer module.

        Parameters:
        d_model (int): The number of expected features in the input (and output) for each token (default is 24*3).
        nhead (int): The number of heads in the multihead attention models (default is 8).
        dim_feedforward (int): The dimension of the feedforward network model (default is 1024).
        dropout (float): The dropout value to apply to layers (default is 0.5).
        batch_first (bool): If True, the input and output tensors are provided as (batch_size, seq_len, feature_dim).
        num_layers (int): The number of sub-encoder-layers in the encoder (default is 2).
        """
        super(Transformer, self).__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=batch_first
        )
        self.encoder = nn.TransformerEncoder(self.layer, num_layers)

    def forward(self, x):
        """
        Initializes the Transformer module.

        Parameters:
        d_model (int): The number of expected features in the input (and output) for each token (default is 24*3).
        nhead (int): The number of heads in the multihead attention models (default is 8).
        dim_feedforward (int): The dimension of the feedforward network model (default is 1024).
        dropout (float): The dropout value to apply to layers (default is 0.5).
        batch_first (bool): If True, the input and output tensors are provided as (batch_size, seq_len, feature_dim).
        num_layers (int): The number of sub-encoder-layers in the encoder (default is 2).
        """
        return self.encoder(x)


class MLP(nn.Module):
    def __init__(self, input_dim=72, hidden_dim=128, output_dim=72, num_hidden_layers=2, dropout=0.5):
        """
        Initializes the MLP with specified dimensions and layers.

        Parameters:
        input_dim (int): Dimension of the input features.
        hidden_dim (int): Dimension of the hidden layers.
        output_dim (int): Dimension of the output layer.
        num_hidden_layers (int): Number of hidden layers.
        dropout (float): Dropout rate for regularization.
        """
        super(MLP, self).__init__()
        
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Forward pass for the MLP.

        Parameters:
        x (torch.Tensor): Input tensor with shape (batch_size, input_dim).

        Returns:
        torch.Tensor: Output tensor with shape (batch_size, output_dim).
        """
        x = F.relu(self.input_layer(x))
        x = self.dropout(x)
        
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
            x = self.dropout(x)
        
        x = self.output_layer(x)

        return x


class AFIN(nn.Module):
    def __init__(self, d_model=72):
        """
        Initializes the AFIN module, which integrates multiple MLPs, Transformers, and CrossAttention layers.

        Parameters:
        d_model (int): The input feature dimension (default is 72).
        """
        super(AFIN, self).__init__()

        self.st_mlp1 = MLP()
        self.st_mlp2 = MLP()
        self.pathway_mlp1 = MLP()
        self.pathway_mlp2 = MLP()

        self.st_transformer = Transformer()
        self.pathway_transformer = Transformer()

        self.st_norm1 = nn.LayerNorm(d_model)
        self.pathway_norm1 = nn.LayerNorm(d_model)
        self.st_norm2 = nn.LayerNorm(d_model)
        self.pathway_norm2 = nn.LayerNorm(d_model)

        self.corss_attention_x = CrossAttention()
        self.corss_attention_y = CrossAttention()

        self.ffn_one = nn.Sequential(
            nn.Linear(d_model,d_model//4),
            nn.GELU(),
            nn.Linear(d_model//4,d_model//2),
        )
        self.ffn_two = nn.Sequential(
            nn.Linear(d_model,d_model//4),
            nn.GELU(),
            nn.Linear(d_model//4,d_model//2),
        )

        self.ln_fuse = nn.LayerNorm(d_model)
        
    def forward(self, st_x, pathway_y):
        """
        Forward pass for the AFIN module.

        Parameters:
        st_x (torch.Tensor): Input tensor for spatial-temporal features.
        pathway_y (torch.Tensor): Input tensor for pathway features.

        Returns:
        torch.Tensor: The fused feature tensor.
        """
        st = self.st_norm1(st_x)
        pathway = self.pathway_norm1(pathway_y)

        st_mlp = self.st_mlp1(st)
        st_transformer = self.st_transformer(self.st_mlp2(st))
        pathway_mlp = self.pathway_mlp1(pathway)
        pathway_transformer = self.pathway_transformer(self.pathway_mlp2(pathway))

        st = self.st_norm2(st_mlp * st_transformer)
        pathway = self.pathway_norm2(pathway_mlp * pathway_transformer)
        st, _ = self.corss_attention_x(st, pathway)
        pathway, _ = self.corss_attention_y(pathway, st)

        feature_st_down = self.ffn_one(st)
        feature_pathway_down = self.ffn_two(pathway)

        fuse_feat = torch.cat([feature_st_down, feature_pathway_down], dim=-1) 
        fuse_feat = self.ln_fuse(fuse_feat + st + pathway)

        return fuse_feat


class DSCAN(nn.Module):
    def __init__(self):
        """
        Initializes the DSCAN module, which integrates STDPAFI, PathwayDPAFI, and AFIN modules.

        This module combines spatial-temporal features and pathway features using specialized processing modules.
        """
        super(DSCAN, self).__init__()

        self.st_dpafi = STDPAFI()
        self.pathway_dpafi = PathwayDPAFI()
        self.afin = AFIN()

        self.final_linear_out = nn.Linear(1728, 1024)
        self.final_linear_st = nn.Linear(1728, 1024)
        self.final_linear_exp = nn.Linear(1728, 1024)

    def forward(self, st, st_emb, exp, exp_emb):
        """
        Forward pass for the DSCAN module.

        Parameters:
        st (torch.Tensor): Input tensor for spatial-temporal data.
        st_emb (torch.Tensor): Embedding tensor associated with the spatial-temporal data.
        exp (torch.Tensor): Input tensor for pathway data.
        exp_emb (torch.Tensor): Embedding tensor associated with the pathway data.

        Returns:
        torch.Tensor: Output tensor after combining features from spatial-temporal and pathway data.
        """
        st_feature = self.st_dpafi(st, st_emb)
        pathway_feature = self.pathway_dpafi(exp, exp_emb)
        out = self.afin(st_feature, pathway_feature)

        out = out.view(out.size(0), -1)
        st_feature = st_feature.view(st_feature.size(0), -1)
        pathway_feature = pathway_feature.view(pathway_feature.size(0), -1)
        
        out = self.final_linear_out(out)
        st = self.final_linear_st(st_feature)
        exp = self.final_linear_exp(pathway_feature)

        return st, exp, out
    