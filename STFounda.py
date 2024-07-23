import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import random
from einops import rearrange
from mamba_ssm import Mamba2
from kan import KAN

# nn.BatchNorm1d()
# img
class STPreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
    

class STFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
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
    

class STFeedForwardKAN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            KAN([dim, hidden_dim, dim])
        )
    def forward(self, x):
        return self.net(x)
    

class STFormerFront(nn.Module):
    def __init__(self, dim=1024, depth=2, mlp_dim=512, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                STPreNorm(dim, STFeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for layers in self.layers:
            for ff in layers:
                x = ff(x) + x
        return x
    
class STFormerBack(nn.Module):
    def __init__(self, dim=128, depth=2, mlp_dim=256, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                STPreNorm(dim, STFeedForwardKAN(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for layers in self.layers:
            for ff in layers:
                x = ff(x) + x
        return x
    

class STReshapeMLP(nn.Module):
    def __init__(self, input_dim=1024, n_tokens=8, dim_features=128):
        super().__init__()
        self.n_tokens = n_tokens
        self.dim_features = dim_features

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, n_tokens * dim_features),
            nn.ReLU(),
            nn.Linear(n_tokens * dim_features, n_tokens * dim_features)
        )

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        out = self.mlp(x)
        # Reshape to (batch_size, n_tokens, dim_features)
        out = out.view(-1, self.n_tokens, self.dim_features)
        return out
    

class STInnerAttention(nn.Module):
    def __init__(self, dim=128, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    

class STOutterAttention(nn.Module):
    def __init__(self, dim=3, heads=8, dim_head=64, dropout=0.):
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

        # Define the MLP layer
        self.mlp = nn.Sequential(
            nn.Linear(dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024)
        )

    def forward(self, x):
        x = rearrange(x, 'b h w c -> b (h w) c')
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        out = self.mlp(out.mean(dim=1))  # Aggregate spatial dimensions and apply MLP
        return out


# exp
class EXPOutterAttention(nn.Module):
    def __init__(self, dim = 128, heads = 8, dim_head = 64, dropout = 0., input_dim=1024, n_tokens=8, dim_features=128):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.n_tokens = n_tokens
        self.dim_features = dim_features

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.mlp1 = nn.Sequential(
            nn.Linear(input_dim, n_tokens * dim_features),
            nn.ReLU(),
            nn.Linear(n_tokens * dim_features, n_tokens * dim_features)
        )

    def forward(self, x):
        x = self.mlp1(x)
        x = x.view(-1, self.n_tokens, self.dim_features)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        out = out.view(out.size(0), -1)
        return out
    

class EXPBottleneckFCWithResidual(nn.Module):
    """
    @author Ling LUO
    @class BottleneckFCWithResidual
    @description Implements a bottleneck fully connected layer with a residual connection.

    Parameters:
    input_dim (int): The size of the input features.
    bottleneck_dim (int): The size of the bottleneck layer. Default is 1024.
    """

    def __init__(self, input_dim=19264, bottleneck_dim=1024):
        super(EXPBottleneckFCWithResidual, self).__init__()
        self.fc1 = nn.Linear(input_dim, bottleneck_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(bottleneck_dim, input_dim)
        self.residual = nn.Identity()  # Residual connection
        self.fc3 = nn.Linear(input_dim, bottleneck_dim)

    def forward(self, x):
        """
        Forward pass for the bottleneck with residual connection.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor after applying the bottleneck and residual connection.
        """
        residual = self.residual(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x += residual  # Residual connection
        x = self.fc3(x)
        return x


class EXPMLPTransform(nn.Module):
    def __init__(self, input_dim=768, output_dim=1024, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.mlp(x)


class EXPPreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
    

class EXPFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
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
    
class EXPFeedForwardKAN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            KAN([dim, hidden_dim, dim])
        )
    def forward(self, x):
        return self.net(x)
    

class EXPFormerFront(nn.Module):
    def __init__(self, dim=1024, depth=2, mlp_dim=512, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                EXPPreNorm(dim, EXPFeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for layers in self.layers:
            for ff in layers:
                x = ff(x) + x
        return x


class EXPFormerBack(nn.Module):
    def __init__(self, dim=128, depth=2, mlp_dim=256, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                EXPPreNorm(dim, EXPFeedForwardKAN(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for layers in self.layers:
            for ff in layers:
                x = ff(x) + x
        return x


class EXPReshapeMLP(nn.Module):
    def __init__(self, input_dim=1024, n_tokens=8, dim_features=128):
        super().__init__()
        self.n_tokens = n_tokens
        self.dim_features = dim_features

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, n_tokens * dim_features),
            nn.ReLU(),
            nn.Linear(n_tokens * dim_features, n_tokens * dim_features)
        )

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        out = self.mlp(x)
        # Reshape to (batch_size, n_tokens, dim_features)
        out = out.view(-1, self.n_tokens, self.dim_features)
        return out


class EXPInnerAttention(nn.Module):
    def __init__(self, dim=128, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class FlattenKAN(nn.Module):
    def __init__(self, input_dim=128, num_tokens=8, combined_dim=256, output_dim=1024):
        super(FlattenKAN, self).__init__()
        self.num_tokens = num_tokens
        self.input_dim = input_dim
        self.combined_dim = combined_dim

        self.fc = nn.Sequential(
            KAN([num_tokens * combined_dim, 512, output_dim])
        )

    def forward(self, x1, x2):
        # 将两个输入张量在特征维度上连接
        x = torch.cat((x1, x2), dim=-1)  # x的形状将是 [128, 8, 256]
        # 将其展平以适应全连接层
        x = x.view(x.size(0), -1)  # x的形状将是 [128, 2048]
        # 通过全连接层
        x = self.fc(x)
        return x


class FlattenMLP(nn.Module):
    def __init__(self, input_dim=128, num_tokens=8, combined_dim=256, output_dim=1024):
        super(FlattenMLP, self).__init__()
        self.num_tokens = num_tokens
        self.input_dim = input_dim
        self.combined_dim = combined_dim

        self.fc = nn.Sequential(
            nn.Linear(num_tokens * combined_dim, output_dim)
        )

    def forward(self, x1, x2):
        # 将两个输入张量在特征维度上连接
        x = torch.cat((x1, x2), dim=-1)  # x的形状将是 [128, 8, 256]
        # 将其展平以适应全连接层
        x = x.view(x.size(0), -1)  # x的形状将是 [128, 2048]
        # 通过全连接层
        x = self.fc(x)
        return x
    

class STModel(nn.Module):
    def __init__(self):
        super(STModel, self).__init__()
        self.outter_atten = STOutterAttention()

        self.atten_path_front_former = STFormerFront()
        self.emb_path_front_former = STFormerFront()

        self.atten_reshape = STReshapeMLP()
        self.emb_reshape = STReshapeMLP()

        self.atten_attention = STInnerAttention()
        self.emb_attention = STInnerAttention()
        self.add_attention = STInnerAttention()

        self.atten_path_back_former = STFormerBack()
        self.add_path_back_former = STFormerBack()
        self.emb_path_back_former = STFormerBack()

    def forward(self, x, embed_x):
        atten_x = self.outter_atten(x)

        atten_x = self.atten_path_front_former(atten_x)
        embed_x = self.emb_path_front_former(embed_x)

        atten_x = self.atten_reshape(atten_x)
        embed_x = self.emb_reshape(embed_x)

        add_x = atten_x + embed_x

        atten_x = self.atten_attention(atten_x)
        embed_x = self.emb_attention(embed_x)
        add_x = self.add_attention(add_x)

        atten_x = self.atten_path_back_former(atten_x)
        embed_x = self.emb_path_back_former(embed_x)
        add_x = self.add_path_back_former(add_x)

        return atten_x + add_x + embed_x



class EXPModel(nn.Module):
    def __init__(self):
        super(EXPModel, self).__init__()
        self.outter_atten = nn.Sequential(
            EXPBottleneckFCWithResidual(),
            EXPOutterAttention()
        )
        self.atten_path_front_former = EXPFormerFront()
        self.emb_path_front_former = nn.Sequential(
            EXPMLPTransform(),
            EXPFormerFront()
        )

        self.atten_reshape = EXPReshapeMLP()
        self.emb_reshape = EXPReshapeMLP()

        self.atten_attention = EXPInnerAttention()
        self.emb_attention = EXPInnerAttention()
        self.add_attention = EXPInnerAttention()

        self.atten_path_back_former = EXPFormerBack()
        self.add_path_back_former = EXPFormerBack()
        self.emb_path_back_former = EXPFormerBack()

    def forward(self, x, embed_x):
        atten_x = self.outter_atten(x)

        atten_x = self.atten_path_front_former(atten_x)
        embed_x = self.emb_path_front_former(embed_x)

        atten_x = self.atten_reshape(atten_x)
        embed_x = self.emb_reshape(embed_x)

        add_x = atten_x + embed_x

        atten_x = self.atten_attention(atten_x)
        embed_x = self.emb_attention(embed_x)
        add_x = self.add_attention(add_x)

        atten_x = self.atten_path_back_former(atten_x)
        embed_x = self.emb_path_back_former(embed_x)
        add_x = self.add_path_back_former(add_x)

        return atten_x + add_x + embed_x



class CrossAttention(nn.Module):
    def __init__(self, emb_dim=128, att_dropout=0.0, dropout=0.0):
        super(CrossAttention, self).__init__()
        self.emb_dim = emb_dim
        self.scale = emb_dim ** -0.5

        self.Wq = nn.Linear(emb_dim, emb_dim)
        self.Wk = nn.Linear(emb_dim, emb_dim)
        self.Wv = nn.Linear(emb_dim, emb_dim)

        self.dropout = nn.Dropout(att_dropout)
        self.proj_out = nn.Linear(emb_dim, emb_dim)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x, context, pad_mask=None):
        '''
        :param x: [batch_size, n_tokens, emb_dim]
        :param context: [batch_size, seq_len, emb_dim]
        :param pad_mask: [batch_size, n_tokens, seq_len]
        :return:
        '''
        Q = self.Wq(x)  # [batch_size, n_tokens, emb_dim]
        K = self.Wk(context)  # [batch_size, seq_len, emb_dim]
        V = self.Wv(context)

        # Compute attention weights
        att_weights = torch.einsum('bqd,bkd -> bqk', Q, K) * self.scale  # [batch_size, n_tokens, seq_len]

        if pad_mask is not None:
            att_weights = att_weights.masked_fill(pad_mask, -1e9)

        att_weights = F.softmax(att_weights, dim=-1)
        att_weights = self.dropout(att_weights)

        out = torch.einsum('bqk,bkd -> bqd', att_weights, V)  # [batch_size, n_tokens, emb_dim]

        out = self.proj_out(out)  # [batch_size, n_tokens, emb_dim]
        out = self.out_dropout(out)

        return out, att_weights


# concat
class DSCAN(nn.Module):
    def __init__(self, d_model=128, d_state=64, headdim=32, d_conv=4, expand=2):
        super(DSCAN, self).__init__()
        self.mamba_st = Mamba2(
            d_model=d_model,
            d_state=d_state,
            headdim=headdim,
            d_conv=d_conv,
            expand=expand,
        )

        self.mamba_exp = Mamba2(
            d_model=d_model,
            d_state=d_state,
            headdim=headdim,
            d_conv=d_conv,
            expand=expand,
        )

        self.corss_atten_st = CrossAttention()
        self.corss_atten_exp = CrossAttention()

        self.mlp = FlattenMLP()

    def forward(self, st_feature, exp_feature):
        '''
        :param x: [batch_size, n_tokens, emb_dim]
        :param context: [batch_size, seq_len, emb_dim]
        :param pad_mask: [batch_size, n_tokens, seq_len]
        :return:
        '''
        exp_feature = self.mamba_exp(exp_feature)
        st_feature = self.mamba_st(st_feature)

        exp_feature_, _ = self.corss_atten_exp(exp_feature, st_feature)
        st_feature_, _ = self.corss_atten_st(st_feature, exp_feature)

        out = self.mlp(st_feature_, exp_feature_)

        return out
    

class STFounda(nn.Module):
    def __init__(self, d_model=128, d_state=64, headdim=32, d_conv=4, expand=2):
        super(STFounda, self).__init__()
        self.st_model = STModel()
        self.exp_model = EXPModel()
        self.d_scan = DSCAN(d_model=d_model, d_state=d_state, headdim=headdim, d_conv=d_conv, expand=expand)

    
    def forward(self, st, st_emb, exp, exp_emb):
        st_feature = self.st_model(st, st_emb)
        exp_feature = self.exp_model(exp, exp_emb)
        fusion_feature = self.d_scan(st_feature, exp_feature)

        return st_feature.view(st_feature.size(0), -1), exp_feature.view(exp_feature.size(0), -1), fusion_feature
