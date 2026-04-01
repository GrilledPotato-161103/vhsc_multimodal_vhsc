import torch
import torch.nn as nn
import torch.nn.functional as F

class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
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

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            mask = mask.bool()
            attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=2.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, mask=None):
        _x, attn = self.attn(self.norm1(x), mask=mask)
        x = x + _x
        x = x + self.mlp(self.norm2(x))
        return x, attn

class TextProcessor(nn.Module):
    def __init__(self, common_dim, latent_dim):
        super().__init__()
        self.common_dim = common_dim
        self.latent_dim = latent_dim
        
        self.fc1 = nn.Linear(self.common_dim, self.latent_dim)
        self.fc2 = nn.Linear(self.latent_dim, self.latent_dim)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x

class ImageProcessor(nn.Module):
    def __init__(self, common_dim, latent_dim):
        super().__init__()
        self.common_dim = common_dim
        self.latent_dim = latent_dim
        
        self.fc1 = nn.Linear(self.common_dim, self.latent_dim)
        self.fc2 = nn.Linear(self.latent_dim, self.latent_dim)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x


class JointProcessor(nn.Module):
    def __init__(self, common_dim, latent_dim):
        super().__init__()
        self.common_dim = common_dim
        self.latent_dim = latent_dim

        self.proj_t = TextProcessor(self.common_dim, self.latent_dim)
        self.proj_i = ImageProcessor(common_dim=common_dim, latent_dim=latent_dim)
        
        self.proj1 = nn.Linear(self.latent_dim*2, self.latent_dim*2)
        self.proj2 = nn.Linear(self.latent_dim*2, self.latent_dim*2)
        self.projector = nn.Linear(self.latent_dim*2, latent_dim)

    def forward(self, batch):
        x_t, x_i = batch

        proj_t = self.proj_t(x_t)
        proj_i = self.proj_i(x_i)

        # (V, A, I)
        last_proj_avi = torch.cat([proj_t, proj_i], dim=1)
        proj_avi = self.proj2(F.dropout(F.gelu(self.proj1(last_proj_avi)), p=0.0, training=self.training))
        proj_avi += last_proj_avi
        
        # Project
        return self.projector(proj_avi)