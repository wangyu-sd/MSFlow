import torch, math
import torch.nn as nn
import torch.nn.functional as F

from model.var_helpers import DropPath, drop_path


import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, block_idx, embed_dim=768, num_heads=12, 
                 attn_drop=0., proj_drop=0., attn_l2_norm=False):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.block_idx = block_idx
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.attn_l2_norm = attn_l2_norm
        
        # 分别定义查询、键、值的投影矩阵[2,4](@ref)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # 条件适应的缩放参数
        if self.attn_l2_norm:
            self.scale = 1
            self.scale_mul_1H11 = nn.Parameter(
                torch.full((1, num_heads, 1, 1), 4.0).log(),
                requires_grad=True)
            self.max_scale_mul = torch.log(torch.tensor(100)).item()
        else:
            self.scale = 0.25 / (self.head_dim ** 0.5)
            
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True) if attn_drop > 0 else nn.Identity()
        self.caching = False
        self.cached_q = None
    
    def q_caching(self, enable: bool):
        self.caching = enable
        if not enable:
            self.cached_q = None

    def forward(self, x, context, attn_bias=None):
        B, L, C = x.shape
        _, L_c, _ = context.shape
        
        # 投影查询、键、值
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, L, D]
        k = self.k_proj(context).view(B, L_c, self.num_heads, self.head_dim).permute(0, 2, 3, 1) # [B, H, D, L_c]
        v = self.v_proj(context).view(B, L_c, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # [B, H, L_c, D]

        # L2归一化处理
        if self.attn_l2_norm:
            scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp()
            q = F.normalize(q, dim=-1).mul(scale_mul)
            k = F.normalize(k, dim=-1)
            
            
        # KV缓存
        if self.caching:
            if self.cached_q is None: 
                self.cached_k = q

            else: 
                q = self.cached_k = torch.cat((self.cached_q, q), dim=2)

        attn = (q @ k) * self.scale  # [B, H, L, L_c]
        if attn_bias is not None:
            attn += attn_bias
            
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # 加权聚合值向量
        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, L, C)  # [B, L, C]
        return self.proj_drop(self.proj(out))

class AdaLNCrossAttn(nn.Module):
    def __init__(self, block_idx, last_drop_p, embed_dim, cond_dim, norm_layer,
                 num_heads, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0., 
                 attn_l2_norm=False):
        super().__init__()
        self.block_idx = block_idx
        self.last_drop_p = last_drop_p
        self.embed_dim = embed_dim
        
        # 跨注意力模块
        self.cross_attn = CrossAttention(
            block_idx, embed_dim, num_heads, 
            attn_drop=attn_drop, proj_drop=drop, 
            attn_l2_norm=attn_l2_norm
        )
        
        # 前馈网络
        self.ffn = FFN(embed_dim, hidden_features=int(embed_dim*mlp_ratio), drop=drop)
        
        # 条件自适应参数[8](@ref)
        self.ln_wo_grad = norm_layer(embed_dim, elementwise_affine=False)
        self.ada_gss = nn.Parameter(torch.randn(1, 1, 6, embed_dim) / embed_dim**0.5)
        
        # DropPath正则化
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x, context, cond_BD, attn_bias=None):
        """
        Args:
            x: 查询序列 [B, L, C]
            context: 上下文序列 [B, L_c, C]
            cond_BD: 条件向量 [B, D]
            attn_bias: 注意力偏置 [B, H, L, L_c]
        """
        # 分解条件参数
        gamma1, gamma2, scale1, scale2, shift1, shift2 = (
            self.ada_gss + cond_BD  # [B,1,D] + [1,1,6,C] → [B,1,6,C]
        ).unbind(2)
        
        # 条件归一化处理
        norm_x = self.ln_wo_grad(x)
        modulated_x = norm_x * (scale1 + 1) + shift1
        
        # 跨注意力计算
        attn_out = self.cross_attn(modulated_x, context, attn_bias)
        x = x + self.drop_path(attn_out * gamma1)
        
        # 前馈网络处理
        norm_x_ffn = self.ln_wo_grad(x)
        modulated_x_ffn = norm_x_ffn * (scale2 + 1) + shift2
        ffn_out = self.ffn(modulated_x_ffn)
        x = x + self.drop_path(ffn_out * gamma2)
        
        return x


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop, inplace=True) if drop > 0 else nn.Identity()
    
    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))


class SelfAttention(nn.Module):
    def __init__(
        self, block_idx, embed_dim=768, num_heads=12,
        attn_drop=0., proj_drop=0., attn_l2_norm=False
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.block_idx, self.num_heads, self.head_dim = block_idx, num_heads, embed_dim // num_heads
        self.attn_l2_norm = attn_l2_norm
        if self.attn_l2_norm:
            self.scale = 1
            self.scale_mul_1H11 = nn.Parameter(torch.full(size=(1, self.num_heads, 1, 1), fill_value=4.0).log(), requires_grad=True)
            self.max_scale_mul = torch.log(torch.tensor(100)).item()
        else:
            self.scale = 0.25 / math.sqrt(self.head_dim)
        
        self.mat_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.q_bias, self.v_bias = nn.Parameter(torch.zeros(embed_dim)), nn.Parameter(torch.zeros(embed_dim))
        self.register_buffer('zero_k_bias', torch.zeros(embed_dim))
        
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True) if proj_drop > 0 else nn.Identity()
        self.attn_drop = attn_drop
        
        # only used during inference
        self.caching, self.cached_k, self.cached_v = False, None, None
    
    def kv_caching(self, enable: bool): 
        self.caching, self.cached_k, self.cached_v = enable, None, None
    
    # NOTE: attn_bias is None during inference because kv cache is enabled
    def forward(self, x, attn_bias):
        B, L, C = x.shape
        qkv = F.linear(input=x, weight=self.mat_qkv.weight, 
                      bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias))
                      ).view(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
        
        if self.attn_l2_norm:
            scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp()
            q = F.normalize(q, dim=-1).mul(scale_mul)
            k = F.normalize(k, dim=-1)
        
        if self.caching:
            if self.cached_k is None: 
                self.cached_k = k
                self.cached_v = v
            else: 
                k = self.cached_k = torch.cat((self.cached_k, k), dim=2)
                v = self.cached_v = torch.cat((self.cached_v, v), dim=2)
        
        attn = q.mul(self.scale) @ k.transpose(-2, -1)
        if attn_bias is not None:
            attn = attn + attn_bias
        
        attn = F.dropout(attn.softmax(dim=-1), p=self.attn_drop if self.training else 0.0)
        out = (attn @ v).transpose(1, 2).reshape(B, L, C)
        
        return self.proj_drop(self.proj(out))


class AdaLNSelfAttn(nn.Module):
    def __init__(
        self, block_idx, last_drop_p, embed_dim, cond_dim, norm_layer,
        num_heads, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0., attn_l2_norm=False,
        flash_if_available=False, fused_if_available=True,
    ):
        super(AdaLNSelfAttn, self).__init__()
        self.block_idx, self.last_drop_p, self.C = block_idx, last_drop_p, embed_dim
        self.C, self.D = embed_dim, cond_dim
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.attn = SelfAttention(block_idx=block_idx, embed_dim=embed_dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop, attn_l2_norm=attn_l2_norm)
        self.ffn = FFN(in_features=embed_dim, hidden_features=round(embed_dim * mlp_ratio), drop=drop)
        
        self.ln_wo_grad = norm_layer(embed_dim, elementwise_affine=False)
        self.ada_gss = nn.Parameter(torch.randn(1, 1, 6, embed_dim) / embed_dim**0.5)
        
        self.fused_add_norm_fn = None
    
    # NOTE: attn_bias is None during inference because kv cache is enabled
    def forward(self, x, cond_BD, attn_bias):   # C: embed_dim, D: cond_dim
        gamma1, gamma2, scale1, scale2, shift1, shift2 = (self.ada_gss + cond_BD).unbind(2) # 116C + B16C =unbind(2)=> 6 B1C
        x = x + self.drop_path(self.attn( self.ln_wo_grad(x).mul(scale1.add(1)).add_(shift1), attn_bias=attn_bias ).mul_(gamma1))
        x = x + self.drop_path(self.ffn( self.ln_wo_grad(x).mul(scale2.add(1)).add_(shift2) ).mul(gamma2)) # this mul(gamma2) cannot be in-placed when FusedMLP is used
        return x


class AdaLNBeforeHead(nn.Module):
    def __init__(self, C, D, norm_layer):   # C: embed_dim, D: cond_dim
        super().__init__()
        self.C, self.D = C, D
        self.ln_wo_grad = norm_layer(C, elementwise_affine=False)
        self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), nn.Linear(D, 2*C))
    
    def forward(self, x_BLC: torch.Tensor, cond_BD: torch.Tensor):
        scale, shift = self.ada_lin(cond_BD).view(-1, 1, 2, self.C).unbind(2)
        return self.ln_wo_grad(x_BLC).mul(scale.add(1)).add_(shift)


class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).view(-1, 1, 6, C)   # B16C