import math
import torch
from torch import nn
from torch.nn import functional as F
from scipy.cluster.vq import kmeans2
from typing import List
import torch.distributed as dist
class ReparameterizedCodebook(nn.Module):
    def __init__(self, codebook_size=128, embedding_dim=512):
        super().__init__()
        # 基向量矩阵 (K x base_dim)
        self.base = nn.Parameter(torch.randn(codebook_size, embedding_dim))  
        # 可学习投影矩阵
        self.proj = nn.Linear(embedding_dim, embedding_dim)  
        
        self.reset_parameters()
        
    def reset_parameters(self):
        # 重置参数
        torch.nn.init.orthogonal_(self.base)
        torch.nn.init.xavier_uniform_(self.proj.weight)
        self.proj.bias.data.zero_()
        
    def forward(self):
        return self.proj(self.base)  # 动态生成codebook向量
class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size, embedding_dim, commitment_cost, init_steps, collect_desired_size, scales):
        super().__init__()

        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.scales = [2**i for i in range(scales+1)]
        
        self.coodbook_generator = ReparameterizedCodebook(codebook_size, embedding_dim)
        self.register_buffer("embedding", self.coodbook_generator())
        # self.embedding = nn.Embedding(codebook_size, embedding_dim)
        # self.embedding.weight.data.uniform_(-1/codebook_size, 1/codebook_size)

        self.init_steps = init_steps
        self.collect_phase = init_steps > 0
        collected_samples = torch.Tensor(0, self.embedding_dim)
        self.collect_desired_size = collect_desired_size
        self.use_prob = True
        self.register_buffer("collected_samples", collected_samples)
        self.register_buffer('usage_counts', torch.zeros(codebook_size, dtype=torch.long))
    
    
    def reset_counts(self):
        self.usage_counts = torch.zeros(self.codebook_size, dtype=torch.long, device=self.embedding.device)
        
    def update_embedding(self):
        self.embedding = self.coodbook_generator()
    
    
    def quantize_input(self, query):
        d_no_grad = torch.sum(query.square(), dim=1, keepdim=True) + torch.sum(self.embedding.data.square(), dim=1, keepdim=False)
        d_no_grad.addmm_(query, self.embedding.data.T, alpha=-2, beta=1)
        idx_N = torch.argmin(d_no_grad, dim=1)
        h_NC = self.embedding[idx_N]
        
        return idx_N, h_NC, d_no_grad

        
    def forward(self, f_BNC, col_samples=False, vae_stage=False):
        f_BCN = f_BNC.permute(0, 2, 1)
        B, C, N = f_BCN.shape

        f_no_grad = f_BCN.detach()
        f_rest = f_no_grad.clone()
        f_hat  = torch.zeros_like(f_rest)
        
        if not vae_stage:
            self.update_embedding()

        with torch.amp.autocast(enabled=False, device_type=f_BCN.device.type):
            mean_q_latent_loss: torch.Tensor = 0.0
            mean_commitment_loss: torch.Tensor = 0.0
            SN = len(self.scales)
            for si, pn in enumerate(self.scales):
                rest_NC = F.interpolate(f_rest, size=(pn), mode='area').permute(0, 2, 1).reshape(-1, C)

                if self.collect_phase and col_samples:
                    self.collected_samples = torch.cat((self.collected_samples, rest_NC), dim=0)
                
                if vae_stage:
                    h_BCn = F.interpolate(rest_NC.reshape(B, -1, C).permute(0, 2, 1), size=(N), mode='linear').contiguous()
                else:                 
                    # d_no_grad = torch.sum(rest_NC.square(), dim=1, keepdim=True) + torch.sum(self.embedding.data.square(), dim=1, keepdim=False)
                    # d_no_grad.addmm_(rest_NC, self.embedding.data.T, alpha=-2, beta=1)
                    # idx_N = torch.argmin(d_no_grad, dim=1)
                    # idx_Bn = idx_N.view(B, pn)
                    idx_N, h_NC, _ = self.quantize_input(rest_NC, self.embedding.data)
                    idx_Bn, h_BnC = idx_N.view(B, pn), h_NC.view(B, pn, C)
                    
                    h_BCn = F.interpolate(h_BnC.permute(0, 2, 1), size=(N), mode='linear').contiguous()
                    batch_counts = torch.bincount(idx_Bn.flatten(), minlength=self.codebook_size)
                    self.batch_counts = self.usage_counts + batch_counts
                    # h_BCn, _ = self.get_softvq(rest_NC, B, pn, C, N)

                f_hat = f_hat + h_BCn
                f_rest -= h_BCn

                mean_commitment_loss += F.mse_loss(f_hat.data, f_BCN).mul_(0.25)
                mean_q_latent_loss += F.mse_loss(f_hat, f_no_grad)
            
            mean_commitment_loss *= 1. / SN
            mean_q_latent_loss *= 1. / SN
            
            f_hat = (f_hat.data - f_no_grad).add_(f_BCN)
            f_hat = f_hat.permute(0, 2, 1) # B, N, C
            # print(self.embedding.weight.data[219, :10])
            # divs_loss = self._calculate_diversity_loss()
            
            return f_hat, mean_commitment_loss, mean_q_latent_loss, 0.
    

    def collect_samples(self, zq):
        # Collect samples
        self.forward(zq, col_samples=True)
        # If enough samples collected, initialise codebook with k++ means
        if self.collected_samples.shape[0] >= self.collect_desired_size:
            self.collected_samples = self.collected_samples[-self.collect_desired_size:]
            self.collect_phase = False
            self.kmeans_init()
            self.collected_samples = torch.zeros(0, self.embedding_dim, device=self.collected_samples.device)
    
    def kmeans_init(self):
        print('K++ means Codebook initialisation starting...')
        device = self.collected_samples.device
        collected_samples = self.collected_samples.cpu().detach().numpy()

        # Perform k-means clustering on the entire embedding space
        k = kmeans2(collected_samples, self.codebook_size, minit='++')[0]
        
        # Update embedding weights with k-means centroids
        self.embedding.weight.data = torch.from_numpy(k).to(device)
        print('K++ Success!')

    def f_to_idxBl(self, f_BNC):
        f_BCN = f_BNC.permute(0, 2, 1)
        B, C, N = f_BCN.shape

        f_rest = f_BCN.detach().clone()
        f_hat = torch.zeros_like(f_rest)

        idx_Bl: List[torch.Tensor] = []
        SN = len(self.scales)
        for si, pn in enumerate(self.scales):
            # Find the nearest embedding
            z_NC = F.interpolate(f_rest, size=(pn), mode='area').permute(0, 2, 1).reshape(-1, C)
            idx_N, h_NC, _ = self.quantize_input(z_NC, self.embedding.data)
            idx_Bn, h_BnC = idx_N.view(B, pn), h_NC.view(B, pn, C)
            h_BCn = F.interpolate(h_BnC.permute(0, 2, 1), size=(N), mode='linear').contiguous()
            f_hat.add_(h_BCn)
            f_rest.sub_(h_BCn)
            
            # z_NC = F.interpolate(f_rest, size=(pn), mode='area').permute(0, 2, 1).reshape(-1, C)
            # h_BCn, d_no_grad = self.get_softvq(z_NC, B, pn, C, N)
            # f_hat.add_(h_BCn)
            # f_rest.sub_(h_BCn)

            idx_Bl.append(idx_Bn.reshape(B, pn))
        
        return idx_Bl
    
    def idxBl_to_var_input(self, gt_idx_Bl):
        next_scales = []
        B, N, C = gt_idx_Bl[0].shape[0], self.scales[-1], self.embedding_dim
        SN = len(self.scales)

        # f_hat = gt_idx_Bl[0].new_zeros(B, C, N, dtype=torch.float32)
        pn_next = self.scales[0]
        for si in range(SN-1):
            h = self.embedding.data[gt_idx_Bl[si]]
            h_BCn = F.interpolate(h.transpose_(1, 2).view(B, C, pn_next), size=(pn_next * 2), mode='linear')
            #From: 0,   1, 1, 2, 2, 2, 2
            #To:   cls, 0, 0, 1, 1, 1, 1  (cls will be added out of this function)
            next_scales.append(h_BCn.transpose(1, 2)) # B, N, C
            pn_next = self.scales[si+1]
            # next_scales.append(F.interpolate(f_hat, size=(pn_next), mode='area').view(B, C, -1).transpose(1, 2))
        
        return torch.cat(next_scales, dim=1) # cat BlCs to BLC
    
    def get_next_autoregressive_input(self, si: int, SN: int, f_hat, h_BCn):
        N = self.scales[-1]
        h = F.interpolate(h_BCn, size=(N), mode='linear')
        f_hat.add_(h)
        if si == SN-1:
            return f_hat, f_hat
        return f_hat, F.interpolate(f_hat, size=(self.scales[si+1]), mode='area')
    
    
    def get_usage_metrics(self):
        """返回codebook使用率诊断指标"""
        # total = self.usage_counts.sum().float()
        used = (self.usage_counts > 0).sum().float()
        return {
            'usage_rate': used / self.codebook.num_embeddings,  # 已使用向量占比
            'entropy': self._calculate_codebook_entropy(),       # 信息熵
            'top5_usage': self.usage_counts.topk(5).values       # 最高频使用向量计数
        }
    
    def _calculate_codebook_entropy(self):
        # 计算codebook使用分布的香农熵
        prob = self.usage_counts.float() / (self.usage_counts.sum() + 1e-6)
        return -torch.sum(prob * torch.log(prob + 1e-6))