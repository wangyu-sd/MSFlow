import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class MultiScaleGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.scale_levels = cfg.scale_levels  # 尺度层级数
        
        # 共享的codebook嵌入层
        # self.codebook = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        
        # GPT解码器架构
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.hidden_size,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.dim_feedforward,
            batch_first=True,
            norm_first=True  # 参考网页1的Pre-LN设计
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, cfg.n_layers)
        
        # 多尺度预测头
        self.proj_heads = nn.ModuleList([
            nn.Linear(cfg.hidden_size, cfg.vocab_size) 
            for _ in range(self.scale_levels)
        ])
        
        # 位置编码
        self.pos_embed = nn.Embedding(cfg.max_len, cfg.hidden_size)

    def _create_scale_mask(self, seq_len, device):
        """创建分层因果掩码(参考网页2/3的块状注意力设计)"""
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
        total_blocks = 2**self.scale_levels -1
        
        for level in range(self.scale_levels):
            block_size = 2**level
            start = 2**level -1
            end = 2**(level+1) -1
            # 块内允许全连接，块间保持因果性
            mask[start:end, start:end] = torch.tril(
                torch.ones(block_size, block_size, dtype=torch.bool)
            )
            
        return mask

    def forward(self, src_indices, src_emb, tgt_indices, codebook):
        """
        输入:
            src_indices: (B, S_pocket) 口袋的VQ indices
            src_emb: (B, S_pocket, D) 口袋的embeddings 
            tgt_indices: (B, L_peptide) 肽链的目标indices
        输出:
            logits: 各尺度的预测logits列表
            loss: 多尺度加权损失
        """
        B = src_indices.size(0)
        
        # 编码器处理(此处假设src_emb已编码)
        memory = src_emb  # (B, S, D)
        
        # 目标序列嵌入
        tgt_emb = codebook(tgt_indices)  # (B, L, D)
        positions = self.pos_embed(torch.arange(tgt_emb.size(1), device=tgt_emb.device))
        tgt_emb = tgt_emb + positions.unsqueeze(0)
        
        # 生成分层因果掩码
        causal_mask = self._create_scale_mask(tgt_emb.size(1), tgt_emb.device)
        
        # 解码器前向
        output = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=~causal_mask,  # 翻转掩码符合PyTorch规范(参考网页7)
            memory_key_padding_mask=None
        )
        
        # 多尺度预测
        scale_logits = []
        for level in range(self.scale_levels):
            # 提取对应尺度的特征
            scale_feat = output[:, 2**level -1 : 2**(level+1) -1, :]
            scale_logits.append(self.proj_heads[level](scale_feat))
        
        # 计算多尺度损失
        loss = 0
        for level, logits in enumerate(scale_logits):
            scale_targets = tgt_indices[:, 2**level -1 : 2**(level+1) -1]
            loss += F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                scale_targets.contiguous().view(-1),
                # ignore_index=0  # 忽略padding
            )
        
        return scale_logits, loss

    @torch.no_grad()
    def generate(self, src_emb, codebook, max_scales=3, temperature=1.0):
        """
        多尺度自回归生成(参考网页3的next scale prediction)
        参数:
            src_emb: 口袋编码 (B, S, D)
            codebook: VQ-VAE的codebook
            max_scales: 最大生成尺度数
        """
        B = src_emb.size(0)
        device = src_emb.device
        current_seq = torch.full((B,1), self.cfg.bos_token_id, device=device)
        
        for scale in range(max_scales):
            # 当前尺度生成长度
            curr_len = current_seq.size(1)
            block_size = 2**scale
            scale_positions = 2**scale -1  # 当前尺度的起始位置
            
            # 生成当前尺度掩码
            total_len = curr_len + block_size
            causal_mask = self._create_scale_mask(total_len, device)
            
            # 获取嵌入
            seq_emb = codebook(current_seq)  # (B, L, D)
            positions = self.pos_embed(torch.arange(total_len, device=device))
            seq_emb = seq_emb + positions[:curr_len].unsqueeze(0)
            
            # 解码器前向
            output = self.decoder(
                tgt=seq_emb,
                memory=src_emb,
                tgt_mask=~causal_mask[:curr_len, :curr_len]
            )
            
            # 预测当前尺度的token
            scale_feat = output[:, scale_positions:scale_positions+block_size, :]
            logits = self.proj_heads[scale](scale_feat) / temperature
            next_tokens = torch.argmax(logits, dim=-1)
            
            # 拼接生成结果
            current_seq = torch.cat([current_seq, next_tokens], dim=1)
            
        return current_seq[:, 1:]  # 去除BOS