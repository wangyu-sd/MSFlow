import torch
import numpy as np

class ProteinGeometryProcessor:
    def __init__(self, eps=1e-8):
        self.eps = eps  # 数值稳定性系数

    def _calc_bond_angle(self, a, b, c):
        """计算三点键角"""
        ba = a - b
        bc = c - b
        cos_theta = torch.sum(ba * bc, dim=-1) / (
            torch.norm(ba, dim=-1) * torch.norm(bc, dim=-1) + self.eps)
        return torch.acos(torch.clamp(cos_theta, -1.0, 1.0)).unsqueeze(-1)

    def _calc_dihedral_angle(self, a, b, c, d):
        """计算四点二面角"""
        b0 = a - b
        b1 = c - b
        b2 = d - c

        v = torch.cross(b0, b1, dim=-1)
        w = torch.cross(b2, b1, dim=-1)
        x = torch.sum(v * w, dim=-1)
        y = torch.sum(torch.cross(v, w) * b1, dim=-1) / (torch.norm(b1, dim=-1) + self.eps)
        
        return torch.atan2(y, x).unsqueeze(-1)

    def compute_geometry(self, coords):
        """
        输入：骨架坐标[B, N, 3] (假设每个位置对应Cα坐标)
        输出：
            dist_matrix: [B, N-1, 3] 相邻残基间距离
            torsion_angles: [B, N-2, 1] 连续三个Cα的键角  
            dihedral_angles: [B, N-2, 4] 四个特征二面角
        """
        B, N, _ = coords.shape
        
        # 距离矩阵计算 (Cα_i到Cα_{i+1}的距离)
        dist_matrix = torch.norm(coords[:, 1:] - coords[:, :-1], dim=-1)
        dist_matrix = dist_matrix.unsqueeze(-1).expand(-1, -1, 3)  # 扩展为3通道
        
        # 键角计算 (连续三个Cα)
        torsion_angles = torch.stack([
            self._calc_bond_angle(coords[:, i-1], coords[:, i], coords[:, i+1])
            for i in range(1, N-1)], dim=1)
        
        # 二面角计算 (四种组合方式)
        dihedral_list = []
        for i in range(2, N-1):
            # 四种二面角组合 (参考网页6的几何特征)
            ang1 = self._calc_dihedral_angle(coords[:,i-2], coords[:,i-1], coords[:,i], coords[:,i+1])
            ang2 = self._calc_bond_angle(coords[:,i-1], coords[:,i], coords[:,i+1])
            ang3 = self._calc_dihedral_angle(coords[:,i-2], coords[:,i], coords[:,i+1], coords[:,i-1])
            ang4 = self._calc_bond_angle(coords[:,i], coords[:,i+1], coords[:,i-1])
            dihedral_list.append(torch.cat([ang1, ang2, ang3, ang4], dim=-1))
        
        dihedral_angles = torch.stack(dihedral_list, dim=1)
        
        return dist_matrix, torsion_angles, dihedral_angles

    def build_local_frames(self, coords):
        """
        构建局部坐标系[6,7](@ref)
        输入：骨架坐标[B, N, 3]
        输出：局部坐标系矩阵[B, N, 3, 3]
        """
        B, N, _ = coords.shape
        device = coords.device
        
        # 生成虚拟N和C原子坐标（若仅有Cα坐标）
        # 此处使用网页7的残基坐标系构建方法
        v_n = torch.zeros_like(coords)
        v_c = torch.zeros_like(coords)
        
        # 主链坐标系构建
        for i in range(1, N-1):
            c_alpha = coords[:, i]
            prev_c = coords[:, i-1]
            next_n = coords[:, i+1]
            
            # 构建局部坐标系三个基向量
            x = next_n - c_alpha
            y = prev_c - c_alpha
            z = torch.cross(x, y, dim=-1)
            
            # 正交化处理（施密特正交化）
            x_norm = x / (torch.norm(x, dim=-1, keepdim=True) + self.eps)
            z_norm = z / (torch.norm(z, dim=-1, keepdim=True) + self.eps)
            y_norm = torch.cross(z_norm, x_norm, dim=-1)