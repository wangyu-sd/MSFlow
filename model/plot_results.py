import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy import stats
import wandb
def calc_statistics(arr, threshold=0.1):
    # 中位数和分位数（兼容空数组）
    median_val = np.median(arr) if len(arr) > 0 else 0
    q0, q25, q50, q75, q100 = np.percentile(arr, [0, 25, 50, 75, 100]) if len(arr) > 0 else (0,0,0)
    
    # 众数计算（处理多众数情况
    mode_res = stats.mode(arr, keepdims=True) if len(arr) > 0 else (np.array([0]), np.array([0]))
    modes = mode_res.mode
    
    usage_rate = (arr > threshold).sum() / len(arr) if len(arr) > 0 else 0
    prob = arr / (np.sum(arr) + 1e-6)
    entropy = -np.sum(prob * np.log(prob + 1e-6)) if len(arr) > 0 else 0
    
    if median_val > 0:
        median_val = np.mean(median_val)
        
    mean = np.mean(arr) if len(arr) > 0 else 0
    
    return {
        'median': median_val,
        'q25': q25,
        'q50': q50,
        'q75': q75,
        'q0': q0,
        'entropy': entropy,
        'usage_rate': usage_rate,
        'q100': q100,
        'modes': modes,
        "median_val": median_val,
        "mean": mean,
    }


def plot_codebook_dist(coodbook_cnt, log_dir, it):
  
  # 初始化样式配置
  plt.style.use('seaborn-v0_8-darkgrid')  # 使用专业图表样式[1,3](@ref)
  plt.rcParams.update({
      'font.size': 12,  # 统一字体尺寸
      'axes.titlesize': 14,  # 标题加粗
      'axes.titleweight': 'bold',
      'figure.figsize': (10, 6)  # 优化画布比例
  })
  # 创建画布和主图
  fig, ax = plt.subplots()
  
  stats_dict = calc_statistics(coodbook_cnt)
  # 绘制主数据曲线（增加阴影效果）
  ax.plot(coodbook_cnt, 
          alpha=0.8, 
          linewidth=2.5,
          color='#2c7fb8', 
          label='Code Usage',
          marker='o',
          markevery=int(len(coodbook_cnt)/20),
          markersize=6)
  
  # 添加统计参考线系统
  stats_lines = {
      'Median': (stats_dict['median'], 'purple', '--', 2),
      '25th %ile': (stats_dict['q25'], 'green', ':', 1.8),
      '75th %ile': (stats_dict['q75'], 'orange', ':', 1.8),
      'Mean': (stats_dict['mean'], 'red', '-.', 1.5)
  }
  
  for label, (value, color, ls, lw) in stats_lines.items():
      ax.axhline(value, color=color, linestyle=ls, 
                linewidth=lw, alpha=0.9, 
                label=f'{label}: {value:.1f}')

  # 图表装饰系统
  ax.set_title(
      f'Codebook Usage Distribution | Iteration {it}\n'
      f'Entropy: {stats_dict["entropy"]:.2f} | '
      f'Usage Rate: {stats_dict["usage_rate"]:.2f}',
      pad=20)  # 增加标题间距[4](@ref)
  
  ax.set_xlabel('Codebook Index', labelpad=12)
  ax.set_ylabel('Usage Count', labelpad=12)
  
  # 增强网格系统
  ax.grid(True, alpha=0.4, which='both',
          linestyle='--', linewidth=0.7)
  
  # 专业图例布局
  leg = ax.legend(bbox_to_anchor=(1.28, 0.95), 
                  frameon=True, 
                  shadow=True,
                  borderpad=1.2,
                  title='Statistics',
                  title_fontsize='13')
  leg.get_frame().set_facecolor('#f5f5f5')

  # 保存优化（增加dpi和压缩比）
  os.makedirs(os.path.join(log_dir, "codebook_cnt"), exist_ok=True)
  save_path = os.path.join(log_dir, "codebook_cnt", f'codebook_cnt_{it}.png')
  plt.savefig(save_path, 
              dpi=300,  # 打印级分辨率[5](@ref)
              bbox_inches='tight', 
              facecolor='white') 
  
  print(f"Saved enhanced visualization to {save_path}")
  plt.close(fig)
  
  wandb.log({'codebook_usage': stats_dict['usage_rate']}, step=it)