# SDOA-Net：不完美阵列下的高效深度学习 DOA 估计

## 1. 研究背景与问题

**DOA（Direction of Arrival，到达角）估计**是雷达、无线通信、通感一体化（ISAC）等系统的核心任务之一。

低成本阵列在实际部署中往往存在多种**不完美因素**：

| 不完美类型 | 含义 |
|-----------|------|
| 位置扰动 | 天线实际位置偏离理想均匀线阵 |
| 幅度不一致 | 各通道增益存在随机偏差 |
| 相位不一致 | 各通道存在随机相位误差 |
| 互耦效应 | 相邻天线之间的电磁耦合 |
| 非线性效应 | 功放等环节的 `tanh` 非线性失真 |

传统方法（FFT、MUSIC、OMP、ANM 等）多基于**理想阵列模型**，在不完美阵列条件下性能会显著下降。

---

## 2. SDOA-Net 核心思想

**SDOA-Net（Super-resolution DOA Network）** 是一种基于深度学习的 DOA 估计方法，主要创新点如下：

1. **输入为采样接收信号**，而非协方差矩阵，直接从原始数据提取特征。
2. **网络输出与目标 DOA 数量无关的向量**（维度 = 2×天线数），再通过字典匹配得到空间谱，因此**同一网络可处理任意数量目标**（最多 3 个）。
3. **低维网络结构**（`spectrumModule`），训练收敛快、实现复杂度低。
4. **超分辨**：在细粒度角度网格（默认 10000 点，-50°~50°）上重建空间谱。

---

## 3. 整体流程

```
 imperfect array 信号生成
        ↓
   加噪（训练/测试）
        ↓
  SDOA-Net (spectrumModule)
        ↓
  输出 2×M 维特征向量 u
        ↓
  与理想导向矢量字典匹配 → 空间谱 S(θ)
        ↓
  峰值检测 → DOA 估计
        ↓
  与 FFT / MUSIC / OMP / ANM 对比评估
```

---

## 4. 信号模型与数据生成（`doasys.py`）

### 4.1 阵列与导向矢量

- 默认：**16 元均匀线阵**，阵元间距 `d = 0.5λ`
- 导向矢量：

\[
\mathbf{a}(\theta) = \exp\left(j2\pi(d\mathbf{n} + \Delta\mathbf{d}) \sin\theta\right)
\]

其中 \(\mathbf{n} = [0,1,\ldots,M-1]^T\)，\(\Delta\mathbf{d}\) 为位置扰动。

### 4.2 不完美阵列建模

对每条样本依次叠加：

1. **位置扰动**：`d_per ~ N(0, max_per_std²)`
2. **幅度/相位误差**：`amp_phase = amp · exp(j·pha)`
3. **互耦矩阵**：下三角衰减结构，强度由 `max_mc` 控制
4. **非线性**：`s = tanh(nonlinear · s)`，再归一化

### 4.3 目标 DOA 生成

- 目标数量：随机 1~3 个
- 角度范围：-45° ~ +45°
- 最小间隔约束，避免过近目标

### 4.4 参考空间谱（训练标签）

在角度网格 \(\{\theta_k\}\) 上，以真实 DOA 为中心构造高斯参考谱：

\[
S_{\text{ref}}(\theta) = \sum_i \exp\left(-\frac{(\theta - \theta_i)^2}{\sigma^2}\right)
\]

其中 \(\sigma = \text{gaussian\_std} / M\)。

---

## 5. 网络结构（`spectrumModule`）

```
输入:  noisy signal  [batch, 2, M]  (实部 + 虚部)
  ↓
Linear: 2M → inner_dim × n_filters  (默认 32×8)
  ↓
Reshape: [batch, n_filters, inner_dim]
  ↓
Conv1d × n_layers (circular padding, BatchNorm, ReLU)  (默认 8 层)
  ↓
Linear: inner_dim × n_filters → 2M
  ↓
输出:  u  [batch, 2M]
```

**默认超参数：**

| 参数 | 默认值 |
|------|--------|
| `n_layers` | 8 |
| `n_filters` | 8 |
| `inner_dim` | 32 |
| `kernel_size` | 3 |

---

## 6. 空间谱重建与 DOA 提取

### 6.1 字典匹配（超分辨）

构建理想阵列导向矢量字典 \(\mathbf{D}(\theta_k)\)，将网络输出 \( \mathbf{u} = u_r + j u_i \) 映射到空间谱：

\[
S(\theta_k) = \left| \Re\{\mathbf{D}^H \mathbf{u}\} \right|^2 + \left| \Im\{\mathbf{D}^H \mathbf{u}\} \right|^2
\]

等价于代码中的 `mm_real`、`mm_imag` 矩阵乘法。

### 6.2 DOA 估计

1. 对归一化空间谱做**峰值检测**（`scipy.signal.find_peaks`）
2. 取幅度最大的若干峰值对应角度
3. 与真实 DOA 做最近邻匹配，计算 **RMSE**

---

## 7. 训练流程（`train.py`）

### 7.1 损失函数

\[
\mathcal{L} = \text{MSE}\big(S(\theta),\; S_{\text{ref}}(\theta)\big)
\]

即网络输出经字典匹配后的空间谱与参考高斯谱之间的均方误差。

### 7.2 课程学习（Curriculum Training）

按 7 种不完美类型**依次训练**，逐步增加难度：

| train_type | 训练内容 |
|------------|----------|
| 0 | 理想阵列（无不完美） |
| 1 | 仅位置扰动 |
| 2 | 仅幅度误差 |
| 3 | 仅相位误差 |
| 4 | 仅互耦 |
| 5 | 仅非线性 |
| 6 | **全部不完美因素综合** |

每种类型训练 `n_epochs` 轮（默认 300），最终模型保存为 `net.pkl`。

### 7.3 训练命令

```bash
py train.py --new_train 1 --n_epochs 300 --n_training 8000
```

**主要训练参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `n_training` | 8000 | 训练样本数 |
| `n_validation` | 64 | 验证样本数 |
| `batch_size` | 64 | 批大小 |
| `lr` | 0.0002 | Adam 学习率 |
| `snr` | 1.0 | 训练噪声（随机 SNR，最大 0 dB） |
| `grid_size` | 10000 | 角度网格点数 |

---

## 8. 测试与评估流程（`main.py`）

### 8.1 对比方法

| 方法 | 实现 |
|------|------|
| **Proposed（SDOA-Net）** | 网络 + 字典匹配 |
| **FFT** | 导向矢量与接收信号直接匹配 |
| **MUSIC** | MATLAB `MUSIConesnapshot.m`（单快照 Hankel） |
| **OMP** | 正交匹配追踪 |
| **ANM** | MATLAB `ANM.m` + CVX 凸优化 |

### 8.2 评估指标

- 在 SNR = 10 / 20 / 30 dB 下做 Monte Carlo 仿真
- 统计 **RMSE（°）** 随 SNR 变化曲线
- 绘制各 SNR 下的**空间频谱对比图**

### 8.3 运行命令

```bash
# 完整评估 + 频谱图 + RMSE 曲线
py main.py --show_fig 1 --n_test 100

# 仅生成 10/20/30 dB 频谱图
py main.py --fig_only 1 --show_fig 1 --snr_list 10,20,30
```

### 8.4 输出文件

| 文件/目录 | 内容 |
|-----------|------|
| `figures/spectrum_SNR_*dB.png` | 各 SNR 空间频谱图 |
| `figures/RMSE_vs_SNR.png` | RMSE 对比曲线 |
| `net.pkl` | 训练好的网络权重 |
| `loss.npz` | 训练/验证损失记录 |

---

## 9. 项目文件结构

```
SDOA-Net/
├── doasys.py          # 信号生成、网络定义、训练/DOA 提取
├── train.py           # 训练入口（课程学习）
├── main.py            # 测试评估入口（对比实验 + 绘图）
├── MUSIConesnapshot.m # MUSIC 基线（MATLAB）
├── ANM.m              # 原子范数最小化基线（MATLAB + CVX）
├── net.pkl            # 预训练模型
├── loss.npz           # 损失记录
├── figures/           # 输出图表
└── requirements.txt   # Python 依赖
```

---

## 10. 方法优势总结

| 对比项 | 传统 DL-DOA | SDOA-Net |
|--------|-------------|----------|
| 输入 | 协方差矩阵 | **原始接收信号** |
| 输出 | 与目标数相关 | **固定维度向量 → 空间谱** |
| 目标数泛化 | 需重新训练 | **同一网络适用 1~3 目标** |
| 不完美阵列 | 多数未专门建模 | **显式建模 5 类不完美因素** |
| 网络规模 | 较大 | **低维，收敛快** |
| 角度分辨率 | 受阵列孔径限制 | **超分辨（10000 点网格）** |

---

## 11. 依赖环境

- **Python 3.9+**：NumPy、SciPy、PyTorch、Matplotlib
- **MATLAB Engine for Python**：MUSIC、ANM 基线
- **CVX 工具箱**（MATLAB）：ANM 求解器

安装：

```bash
py -m pip install numpy scipy matplotlib torch matlabengine
```

---

## 12. 一句话总结

> SDOA-Net 以**不完美阵列下的单快照接收信号**为输入，通过轻量级卷积网络输出与 DOA 数量无关的特征向量，再经**导向矢量字典匹配**重建超分辨空间谱并完成 DOA 估计；采用**课程学习**逐步适应各类阵列缺陷，在不完美阵列条件下优于 FFT、MUSIC、OMP、ANM 等传统方法。
