# 🧠 Mini-Telepathy: Motor Cortex (M1) Spike-to-Cursor BCI Simulator & Decoder

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](#)
[![Interactive Web Studio](https://img.shields.io/badge/Live_Demo-HTML5_Canvas-00f2fe?logo=html5&logoColor=white)](#2-零依赖网页实时交互台推荐)
[![Field](https://img.shields.io/badge/Neuroscience-Invasive_BCI_%2F_Neuralink-a855f7)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-10b981.svg)](LICENSE)

> **复现 Neuralink "Telepathy" 脑机接口最核心的算法链路**：从 64 通道运动皮层（Motor Cortex, M1）微电极阵列的动作电位脉冲（Action Potential Spikes），通过状态空间**卡尔曼滤波器（Kalman Filter）**与**最优线性估计器（Ridge OLE）**实时解码二维屏幕光标速度与运动轨迹。

![Mini-Telepathy Decoding Result](assets/decoding_result.png)

---

## ✨ 项目亮点 (Key Highlights)

1. **端到端生物物理仿真 (Biophysical M1 Simulation)**：
   - 基于 **Flash & Hogan (1985)** 最小急动度模型（Minimum-Jerk Trajectory）生成 8 方向放射状靶标追踪任务（Center-Out Reaching Task）。
   - 基于 **Georgopoulos et al. (1986)** 运动皮层神经元**余弦方向调谐定律（Cosine Directional Tuning）**与非齐次泊松过程（Inhomogeneous Poisson Process）模拟 64 通道微电极锋电位（Spikes）。
2. **双经典侵入式脑机解码器 (Clinical-Grade BCI Decoders)**：
   - **State-Space Kalman Filter (`KalmanFilterDecoder`)**：利用矩阵求逆引理（Woodbury Identity）实现每步 **$< 30\,\mu\text{s}$** 的超低延迟递归状态更新，平滑追踪 $[p_x, p_y, v_x, v_y]^\top$。
   - **Multi-Lag Ridge Wiener Filter (`RidgeVelocityDecoder`)**：融合多时间步历史神经发放率窗口的 $L_2$ 正则化最优线性解码器。
3. **开箱即用的双形态体验 (Python 科学计算栈 + 零依赖 Web 实时交互台)**：
   - **`demo.py`**：一键完成 32 轮校准训练 + 8 轮留出测试评估，自动输出量化指标表格并绘制 4 面板科研级总结大图。
   - **`index.html`**：无需安装任何环境，双击或在 **GitHub Pages** 上即可打开 60 FPS 实时神经解码控制台，支持 **「自动靶标追踪」** 与 **「鼠标实时脑控意图模式」**！

---

## 🔬 核心数学原理 (Mathematical Formulation)

```mermaid
flowchart LR
    A["运动意图 (Motor Intent)\n速度 v(t) = [vx, vy]ᵀ"] --> B["M1 余弦方向调谐\nλi(t) = bi + mi(v(t)·ui)"]
    B --> C["泊松脉冲生成 (Poisson Spikes)\nyi(t) ~ Poisson(λi(t)Δt)"]
    C --> D["因果指数平滑滤波\nzi(t) = α·yi(t)/Δt + (1-α)zi(t-1)"]
    D --> E["卡尔曼滤波状态空间解码\nx̂(t) = Ax̂(t-1) + K(z(t) - CAx̂(t-1))"]
    E --> F["2D 脑机光标轨迹\n[p̂x(t), p̂y(t), v̂x(t), v̂y(t)]"]
```

### 1. 运动皮层神经元余弦调谐编码 (Encoding Model)
初级运动皮层（M1）中第 $i$ 个神经元具有特定的偏好方向角 $\theta_{p,i} \in [0, 2\pi)$（对应单位方向向量 $\mathbf{u}_i = [\cos\theta_{p,i}, \sin\theta_{p,i}]^\top$）。当受试者意图以速度 $\mathbf{v}(t) = [v_x(t), v_y(t)]^\top$ 移动光标时，该神经元的瞬时发放率 $\lambda_i(t)$ 服从余弦调谐关系：

$$\lambda_i(t) = \max\Big(0,\; b_i + m_i \big(\mathbf{v}(t) \cdot \mathbf{u}_i\big) + s_i \|\mathbf{v}(t)\|_2 + \epsilon_i(t)\Big)$$

其中 $b_i$ 为基线发放率（Hz），$m_i$ 为方向调制深度（Hz），在时间窗 $\Delta t = 20\text{ ms}$ 内的脉冲计数服从泊松分布 $y_i(t) \sim \text{Poisson}(\lambda_i(t)\Delta t)$。

### 2. 状态空间卡尔曼滤波解码 (Kalman Filter Decoding)
定义时刻 $t$ 的运动状态向量为 $\mathbf{x}_t = [p_x(t), p_y(t), v_x(t), v_y(t)]^\top \in \mathbb{R}^4$，多通道平滑发放率观测向量为 $\mathbf{z}_t \in \mathbb{R}^{64}$：

$$\begin{aligned}
\text{状态转移方程 (State Transition):} \quad \mathbf{x}_t &= \mathbf{A}\mathbf{x}_{t-1} + \mathbf{w}_t, \quad \mathbf{w}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{W}) \\
\text{神经观测方程 (Neural Observation):} \quad \mathbf{z}_t &= \mathbf{C}\mathbf{x}_t + \mathbf{q}_t, \quad \mathbf{q}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{Q})
\end{aligned}$$

在校准阶段（Calibration Trials）通过闭式最大似然估计（Closed-Form MLE）拟合参数矩阵 $(\mathbf{A}, \mathbf{W}, \mathbf{C}, \mathbf{Q})$，在实时解码阶段通过递归预测-更新步输出最优光标位置与速度估计。

---

## 🚀 快速开始 (Quickstart)

### 1. 运行 Python 端到端解码与 Benchmark
```bash
# 1. 进入项目目录并安装依赖
cd mini-telepathy
pip install -r requirements.txt

# 2. 运行校准、测试与 4 面板科研大图生成
python demo.py
```

终端将输出留出测试集（Held-Out Test Split）的定量性能对比：

| Decoder Algorithm | Vel $R^2$ | Pos $R^2$ | Pearson $r$ | Vel RMSE | Latency / bin |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Kalman Filter (State-Space)** | **~0.91** | **~0.93** | **~0.95** | **~0.28** | **~25 µs** |
| **Ridge OLE (5-Lag Wiener)** | ~0.87 | ~0.88 | ~0.93 | ~0.34 | ~15 µs |

### 2. 零依赖网页实时交互台（推荐）
直接用浏览器打开仓库根目录下的 [`index.html`](index.html)（或者将仓库推送到 GitHub 后开启 **GitHub Pages**）：
- **🎯 自动靶标追踪模式**：观察 64 通道神经元脉冲如何实时驱动光标完成 8 方向靶标捕获，并实时计算 Fitts 定律脑机通信比特率（bps）。
- **🖱️ 鼠标脑控意图模式**：在左侧竞技场内移动鼠标代表你的“运动皮层意图”，系统会实时将其编码为 64 个通道的带噪神经脉冲，并由卡尔曼滤波器解码还原出蓝光脑机光标！
- **实时参数调节**：自由滑动调整脉冲平滑时间常数 $\tau$、背景神经噪声 $\sigma$ 和解码增益 $K$，直观感受信噪比与响应延迟之间的工程权衡。

---

## 📂 项目目录结构 (Repository Structure)

```text
mini-telepathy/
├── index.html                  # 零依赖 HTML5 实时脑机接口仿真与解码交互台 (支持 GitHub Pages)
├── demo.py                     # 一键运行 Python 端到端训练、评估与 4 面板可视化脚本
├── requirements.txt            # Python 科学计算依赖 (numpy, matplotlib)
├── src/
│   ├── __init__.py
│   ├── simulator.py            # M1 运动皮层余弦调谐脉冲仿真器与 Minimum-Jerk 轨迹生成
│   └── decoder.py              # 状态空间卡尔曼滤波解码器与多延迟 Ridge OLE 解码器
├── assets/
│   ├── decoding_result.png     # 自动生成的 4 面板高分辨率解码效果图
│   └── benchmark_summary.txt   # 留出测试集定量评估报告
└── LICENSE                     # MIT 开源许可证
```

---

## 🛣️ 后续扩展方向 (Roadmap for Next Versions)

如果你想在这个项目基础上继续深入扩展，可以依次添加以下模块：
- [ ] **接入真实公开猕猴脑机数据集**：对接 [Neural Latents Benchmark (NLB)](https://neurallatents.github.io/) 或 DANDI Archive 上的真实 `MC_Maze` NWB 格式数据。
- [ ] **非线性深度学习解码器**：增加轻量级 `GRU` / `LSTM` 或 `Causal Transformer` 解码器，与卡尔曼滤波器进行同台对比。
- [ ] **神经非平稳性自适应重校准 (Closed-Loop Recalibration)**：模拟微电极微位移导致的跨天信号漂移（Neural Drift），实现无监督自校准。
