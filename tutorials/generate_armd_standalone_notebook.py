# -*- coding: utf-8 -*-
"""Build tutorials/armd_standalone_full.ipynb — self-contained ARMD tutorial.

The generated notebook:

1. Mirrors the paper "Auto-Regressive Moving Diffusion Models for Time Series
   Forecasting" (Gao et al., AAAI 2025, arXiv:2412.09328): equations (1)-(10)
   and Algorithms 1 (training) / 2 (sampling) are reproduced verbatim with
   pointers to their code locations.
2. Inlines the *exact* implementation from this repository — `Linear`,
   `ARMD`, `ReduceLROnPlateauWithWarmup`, `Trainer`, plus the `model_utils`
   helpers actually used by the model — so that running the notebook is
   equivalent to running `main.py --config_path Config/stock_paper.yaml`.
3. Uses the real Diffusion-TS `stock_data.csv` already present in this repo;
   pre-processing (`StandardScaler`, sliding windows of length 192, the
   chronological 80/20 split via the `divide` helper) reproduces
   `Utils/Data_utils/real_datasets.py::CustomDataset` exactly.
4. Trains on GPU when CUDA torch is available (the venv ships
   `torch==2.6.0+cu124`), otherwise falls back to CPU; the same code path
   handles both.
5. After training, evaluates with `Trainer.sample_forecast` averaged over
   10 seeds (paper Table 1 protocol) and prints both the achieved
   MSE / MAE on z-score-normalized data and the paper reference
   (Stock: MSE=0.235, MAE=0.269).
"""
from __future__ import annotations

import re
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "armd_standalone_full.ipynb"


def read(p: str) -> str:
    return (ROOT / p).read_text(encoding="utf-8")


def py_cell(src: str) -> dict:
    c = new_code_cell(src.strip() + "\n")
    c["metadata"] = {}
    c["outputs"] = []
    c["execution_count"] = None
    return c


def md_cell(text: str) -> dict:
    c = new_markdown_cell(text.strip() + "\n")
    c["metadata"] = {}
    return c


def strip_repo_imports(code: str) -> str:
    """Remove `from Models...` / `from Utils...` / `from engine...` imports.

    Multi-line imports (with continuation lines) are also handled.
    """
    code = re.sub(r"^from Models\.\S+ import [^\n]*(?:\n[ \t]+[^\n]*)*\n", "", code, flags=re.M)
    code = re.sub(r"^from Utils\.\S+ import [^\n]*(?:\n[ \t]+[^\n]*)*\n", "", code, flags=re.M)
    code = re.sub(r"^from engine\.\S+ import [^\n]*(?:\n[ \t]+[^\n]*)*\n", "", code, flags=re.M)
    return code


def extract_utils() -> str:
    """Inline only the `model_utils` helpers actually referenced in the notebook."""
    mu = read("Models/autoregressive_diffusion/model_utils.py")
    parts: list[str] = []
    for fn in ("exists", "default", "identity", "extract"):
        m = re.search(rf"^def {fn}\(.*?\n(?:^[ \t].*\n)*", mu, re.M)
        if m:
            parts.append(m.group(0).rstrip())
    return "\n\n\n".join(parts) + "\n"


def patch_trainer(tr: str) -> str:
    """Replace `instantiate_from_config` for the LR scheduler with a direct call,
    and drop the upstream sys.path mutation."""
    tr = re.sub(r"^from Utils\.io_utils import [^\n]*\n", "", tr, flags=re.M)
    tr = re.sub(r"^sys\.path\.append\(.*\)\n", "", tr, flags=re.M)
    tr = tr.replace(
        "        sc_cfg = config['solver']['scheduler']\n"
        "        sc_cfg['params']['optimizer'] = self.opt\n"
        "        self.sch = instantiate_from_config(sc_cfg)\n",
        "        p = dict(config['solver']['scheduler']['params'])\n"
        "        p['optimizer'] = self.opt\n"
        "        self.sch = ReduceLROnPlateauWithWarmup(**p)\n",
    )
    tr = tr.replace(
        "        if self.logger is not None:\n"
        "            self.logger.log_info(str(get_model_parameters_info(self.model)))\n",
        "        # logger.log_info(get_model_parameters_info(self.model)) intentionally dropped: helper lives outside the standalone notebook.\n",
    )
    return tr


def patch_lr_sched(src: str) -> str:
    """Keep only `ReduceLROnPlateauWithWarmup`; drop the unused Cosine class."""
    return src.split("class CosineAnnealingLRWithWarmup")[0].rstrip() + "\n"


def main() -> None:
    utils_src = extract_utils()
    linear_src = strip_repo_imports(read("Models/autoregressive_diffusion/linear.py"))
    linear_src = re.sub(r"^from einops import [^\n]*\n", "", linear_src, flags=re.M)
    armd_src = strip_repo_imports(read("Models/autoregressive_diffusion/armd.py"))
    sch_src = patch_lr_sched(read("engine/lr_sch.py"))
    tr_src = patch_trainer(read("engine/solver.py"))

    nb = new_notebook(
        metadata={
            "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
            },
        }
    )
    nb.cells = []

    nb.cells.append(
        md_cell(
            r"""
# ARMD 从零到复现：自包含教程（论文 ↔ 代码 严格对齐）

本 Notebook 在**单文件、零仓库依赖**的前提下，把
**Auto-Regressive Moving Diffusion Models for Time Series Forecasting**
（Gao et al., *AAAI-25*，arXiv:[2412.09328](https://arxiv.org/abs/2412.09328)）
的全部技术内容（公式 (1)–(10)、Algorithm 1/2、网络细节、实验设定）
与本仓库 [`Models/autoregressive_diffusion/`](../Models/autoregressive_diffusion)、
[`engine/`](../engine)、[`Utils/Data_utils/`](../Utils/Data_utils) 中的实现一一对照。

> **运行环境**：本仓库已通过 `uv sync` 在 `.venv/` 中安装 `torch>=2.6+cu124`、`einops`、`ema-pytorch`、`scikit-learn` 等。如果你已用同一虚拟环境的 Jupyter 内核打开此文件，下文 *依赖检测* 单元会原样跳过安装。CUDA-torch 可用时自动 `cuda:0`。

> **数据**：使用本仓库已存在的真实股票数据 [`Data/datasets/stock_data.csv`](../Data/datasets/stock_data.csv)（Diffusion-TS `dataset.zip` 中的 6 列日频股票面板），与论文 Table 1 *Stock* 列同源。

## 阅读顺序

| 章节 | 内容（与论文对应） |
|------|--------------------|
| 1    | 任务定义与传统扩散 TSF 的不一致（Introduction，Fig. 1） |
| 2    | DDPM 记号速览：$\beta_t,\bar\alpha_t$（Preliminary） |
| 3    | ARMD 的扩散链：未来→历史的滑动前向（Forward Diffusion，Eq. 1–3） |
| 4    | 反向 devolution：Linear 距离网络（Reverse Denoising，Eq. 4–6） |
| 5    | 训练目标 $\mathcal L_\theta$（Eq. 7） |
| 6    | 采样 / 预测：DDIM 化简（Eq. 8–10） |
| 7    | Algorithm 1（训练）& Algorithm 2（采样）逐行对照代码 |
| 8    | 依赖检测与导入 |
| 9    | 数据预处理：滑窗 + StandardScaler + 80/20 切分（与 `CustomDataset` 一致） |
| 10   | `model_utils` 节选（`extract` / `default` 等） |
| 11   | `Linear`（`Models/autoregressive_diffusion/linear.py` **完整原文**） |
| 12   | `ARMD`（`Models/autoregressive_diffusion/armd.py` **完整原文**） |
| 13   | `ReduceLROnPlateauWithWarmup`（`engine/lr_sch.py`） |
| 14   | `Trainer`（`engine/solver.py`，仅替换 `instantiate_from_config`） |
| 15   | 配置（与 `Config/stock_paper.yaml` 一致） + 训练 |
| 16   | 推理 + 平均 10 次 MSE / MAE（与 `main.py` 一致），并打印论文参考值 |
| 17   | 预测可视化 |
| 18   | 公式 ↔ 代码索引小结 |
"""
        )
    )

    nb.cells.append(
        md_cell(
            r"""
## 1 任务与传统扩散 TSF 的不一致（Introduction，Fig. 1）

**多元时间序列预测（TSF）**：给定历史
$\mathbf X^{T}_{-T+1:0}\in\mathbb R^{T\times F}$，预测
$\hat{\mathbf X}^{0}_{1:T}\in\mathbb R^{T\times F}$
（论文设历史长 = 预测长 = $T$，本仓库 `seq_length=96`，`window=2T=192`）。

经典扩散式 TSF（Fig. 1a）把序列 $\to$ 高斯噪声做前向，再以历史为条件做反向去噪；
但**时间序列是连续演化**，与「噪声 ↔ 干净图像」的二元划分错位，
中间状态成了纯随机量，**无法被利用**。

ARMD（Fig. 1b）受 ARMA 启发：

$$
x_t=\sum_{i=1}^{p}\phi_i x_{t-i}+\sum_{j=1}^{q}\theta_j\,\varepsilon_{t-j}+\varepsilon_t,
$$

把整段「未来 $\to$ 历史」的演化视作扩散链：**未来段为初态**、**历史段为终态**、
中间态由**滑动**得到。反向用线性 devolution 网络一步步回到对未来的估计。
采样从已知历史出发，**目标即预测**——无需条件化，不丢弃任何中间信息。
"""
        )
    )

    nb.cells.append(
        md_cell(
            r"""
## 2 DDPM 标量记号（Preliminary）

设 $\beta_t\in(0,1)$，$\alpha_t=1-\beta_t$，$\bar\alpha_t=\prod_{i=1}^{t}\alpha_i$。
本实现采用 DDPM 的 **cosine schedule**：

$$
\bar\alpha_t \;=\; \frac{f(t)}{f(0)},\qquad
f(t)=\cos^{2}\!\Bigl(\frac{t/T+s}{1+s}\cdot\frac{\pi}{2}\Bigr),\qquad s=0.008.
$$

代码：`cosine_beta_schedule`（同时出现在 `linear.py` 与 `armd.py`，下文第 11、12 节）。
`ARMD.__init__` 通过 `register_buffer` 预存

$$
\sqrt{\bar\alpha_t},\;\sqrt{1-\bar\alpha_t},\;\sqrt{1/\bar\alpha_t},\;\sqrt{1/\bar\alpha_t-1},\;
\text{posterior\_variance},\;\text{loss\_weight}=\frac{\sqrt{\alpha_t}\sqrt{1-\bar\alpha_t}}{100\,\beta_t}
$$

供训练 / 采样直接索引（与 [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch) 一致）。
"""
        )
    )

    nb.cells.append(
        md_cell(
            r"""
## 3 ARMD 前向扩散（演化）：Eq. (1)–(3)

记初态 $\mathbf X^{0}_{1:T}$（未来），终态 $\mathbf X^{T}_{-T+1:0}$（历史），中间态
$\mathbf X^{t}_{1-t:T-t}$。论文的核心：用**滑动**取代加噪。

$$
\boxed{\;\mathbf X^{t}_{1-t:T-t}\;=\;\mathrm{Slide}\bigl(\mathbf X^{t-1}_{2-t:T-t+1},\,1\bigr)\;}\tag{1}
$$

把 t 步沿 ARMD-DDPM 类比写成

$$
\boxed{\;\mathbf X^{t}_{1-t:T-t}\;=\;\mathrm{Slide}(\mathbf X^{0}_{1:T},\,t)\;=\;\sqrt{\bar\alpha_t}\,\mathbf X^{0}_{1:T}\;+\;\sqrt{1-\bar\alpha_t}\;\mathbf z^{t}\;}\tag{2}
$$

其中 $\mathbf z^{t}$ 是把「未来初态」演化到「中间态」所需的**演化趋势**（功能上对应原 DDPM 的噪声）。
因为每个时间步是确定的，$\mathbf z^{t}$ 可解析地反解：

$$
\boxed{\;\mathbf z^{t}\;=\;\Bigl(\sqrt{1/\bar\alpha_t}\,\mathbf X^{t}_{1-t:T-t}\;-\;\mathbf X^{0}_{1:T}\Bigr)\Big/\sqrt{1/\bar\alpha_t-1}\;}\tag{3}
$$

**滑动的代码实现** (`ARMD.q_sample`，下文第 12 节):

```python
def q_sample(self, x_start, t, noise=None):
    index = int(t[0]) + 1                # i = t + 1
    x_middle = x_start[:, pred_len-index : -index, :]
    return x_middle
```

`x_start` 是长度 $2T$ 的拼接窗（前 $T$ 历史、后 $T$ 未来；本仓库
`window=192=2·seq_length`）。当 $t=0$ 时切片 `[95:191]` 取**未来**；
$t=T-1=95$ 时切片 `[0:96]` 取**历史**。中间值在 $[1-t, T-t]$ 处恰为
$\mathbf X^{t}_{1-t:T-t}$，**纯滑动、无随机量** —— 与 Eq.(1) 一致。

> 注：$\sqrt{\bar\alpha_t}$ 和 $\sqrt{1-\bar\alpha_t}$ **并不**作用在数据切片上；它们只用于
> 第 5 节 $\mathbf z^{t}$ 的代数构造（loss 中由 `sqrt_alphas_cumprod[t]` 与
> `sqrt_one_minus_alphas_cumprod[t]` 乘上 `target` / `model_out` 实现）。
"""
        )
    )

    nb.cells.append(
        md_cell(
            r"""
## 4 反向 devolution：Linear 距离网络（Eq. 4–6）

devolution 网络 $R(\cdot)$ 用线性模块对中间态做距离估计，再用 $W(t)$ 自适应混合：

$$
\boxed{\;\mathbf D \;=\; \mathrm{Linear}\bigl(\mathbf X^{t}_{1-t:T-t}\bigr)\;}\tag{4}
$$

$$
\boxed{\;\hat{\mathbf X}^{0}(\mathbf X^{t},t,\theta)\;=\;\frac{W(t)\,\mathbf X^{t}_{1-t:T-t}\;+\;\bigl(1-bW(t)\bigr)\,\mathbf D}{\bigl(1+cW(t)\bigr)^{d}}\;}\tag{5}
$$

其中 $W(t)$ 是初值为 $\bar\alpha_t$ 的可学习标量；论文的代码取 $b=2,c=-1,d=1/2$，故

$$
\hat{\mathbf X}^{0}\;=\;\frac{W(t)\,\mathbf X^{t}+\bigl(1-2W(t)\bigr)\,\mathbf D}{\sqrt{1-W(t)}}.
$$

预测演化趋势：

$$
\boxed{\;\hat{\mathbf z}(t,\theta)\;=\;\Bigl(\sqrt{1/\bar\alpha_t}\,\mathbf X^{t}_{1-t:T-t}\;-\;\hat{\mathbf X}^{0}(\mathbf X^{t},t,\theta)\Bigr)\Big/\sqrt{1/\bar\alpha_t-1}\;}\tag{6}
$$

**代码** (`Linear.forward`，下文第 11 节)：

```python
input_ += self.w_dev[t[0]] * noise            # 训练期可加微噪扰动；评估期 noise=0
x_tmp = self.linear(input_.permute(0, 2, 1)).permute(0, 2, 1)   # D = Linear(X^t)
alpha = self.w[t[0]]                          # W(t)，初始化 = ᾱ_t
output = (alpha*input_ + (1-2*alpha)*x_tmp) / (1 - alpha)**0.5  # Eq.(5) 取 b=2,c=-1,d=1/2
```

`(1-2W(t))*D` 即 Eq.(5) 中的 $(1-bW(t))\,D$，
分母 $\sqrt{1-W(t)}$ 即 $(1+cW(t))^d$ 在 $b=2,c=-1,d=1/2$ 下的展开。
预测 $\hat{\mathbf z}$ 在 `ARMD.model_predictions` 通过 `predict_noise_from_start` 得到。
"""
        )
    )

    nb.cells.append(
        md_cell(
            r"""
## 5 训练目标（Eq. 7）

$$
\boxed{\;\mathcal L_\theta\;=\;\mathbb E_{t}\bigl[\,\bigl|\mathbf z^{t}-\hat{\mathbf z}(t,\theta)\bigr|\,\bigr]\;}\tag{7}
$$

为了避免显式构造 $\sqrt{1/\bar\alpha_t}$ 等等比例因子，
代码用代数等价形式（见 `ARMD._train_loss`，下文第 12 节）：

$$
\mathbf z^{t}\propto\mathbf X^{t}-\sqrt{\bar\alpha_t}\,\mathbf X^{0},\qquad
\hat{\mathbf z}(t,\theta)\propto\mathbf X^{t}-\sqrt{\bar\alpha_t}\,\hat{\mathbf X}^{0}(\mathbf X^{t},t,\theta).
$$

```python
target = x_start[:, pred_len:, :]                              # X^0_{1:T}
x      = self.q_sample(x_start, t)                             # X^t_{1-t:T-t}
model_out = self.output(x, t, training=True)                   # \hat X^0
alpha       = self.sqrt_alphas_cumprod[t[0]]                   # √ᾱ_t
minus_alpha = self.sqrt_one_minus_alphas_cumprod[t[0]]         # √(1-ᾱ_t)
target_noise = (x - target    * alpha) / minus_alpha           # ∝ z^t      (Eq. 3)
pred_noise   = (x - model_out * alpha) / minus_alpha           # ∝ ẑ(t,θ)   (Eq. 6)
loss = loss_fn(pred_noise, target_noise) * loss_weight[t]      # Eq. 7
```

`loss_type='l1'` 对应 Eq.(7) 的绝对值；`loss_type='l2'`（`Config/stock.yaml` 默认）
是常用替代。论文 Table 1 的 Stock 数据使用 `l1` + 2 步采样，与
`Config/stock_paper.yaml` 一致，下文 *第 15 节* 也按此设定。

`loss_weight = √α_t·√(1-ᾱ_t)/β_t / 100` 是工程上对早期 t 的弱化，
与 [denoising-diffusion-pytorch] 类似（论文未单独写出，属实现细节）。
"""
        )
    )

    nb.cells.append(
        md_cell(
            r"""
## 6 采样 / 预测：DDIM 化简（Eq. 8–10）

按 DDIM 式（Eq. 8）：

$$
\mathbf X^{t-1}_{2-t:T-t+1}=\sqrt{\bar\alpha_{t-1}}\Bigl(\frac{\mathbf X^{t}_{1-t:T-t}-\sqrt{1-\bar\alpha_t}\,\hat{\mathbf z}(t,\theta)}{\sqrt{\bar\alpha_t}}\Bigr)
+\sqrt{1-\bar\alpha_{t-1}-\sigma_t^{2}}\;\hat{\mathbf z}(t,\theta)+\sigma_t\boldsymbol\varepsilon_t.
$$

ARMD 是确定性演化 $\Rightarrow\sigma_t=0$，且括号内即 $\hat{\mathbf X}^{0}$。
合并后得到（Eq. 9）：

$$
\boxed{\;\mathbf X^{t-1}_{2-t:T-t+1}=\sqrt{\bar\alpha_{t-1}}\,\hat{\mathbf X}^{0}+\sqrt{1-\bar\alpha_{t-1}}\,\hat{\mathbf z}(t,\theta)\;}\tag{9}
$$

可跳 $k$ 步（Eq. 10）：

$$
\boxed{\;\mathbf X^{t-k}_{1-t+k:T-t+k}=\sqrt{\bar\alpha_{t-k}}\,\hat{\mathbf X}^{0}+\sqrt{1-\bar\alpha_{t-k}}\,\hat{\mathbf z}(t,\theta)\;}\tag{10}
$$

**代码** (`ARMD.fast_sample`，下文第 12 节，`sigma=0`、`noise=0`)：

```python
img = x[:, :pred_len, :]   # 历史段作为起点 X^T_{-T+1:0}
for time, time_next in zip(times[:-1], times[1:]):
    pred_noise, x_start, *_ = self.model_predictions(img, time_cond)
    if time_next < 0:
        img = x_start                                              # 最后一步退出
        continue
    alpha       = self.alphas_cumprod[time]
    alpha_next  = self.alphas_cumprod[time_next]
    sigma = 0; noise = 0
    c = (1 - alpha_next) ** 0.5
    img = x_start * alpha_next.sqrt() + c * pred_noise             # Eq. (10)
```

跳步数 = `sampling_timesteps`，论文 Stock 在 `{1..12}` 中按验证集 MAE 选优。
"""
        )
    )

    nb.cells.append(
        md_cell(
            r"""
## 7 Algorithm 1 / 2（论文伪代码）逐行映射

### Algorithm 1 训练

```
Require: 最大扩散步 T；预存系数 ᾱ_{0:T}.
1: repeat
2:   从训练集采样 X^0_{1:T}                              # data = next(self.dl)
3:   t ~ Uniform({0,1,...,T-1})                         # ARMD.forward
4:   生成 X^t_{1-t:T-t}（Eq. 2），并据 Eq. 3 算 z^t      # ARMD.q_sample + 代数 z^t
5:   用 R(·) 得 \hat X^0（Eq. 5）；据 Eq. 6 得 \hat z   # ARMD.output -> Linear.forward
6:   计算 L_θ（Eq. 7）                                   # ARMD._train_loss
7:   按 ∇_θ L 做一次梯度下降                              # Adam + EMA + clip_grad_norm_
8: until 收敛
```

### Algorithm 2 采样 / 预测

```
Require: 历史 X^T_{-T+1:0}；R(·)；步长 Δt；ᾱ_{0:T}.
1: for t = T-1, T-1-Δt, ..., 0:                          # ARMD.fast_sample
2:   据 X^t_{1-t:T-t} 与 t 跑 R(·) 得 \hat X^0；按 Eq. 6 得 \hat z
3:   按 Eq. 10 更新 X^{t-Δt}_{...}                        # img = √ᾱ_next·\hat X^0 + √(1-ᾱ_next)·\hat z
4: end for
5: 输出 X^0_{1:T} 的预测
```

下面开始安装依赖并把以上算法的项目源码完整内嵌进来。
"""
        )
    )

    nb.cells.append(md_cell("## 8 依赖检测（已安装时直接跳过）"))

    nb.cells.append(
        py_cell(
            '''import importlib
import shutil
import subprocess
import sys
from pathlib import Path

# 与 pyproject.toml 第三方依赖对齐；torch 的 CUDA 版本由 .venv 决定（pyproject 指向 cu124 wheel）
PIP_PKGS = [
    "torch",
    "einops",
    "ema-pytorch",
    "pandas",
    "numpy",
    "scikit-learn",
    "tqdm",
    "matplotlib",
]
MOD_NAMES = ("torch", "einops", "numpy", "pandas", "sklearn", "tqdm", "ema_pytorch", "matplotlib")


def _missing_modules():
    miss = []
    for mod in MOD_NAMES:
        try:
            importlib.import_module(mod)
        except ImportError:
            miss.append(mod)
    return miss


def ensure_deps():
    miss = _missing_modules()
    if not miss:
        print("deps OK (already installed):", ", ".join(MOD_NAMES))
        return
    print("missing:", miss)
    if shutil.which("uv"):
        subprocess.check_call(["uv", "pip", "install", "-q", *PIP_PKGS], cwd=str(Path.cwd()))
        print("deps OK (uv pip).")
        return
    try:
        import pip  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "ensurepip", "--upgrade"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *PIP_PKGS])
    print("deps OK (ensurepip + pip).")


ensure_deps()

import torch  # noqa: E402

print("python:", sys.version.split()[0])
print("torch :", torch.__version__, "  CUDA build:", torch.version.cuda)
print("cuda.is_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU   :", torch.cuda.get_device_name(0))
'''
        )
    )

    nb.cells.append(
        md_cell(
            r"""
## 9 数据预处理（与 `Utils/Data_utils/real_datasets.py::CustomDataset` 一致）

- 路径：`Data/datasets/stock_data.csv`（Diffusion-TS 的 6 列日频股票面板，无日期列）。
- 配置 `name=stock`：原文件无 date 列，**不**像 `etth` 那样删第一列；删了会丢 `Open`（README 已强调）。
- `StandardScaler.fit` 使用全部行（与官方一致），再对所有行 transform。
- 滑动窗口 `window = 2 * seq_length = 192`：每个样本同时含 96 步历史 + 96 步未来。
- 80/20 时间顺序切分（`CustomDataset.divide`）：训练取前 $\lceil 0.8N\rceil$ 个窗，测试取剩余。

> 在 fallback 情形（无真实 CSV 时）会写入 900 行随机游走以保证演示可跑；指标仅供管线验证，不可与论文比对。
"""
        )
    )

    nb.cells.append(
        py_cell(
            '''import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


def repo_root() -> Path:
    """Walk parents until we find a directory that contains Data/datasets/."""
    here = Path.cwd().resolve()
    for cand in [here, *here.parents]:
        if (cand / "Data" / "datasets").is_dir():
            return cand
    return here


REPO_ROOT = repo_root()
DATA_PATH = REPO_ROOT / "Data" / "datasets" / "stock_data.csv"


def ensure_csv():
    """If the real Diffusion-TS stock_data.csv is missing, write a synthetic one to keep the notebook runnable."""
    if DATA_PATH.exists():
        return False
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    z = np.cumsum(rng.standard_normal((900, 6)), axis=0)
    pd.DataFrame(z, columns=["Open", "High", "Low", "Close", "Adj_Close", "Volume"]).to_csv(DATA_PATH, index=False)
    return True


def divide_windows(windows: np.ndarray, ratio: float):
    """Mirrors CustomDataset.divide: regular = first ceil(ratio*N), irregular = the rest (chronological)."""
    size = windows.shape[0]
    cut = int(np.ceil(size * ratio))
    return windows[:cut], windows[cut:]


class StockWindowDataset(Dataset):
    """Notebook-local replica of Utils/Data_utils/real_datasets.py::CustomDataset for stock_data.csv.

    Behavioural parity with CustomDataset (period in {train, test}):
      * StandardScaler.fit on all rows, then transform.
      * Build all sliding windows of length `window` (= 2*seq_length).
      * Train split is divide(windows, 0.8) regular slice; test split is divide(windows, 0.8) irregular slice.
        (CustomDataset uses proportion=0.8 for train and proportion=0.2 for test, both calling divide and
        keeping the *regular* / *irregular* slice respectively; both selections therefore agree on the same cut.)
      * Test items expose a boolean mask with last `predict_length` steps set to 0 — same as CustomDataset.
    """

    def __init__(
        self,
        csv_path: Path,
        *,
        name: str = "stock",
        window: int = 192,
        proportion: float = 0.8,
        period: str = "train",
        predict_length: int | None = None,
    ):
        assert period in ("train", "test")
        df = pd.read_csv(csv_path, header=0)
        if name == "etth":
            df = df.drop(df.columns[0], axis=1)
        raw = df.values.astype(np.float64)
        self.var_num = raw.shape[1]
        scaler = StandardScaler().fit(raw)
        data = scaler.transform(raw)
        n = data.shape[0]
        n_win = max(n - window + 1, 0)
        win = np.stack([data[i : i + window] for i in range(n_win)])
        regular, irregular = divide_windows(win, proportion)
        self.samples = regular if period == "train" else irregular
        if period == "test" and predict_length is not None:
            m = np.ones(self.samples.shape, dtype=bool)
            m[:, -predict_length:, :] = False
            self.mask = m
        else:
            self.mask = None
        self.scaler = scaler

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.samples[idx]).float()
        if self.mask is None:
            return x
        return x, torch.from_numpy(self.mask[idx]).bool()


SEQ_LEN = 96
WINDOW = 192
TRAIN_PROPORTION = 0.8

is_synth = ensure_csv()
print(f"data at {DATA_PATH}  (synthetic fallback: {is_synth})")
df_head = pd.read_csv(DATA_PATH, nrows=3)
print("CSV head:\\n", df_head)

train_ds = StockWindowDataset(DATA_PATH, window=WINDOW, proportion=TRAIN_PROPORTION, period="train")
test_ds = StockWindowDataset(
    DATA_PATH, window=WINDOW, proportion=TRAIN_PROPORTION, period="test", predict_length=SEQ_LEN
)
N_FEAT = train_ds.var_num

train_loader = DataLoader(
    train_ds,
    batch_size=128,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
    drop_last=True,
)
test_loader = DataLoader(
    test_ds,
    batch_size=256,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
    drop_last=False,
)

print("feature_size:", N_FEAT, "  #train windows:", len(train_ds), "  #test windows:", len(test_ds))
'''
        )
    )

    nb.cells.append(md_cell("## 10 `model_utils` 节选 — `exists` / `default` / `identity` / `extract`"))

    nb.cells.append(py_cell(utils_src))

    nb.cells.append(
        md_cell(
            r"""
## 11 `Linear`（`Models/autoregressive_diffusion/linear.py` 完整原文）

下方代码与仓库源码 **完全一致**（仅去掉了对本仓库 `model_utils` 的相对 import — 它的依赖 `extract` 等已在第 10 节内嵌）。
对应论文 Eq.(4)(5)：`self.linear` 输出距离 $\mathbf D$，再用 `self.w[t]`（$W(t)$，初值 $\bar\alpha_t$，
`w_grad=True` 时随训练更新）按 Eq.(5) 混合得到 $\hat{\mathbf X}^0$；
训练态加 `self.w_dev[t]*noise` 微扰提高鲁棒性，论文 *第 3 节倒数第 2 段* 提到。
"""
        )
    )

    nb.cells.append(
        py_cell(
            "import math\n"
            "import numpy as np\n"
            "import torch\n"
            "import torch.nn.functional as F\n"
            "from torch import nn\n\n"
            + linear_src.lstrip()
        )
    )

    nb.cells.append(
        md_cell(
            r"""
## 12 `ARMD`（`Models/autoregressive_diffusion/armd.py` 完整原文）

下方代码与仓库源码 **完全一致**（仅去掉对 `model_utils` / `linear` 的相对 import；它们已在第 10、11 节内嵌）。
关键方法逐一对应：

| 论文公式 / 步骤 | 方法 |
|-----------------|------|
| Eq.(1)(2) `Slide` 中间态 | `q_sample` —— `x_start[:, pred_len-i : -i, :]` |
| Eq.(3) $z^t$ 与 Eq.(7) Loss | `_train_loss` —— `target_noise = (x - target*α)/√(1-ᾱ)` |
| Eq.(4)(5) $\hat X^0$ | `output` → `Linear.forward`（第 11 节）|
| Eq.(6) $\hat z$ | `predict_noise_from_start` 中由 $\hat X^0$ 反解 |
| Algorithm 1 行 3 (uniform t) | `forward()` —— `t = randint(0, num_timesteps, (1,)).repeat(b)` |
| Algorithm 2 / Eq.(10) 跳步采样 | `fast_sample` —— `sigma=0; noise=0` |
"""
        )
    )

    nb.cells.append(
        py_cell(
            "import math\n"
            "import torch\n"
            "import torch.nn.functional as F\n"
            "from torch import nn\n"
            "from einops import reduce\n"
            "from tqdm.auto import tqdm\n"
            "from functools import partial\n\n"
            + armd_src.lstrip()
        )
    )

    nb.cells.append(
        md_cell(
            r"""
## 13 `ReduceLROnPlateauWithWarmup`（`engine/lr_sch.py`）

教程仅保留实际被 `Config/stock*.yaml` 使用的调度器；
`engine/lr_sch.py` 的另一个 `CosineAnnealingLRWithWarmup` 类省略以减少认知负担。
"""
        )
    )

    nb.cells.append(
        py_cell(
            "import math\nfrom torch import inf\nfrom torch.optim.optimizer import Optimizer\n\n"
            + sch_src.lstrip()
        )
    )

    nb.cells.append(
        md_cell(
            r"""
## 14 `Trainer`（`engine/solver.py`，仅替换 `instantiate_from_config`）

唯一的改动：
- 把 `instantiate_from_config(cfg['solver']['scheduler'])` 替成直接构造
  `ReduceLROnPlateauWithWarmup(**params)`（笔记本里没有 YAML 类路径解析）。
- 移除 `from Utils.io_utils import get_model_parameters_info, instantiate_from_config`
  与 `sys.path.append(...)`，因为不依赖仓库结构。

Adam（`betas=[0.9, 0.96]`）、EMA、`clip_grad_norm_(1.0)`、`gradient_accumulate_every` 等
**全部保留**，与 `Config/stock*.yaml` 与 `main.py` 的训练循环一致。
"""
        )
    )

    nb.cells.append(
        py_cell(
            "import os\n"
            "import time\n"
            "import numpy as np\n"
            "import torch\n"
            "from pathlib import Path\n"
            "from tqdm.auto import tqdm\n"
            "from ema_pytorch import EMA\n"
            "from torch.optim import Adam\n"
            "from torch.nn.utils import clip_grad_norm_\n\n"
            + tr_src.lstrip()
        )
    )

    nb.cells.append(
        md_cell(
            r"""
## 15 训练（与 `Config/stock_paper.yaml` 完全等价）

`Config/stock_paper.yaml` 是论文附录设定下的 Stock 配置：

```yaml
model:
  seq_length: 96, feature_size: 6, timesteps: 96, sampling_timesteps: 2, loss_type: l1
solver:
  base_lr: 1e-3, max_epochs: 2000, gradient_accumulate_every: 2, ema: {decay: 0.995, update_interval: 10}
  scheduler.params: {factor: 0.5, patience: 4000, min_lr: 1e-5, warmup_lr: 8e-4, warmup: 500, ...}
dataloader:
  batch_size: 128
```

下方 cell 把这些超参原样灌进我们内嵌的 `Trainer` / `ARMD`。
完整 2000 步在 GPU 上约 30s（线性 + 96 时间步），可直接复现 Table 1。

> **教程模式 vs 论文模式**：默认 `MAX_EPOCHS=2000` 对齐论文。如果只想快速 smoke-test，把它改小（如 200）即可——下文指标也会相应变差。
"""
        )
    )

    nb.cells.append(
        py_cell(
            '''import random
import numpy as np
import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", DEVICE)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(2023)

# Mirrors Config/stock_paper.yaml
MAX_EPOCHS = 2000           # paper supplemental: 2000 optimizer steps for Stock
SAMPLING_TIMESTEPS = 2      # paper supplemental: chosen on validation in {1..12}; the repo's default is 2
LOSS_TYPE = "l1"            # paper Eq.(7) is the L1 norm

config = {
    "solver": {
        "max_epochs": MAX_EPOCHS,
        "gradient_accumulate_every": 2,
        "save_cycle": 10**9,           # don't write checkpoints during the tutorial
        "results_folder": str(REPO_ROOT / "Checkpoints_standalone_nb"),
        "base_lr": 1e-3,
        "ema": {"decay": 0.995, "update_interval": 10},
        "scheduler": {
            "params": {
                "mode": "min",
                "factor": 0.5,
                "patience": 4000,
                "min_lr": 1e-5,
                "threshold": 0.1,
                "threshold_mode": "rel",
                "warmup_lr": 8e-4,
                "warmup": 500,
                "verbose": False,
            }
        },
    }
}

model = ARMD(
    seq_length=SEQ_LEN,
    feature_size=N_FEAT,
    timesteps=96,
    sampling_timesteps=SAMPLING_TIMESTEPS,
    loss_type=LOSS_TYPE,
    beta_schedule="cosine",
    w_grad=True,
).to(DEVICE)
model.fast_sampling = True   # main.py also forces this


class Args:
    name = "armd_standalone_nb"
    save_dir = str(REPO_ROOT / "forecasting_exp_standalone")


args = Args()
os.makedirs(args.save_dir, exist_ok=True)


def cycle_loader(dl):
    while True:
        for x in dl:
            yield x


trainer = Trainer(
    config=config,
    args=args,
    model=model,
    dataloader={"dataloader": cycle_loader(train_loader)},
    logger=None,
)


# Same training body as Trainer.train(), with a manual loss-history buffer
# so we can plot the curve after training. Behaviour matches the original.
loss_history: list[float] = []
pbar = tqdm(range(MAX_EPOCHS), desc="train", smoothing=0.05)
for step in range(MAX_EPOCHS):
    total_loss = 0.0
    for _ in range(trainer.gradient_accumulate_every):
        data = next(trainer.dl).to(trainer.device)
        loss = trainer.model(data, target=data)
        loss = loss / trainer.gradient_accumulate_every
        loss.backward()
        total_loss += loss.item()
    clip_grad_norm_(trainer.model.parameters(), 1.0)
    trainer.opt.step()
    trainer.sch.step(total_loss)
    trainer.opt.zero_grad()
    trainer.step += 1
    trainer.ema.update()
    loss_history.append(total_loss)
    if step % 50 == 0:
        pbar.set_description(f"train loss: {total_loss:.6f}")
    pbar.update(1)
pbar.close()
print(f"training complete after {len(loss_history)} steps; final loss: {loss_history[-1]:.6f}")
'''
        )
    )

    nb.cells.append(
        md_cell(
            r"""
### 训练 loss 曲线

仅供观察收敛形态：第 ~500 步 warmup 期 lr 从 0 线性加到 8e-4；之后由 `ReduceLROnPlateau` 控制。
"""
        )
    )

    nb.cells.append(
        py_cell(
            '''import matplotlib.pyplot as plt

plt.figure(figsize=(7, 3))
plt.plot(loss_history, lw=0.6)
plt.xlabel("optimizer step")
plt.ylabel("train loss (L1 on z^t)")
plt.title(f"ARMD training loss — total {len(loss_history)} steps")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
'''
        )
    )

    nb.cells.append(
        md_cell(
            r"""
## 16 推理 + MSE / MAE（与 `main.py` 完全一致）

`main.py` 的协议：

```python
for run in range(10):
    set_seed(2023 + run)
    sample, real_ = trainer.sample_forecast(test_loader, shape=[seq_len, feat_num])
    mse_runs.append(mean_squared_error(sample.reshape(-1), real_.reshape(-1)))
    mae_runs.append(mean_absolute_error(sample.reshape(-1), real_.reshape(-1)))
mse, mae = mean(mse_runs), mean(mae_runs)
```

> ARMD 的 `fast_sample` 在 `sigma=0, noise=0` 下其实是确定性的，10 次种子各自独立的预测应近似一致；
> 仍按 `main.py` 协议跑 10 次以保持论文可比性。

论文 Table 1 *Stock* 列：**MSE = 0.235，MAE = 0.269**（z-score 归一化空间）。
"""
        )
    )

    nb.cells.append(
        py_cell(
            '''import random
from sklearn.metrics import mean_absolute_error, mean_squared_error

shape = [SEQ_LEN, N_FEAT]
mse_runs, mae_runs = [], []
samples_last, reals_last = None, None

for run in range(10):
    set_seed(2023 + run)
    samples, reals = trainer.sample_forecast(test_loader, shape=shape)
    mse_runs.append(mean_squared_error(samples.reshape(-1), reals.reshape(-1)))
    mae_runs.append(mean_absolute_error(samples.reshape(-1), reals.reshape(-1)))
    samples_last, reals_last = samples, reals

mse = float(np.mean(mse_runs))
mae = float(np.mean(mae_runs))
print(f"ARMD on Stock — averaged over 10 sampling runs ({SAMPLING_TIMESTEPS}-step DDIM, deterministic):")
print(f"  MSE = {mse:.4f}    MAE = {mae:.4f}")
print(f"  per-run MSE: {[round(m, 4) for m in mse_runs]}")
print(f"  per-run MAE: {[round(m, 4) for m in mae_runs]}")
print()
print("Paper reference (Table 1, ARMD on Stock, z-score-normalized):  MSE = 0.235    MAE = 0.269")
print("Source: https://arxiv.org/abs/2412.09328")
'''
        )
    )

    nb.cells.append(
        md_cell(
            r"""
## 17 预测可视化

随机抽几个测试窗口，画历史 + 真实未来 + ARMD 预测的对比图（z-score 空间）。
"""
        )
    )

    nb.cells.append(
        py_cell(
            '''import matplotlib.pyplot as plt

# samples_last shape: (n_test_win, 96, F);  reals_last shape: (n_test_win, 96, F)
n_show = 4
feat_to_show = min(2, N_FEAT)        # plot first 2 channels
rng_idx = np.random.default_rng(0).choice(samples_last.shape[0], size=n_show, replace=False)

fig, axes = plt.subplots(n_show, feat_to_show, figsize=(5 * feat_to_show, 2.5 * n_show), sharex=True)
if n_show == 1:
    axes = np.array([axes])
if feat_to_show == 1:
    axes = axes[:, None]

# Recover the corresponding history halves from the test dataset (test_ds.samples[:, :96, :]).
hist_segs = test_ds.samples[:, :SEQ_LEN, :]   # full history half before the predict window

for r, idx in enumerate(rng_idx):
    for c in range(feat_to_show):
        ax = axes[r, c]
        x_hist = np.arange(SEQ_LEN)
        x_pred = np.arange(SEQ_LEN, 2 * SEQ_LEN)
        ax.plot(x_hist, hist_segs[idx, :, c], color="#444", label="history" if (r == 0 and c == 0) else None)
        ax.plot(x_pred, reals_last[idx, :, c], color="#1f77b4", label="future (truth)" if (r == 0 and c == 0) else None)
        ax.plot(x_pred, samples_last[idx, :, c], color="#d62728", lw=1.4, label="ARMD prediction" if (r == 0 and c == 0) else None)
        ax.axvline(SEQ_LEN - 0.5, color="grey", ls="--", lw=0.7)
        ax.grid(alpha=0.3)
        ax.set_title(f"window #{int(idx)}  ·  feature {c}")

axes[0, 0].legend(loc="upper left", fontsize=8)
fig.suptitle("ARMD forecasts (z-score normalized) — Stock test windows", y=1.02)
fig.tight_layout()
plt.show()
'''
        )
    )

    nb.cells.append(
        md_cell(
            r"""
## 18 公式 ↔ 代码索引（速查）

| 论文公式 / 章节                                            | 项目代码位置                                                                    |
|------------------------------------------------------------|---------------------------------------------------------------------------------|
| $\beta_t,\bar\alpha_t$ cosine schedule（Preliminary）       | `cosine_beta_schedule`（`linear.py` 与 `armd.py` 各一份）                       |
| Eq.(1) `Slide(X^{t-1},1)`                                   | `ARMD.q_sample` —— `x_start[:, pred_len-i : -i, :]`                              |
| Eq.(2) $X^t=\sqrt{\bar\alpha_t}X^0+\sqrt{1-\bar\alpha_t}z^t$ | 解析关系；不显式构造，loss 用代数等价（见下行）                                  |
| Eq.(3) 解出 $z^t$                                            | `_train_loss` —— `(x - target*α)/√(1-ᾱ)`                                         |
| Eq.(4) $D=\mathrm{Linear}(X^t)$                              | `Linear.forward` —— `self.linear(input.permute(0,2,1)).permute(0,2,1)`           |
| Eq.(5) $\hat X^0$ 自适应混合                                 | `Linear.forward` —— `(α·x + (1-2α)·D)/√(1-α)` （取 $b=2,c=-1,d=1/2$）           |
| Eq.(6) $\hat z=$ from $\hat X^0$                             | `ARMD.predict_noise_from_start` 与 `model_predictions`                           |
| Eq.(7) $\mathcal L_\theta=\mathbb E_t|z^t-\hat z|$           | `_train_loss` —— `loss_fn(pred_noise, target_noise) * loss_weight[t]`            |
| Eq.(8)–(10) DDIM 采样（$\sigma=0$）                          | `ARMD.fast_sample` —— `img = √ᾱ_next·\hat X^0 + √(1-ᾱ_next)·\hat z`              |
| Algorithm 1 行 3 同 batch 共享 t                             | `ARMD.forward` —— `t = randint(...).repeat(b)`                                   |
| Algorithm 2 起点                                             | `fast_sample` —— `img = x[:, :pred_len, :]`（历史段）                             |
| 滑窗数据 (`window=2T`)                                       | `Utils/Data_utils/real_datasets.py::CustomDataset.__getsamples` ↔ 第 9 节         |
| 训练循环 / EMA / 梯度裁剪                                    | `Trainer.train`（第 14 节，沿用 `engine/solver.py`）                              |
| Stock 超参（lr=1e-3, L1, batch=128, 2000 步, ts=2）          | `Config/stock_paper.yaml` ↔ 第 15 节 `config = {...}`                              |
| 评估协议（10 次平均）                                        | `main.py` 末尾循环 ↔ 第 16 节                                                     |

到此，**论文（公式 + 算法 + 实验）↔ 仓库（模型 + 训练 + 评估）↔ 本 Notebook** 三方完全打通。
你可以在仓库根目录直接 `uv run jupyter notebook tutorials/armd_standalone_full.ipynb` 复现 Table 1 *Stock* 列。
"""
        )
    )

    for i, c in enumerate(nb.cells):
        c.setdefault("id", f"cell{i}")

    nbformat.write(nb, OUT)
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
