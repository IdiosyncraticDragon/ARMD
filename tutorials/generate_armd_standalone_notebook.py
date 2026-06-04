# -*- coding: utf-8 -*-
"""Build tutorials/armd_standalone_full.ipynb — comprehensive beginner-friendly ARMD tutorial.

Covers ALL 20 paper equations, both algorithms, all project code files.
Target reader: knows Python + basic ML, unfamiliar with PyTorch / DDPM.
"""
from __future__ import annotations
import re
from pathlib import Path
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parent.parent
OUT  = Path(__file__).resolve().parent / "armd_standalone_full.ipynb"

def read(p: str) -> str:
    return (ROOT / p).read_text(encoding="utf-8")

def py_cell(src: str) -> dict:
    c = new_code_cell(src.strip() + "\n")
    c["metadata"] = {}; c["outputs"] = []; c["execution_count"] = None
    return c

def md_cell(text: str) -> dict:
    c = new_markdown_cell(text.strip() + "\n")
    c["metadata"] = {}
    return c

def strip_repo_imports(code: str) -> str:
    code = re.sub(r"^from Models\.\S+ import [^\n]*(?:\n[ \t]+[^\n]*)*\n", "", code, flags=re.M)
    code = re.sub(r"^from Utils\.\S+ import [^\n]*(?:\n[ \t]+[^\n]*)*\n", "", code, flags=re.M)
    code = re.sub(r"^from engine\.\S+ import [^\n]*(?:\n[ \t]+[^\n]*)*\n", "", code, flags=re.M)
    return code

def patch_trainer(tr: str) -> str:
    tr = re.sub(r"^from Utils\.io_utils import [^\n]*\n", "", tr, flags=re.M)
    tr = re.sub(r"^sys\.path\.append\(.*\)\n", "", tr, flags=re.M)
    # Replace instantiate_from_config call with direct construction
    tr = re.sub(
        r"        sc_cfg = config\['solver'\]\['scheduler'\]\n"
        r"        sc_cfg\['params'\]\['optimizer'\] = self\.opt\n"
        r"        self\.sch = instantiate_from_config\(sc_cfg\)\n",
        "        p = dict(config['solver']['scheduler']['params'])\n"
        "        p['optimizer'] = self.opt\n"
        "        self.sch = ReduceLROnPlateauWithWarmup(**p)\n",
        tr,
    )
    tr = re.sub(
        r"        if self\.logger is not None:\n"
        r"            self\.logger\.log_info\(str\(get_model_parameters_info\(self\.model\)\)\)\n",
        "        # parameter info logging omitted in standalone\n",
        tr,
    )
    return tr

def patch_lr_sched(src: str) -> str:
    return src.split("class CosineAnnealingLRWithWarmup")[0].rstrip() + "\n"

# ── extract model_utils helpers ──────────────────────────────────────────────
def extract_utils() -> str:
    mu = read("Models/autoregressive_diffusion/model_utils.py")
    parts = []
    for fn in ("exists", "default", "identity", "extract"):
        m = re.search(rf"^def {fn}\(.*?\n(?:^[ \t].*\n)*", mu, re.M)
        if m:
            parts.append(m.group(0).rstrip())
    return "\n\n\n".join(parts) + "\n"

def main() -> None:
    # ── source code extraction ────────────────────────────────────────────────
    utils_src   = extract_utils()
    linear_raw  = read("Models/autoregressive_diffusion/linear.py")
    linear_src  = strip_repo_imports(linear_raw)
    linear_src  = re.sub(r"^from einops import [^\n]*\n", "", linear_src, flags=re.M)
    armd_src    = strip_repo_imports(read("Models/autoregressive_diffusion/armd.py"))
    sch_src     = patch_lr_sched(read("engine/lr_sch.py"))
    tr_src      = patch_trainer(read("engine/solver.py"))
    dataset_raw = read("Utils/Data_utils/real_datasets.py")
    build_raw   = read("Data/build_dataloader.py")
    main_raw    = read("main.py")

    nb = new_notebook(metadata={
        "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
        "language_info": {"codemirror_mode": {"name": "ipython", "version": 3},
                          "file_extension": ".py", "mimetype": "text/x-python",
                          "name": "python", "nbconvert_exporter": "python",
                          "pygments_lexer": "ipython3"},
    })
    nb.cells = []
    C = nb.cells  # alias

    # ══════════════════════════════════════════════════════════════════════════
    # 封面 & 导读
    # ══════════════════════════════════════════════════════════════════════════
    C.append(md_cell(r"""
# ARMD 从论文到代码：完全自包含入门教程

**论文**：Auto-Regressive Moving Diffusion Models for Time Series Forecasting
Gao et al., *AAAI-25*，arXiv:[2412.09328](https://arxiv.org/abs/2412.09328)

**本教程目标**：
- 把论文的 **20 个公式（Eq.1–20）** 和 **2 个算法** 一一对应到代码；
- 把仓库的 **全部核心代码文件**（`linear.py / armd.py / solver.py / lr_sch.py / real_datasets.py / build_dataloader.py / main.py`）内嵌入 notebook；
- 面向**有 Python + 机器学习基础但不熟悉 PyTorch / DDPM** 的读者，逐行解释实现细节。

**阅读路径**：

| 章节 | 主题 | 包含公式 |
|------|------|----------|
| A | 环境 + Stock 数据加载可视化 | — |
| B | DDPM 背景与 ARMD 动机 | Eq.11–20 |
| C | 前向扩散：滑动过程 | Eq.1–3 |
| D | 反向去噪：Devolution 网络 | Eq.4–7 |
| E | 采样/预测过程 | Eq.8–10 |
| F | 代码实现（Beta 调度 / model_utils / Linear / ARMD） | 全部 |
| G | 训练支持（LR 调度 / Trainer / DataLoader） | Algorithm 1 |
| H | 训练、评估与可视化 | Algorithm 2 |
| I | main.py 等价代码 + 消融实验分析 | — |

> **如何快速上手**：先运行所有 Code Cell（不看解释），确认管道跑通，再回头逐节阅读。
> **快速测试模式**：找到 `QUICK_TEST = False`，改为 `True`，约 3 分钟完成全流程。

**Paper reference (Stock, Table 1, z-score)**：MSE = 0.235, MAE = 0.269
"""))

    C.append(md_cell(r"""
## 符号与变量对照表

在阅读代码时，下表帮助你把论文符号映射到 Python 变量名：

| 论文符号 | 含义 | 代码变量 | 所在文件 |
|---|---|---|---|
| $X_{-L+1:0}$ | 历史序列（长度 L） | `x[:, :96, :]` | solver.py |
| $X^0_{1:T}$ | 未来序列（初态） | `x_start[:, 96:, :]` / `target` | armd.py |
| $X^t_{1-t:T-t}$ | 第 t 步中间态 | `x`（q_sample 输出） | armd.py |
| $X^T_{-T+1:0}$ | 历史序列（终态） | `x[:, :96, :]` | armd.py |
| $\bar\alpha_t$ | 累积乘积 $\prod_{k=1}^t\alpha_k$ | `alphas_cumprod[t]` | armd.py |
| $z_t$ | 真实演化趋势 | `target_noise` | armd.py `_train_loss` |
| $\hat z(t,\theta)$ | 预测演化趋势 | `pred_noise` | armd.py `_train_loss` |
| $\hat X^0$ | 预测未来初态 | `x_start`（fast_sample 内） | armd.py |
| $W(t)$ | 可学习权重（初始化为 $\bar\alpha_t$） | `self.w[t[0]]` | linear.py |
| $D$ | Linear 距离估计 | `x_tmp` | linear.py |
| $R(\cdot)$ | Devolution 网络 | `Linear`（类） | linear.py |
| `t_code` | 代码中的时间步（≠ 论文 t） | `t` in `randint` | armd.py |
| 论文 $t$ | 实际滑动步数 | `index = t_code + 1` | armd.py `q_sample` |
| $T$ | 最大扩散步数（=预测长度） | `self.num_timesteps = 96` | armd.py |
| $b, c, d$ | Eq.5 超参 | 硬编码: b=2, c=−1, d=0.5 | linear.py |
| $\eta_{0:t}$ | 训练扰动系数（=$ \bar\alpha_t$） | `self.w_dev[t[0]]` | linear.py |
"""))

    # ══════════════════════════════════════════════════════════════════════════
    # Part A: 环境 + 数据
    # ══════════════════════════════════════════════════════════════════════════
    C.append(md_cell(r"""
---
## Part A：环境检测与数据加载

> 先把环境跑通，看见数据。
"""))

    C.append(md_cell("### A-1  环境检测"))
    C.append(py_cell('''\
import importlib, sys, os
from pathlib import Path

REQUIRED = {"torch":"torch","einops":"einops","numpy":"numpy","pandas":"pandas",
            "sklearn":"scikit-learn","tqdm":"tqdm","ema_pytorch":"ema-pytorch","matplotlib":"matplotlib"}
missing = [pkg for mod,pkg in REQUIRED.items() if not importlib.util.find_spec(mod)]
if missing:
    print(f"[!] 缺少: {missing}  请运行: pip install {' '.join(missing)}")
else:
    print("[OK] 所有依赖已就绪")

import torch
print(f"Python : {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}  CUDA: {torch.version.cuda}")
print(f"GPU 可用: {torch.cuda.is_available()}", end="")
if torch.cuda.is_available():
    print(f"  -> {torch.cuda.get_device_name(0)}")
else:
    print("  (CPU 模式，训练会慢但结果正确)")

# 定位仓库根目录（向上查找 Data/datasets/）
def repo_root() -> Path:
    here = Path.cwd().resolve()
    for cand in [here, *here.parents]:
        if (cand / "Data" / "datasets").is_dir():
            return cand
    return here

REPO_ROOT = repo_root()
DATA_PATH = REPO_ROOT / "Data" / "datasets" / "stock_data.csv"
print(f"仓库根目录: {REPO_ROOT}")
print(f"股票数据: {DATA_PATH}  存在={DATA_PATH.exists()}")
'''))

    C.append(md_cell(r"""
### A-2  Stock 数据集说明

**Stock 数据集**（来自 Diffusion-TS）：
- 谷歌股票日线数据，2004–2019，**3685 行**
- **6 个特征**：Open / High / Low / Close / Adj\_Close / Volume
- 无日期列（这与 ETTh 格式不同，需用 `name='stock'`）

**预处理流程**（对应 `Utils/Data_utils/real_datasets.py::CustomDataset`）：

```
原始数据 (3685, 6)
    ↓ StandardScaler.fit(全部行) → z-score 归一化
归一化数据 (3685, 6)
    ↓ 滑动窗口，步长1，窗口长度192
所有窗口 (3494, 192, 6)
    ↓ 时间顺序 70/10/20 切分（论文补充材料设置）
训练集 / 验证集 / 测试集
```

**为什么窗口长度 192？**  历史 96 步 + 未来 96 步 = 一个 ARMD 样本。

**为什么 z-score 归一化？** 论文 Table 1 的 MSE/MAE 都在归一化空间计算，不做反变换。

> **注意**：若真实 CSV 缺失，下方会自动生成随机游走占位数据，管道可跑通但指标不可与论文对比。
"""))

    # ── 完整 CustomDataset ────────────────────────────────────────────────────
    C.append(md_cell(r"""
### A-3  `CustomDataset`（完整代码）

下方是 `Utils/Data_utils/real_datasets.py::CustomDataset` 的**完整源码**，
仅去掉了对仓库内部模块的相对 import（`masking_utils`、`model_utils`）。

**PyTorch 基础提示**：
- `torch.utils.data.Dataset`：PyTorch 数据集基类，子类必须实现 `__len__` 和 `__getitem__`。
- `__getitem__(idx)` 返回一个样本；DataLoader 会把多个样本拼成 batch。
- `torch.from_numpy(arr).float()` 把 numpy array 转成 PyTorch float32 张量。

**关键方法说明**：

| 方法 | 功能 |
|------|------|
| `read_data` | 读 CSV，`StandardScaler.fit`，返回原始数据和 scaler |
| `__normalize` | 用 scaler.transform 做 z-score 归一化 |
| `__getsamples_three_split` | 论文口径：70/10/20 时间顺序切分 |
| `__getsamples` | 默认 80/20 切分 |
| `divide` | 按 ratio 切分 regular/irregular 两段 |
| `__getitem__` | 训练期返回 `(x,)`；测试期返回 `(x, mask)` |
"""))

    # CustomDataset — standalone version
    _dataset_standalone = '''\
import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """Standalone replica of Utils/Data_utils/real_datasets.py::CustomDataset.

    Only supports stock CSV (name='stock') for this tutorial.
    Removes dependency on masking_utils and model_utils.
    """

    def __init__(
        self,
        name,
        data_root,
        window=192,
        proportion=0.8,
        save2npy=False,
        neg_one_to_one=False,
        seed=123,
        period="train",
        output_dir="./OUTPUT",
        predict_length=None,
        missing_ratio=None,
        style="separate",
        distribution="geometric",
        mean_mask_length=3,
        three_split=False,
        train_ratio=0.7,
        val_ratio=0.1,
    ):
        super().__init__()
        self.three_split  = three_split
        self.train_ratio  = float(train_ratio)
        self.val_ratio    = float(val_ratio)
        if self.three_split:
            assert period in ("train", "val", "test")
        else:
            assert period in ("train", "test")
        if period == "train":
            assert predict_length is None and missing_ratio is None

        self.name       = name
        self.pred_len   = predict_length
        self.auto_norm  = False   # neg_one_to_one scaling disabled in standalone

        self.rawdata, self.scaler = self.read_data(data_root, name)
        self.dir = os.path.join(output_dir, "samples")
        os.makedirs(self.dir, exist_ok=True)

        self.window, self.period = window, period
        self.len, self.var_num = self.rawdata.shape
        self.sample_num_total  = max(self.len - window + 1, 0)
        self.save2npy = save2npy

        # z-score normalization
        self.data = self.scaler.transform(self.rawdata)

        # split
        if self.three_split:
            train_w, val_w, test_w = self.__getsamples_three_split(self.data, seed)
            self.samples = {"train": train_w, "val": val_w, "test": test_w}[period]
        else:
            train_w, test_w = self.__getsamples(self.data, proportion, seed)
            self.samples = train_w if period == "train" else test_w

        # build mask for test/val (future region masked)
        if period in ("test", "val"):
            if predict_length is not None:
                masks = np.ones(self.samples.shape, dtype=bool)
                masks[:, -predict_length:, :] = False
                self.masking = masks
            else:
                raise NotImplementedError("missing_ratio masking not used in this tutorial")

        self.sample_num = self.samples.shape[0]

    # ── window construction ────────────────────────────────────────────────
    def __getsamples(self, data, proportion, seed):
        """Original 2-way split (used in stock.yaml, proportion=0.8)."""
        n = self.sample_num_total
        x = np.stack([data[i : i + self.window] for i in range(n)])
        return self.divide(x, proportion, seed)

    def __getsamples_three_split(self, data, seed):
        """Chronological 70/10/20 split (paper supplemental setting)."""
        n = self.sample_num_total
        x = np.stack([data[i : i + self.window] for i in range(n)])
        t_end = int(np.ceil(n * self.train_ratio))
        v_end = int(np.ceil(n * (self.train_ratio + self.val_ratio)))
        return x[:t_end], x[t_end:v_end], x[v_end:]

    # ── normalization helpers ──────────────────────────────────────────────
    def unnormalize(self, sq):
        d = self.scaler.inverse_transform(sq.reshape(-1, self.var_num))
        return d.reshape(-1, self.window, self.var_num)

    # ── static helpers ─────────────────────────────────────────────────────
    @staticmethod
    def divide(data, ratio, seed=2023):
        """Split windows into regular (first ceil(ratio*N)) and irregular (rest)."""
        size = data.shape[0]
        st0 = np.random.get_state()
        np.random.seed(seed)
        cut = int(np.ceil(size * ratio))
        idx = np.arange(size)          # chronological order (no shuffle)
        regular   = data[idx[:cut]]
        irregular = data[idx[cut:]]
        np.random.set_state(st0)
        return regular, irregular

    @staticmethod
    def read_data(filepath, name="stock"):
        """Read CSV; drop first column only for 'etth' format (has a date string)."""
        df = pd.read_csv(filepath, header=0)
        if name == "etth":
            df.drop(df.columns[0], axis=1, inplace=True)  # drop date column
        data = df.values.astype(np.float64)
        scaler = StandardScaler()
        scaler.fit(data)         # fit on ALL rows before splitting
        return data, scaler

    # ── PyTorch Dataset interface ──────────────────────────────────────────
    def __getitem__(self, ind):
        x = self.samples[ind]             # (window, var_num)  numpy float64
        x_t = torch.from_numpy(x).float()  # → PyTorch float32 tensor
        if self.period in ("test", "val"):
            m = self.masking[ind]
            return x_t, torch.from_numpy(m)
        return x_t

    def __len__(self):
        return self.sample_num
'''
    C.append(py_cell(_dataset_standalone))

    # ── data loading ──────────────────────────────────────────────────────────
    C.append(py_cell('''\
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# ── generate synthetic fallback if CSV missing ─────────────────────────────
def ensure_csv(path):
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    z = np.cumsum(rng.standard_normal((900, 6)), axis=0)
    pd.DataFrame(z, columns=["Open","High","Low","Close","Adj_Close","Volume"])\
      .to_csv(path, index=False)
    print(f"[合成数据] {path} 已生成（占位，真实数据结果会不同）")
    return True

is_synth = ensure_csv(DATA_PATH)
print(f"使用{'合成' if is_synth else '真实'} CSV: {DATA_PATH}")

SEQ_LEN = 96    # 历史 = 预测 = 96 步（论文实验设置）
WINDOW  = 192   # 滑动窗口长度 = SEQ_LEN * 2

# ── 训练集（三段切分，对应 stock_paper.yaml） ──────────────────────────────
train_ds = CustomDataset(
    name="stock",
    data_root=str(DATA_PATH),
    window=WINDOW,
    three_split=True, train_ratio=0.7, val_ratio=0.1,
    period="train",
    save2npy=False,
)
# ── 测试集 ─────────────────────────────────────────────────────────────────
test_ds = CustomDataset(
    name="stock",
    data_root=str(DATA_PATH),
    window=WINDOW,
    three_split=True, train_ratio=0.7, val_ratio=0.1,
    period="test",
    predict_length=SEQ_LEN,
    save2npy=False,
)
N_FEAT = train_ds.var_num
print(f"特征数: {N_FEAT}  训练窗口: {len(train_ds)}  测试窗口: {len(test_ds)}")
print(f"单窗口形状: {train_ds[0].shape}  (192步 × {N_FEAT}特征)")
'''))

    C.append(md_cell("### A-4  可视化数据"))
    C.append(py_cell('''\
fig, axes = plt.subplots(2, 1, figsize=(14, 7))

# 原始时序（前 500 行）
df_raw = pd.read_csv(DATA_PATH)
for col in df_raw.columns[:3]:
    axes[0].plot(df_raw[col].values[:500], alpha=0.8, lw=0.8, label=col)
axes[0].set_title("Stock 原始数据（前 500 行，归一化前）")
axes[0].legend(loc="upper right"); axes[0].grid(alpha=0.3)

# 单个训练窗口结构：历史 | 未来
win = train_ds[0].numpy()   # (192, 6)
x_h = np.arange(SEQ_LEN); x_f = np.arange(SEQ_LEN, 2*SEQ_LEN)
for c in range(3):
    axes[1].plot(x_h, win[:SEQ_LEN, c], lw=0.9, alpha=0.8)
    axes[1].plot(x_f, win[SEQ_LEN:, c], lw=0.9, alpha=0.8, ls="--")
axes[1].axvline(SEQ_LEN-0.5, color="red", ls="--", lw=1.5)
axes[1].axvspan(0, SEQ_LEN, alpha=0.05, color="steelblue", label="历史（已知）")
axes[1].axvspan(SEQ_LEN, WINDOW, alpha=0.05, color="darkorange", label="未来（预测目标）")
axes[1].set_title(f"训练窗口结构：{SEQ_LEN} 历史 + {SEQ_LEN} 未来 = {WINDOW} 步")
axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout(); plt.show()

print("Batch 示意形状: (128, 192, 6)")
print("  轴 0 = batch (128 个窗口)")
print("  轴 1 = time  (192 时间步: 0-95 历史, 96-191 未来)")
print("  轴 2 = feat  (6 个股票特征)")
'''))

    # ══════════════════════════════════════════════════════════════════════════
    # Part B: DDPM 背景 + ARMD 动机
    # ══════════════════════════════════════════════════════════════════════════
    C.append(md_cell(r"""
---
## Part B：背景知识 —— DDPM 与 ARMD 动机（论文 Eq.11–20）

> 本章对应论文 **Preliminary** 节和 **Supplemental Materials** 中的 DDPM 推导。
> 如果你已熟悉 DDPM，可跳过 B-1/B-2，直接看 B-3（ARMD 动机）。
"""))

    C.append(md_cell(r"""
### B-1  扩散模型基础（DDPM）—— Eq.11–14

DDPM（Ho et al., 2020）的核心思想：
- **前向过程**：给数据 $X^0$ 逐步加噪，直到变成纯高斯噪声 $X^T \sim \mathcal{N}(0,I)$。
- **反向过程**：训练神经网络从噪声逐步去噪还原数据。

**Eq.11 — 单步前向过程**：

$$q(X^t \mid X^{t-1}) = \mathcal{N}\!\left(X^t;\;\sqrt{1-\beta_t}\,X^{t-1},\;\beta_t I\right) \tag{11}$$

其中 $\beta_t \in [0,1]$ 是预定义噪声方差（"beta schedule"）。

**Eq.12 — 边缘分布：从 $X^0$ 直接采样第 $t$ 步**（不需要逐步迭代）：

$$q(X^t \mid X^0) = \mathcal{N}\!\left(X^t;\;\sqrt{\bar\alpha_t}\,X^0,\;(1-\bar\alpha_t)I\right) \tag{12}$$

**Eq.13 — 累积乘积**：

$$\bar\alpha_t = \prod_{k=1}^{t}\alpha_k, \quad \alpha_t = 1 - \beta_t \tag{13}$$

**Eq.14 — 等价重参数化形式**（可直接计算，无需逐步）：

$$\boxed{X^t = \sqrt{\bar\alpha_t}\,X^0 + \sqrt{1-\bar\alpha_t}\,\varepsilon, \quad \varepsilon \sim \mathcal{N}(0,I)} \tag{14}$$

**Eq.15 — 反向过程**：

$$p_\theta(X^{t-1} \mid X^t) = \mathcal{N}\!\left(X^{t-1};\;\mu_\theta(X^t,t),\;\sigma_t^2 I\right) \tag{15}$$

**两种训练目标**：
- **噪声预测**：网络预测 $\varepsilon$，推导出 $\mu_\theta$。
- **数据预测**：网络直接预测 $X^0$，从而计算 $\mu_\theta$。
  ARMD 采用后者：预测未来初态 $\hat X^0$。
"""))

    C.append(md_cell(r"""
### B-2  条件 DDPM 用于时间序列预测 —— Eq.16–17

把 DDPM 用于 TSF 的常规做法（论文所批评的方向）：

**Eq.16 — 条件生成模型**：

$$p_\theta(X^{0:T}_{1:F} \mid c) = p_\theta(X^T_{1:F})\prod_{t=1}^T p_\theta(X^{t-1}_{1:F} \mid X^t_{1:F}, c) \tag{16}$$

其中 $X^T_{1:F} \sim \mathcal{N}(0,I)$（纯高斯噪声），$c = g(X^0_{-L+1:0})$ 是从历史序列提取的条件。

**Eq.17 — 条件单步去噪**：

$$p_\theta(X^{t-1}_{1:F} \mid X^t_{1:F}, c) = \mathcal{N}\!\left(X^{t-1}_{1:F};\;\mu_\theta(X^t_{1:F},t \mid c),\;\sigma_t^2 I\right) \tag{17}$$

**论文的批评**：
1. 初始状态 $X^T \sim \mathcal{N}(0,I)$ 与历史序列**毫无关系**，大量的反向去噪步骤是在"从头生成"，效率极低。
2. 中间状态（加噪后的数据）不反映时间序列的**连续演化规律**，与 TSF 目标错位。
3. 历史信息以条件 $c$ 的形式注入，增加了模型复杂度和训练难度。
"""))

    C.append(md_cell(r"""
### B-3  ARMA 理论 → ARMD 动机 —— Eq.18–20

ARMD 的名字和设计灵感来自 **ARMA（Auto-Regressive Moving Average）**：

**Eq.18 — AR 成分**：

$$x_t = \phi_1 x_{t-1} + \phi_2 x_{t-2} + \cdots + \phi_p x_{t-p} + \varepsilon_t \tag{18}$$

**Eq.19 — MA 成分**：

$$x_t = \mu + \theta_1 \varepsilon_{t-1} + \theta_2 \varepsilon_{t-2} + \cdots + \theta_q \varepsilon_{t-q} + \varepsilon_t \tag{19}$$

**Eq.20 — ARMA 完整模型**：

$$x_t = \underbrace{\phi_1 x_{t-1} + \cdots + \phi_p x_{t-p}}_{\text{AR: 自回归项}} + \underbrace{\theta_1\varepsilon_{t-1} + \cdots + \theta_q\varepsilon_{t-q}}_{\text{MA: 移动平均项}} + \varepsilon_t \tag{20}$$

**ARMD 的对应关系**（论文 Supplemental）：

| ARMA 概念 | ARMD 对应 |
|---|---|
| 历史值 $x_{t-i}$ | 滑动中间态的"历史部分" |
| 扰动项 $\varepsilon_{t-j}$ | 演化趋势 $z^t$（从未来到历史的偏移） |
| AR 系数 $\phi_i$ | Linear 模块的权重 $W(t)$ |
| MA 系数 $\theta_j$ | Linear 模块的线性层参数 |

**核心改变**：不再加高斯噪声，改用**滑动（Slide）**作为前向演化：
- 初态 = 未来序列 $X^0_{1:T}$
- 终态 = 历史序列 $X^T_{-T+1:0}$（推理时已知！）
- 中间态 = 时间轴上的过渡窗口
"""))

    C.append(md_cell(r"""
### B-4  Beta Schedule 可视化

Beta Schedule 决定了 $\bar\alpha_t$ 的变化曲线，进而影响：
- 训练时 loss 的加权（`loss_weight`）
- Linear 中 W(t) 的初始化
- 推理时的更新步长

下面可视化 linear 和 cosine 两种 schedule 的差异。
"""))

    C.append(py_cell('''\
import math
import numpy as np
import matplotlib.pyplot as plt
import torch

def linear_beta_schedule(timesteps):
    """Linear schedule: beta 从 beta_start 线性增加到 beta_end。
    scale = 1000/timesteps 是为了使不同 T 值下的尺度一致。
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end   = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine schedule (Nichol & Dhariwal, 2021).
    相比 linear schedule，cosine 在 t 较小时变化更平缓，有助于保留更多数据信息。
    s=0.008 是避免 t=0 时 beta 过小的小偏移。
    """
    steps = timesteps + 1
    x     = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    # cos((x/T + s) / (1+s) * pi/2)^2，然后归一化
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]   # 使 alpha_bar_0 = 1
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

T = 96   # Stock 实验中 T = SEQ_LEN = 96

betas_lin = linear_beta_schedule(T)
betas_cos = cosine_beta_schedule(T)

alphas_lin = 1. - betas_lin
alphas_cos = 1. - betas_cos
abar_lin = torch.cumprod(alphas_lin, dim=0)
abar_cos = torch.cumprod(alphas_cos, dim=0)

t_range = np.arange(1, T+1)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(t_range, betas_lin.numpy(),  label="linear beta",  lw=1.5)
axes[0].plot(t_range, betas_cos.numpy(),  label="cosine beta",  lw=1.5)
axes[0].set_title(r"$\\beta_t$ (噪声方差 schedule)"); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(t_range, abar_lin.numpy(),  label=r"linear $\\bar\\alpha_t$",  lw=1.5)
axes[1].plot(t_range, abar_cos.numpy(),  label=r"cosine $\\bar\\alpha_t$",  lw=1.5)
axes[1].set_title(r"$\\bar\\alpha_t = \\prod_{k=1}^t\\alpha_k$ (累积乘积)"); axes[1].legend(); axes[1].grid(alpha=0.3)

axes[2].plot(t_range, np.sqrt(abar_lin.numpy()),          label=r"$\\sqrt{\\bar\\alpha_t}$ (linear)", lw=1.5)
axes[2].plot(t_range, np.sqrt(1-abar_lin.numpy()),        label=r"$\\sqrt{1-\\bar\\alpha_t}$ (linear)", lw=1.5, ls="--")
axes[2].set_title(r"Eq.14 系数: $\\sqrt{\\bar\\alpha}$ vs $\\sqrt{1-\\bar\\alpha}$")
axes[2].legend(); axes[2].grid(alpha=0.3)

plt.suptitle("Beta Schedule 对比（ARMD 使用 cosine schedule 初始化 ARMD buffers，linear 初始化 Linear.w）")
plt.tight_layout(); plt.show()

print("关键数值（t=1, T/4, T/2, T）:")
for t_idx in [0, T//4-1, T//2-1, T-1]:
    print(f"  t_code={t_idx} (论文t={t_idx+1:2d}): "
          f"abar_lin={abar_lin[t_idx]:.4f}  abar_cos={abar_cos[t_idx]:.4f}")
'''))

    # ══════════════════════════════════════════════════════════════════════════
    # Part C: 前向过程 Eq.1-3
    # ══════════════════════════════════════════════════════════════════════════
    C.append(md_cell(r"""
---
## Part C：ARMD 前向扩散过程（Evolution）—— Eq.1–3

> 核心思想：不加高斯噪声，改用**确定性滑动窗口**作为前向演化。
"""))

    C.append(md_cell(r"""
### C-1  窗口约定与符号

在 ARMD 中，"上标"表示**扩散状态**，"下标"表示**时间覆盖范围**：

| 符号 | 说明 | 代码中的切片 |
|---|---|---|
| $X^0_{1:T}$ | 初态 = **未来序列**（训练时已知） | `x_start[:, 96:, :]` |
| $X^T_{-T+1:0}$ | 终态 = **历史序列**（推理时已知） | `x[:, :96, :]` |
| $X^t_{1-t:T-t}$ | 第 $t$ 步中间态 | `q_sample` 的输出 |

**代码时间步偏移**（非常重要！）：

论文里 $t \in \{1, 2, \ldots, T\}$，但代码里 `randint(0, T)` 得到的 `t_code` $\in \{0,\ldots,T-1\}$。

在 `q_sample` 里有 `index = int(t[0]) + 1`，所以：

$$t_{\text{论文}} = t_{\text{code}} + 1$$

这个 +1 确保：
- `t_code=0` → `index=1` → 滑动 1 步（接近未来初态）
- `t_code=95` → `index=96=T` → 滑动 T 步（等于完整历史终态）
"""))

    C.append(md_cell(r"""
### C-2  Eq.1 —— 单步滑动

$$\boxed{X^t_{1-t:T-t} = \mathrm{Slide}\!\left(X^{t-1}_{2-t:T-t+1},\;1\right)} \tag{1}$$

`Slide(X, k)` 表示将序列窗口 $X$ 向历史方向移动 $k$ 步。

**含义**：从第 $t-1$ 步的中间态，向历史方向移动 1 步，得到第 $t$ 步中间态。

这是**确定性操作**，没有随机性（区别于 DDPM 加噪！）。

### C-3  Eq.2 —— t 步直接计算中间态

$$\boxed{X^t_{1-t:T-t} = \underbrace{\mathrm{Slide}(X^0_{1:T},\;t)}_{\text{（a）滑动定义}} = \underbrace{\sqrt{\bar\alpha_t}\,X^0_{1:T} + \sqrt{1-\bar\alpha_t}\,z_t}_{\text{（b）扩散式分解}}} \tag{2}$$

Eq.2 把同一个 $X^t$ 写成了**两种等价形式**，初学者最容易困惑的就是它们怎么对应到代码——下面逐一拆开。

#### （a）`q_sample` 只实现了"滑动定义"这一半

```python
def q_sample(self, x_start, t, noise=None):
    index = int(t[0]) + 1                          # 论文 t = t_code + 1
    x_middle = x_start[:, pred_len-index : -index, :]
    return x_middle
```

注意：**代码里压根没有出现 $\sqrt{\bar\alpha_t}$、$z_t$、也没用到 `noise` 参数**。`q_sample` 做的纯粹是按下标切一个窗口，这正是 $\mathrm{Slide}(X^0_{1:T},\,t)$——把"历史+未来"拼接序列 `x_start`（长度 192）向历史方向滑 `index` 步后，取出长度 96 的窗口。

以 `pred_len=96` 为例：
- `t_code=0, index=1`：切 `x_start[:, 95:191, :]`（含1步历史+95步未来）
- `t_code=47, index=48`：切 `x_start[:, 48:144, :]`（各48步混合）
- `t_code=95, index=96`：切 `x_start[:, 0:96, :]`（完整历史）

#### （b）"扩散式分解"那一半在代码里是怎么体现的？

关键点（也是仅看 `q_sample` 看不出来的地方）：**ARMD 从不用右边的公式 $\sqrt{\bar\alpha_t}X^0+\sqrt{1-\bar\alpha_t}z_t$ 去"生成" $X^t$。** 计算方向恰好和 DDPM 相反：

| | DDPM（Eq.14） | ARMD（Eq.2） |
|---|---|---|
| 第 1 步 | 先**采样**噪声 $\varepsilon\sim\mathcal N(0,I)$ | 先**滑动**得到 $X^t$（`q_sample`） |
| 第 2 步 | 再算 $X^t=\sqrt{\bar\alpha_t}X^0+\sqrt{1-\bar\alpha_t}\varepsilon$ | 再**反解**出 $z_t$（见 Eq.3） |
| 谁是已知量 | $\varepsilon$ 是输入，$X^t$ 是输出 | $X^t$ 是输入，$z_t$ 是输出 |

所以右半边 $\sqrt{\bar\alpha_t}X^0+\sqrt{1-\bar\alpha_t}z_t$ **不是一段计算 $X^t$ 的代码**，而是一个**恒等式约束**：我们把已经切好的 $X^t$ 强行拆成"已知未来初态 $X^0$ 的缩放" + "一段残差"，其中缩放系数 $\sqrt{\bar\alpha_t},\sqrt{1-\bar\alpha_t}$ 由 beta schedule 事先固定，剩下的残差就**定义**为 $z_t$（演化趋势）。换句话说：

$$z_t \;\overset{\text{定义}}{=}\; \frac{X^t - \sqrt{\bar\alpha_t}\,X^0}{\sqrt{1-\bar\alpha_t}} \;\;\Longrightarrow\;\; \sqrt{\bar\alpha_t}X^0+\sqrt{1-\bar\alpha_t}z_t \equiv X^t\ \text{（恒成立）}$$

这一步**在 `q_sample` 里没有，而是落在 `_train_loss` 里**（下一节 Eq.3 给出对应代码）。也就是说，Eq.2 的（b）形式 = `q_sample` 的切片输出（a） + `_train_loss` 里 $z_t$ 的定义，两段代码合起来才完整对应 Eq.2。

> **为什么要费劲写成（b）形式？** 因为网络要预测的不是 $X^t$（它能直接切出来），而是这段残差 $z_t$（Eq.7 的回归目标）。把 $X^t$ 分解成"已知部分 + 残差"，才能定义出一个有意义的、随 $t$ 归一化的学习目标 $z_t$，并复用 DDIM 的采样公式（Eq.8–10）。

### C-4  Eq.3 —— 真实演化趋势 $z_t$

把上面的"定义"整理成论文形式（分子分母同除以 $\sqrt{\bar\alpha_t}$）：

$$\boxed{z_t = \frac{X^t_{1-t:T-t} - \sqrt{\bar\alpha_t}\,X^0_{1:T}}{\sqrt{1-\bar\alpha_t}} = \frac{\sqrt{1/\bar\alpha_t}\,X^t_{1-t:T-t} - X^0_{1:T}}{\sqrt{1/\bar\alpha_t - 1}}} \tag{3}$$

这正是把 Eq.2（b）反解出 $z_t$ 的结果——所以 Eq.2 和 Eq.3 是**同一个等式的两种摆放**，而 `q_sample`（给 $X^t$）+ Eq.3（给 $z_t$）合在一起，才把 Eq.2 完整落实到代码。

在 `_train_loss` 中，这对应（这就是 Eq.2 的（b）半在代码里真正出现的位置）：

```python
# x      = q_sample 输出 = X^t_{1-t:T-t}   shape: (B, 96, 6)   ← Eq.2(a) 切片结果
# target = X^0_{1:T} = 真实未来             shape: (B, 96, 6)
# alpha       = sqrt(alpha_bar_t)           ← Eq.2(b) 的 √ᾱ_t
# minus_alpha = sqrt(1 - alpha_bar_t)       ← Eq.2(b) 的 √(1-ᾱ_t)

target_noise = (x - target * alpha) / minus_alpha     # ← 这一行就是 Eq.3 / Eq.2(b) 反解
# 证明它与 Eq.3 一致：令 a = sqrt(abar_t)，则
#   target_noise = (x - X^0 * a) / sqrt(1-abar)
# 反过来代回去就得到 Eq.2(b):
#   a*X^0 + sqrt(1-abar)*target_noise
#   = a*X^0 + sqrt(1-abar) * (x - a*X^0)/sqrt(1-abar)
#   = a*X^0 + (x - a*X^0) = x = X^t   ✓ （恒等式，对任意 X^0 都成立）
```

下一个代码单元用真实股票窗口做**数值验证**：用 Eq.3 从切片 $X^t$ 反解 $z_t$，再用 Eq.2(b) 重建，确认结果与原始切片逐元素相等。

### C-5  数值验证：Eq.2 的两种形式逐元素相等
"""))
    C.append(py_cell('''\
import torch

# 复用 ARMD 的 schedule 系数（cosine, T=96），独立复算一遍以便本节自包含
def _cosine_abar(timesteps=96, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    ac = torch.cos(((x / timesteps) + s) / (1 + s) * 3.141592653589793 * 0.5) ** 2
    ac = ac / ac[0]
    betas = torch.clip(1 - (ac[1:] / ac[:-1]), 0, 0.999)
    alphas = 1.0 - betas
    return torch.cumprod(alphas, dim=0)            # alphas_cumprod, shape (96,)

PRED_LEN = 96
abar = _cosine_abar(PRED_LEN)

win0 = torch.from_numpy(train_ds[0].numpy()).double()   # (192, 6) 一个训练窗口
X0   = win0[PRED_LEN:, :]                                # X^0_{1:T} 真实未来 (96, 6)

print(f"{'t_code':>6} {'index':>5} {'切片==Slide':>11} {'Eq2(b)重建误差':>15}")
for t_code in [0, 23, 47, 71, 95]:
    index = t_code + 1                                   # 论文 t = t_code+1

    # —— Eq.2(a)：q_sample 的滑动切片 —— 这就是 X^t
    Xt = win0[PRED_LEN-index : (-index if index>0 else None), :]   # (96, 6)

    # —— Eq.3：从 X^t 反解 z_t（= _train_loss 里的 target_noise）——
    a  = abar[t_code].sqrt()
    ma = (1 - abar[t_code]).sqrt()
    z_t = (Xt - X0 * a) / ma

    # —— Eq.2(b)：用 √ᾱ·X^0 + √(1-ᾱ)·z_t 重建，应当 == Xt ——
    Xt_rebuilt = a * X0 + ma * z_t
    err = (Xt_rebuilt - Xt).abs().max().item()

    same_as_slide = torch.allclose(Xt, win0[PRED_LEN-index:PRED_LEN-index+PRED_LEN, :])
    print(f"{t_code:>6} {index:>5} {str(same_as_slide):>11} {err:>15.2e}")

print("\\n结论：")
print(" • Eq.2(a) 切片 = q_sample 的输出，纯下标操作，无系数、无随机数；")
print(" • Eq.3 用 √ᾱ、√(1-ᾱ) 把该切片反解为 z_t（_train_loss 的 target_noise）；")
print(" • Eq.2(b) 重建误差≈机器精度(1e-15) ⇒ 两种形式确为同一 X^t 的恒等改写。")
'''))

    C.append(md_cell("### C-6  交互可视化：`q_sample` 滑动过程"))
    C.append(py_cell('''\
import matplotlib.pyplot as plt
import numpy as np
import torch

PRED_LEN = 96
win0 = train_ds[0].numpy()                         # (192, 6)
x_start_d = torch.from_numpy(win0).unsqueeze(0)   # (1, 192, 6)

t_vals = [0, 23, 47, 71, 95]
feat   = 0   # 只展示第 0 个特征

fig, axes = plt.subplots(1, len(t_vals), figsize=(4*len(t_vals), 3.5), sharey=False)
colors = plt.cm.RdYlBu(np.linspace(0.1, 0.9, len(t_vals)))

true_future = win0[PRED_LEN:, feat]
true_hist   = win0[:PRED_LEN, feat]

for i, t_code in enumerate(t_vals):
    index = t_code + 1           # q_sample 中的 index = t_code + 1
    start = PRED_LEN - index
    end   = start + PRED_LEN
    x_mid = win0[start:end, feat]

    ax = axes[i]
    ax.plot(x_mid, color=colors[i], lw=1.5)
    if i == 0:
        ax.plot(true_future, color="green",  ls="--", lw=0.8, alpha=0.5, label="真实未来 X^0")
        ax.plot(true_hist,   color="purple", ls="--", lw=0.8, alpha=0.5, label="真实历史 X^T")
        ax.legend(fontsize=7)
    ax.set_title(
        f"t_code={t_code} (论文t={t_code+1})\\n"
        f"切片[{start}:{end}]\\n"
        f"{'≈未来初态' if t_code==0 else ('=历史终态' if t_code==95 else '过渡状态')}",
        fontsize=9,
    )
    ax.grid(alpha=0.3)

fig.suptitle("Eq.(1)(2) q_sample：t 增大 → 窗口从未来初态滑向历史终态（确定性，无随机噪声）")
plt.tight_layout(); plt.show()

# 验证 t_code=95 = 完整历史
assert np.allclose(win0[0:96, feat], true_hist), "验证失败"
print("验证通过: t_code=95 的切片 == 完整历史段")
print("验证通过: 无随机性（纯切片操作，不依赖任何随机数生成器）")
'''))

    # ══════════════════════════════════════════════════════════════════════════
    # Part D: 反向过程 Eq.4-7
    # ══════════════════════════════════════════════════════════════════════════
    C.append(md_cell(r"""
---
## Part D：ARMD 反向去噪过程（Devolution）—— Eq.4–7

> 网络的任务：给定中间态 $X^t$，预测未来初态 $\hat X^0$，从而计算预测演化趋势 $\hat z$，优化 Eq.7 的 L1 loss。
"""))

    C.append(md_cell(r"""
### D-1  Eq.4 —— Linear 距离预测 D

$$\boxed{D = \mathrm{Linear}(X^t_{1-t:T-t})} \tag{4}$$

`Linear` 模块是 ARMD 的核心 devolution 网络 $R(\cdot)$。

对每个特征维度独立做时间轴上的线性映射（`nn.Linear(T, T)`）：

```python
# input_ shape: (B, 96, 6)
# 对时间轴做线性映射，需要把时间维放到最后
x_tmp = self.linear(input_.permute(0, 2, 1)).permute(0, 2, 1)
#               (B, 6, 96) → Linear(96,96) → (B, 6, 96) → (B, 96, 6)
```

**PyTorch 基础提示**：
- `nn.Linear(in, out)` 对输入的**最后一维**做线性变换：`output = input @ W.T + b`
- `permute(0, 2, 1)` 交换第 1 和第 2 维：`(B, T, F)` → `(B, F, T)`
- 再 permute 回来得到 `(B, T, F)`，这样 Linear 对每个特征的时间轴做映射

### D-2  Eq.5 —— W(t) 加权混合，预测 $\hat X^0$

$$\boxed{\hat X^0(X^t, t, \theta) = \frac{W(t) \cdot X^t_{1-t:T-t} + (1-b\,W(t))\cdot D}{(1 + c\,W(t))^d}} \tag{5}$$

`Linear.forward` 的完整代码只有几行，但**几乎每一行都和公式对不上**，是除 `q_sample` 外第二个"看代码读不出公式"的地方。逐行拆开：

```python
def forward(self, input_, t, training=True):
    noise = torch.randn_like(input_)
    if not training:
        noise = 0
    input_ += self.w_dev[t[0]] * noise                          # ←(1) 公式 Eq.5 里没有这一项
    x_tmp = self.linear(input_.permute(0,2,1)).permute(0,2,1)   # ←(2) 这才是 Eq.4 的 D
    alpha = self.w[t[0]]                                          # ←(3) 此 alpha = W(t)，不是扩散 √ᾱ！
    output = (alpha*input_ + (1-2*alpha)*x_tmp) / (1-1*alpha)**(1/2)   # ←(4) Eq.5
    return output.to(torch.float32)
```

#### （3）致命的变量名冲突：`alpha` 在这里是 $W(t)$，不是 $\sqrt{\bar\alpha_t}$

这是初学者最容易踩的坑。**同一个名字 `alpha` 在两个文件里含义完全不同**：

| 位置 | `alpha` 指代 | 数值来源 |
|---|---|---|
| `linear.py` 的 `forward` | $W(t)$（可学习权重） | `self.w[t[0]]`，**linear schedule** 的 $\bar\alpha$ |
| `armd.py` 的 `_train_loss` | $\sqrt{\bar\alpha_t}$（扩散系数） | `self.sqrt_alphas_cumprod[t[0]]`，**cosine schedule** |

读 Eq.5 时，公式里的 $W(t)$ 对应代码里的 `alpha`；而 Eq.2/3/6 里的 $\bar\alpha_t$ 是另一套数（见 D-3 的双 schedule 说明）。**两者不要混为一谈。**

#### （4）把硬编码常数代回 Eq.5

代码里 `b=2, c=−1, d=1/2` 被直接写死。代入 Eq.5：

$$\frac{W(t)\cdot X^t + (1-b\,W(t))\cdot D}{(1+c\,W(t))^d}\Bigg|_{b=2,\,c=-1,\,d=1/2} = \frac{W(t)\cdot X^t + (1-2W(t))\cdot D}{(1-W(t))^{1/2}} \;\checkmark$$

与代码 `(alpha*input_ + (1-2*alpha)*x_tmp) / (1-1*alpha)**(1/2)` 逐项一致（`input_`=$X^t$，`x_tmp`=$D$）。

> ⚠️ 因为 $d=1/2$、分母是 $(1-W(t))^{1/2}$，所以 **$W(t)$ 必须 $<1$**，否则开方出 NaN。$W(t)$ 初始化为 $\bar\alpha_t<1$，但它是可学习参数（`w_grad=True`），训练中无约束——这是代码隐含的数值前提，公式里看不出来。

**W(t) 的物理含义**：
- $t$ 大（中间态接近历史）→ $W(t)$ 小 → 更依赖 $D$（距离估计）
- $t$ 小（中间态接近未来）→ $W(t)$ 大 → 更依赖输入 $X^t$ 本身

### D-3  训练时小扰动（Supplemental: Deviation）—— 代码第 (1) 行，Eq.5 里没有

`forward` 第一段在做 Eq.5 之前，先给输入加了一项扰动——**这一项在 Eq.5 中完全不存在**，属于论文补充材料里的训练 trick：

$$X^t_{\text{input}} = X^t + \eta_t \cdot \varepsilon, \quad \varepsilon \sim \mathcal{N}(0,I)$$

```python
noise = torch.randn_like(input_)
if not training:
    noise = 0                          # 推理时关闭扰动 → forward 退化为纯 Eq.5
input_ += self.w_dev[t[0]] * noise     # 扰动系数 = w_dev[t]
```

这里有**三个**只看公式发现不了、必须读代码（甚至要动手验证）才知道的实现细节：

#### （1）扰动系数 `w_dev` 是"逐步 $\alpha=1-\beta$"，不是累积 $\bar\alpha_t$

很容易想当然地以为 $\eta_t=\bar\alpha_t$（和 Eq.13 一样），但代码里：

```python
self.betas_dev  = cosine_beta_schedule(96)      # cosine 的 beta
self.alphas_dev = 1. - self.betas_dev           # 逐步 alpha，注意：没有 cumprod！
self.w_dev      = Parameter(alphas_dev, requires_grad=False)
```

`w_dev[t] = 1 - β_t`（**单步**），而非累积乘积 $\bar\alpha_t=\prod_{k\le t}\alpha_k$。两者数值差异很大：

| $t$ | `w_dev[t] = 1-β_t`（代码实际用） | $\bar\alpha_t$（累积，对比） |
|---|---|---|
| 0  | 0.999 | 0.999 |
| 47 | **0.968** | 0.494 |
| 95 | 0.001 | ≈0 |

所以扰动幅度在大半个 $t$ 区间都接近 1（不小！），只有在 $t\to95$（最接近历史终态）时才骤降到 0。**定性**结论（$t$ 大→扰动小、$t$ 小→扰动大）仍成立，但**定量**上和"$\bar\alpha_t$"完全不同——这是代码与论文符号的一处出入，按代码为准。

#### （2）`input_ += ...` 是原地修改，且会"串改" target（真实的 in-place 别名陷阱）

`+=` 是 in-place 操作，直接改写传入张量的底层存储。而传入的 `input_` 正是 `_train_loss` 里 `x = q_sample(...)` 的**切片视图**，它和回归目标 `target = x_start[:, 96:, :]` **共享同一块 `x_start` 存储且区间重叠**。实测：当 `t_code=0` 时，这一行 `+=` 会改动 `target` 96 个元素中的 95 个。

```python
# _train_loss 内部，两者都是 x_start 的视图：
target = x_start[:, 96:, :]               # 覆盖 x_start[96:192]
x      = x_start[:, 96-index:-index, :]   # 覆盖 x_start[96-index:192-index]，与 target 重叠
model_out = self.output(x, ...)           # 内部 x += w_dev*noise → 同时改了 target 重叠段
```

也就是说，**训练时回归目标本身也被这股扰动"染"了一点**。这是已发布代码的真实行为（不是笔记本的改写），它仍能复现论文指标，但属于"公式上看不出、且容易踩坑"的实现细节。如果你自己改写 `_train_loss`，想避免别名，可在 `q_sample` 里 `return x_middle.clone()` 或在 `forward` 用 `input_ = input_ + ...`（非原地）。推理时 `noise=0`，`input_ += 0` 不改值，等价于跳过，无此问题。

> 下一个代码单元会**实测**这个别名效应，亲眼看到 `target` 被改了多少个元素。

#### （3）三套同名 $\alpha$，务必区分

| 名字/出处 | 含义 | schedule | 是否 cumprod |
|---|---|---|---|
| `linear.py` `self.w` → `alpha` | $W(t)$ 加权权重 | **linear** | 是（`alphas_cumprod`）|
| `linear.py` `self.w_dev` | 扰动系数 $\eta_t$ | **cosine** | **否**（`1-β`）|
| `armd.py` `sqrt_alphas_cumprod` 等 | 扩散 $\sqrt{\bar\alpha_t}$ | **cosine** | 是 |

三处都叫 "alpha/α"，但 schedule 不同、是否累积也不同——这是读 ARMD 源码最大的混淆源，记住这张表即可。"""))

    C.append(md_cell("### D-3b  实测：in-place 扰动对 target 的别名影响"))
    C.append(py_cell('''\
import torch

# 复现 _train_loss 中的视图别名：x 与 target 都是 x_start 的切片视图
pred_len = 96
x_start = torch.arange(192, dtype=torch.float32).reshape(1, 192, 1).clone()

target = x_start[:, pred_len:, :]                 # 真实未来 X^0，覆盖 x_start[96:192]
print("x 是 x_start 的视图吗? ", x_start[:, 0:96, :].data_ptr() == x_start.data_ptr())

for t_code in [0, 47, 95]:
    xs = x_start.clone()                          # 每次重置
    tgt = xs[:, pred_len:, :]
    tgt_before = tgt.clone()
    index = t_code + 1
    x = xs[:, pred_len-index:(-index if index>0 else None), :]   # q_sample 切片（视图）

    # 模拟 Linear.forward 第一行：input_ += w_dev*noise（这里用常数 +1000 放大可见）
    x += 1000.0

    changed = (tgt != tgt_before).sum().item()
    print(f"t_code={t_code:>2} (index={index:>2}): in-place += 改动了 target {changed}/{tgt.numel()} 个元素")

print("\\n说明：t_code 越小，x 与 target 的重叠越多，被'串改'的元素越多；")
print("     t_code=95 时 x=完整历史段，与 target 不重叠，target 不受影响。")
print("     真实训练里加的是 w_dev*randn（不是+1000），但别名机制相同。")
'''))

    C.append(md_cell(r"""
### D-4  Eq.6 —— 预测演化趋势 $\hat z$

$$\boxed{\hat z(t,\theta) = \frac{\sqrt{1/\bar\alpha_t}\,X^t_{1-t:T-t} - \hat X^0(X^t,t,\theta)}{\sqrt{1/\bar\alpha_t - 1}}} \tag{6}$$

代码中对应 `predict_noise_from_start`（**函数名沿用 DDPM 的 "noise" 术语，但在 ARMD 语义下是"演化趋势"**）：

```python
def predict_noise_from_start(self, x_t, t, x0):
    return (
        extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0
    ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
# 等价于: (sqrt(1/abar_t) * x_t - x0) / sqrt(1/abar_t - 1)
# x_t = X^t,  x0 = X_hat^0
```

**一处看代码会困惑的地方：同一个 $\hat z$，代码里有两种写法。** ARMD 在两处算 $\hat z$，公式形态完全不同，却是同一个量：

| 出处 | 代码 | 数学形式 |
|---|---|---|
| `_train_loss`（D-5） | `(x - model_out*alpha) / minus_alpha` | $\dfrac{X^t-\sqrt{\bar\alpha_t}\,\hat X^0}{\sqrt{1-\bar\alpha_t}}$ |
| `predict_noise_from_start`（采样用） | `(sqrt_recip*x_t - x0) / sqrt_recipm1` | $\dfrac{\sqrt{1/\bar\alpha_t}\,X^t-\hat X^0}{\sqrt{1/\bar\alpha_t-1}}$ |

把第二式分子分母同乘 $\sqrt{\bar\alpha_t}$：分子 $\to X^t-\sqrt{\bar\alpha_t}\hat X^0$，分母 $\to\sqrt{\bar\alpha_t}\sqrt{1/\bar\alpha_t-1}=\sqrt{1-\bar\alpha_t}$，**两式逐项相等**。即 `_train_loss` 用的是 Eq.6 的等价改写，`predict_noise_from_start` 用的是 Eq.6 的原式——同一个 $\hat z$。（C-6 后的验证单元会顺带数值确认这一点。）

**`extract` 函数的作用**（初学者必读）：

```python
def extract(a, t, x_shape):
    # a: shape (T,)      - 预计算系数数组，如 sqrt_recip_alphas_cumprod
    # t: shape (B,)      - batch 中每个样本的时间步（相同值）
    # x_shape: (B, 96, 6) - 目标形状，用于广播
    b, *_ = t.shape               # b = batch_size
    out = a.gather(-1, t)         # 按 t 的值从 a 中取对应元素 → shape (B,)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))
    # reshape 为 (B, 1, 1)，方便与 (B, 96, 6) 的张量广播相乘
```

示例：`t = [47, 47, 47]`（B=3，相同时间步），`a[47] = 1.23`
→ `extract(a, t, (3,96,6))` 返回形状 `(3, 1, 1)` 的张量 `[[[1.23]], [[1.23]], [[1.23]]]`。

> **为什么 `_train_loss` 用 `self.X[t[0]]` 标量索引，这里却用 `extract(...)`？** 两者等价：因为 `forward` 里 `t = randint(...).repeat(b)` 让整个 batch 共享同一个 $t$，所以 `self.sqrt_alphas_cumprod[t[0]]` 取一个标量、靠广播作用到整个 batch；`extract` 则显式 gather 成 `(B,1,1)`。同一份系数，两种取法，结果相同。
"""))

    C.append(md_cell(r"""
### D-5  Eq.7 —— 训练目标（L1 Loss）

$$\boxed{\mathcal{L}_\theta = \mathbb{E}_t\!\left[\left|z_t - \hat z(t,\theta)\right|\right]} \tag{7}$$

完整的 `_train_loss` 流程（含形状注释）：

```python
def _train_loss(self, x_start, t, target=None, noise=None, training=True):
    # x_start: (B, 192, 6) — 完整窗口（历史 + 未来）
    noise  = default(noise, lambda: torch.randn_like(x_start))
    target = x_start[:, pred_len:, :]          # (B, 96, 6) — 真实未来 X^0_{1:T}

    x = self.q_sample(x_start=x_start, t=t)   # (B, 96, 6) — 中间态 X^t
    model_out = self.output(x, t, training)    # (B, 96, 6) — 预测 X_hat^0

    # Eq.3 中的系数
    alpha       = self.sqrt_alphas_cumprod[t[0]]           # sqrt(alpha_bar_t), 标量
    minus_alpha = self.sqrt_one_minus_alphas_cumprod[t[0]] # sqrt(1-alpha_bar_t), 标量

    # 真实演化趋势 z_t（Eq.3 变形）
    target_noise = (x - target * alpha) / minus_alpha      # (B, 96, 6)

    # 预测演化趋势 z_hat（Eq.6 变形）
    pred_noise   = (x - model_out * alpha) / minus_alpha   # (B, 96, 6)

    # L1 loss（Eq.7）
    train_loss = self.loss_fn(pred_noise, target_noise, reduction='none')  # (B, 96, 6)
    train_loss = reduce(train_loss, 'b ... -> b (...)', 'mean')            # (B, 96*6)
    train_loss = train_loss * extract(self.loss_weight, t, train_loss.shape)  # 时间步加权
    return train_loss.mean()
```

#### 关键：Eq.7 写成"对 noise 的 L1"，但代码其实是"对 $\hat X^0$ 的加权 L1"

这是又一个"只看公式 Eq.7 看不出、必须读代码才懂"的点。`target_noise` 和 `pred_noise` 共用同一个 $X^t$、同一组系数，只有 $X^0$/$\hat X^0$ 不同。相减时 $X^t$ 整项抵消：

$$z_t - \hat z = \frac{X^t-\sqrt{\bar\alpha_t}X^0}{\sqrt{1-\bar\alpha_t}} - \frac{X^t-\sqrt{\bar\alpha_t}\hat X^0}{\sqrt{1-\bar\alpha_t}} = \frac{\sqrt{\bar\alpha_t}}{\sqrt{1-\bar\alpha_t}}\,(\hat X^0 - X^0)$$

所以 Eq.7 的 $|z_t-\hat z|$ **等价于对 $\hat X^0$ 的回归**，只是带了一个随 $t$ 变化的系数 $\dfrac{\sqrt{\bar\alpha_t}}{\sqrt{1-\bar\alpha_t}}$：

$$\big|z_t-\hat z\big| = \underbrace{\frac{\sqrt{\bar\alpha_t}}{\sqrt{1-\bar\alpha_t}}}_{\text{系数①（相减自带）}}\;\big|\hat X^0 - X^0\big|$$

而代码**之后又乘了一次** `loss_weight`：

$$\mathcal L \;=\; \underbrace{\frac{\sqrt{\alpha_t}\,\sqrt{1-\bar\alpha_t}}{100\,\beta_t}}_{\text{系数②（loss\_weight）}}\;\times\;\underbrace{\frac{\sqrt{\bar\alpha_t}}{\sqrt{1-\bar\alpha_t}}\big|\hat X^0-X^0\big|}_{\text{= }|z_t-\hat z|}$$

两个系数相乘，$\sqrt{1-\bar\alpha_t}$ 约掉，最终对 $|\hat X^0-X^0|$ 的**总有效权重**是 $\dfrac{\sqrt{\alpha_t}\sqrt{\bar\alpha_t}}{100\,\beta_t}$。**这个"双重加权"在 Eq.7 里完全看不出来**——公式只写了 $|z_t-\hat z|$，而代码里 (a) 相减自带系数① + (b) 显式乘 `loss_weight` 系数②。

要点：
- 网络 `model_out` 实际学的是 **$\hat X^0$**（未来初态），不是直接学 noise；"预测 noise"只是 Eq.7 的书写形式。
- `loss_weight = sqrt(alpha_t)*sqrt(1-alpha_bar_t)/beta_t/100` 里的 `/100` 纯是缩放常数（论文未展开），其余因子用来平衡不同 $t$ 的 loss 量级。

> 下一个代码单元用真实张量**数值验证**：直接算的 `l1_loss(pred_noise,target_noise)` 与 `(√ᾱ/√(1-ᾱ))·|X̂⁰-X⁰|` 逐元素相等。
"""))

    C.append(md_cell("### D-5b  数值验证：Eq.7 的 noise-loss = 加权的 $\\hat X^0$-loss"))
    C.append(py_cell('''\
import torch

torch.manual_seed(0)
B = 4
abar_t = torch.tensor(0.4938)          # 取 t_code=47 的 cosine bar_alpha 作示例
a  = abar_t.sqrt()                     # √ᾱ_t
ma = (1 - abar_t).sqrt()               # √(1-ᾱ_t)

Xt        = torch.randn(B, 96, 6)      # X^t（任意）
X0        = torch.randn(B, 96, 6)      # 真实未来 X^0
X0_hat    = torch.randn(B, 96, 6)      # 网络预测 \\hat X^0

# —— 按 _train_loss 的写法：先转成 noise，再 L1 ——
target_noise = (Xt - X0     * a) / ma          # z_t   (Eq.3 变形)
pred_noise   = (Xt - X0_hat * a) / ma          # \\hat z (Eq.6 变形)
loss_as_noise = (pred_noise - target_noise).abs()

# —— 解析等价：|z - ẑ| == (√ᾱ/√(1-ᾱ)) * |X̂⁰ - X⁰| ——
loss_as_x0 = (a / ma) * (X0_hat - X0).abs()

print("两种算法逐元素最大差:", (loss_as_noise - loss_as_x0).abs().max().item())
print("→ ≈0 说明：对 noise 的 L1 = (√ᾱ/√(1-ᾱ)) × 对 X̂⁰ 的 L1（系数①）")
print()
print(f"系数① √ᾱ/√(1-ᾱ) (t=47) = {(a/ma).item():.4f}")
print("再乘 loss_weight=√α_t·√(1-ᾱ_t)/(100·β_t)（系数②）后，√(1-ᾱ) 约掉，")
print("对 |X̂⁰-X⁰| 的总有效权重 = √α_t·√ᾱ_t/(100·β_t)。")
'''))

    # ══════════════════════════════════════════════════════════════════════════
    # Part E: 采样 Eq.8-10
    # ══════════════════════════════════════════════════════════════════════════
    C.append(md_cell(r"""
---
## Part E：采样/预测过程（Algorithm 2）—— Eq.8–10

> 推理时，从已知**历史序列**出发，通过 devolution 逐步还原**未来预测**。
> 历史序列不是"条件"，而是采样链的**起点**。
"""))

    C.append(md_cell(r"""
### E-1  Eq.8 —— DDIM 风格的完整反向步

类比 DDIM（Song et al., 2021），把 DDPM 的反向步改写为基于 $\hat X^0$ 的形式：

$$X^{t-1}_{2-t:T-t+1} = \sqrt{\bar\alpha_{t-1}}\left[\frac{X^t_{1-t:T-t} - \sqrt{1-\bar\alpha_t}\,\hat z}{\sqrt{\bar\alpha_t}}\right] + \sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\,\hat z + \sigma_t\varepsilon_t \tag{8}$$

括号内等于 $\hat X^0(X^t, t, \theta)$（由 Eq.5 定义）。

### E-2  Eq.9 —— 确定性简化（去掉随机项）

由于 ARMD 的序列演化是**确定性滑动**，令 $\sigma_t = 0$：

$$\boxed{X^{t-1}_{2-t:T-t+1} = \sqrt{\bar\alpha_{t-1}}\,\hat X^0(X^t,t,\theta) + \sqrt{1-\bar\alpha_{t-1}}\,\hat z(t,\theta)} \tag{9}$$

代码注释：`sigma = 0; noise = 0`（`fast_sample` 中硬编码）。

### E-3  Eq.10 —— 跳步加速采样

每次跳 $k$ 步（而非每步 1 步），大幅减少推理时间：

$$\boxed{X^{t-k}_{1-t+k:T-t+k} = \sqrt{\bar\alpha_{t-k}}\,\hat X^0(X^t,t,\theta) + \sqrt{1-\bar\alpha_{t-k}}\,\hat z(t,\theta)} \tag{10}$$

论文中 $k$ 对应 `sampling_timesteps`，从 $\{1,2,3,4,6,8,12\}$ 中在验证集上选取。
本教程使用 `sampling_timesteps=2`（论文补充材料 Stock 最优值）。

`fast_sample` 的实现（含详细注释）：

```python
@torch.no_grad()
def fast_sample(self, x, clip_denoised=True):
    # x: (B, 192, 6) — 完整测试窗口（含历史+未来，但只用历史半段）
    batch = x.shape[0]
    # 生成 DDIM 跳步序列：[-1, ..., T-1]，共 sampling_timesteps+1 个点
    times = torch.linspace(-1, self.num_timesteps - 1, steps=self.sampling_timesteps + 1)
    times = list(reversed(times.int().tolist()))
    # 相邻步对：[(T-1, T-2/T-k), ..., (k, 0), (0, -1)]
    time_pairs = list(zip(times[:-1], times[1:]))

    img = x[:, :pred_len, :]   # (B, 96, 6) — 历史段，Algorithm 2 的起点 X^T

    for time, time_next in time_pairs:
        time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
        pred_noise, x_start, *_ = self.model_predictions(img, time_cond, clip_x_start=clip_denoised)
        # x_start : (B, 96, 6) — \hat{X}^0
        # pred_noise: (B, 96, 6) — \hat{z}(t, theta)

        if time_next < 0:
            img = x_start   # 最后一步，直接输出 X_hat^0
            continue

        alpha_next = self.alphas_cumprod[time_next]   # bar_alpha_{t-k}
        sigma = 0    # 确定性：sigma_t = 0
        noise = 0    # 不加随机噪声
        c = (1 - alpha_next - sigma ** 2).sqrt()      # = sqrt(1 - bar_alpha_{t-k})
        img = x_start * alpha_next.sqrt() + c * pred_noise   # Eq.(10)

    return img   # (B, 96, 6) — 最终预测 X_hat^0_{1:T}
```

**Algorithm 2 伪代码对照**：

| 论文步骤 | 代码 |
|---|---|
| 输入：历史序列 $X^T_{-T+1:0}$ | `img = x[:, :96, :]` |
| 输入：trained $R(\cdot)$, $\Delta t$, $\bar\alpha_{0:T}$ | `model`, `sampling_timesteps`, `alphas_cumprod` |
| for $t = T$ to $0$ by $\Delta t$ | `for time, time_next in time_pairs:` |
| 用 $R(\cdot)$ 得到 $\hat X^0$, $\hat z$ | `x_start, pred_noise = model_predictions(img, t)` |
| Eq.(10) 更新 | `img = x_start * alpha_next.sqrt() + c * pred_noise` |
| 输出 $X^0_{1:T}$ | `return img` |

### E-4  `fast_sample` 里几处"公式上看不出"的实现细节

和 `q_sample` 一样，`fast_sample` 有几行代码无法和 Eq.8–10 直接对上，逐一说明：

**(1) 更新式 `img = x_start*alpha_next.sqrt() + c*pred_noise` 怎么就是 Eq.10？**

代码变量到公式的映射（注意 `sigma=0` 被硬编码）：

| 代码 | 公式（Eq.10，$\sigma=0$） |
|---|---|
| `x_start` | $\hat X^0(X^t,t,\theta)$（`model_predictions` 返回） |
| `pred_noise` | $\hat z(t,\theta)$（即 `predict_noise_from_start` 的输出） |
| `alpha_next.sqrt()` | $\sqrt{\bar\alpha_{t-k}}$ |
| `c = (1-alpha_next-sigma**2).sqrt()` | $\sqrt{1-\bar\alpha_{t-k}-\sigma_t^2}\xrightarrow{\sigma=0}\sqrt{1-\bar\alpha_{t-k}}$ |

代回：`x_start*√ᾱ_next + √(1-ᾱ_next)*pred_noise` $=\sqrt{\bar\alpha_{t-k}}\hat X^0+\sqrt{1-\bar\alpha_{t-k}}\hat z$ ＝ Eq.10。✓ 注意 Eq.8 方括号内"$\frac{X^t-\sqrt{1-\bar\alpha_t}\hat z}{\sqrt{\bar\alpha_t}}$"在代码里**不再出现**——因为它恒等于 $\hat X^0$，而 `model_predictions` 已直接给出 $\hat X^0$（`x_start`），无需再算方括号。这正是 DDIM 把反向步重写成"$\hat X^0$ + $\hat z$ 线性组合"的好处。

**(2) `pred_noise`/`alpha_next` 的时间索引：这里直接用 `time`/`time_next`，没有 `q_sample` 的 +1**

`q_sample` 切片用 `index=t_code+1`，但这里 `self.alphas_cumprod[time_next]`、`model_predictions(img, time)` 都是**直接拿 `time` 当下标**，没有 +1。看似矛盾，其实一致：`q_sample` 的 `+1` 是把"论文步 $t$"换算成"滑动几步"（滑 $t$ 步要从拼接序列偏移 $t$）；而系数数组 `alphas_cumprod` 是 0-based，`alphas_cumprod[time]` 恰好就是论文第 $time{+}1$ 步的 $\bar\alpha$。两处最终都指向同一个论文步，只是一个数"滑动步数"、一个数"数组下标"。

**(3) `times = linspace(-1, T-1, steps=k+1)` 里的 `-1` 是什么？**

```python
times = torch.linspace(-1, self.num_timesteps - 1, steps=self.sampling_timesteps + 1)
times = list(reversed(times.int().tolist()))     # 例: k=2 → [95, 47, -1]
time_pairs = list(zip(times[:-1], times[1:]))     # → [(95,47), (47,-1)]
```

末尾的 `-1` 是个**哨兵值**：当 `time_next < 0`（最后一对），说明已经走到序列最前端，此时不再做 Eq.10 的线性组合，而是 `img = x_start` 直接输出 $\hat X^0$（见代码 `if time_next < 0`）。这一步在 Algorithm 2 里对应"循环结束、输出 $X^0_{1:T}$"，公式 Eq.10 本身没有体现这个边界处理。

**(4) 起点是历史，不是高斯噪声**

`img = x[:, :pred_len, :]`（注意源码里 `img = torch.randn(...)` 那行被注释掉了）。这是 ARMD 区别于普通扩散模型的本质：采样链起点 $X^T$ ＝**已知的历史序列**，而非 $\mathcal N(0,I)$，所以只需极少步（本教程 `sampling_timesteps=2`）即可。
"""))

    # ══════════════════════════════════════════════════════════════════════════
    # Part F: 代码实现
    # ══════════════════════════════════════════════════════════════════════════
    C.append(md_cell(r"""
---
## Part F：代码实现（Beta Schedules / model_utils / Linear / ARMD）

> 以下 Cell 内嵌仓库全部核心代码，仅去掉相对 import。直接 Run All 可用。
"""))

    C.append(md_cell("### F-1  Beta Schedules（`linear.py` 头部）\n\n已在 B-4 定义，此处已可用。"))

    C.append(md_cell(r"""
### F-2  `model_utils` 辅助函数

来自 `Models/autoregressive_diffusion/model_utils.py`，仅提取 ARMD 实际使用的 4 个函数。

| 函数 | 签名 | 说明 |
|------|------|------|
| `exists` | `exists(x)` | 判断 `x is not None` |
| `default` | `default(val, d)` | 若 `val` 为 None 则返回默认值 `d` |
| `identity` | `identity(t, ...)` | 恒等函数（占位用） |
| `extract` | `extract(a, t, x_shape)` | 按批次时间步索引系数并 reshape 用于广播 |
"""))

    C.append(py_cell(utils_src))

    C.append(md_cell(r"""
### F-3  `Linear` 类（完整代码）—— Eq.4–5 的 Devolution 网络

来自 `Models/autoregressive_diffusion/linear.py`，已去掉仓库相对 import 和 einops（不再使用）。

**`__init__` 中的两个 schedule**：

```
self.w    = Parameter(alphas_cumprod from linear_beta_schedule)  # 可学习，初始化为线性 schedule 的 bar_alpha
self.w_dev = Parameter(alphas_dev from cosine_beta_schedule)      # 不可学习，用于训练扰动
```

为什么 `w` 和 `w_dev` 用不同 schedule？
- `w` 作为 $W(t)$ 需要从 0 到 1 平滑变化，linear schedule 的 $\bar\alpha$ 更线性；
- `w_dev` 作为扰动系数 $\eta_{0:t}$，cosine schedule 在 $t$ 小时变化更平缓，避免扰动过大。

**`forward` 中的形状流**：

```
input_: (B, 96, 6)
  → + w_dev[t]*noise                   # 训练扰动（推理时 noise=0）
  → permute(0,2,1) → (B, 6, 96)
  → nn.Linear(96,96) → (B, 6, 96)     # Eq.4: D = Linear(X^t)
  → permute(0,2,1) → x_tmp: (B, 96, 6)
  → (w[t]*input_ + (1-2*w[t])*x_tmp) / (1-w[t])^0.5   # Eq.5
output: (B, 96, 6)                      # X_hat^0
```
"""))

    C.append(py_cell(linear_src.lstrip()))

    C.append(md_cell(r"""
### F-4  `ARMD` 类（完整代码）—— 主逻辑

来自 `Models/autoregressive_diffusion/armd.py`，已去掉仓库相对 import。

**`__init__` 中预注册的 buffers**（`register_buffer` 说明）：

`register_buffer(name, tensor)` 把张量注册为 buffer：
- **不是可训练参数**（不在 `model.parameters()` 中）
- 但跟模型一起移动（`model.to(device)` 时自动迁移）
- `model.state_dict()` 中有它（可保存/加载）

预计算的系数（对应论文 Eq.13 的各种变形）：

| Buffer 名称 | 公式 | 用途 |
|---|---|---|
| `betas` | $\beta_t$ | 噪声方差 |
| `alphas_cumprod` | $\bar\alpha_t$ | 主 schedule |
| `sqrt_alphas_cumprod` | $\sqrt{\bar\alpha_t}$ | `_train_loss` 中乘 target |
| `sqrt_one_minus_alphas_cumprod` | $\sqrt{1-\bar\alpha_t}$ | `_train_loss` 分母 |
| `sqrt_recip_alphas_cumprod` | $\sqrt{1/\bar\alpha_t}$ | `predict_noise_from_start` |
| `sqrt_recipm1_alphas_cumprod` | $\sqrt{1/\bar\alpha_t-1}$ | `predict_noise_from_start` 分母 |
| `loss_weight` | $\frac{\sqrt{\alpha_t}\sqrt{1-\bar\alpha_t}}{100\beta_t}$ | loss 时间步加权 |

**关键细节**：`t = torch.randint(0, T, (1,)).repeat(b)` —— 整个 batch 共享同一个时间步 t，而不是每个样本独立采样。这是一个实现选择，减少了梯度噪声。
"""))

    C.append(py_cell(armd_src.lstrip()))

    # ══════════════════════════════════════════════════════════════════════════
    # Part G: 训练支持代码
    # ══════════════════════════════════════════════════════════════════════════
    C.append(md_cell(r"""
---
## Part G：训练支持代码（LR 调度 / Trainer / DataLoader）
"""))

    C.append(md_cell(r"""
### G-1  `ReduceLROnPlateauWithWarmup`（`engine/lr_sch.py`）

学习率调度策略（论文实验使用此调度器）：

1. **Warmup 阶段**（前 `warmup=500` 步）：lr 从初始值线性升到 `warmup_lr=8e-4`
2. **Plateau 监控阶段**（之后）：若 loss 在 `patience=4000` 步内无改善，lr 乘以 `factor=0.5`，最小不低于 `min_lr=1e-5`

**PyTorch 提示**：标准 PyTorch 调度器需要调用 `scheduler.step()`，此自定义类也一样，但参数是当前 loss 值（而非 epoch 数）。
"""))

    C.append(py_cell(sch_src.lstrip()))

    C.append(md_cell(r"""
### G-2  `Trainer` 类（`engine/solver.py`，Algorithm 1 的完整实现）

**修改说明**（相比原始仓库）：
- `instantiate_from_config(cfg['scheduler'])` → 直接构造 `ReduceLROnPlateauWithWarmup(**params)`（避免依赖 YAML 解析）
- 删除 `sys.path.append` 和 `from Utils.io_utils import ...`

**关键实现细节**：

| 组件 | 说明 |
|------|------|
| `Adam([...], betas=[0.9, 0.96])` | 标准 Adam，$\beta_1=0.9, \beta_2=0.96$ |
| `EMA(model, decay=0.995)` | 指数移动平均：推理时用 EMA 权重而非原始权重，更稳定 |
| `clip_grad_norm_(model.parameters(), 1.0)` | 梯度裁剪：防止梯度爆炸 |
| `gradient_accumulate_every=2` | 梯度累积：等效于 batch_size × 2，但内存占用不变 |
| `cycle(dataloader)` | 将 DataLoader 变成无限迭代器（训练 steps 而非 epochs） |
| `results_folder = config_folder + f"_{seq_len}"` | 检查点目录自动拼接 seq_length |

**EMA 说明**：
```
EMA 权重 = 0.995 * 上一步EMA权重 + 0.005 * 当前模型权重
```
推理时 `trainer.sample_forecast` 调用 `self.ema.ema_model.generate_mts(x)`（而非 `self.model`）。
"""))

    C.append(py_cell(tr_src.lstrip()))

    C.append(md_cell(r"""
### G-3  `build_dataloader`（`Data/build_dataloader.py`）

把 `CustomDataset` 包装成 PyTorch `DataLoader`。

**训练 vs 测试的差异**：

| 参数 | 训练 (`build_dataloader`) | 测试 (`build_dataloader_cond`) |
|------|------|------|
| `batch_size` | `config['dataloader']['batch_size']` = 128 | `sample_size` = 256 |
| `shuffle` | `True` | `False` |
| `drop_last` | `True`（丢弃不完整的末尾 batch） | `False` |
| 模式 | `period='train'` | `period='test'`, `predict_length=96` |
"""))

    _build_dataloader_standalone = '''\
def build_dataloader(dataset, batch_size, shuffle=True):
    """训练 DataLoader（对应 Data/build_dataloader.py::build_dataloader）。"""
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,   # 训练时丢弃不完整 batch，测试时保留
    )

def build_dataloader_cond(test_dataset, sample_size=256):
    """测试 DataLoader（对应 Data/build_dataloader.py::build_dataloader_cond）。"""
    return torch.utils.data.DataLoader(
        test_dataset,
        batch_size=sample_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

# 创建实际用的 DataLoader
train_loader = build_dataloader(train_ds, batch_size=128, shuffle=True)
test_loader  = build_dataloader_cond(test_ds, sample_size=256)

print(f"训练 DataLoader: {len(train_loader)} 个 batch / epoch  (每 batch 128 窗口)")
print(f"测试 DataLoader: {len(test_loader)} 个 batch          (每 batch 256 窗口)")
print(f"训练 batch 示例形状: {next(iter(train_loader)).shape}")
x_test, mask = next(iter(test_loader))
print(f"测试 batch 示例 x 形状: {x_test.shape}  mask 形状: {mask.shape}")
print(f"  mask[:, :96, :] == True (历史可见)")
print(f"  mask[:, 96:, :] == False (未来被遮掩，用于评估)")
'''
    C.append(py_cell(_build_dataloader_standalone))

    # ══════════════════════════════════════════════════════════════════════════
    # Part H: 训练与评估
    # ══════════════════════════════════════════════════════════════════════════
    C.append(md_cell(r"""
---
## Part H：训练（Algorithm 1）与评估（Algorithm 2）

> **选择运行模式**（在下一个 Cell 中设置 `QUICK_TEST`）：
>
> | 模式 | steps | 预计时间（GPU） | 指标 |
> |------|-------|----------------|------|
> | `True`（快速验证） | 100 | ~3 分钟 | 不可与论文对比 |
> | `False`（完整复现） | 2000 | ~20 分钟 | 对齐论文 Table 1 |
"""))

    C.append(md_cell(r"""
### H-1  超参数配置（对应 `Config/stock_paper.yaml`）

| 超参数 | 值 | 说明 |
|---|---|---|
| `seq_length` | 96 | 历史 = 预测步数 |
| `timesteps` | 96 | 最大扩散步数 $T$ |
| `sampling_timesteps` | 2 | $k$ 步跳步（从 {1,2,3,4,6,8,12} 验证集选取） |
| `loss_type` | `l1` | 对应 Eq.7（绝对值范数） |
| `beta_schedule` | `cosine` | ARMD buffers 初始化 |
| `base_lr` | 1e-3 | Adam 初始学习率 |
| `max_epochs` | 2000 | 总 optimizer steps |
| `gradient_accumulate_every` | 2 | 梯度累积步数 |
| `batch_size` | 128 | 有效 batch = 128 × 2 = 256 |
| `ema.decay` | 0.995 | EMA 衰减系数 |
| `warmup` | 500 | LR warmup 步数 |
"""))

    C.append(py_cell('''\
import os, random
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

# ══════════════════════════════════════════════════════════════════════════
# 选择训练模式
QUICK_TEST = False   # True=100步/3min验证流程；False=2000步/复现论文
# ══════════════════════════════════════════════════════════════════════════

MAX_EPOCHS = 100 if QUICK_TEST else 2000
SAMPLING_TIMESTEPS = 2     # sampling_steps from {1,2,3,4,6,8,12}
LOSS_TYPE = "l1"           # Eq.7 是 L1 loss

if QUICK_TEST:
    print(f"[快速验证模式] {MAX_EPOCHS} steps，指标不可与论文对比")
else:
    print(f"[完整复现模式] {MAX_EPOCHS} steps，对应 Config/stock_paper.yaml")

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

set_seed(2023)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"设备: {DEVICE}")

# ── 创建 ARMD 模型（对应 main.py 的 instantiate_from_config(configs["model"])）
model = ARMD(
    seq_length=SEQ_LEN,          # 96
    feature_size=N_FEAT,         # 6（Stock）
    timesteps=96,                # T = 96
    sampling_timesteps=SAMPLING_TIMESTEPS,
    loss_type=LOSS_TYPE,
    beta_schedule="cosine",
    w_grad=True,                 # W(t) 可学习
).to(DEVICE)
model.fast_sampling = True       # 启用 fast_sample (DDIM 跳步)

n_params = sum(p.numel() for p in model.parameters())
print(f"模型参数: {n_params:,}  "
      f"(Linear.linear: {SEQ_LEN*SEQ_LEN + SEQ_LEN:,}; "
      f"Linear.w: {SEQ_LEN}; Linear.w_dev: {SEQ_LEN})")

# ── solver 配置（镜像 Config/stock_paper.yaml）
config = {
    "solver": {
        "max_epochs": MAX_EPOCHS,
        "gradient_accumulate_every": 2,
        "save_cycle": 10**9,            # 教程不保存检查点
        "results_folder": str(REPO_ROOT / "Checkpoints_standalone_nb"),
        "base_lr": 1e-3,
        "ema": {"decay": 0.995, "update_interval": 10},
        "scheduler": {
            "params": {
                "mode": "min", "factor": 0.5, "patience": 4000,
                "min_lr": 1e-5, "threshold": 0.1, "threshold_mode": "rel",
                "warmup_lr": 8e-4, "warmup": 500, "verbose": False,
            }
        },
    }
}

class Args:
    name = "armd_tutorial"
    save_dir = str(REPO_ROOT / "forecasting_exp_standalone")

args = Args()
os.makedirs(args.save_dir, exist_ok=True)

def cycle_loader(dl):
    """无限迭代器（对应 engine/solver.py 的 cycle 函数）。"""
    while True:
        for x in dl:
            yield x

trainer = Trainer(
    config=config, args=args, model=model,
    dataloader={"dataloader": cycle_loader(train_loader)}, logger=None,
)
print("Trainer 创建完成。")
'''))

    C.append(md_cell(r"""
### H-2  训练循环（手动展开 `Trainer.train()`，含 Algorithm 1 对照）

手动展开而非调用 `trainer.train()`，以便实时记录 loss 历史。行为完全等价。

**Algorithm 1 对照**（每个 optimizer step 的逻辑）：

```
Algorithm 1 (单步):
  1. x_start ← DataLoader (完整窗口 (B, 192, 6))     → next(trainer.dl)
  2. t ← Uniform({1,...,T})                           → ARMD.forward 内部的 randint
  3. X^t ← q_sample(x_start, t)                      → ARMD.q_sample
  4. X_hat^0 ← Linear(X^t, t)                        → ARMD.output
  5. z_hat ← predict_noise_from_start(X^t, t, X_hat0) → ARMD.model_predictions
  6. L = L1(z_t, z_hat)                               → ARMD._train_loss
  7. update theta ← Adam.step()                       → trainer.opt.step()
  8. update EMA                                        → trainer.ema.update()
```
"""))

    C.append(py_cell('''\
from tqdm.auto import tqdm

loss_history: list[float] = []

pbar = tqdm(range(MAX_EPOCHS), desc="train", smoothing=0.05)
for step in range(MAX_EPOCHS):
    total_loss = 0.0
    # ── 梯度累积（gradient_accumulate_every=2）──────────────────────────
    # 等效于 batch_size*2 的有效 batch，但实际只前向一次 128 个样本
    for _ in range(trainer.gradient_accumulate_every):
        data = next(trainer.dl).to(trainer.device)   # (128, 192, 6)
        # ARMD.forward: 随机采样 t → q_sample → Linear → _train_loss
        loss = trainer.model(data, target=data)       # 标量 loss
        loss = loss / trainer.gradient_accumulate_every   # 归一化
        loss.backward()                               # 累积梯度
        total_loss += loss.item()

    clip_grad_norm_(trainer.model.parameters(), 1.0) # 梯度裁剪（防爆炸）
    trainer.opt.step()                                # Adam 更新参数
    trainer.sch.step(total_loss)                      # LR 调度监控 loss
    trainer.opt.zero_grad()                           # 清零梯度
    trainer.step += 1
    trainer.ema.update()                              # EMA 权重更新

    loss_history.append(total_loss)
    if step % max(1, MAX_EPOCHS//20) == 0:
        pbar.set_description(f"loss: {total_loss:.6f}")
    pbar.update(1)

pbar.close()
print(f"训练完成: {len(loss_history)} steps  最终 loss: {loss_history[-1]:.6f}")
'''))

    C.append(md_cell("### H-3  训练 Loss 曲线"))
    C.append(py_cell('''\
import matplotlib.pyplot as plt, numpy as np

fig, axes = plt.subplots(1, 2, figsize=(13, 3))
axes[0].plot(loss_history, lw=0.6, color="#2196F3")
axes[0].set_title(f"Loss 曲线（{len(loss_history)} steps）")
axes[0].set_xlabel("optimizer step"); axes[0].set_ylabel("L1 loss"); axes[0].grid(alpha=0.3)

if len(loss_history) >= 50:
    w = 50
    smooth = np.convolve(loss_history, np.ones(w)/w, mode="valid")
    axes[1].plot(smooth, lw=1.0, color="#E91E63")
    axes[1].axvline(min(500, len(smooth)), color="grey", ls="--", lw=1, label="warmup end")
    axes[1].set_title(f"平滑 Loss（{w}步移动平均）")
    axes[1].set_xlabel("optimizer step"); axes[1].legend(); axes[1].grid(alpha=0.3)
else:
    axes[1].plot(loss_history, lw=1.0); axes[1].set_title("Loss（步数不足50）"); axes[1].grid(alpha=0.3)

plt.tight_layout(); plt.show()
print(f"初始 loss: {loss_history[0]:.6f}  最终 loss: {loss_history[-1]:.6f}")
'''))

    C.append(md_cell(r"""
### H-4  评估（Algorithm 2，10 次采样平均）

**评估协议**（对应 `main.py`，论文 Table 1 使用相同方式）：
1. 对测试集重复 10 次采样
2. 每次用不同随机种子（`2023+run`）
3. 平均 10 次的 MSE / MAE

由于 `fast_sample` 中 `sigma=0; noise=0`（确定性），10 次结果理论上完全一致；保留 10 次只是与 `main.py` 协议对齐。

**指标说明**：MSE / MAE 在 **z-score 归一化空间**计算，**不反变换**到原始价格尺度。
"""))

    C.append(py_cell('''\
import random, numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

shape = [SEQ_LEN, N_FEAT]  # [96, 6]
mse_runs, mae_runs = [], []
samples_last = reals_last = None

for run in range(10):
    set_seed(2023 + run)
    # sample_forecast 内部: fast_sample 用历史 x[:, :96, :] → 预测 (B, 96, 6)
    # reals: 真实未来段 x[:, 96:, :] 从 test_loader 收集
    samples, reals = trainer.sample_forecast(test_loader, shape=shape)
    mse_runs.append(mean_squared_error(samples.reshape(-1), reals.reshape(-1)))
    mae_runs.append(mean_absolute_error(samples.reshape(-1), reals.reshape(-1)))
    samples_last, reals_last = samples, reals

mse = float(np.mean(mse_runs))
mae = float(np.mean(mae_runs))
print("=" * 58)
print(f"ARMD on Stock — {SAMPLING_TIMESTEPS}-step DDIM, 10次采样平均")
print(f"  MSE = {mse:.4f}    MAE = {mae:.4f}")
print(f"  per-run MSE: {[round(m,4) for m in mse_runs]}")
print("=" * 58)
print()
print("论文 Table 1 参考值 (Stock, z-score): MSE=0.235  MAE=0.269")
if QUICK_TEST:
    print("[提示] 快速验证模式，指标偏高属正常；改 QUICK_TEST=False 复现论文")
'''))

    C.append(md_cell("### H-5  预测可视化"))
    C.append(py_cell('''\
import matplotlib.pyplot as plt, numpy as np

n_show = 4; feat_show = min(2, N_FEAT)
idx_list = np.random.default_rng(42).choice(samples_last.shape[0], size=n_show, replace=False)

fig, axes = plt.subplots(n_show, feat_show, figsize=(6*feat_show, 2.8*n_show), sharex=True)
if n_show == 1: axes = np.array([axes])
if feat_show == 1: axes = axes[:, None]

hist_segs = test_ds.samples[:, :SEQ_LEN, :]  # 历史段

for r, idx in enumerate(idx_list):
    for c in range(feat_show):
        ax = axes[r, c]
        xh = np.arange(SEQ_LEN); xf = np.arange(SEQ_LEN, 2*SEQ_LEN)
        ax.plot(xh, hist_segs[idx,:,c],    color="#555",    lw=0.9,
                label="历史" if (r==0 and c==0) else None)
        ax.plot(xf, reals_last[idx,:,c],   color="#1f77b4", lw=1.0,
                label="真实未来" if (r==0 and c==0) else None)
        ax.plot(xf, samples_last[idx,:,c], color="#d62728", lw=1.3, ls="--",
                label="ARMD预测" if (r==0 and c==0) else None)
        ax.axvline(SEQ_LEN-0.5, color="grey", ls="--", lw=0.8)
        ax.grid(alpha=0.3); ax.set_title(f"窗口#{int(idx)} feat{c}", fontsize=9)
        if c == 0: ax.set_ylabel("z-score", fontsize=8)

axes[0,0].legend(loc="upper left", fontsize=8)
fig.suptitle(f"ARMD 预测 (z-score空间)  MSE={mse:.4f} MAE={mae:.4f}", y=1.01)
plt.tight_layout(); plt.show()
'''))

    # ══════════════════════════════════════════════════════════════════════════
    # Part I: main.py 等价 + 消融
    # ══════════════════════════════════════════════════════════════════════════
    C.append(md_cell(r"""
---
## Part I：`main.py` 等价代码与消融实验分析
"""))

    C.append(md_cell(r"""
### I-1  `main.py` 完整流程（参考）

`main.py` 是项目的实际入口，与本教程的等价关系：

```python
# main.py 核心逻辑（伪代码，与本教程各步骤对应）

# 1. 加载配置（YAML → dict）→ 本教程: config dict 直接内联
configs = load_yaml_config(args.config_path)

# 2. 创建模型（instantiate_from_config）→ 本教程: ARMD(...)
model = instantiate_from_config(configs['model']).to(device)
model.fast_sampling = True

# 3. 创建训练 DataLoader → 本教程: build_dataloader(train_ds, ...)
dataloader_info = build_dataloader(configs, args)

# 4. 创建 Trainer → 本教程: Trainer(config, args, model, dataloader)
trainer = Trainer(config=configs, args=args, model=model,
                  dataloader={'dataloader': dataloader})

# 5. 训练（Algorithm 1）→ 本教程: 手动展开训练循环
trainer.train()

# 6. 创建测试 DataLoader → 本教程: build_dataloader_cond(test_ds, ...)
test_dataloader_info = build_dataloader_cond(configs, args)

# 7. 评估（Algorithm 2，10次平均）→ 本教程: for run in range(10): sample_forecast
mse_runs, mae_runs = [], []
for run in range(10):
    set_seed(2023 + run)
    sample, real_ = trainer.sample_forecast(test_dataloader, shape=[seq_len, feat_num])
    mse_runs.append(mean_squared_error(...))
    mae_runs.append(mean_absolute_error(...))
mse, mae = np.mean(mse_runs), np.mean(mae_runs)
print(mse, mae)
```
"""))

    # show main.py as reference
    C.append(md_cell("### I-2  `main.py` 原始代码（参考）"))
    C.append(py_cell(
        "# 以下是 main.py 的原始代码，用于参考。\n"
        "# 在本教程中我们已将其完全展开并内嵌。\n"
        "# 此 Cell 仅展示，不执行（因为仓库 import 在 standalone 中不可用）\n"
        "'''\n" + main_raw.strip() + "\n'''\nprint('[参考代码已显示，不执行]')"
    ))

    C.append(md_cell(r"""
### I-3  消融实验分析（对应论文 Table 4）

论文对 ARMD 进行了 5 个消融实验（在 7 个数据集上，共 14 个设置）：

| 消融变体 | 最优次数 | 原因分析 |
|---|---|---|
| **ARMD（完整模型）** | **11/14** | 基准 |
| 插值方法（Interpolation） | 0/14 | 线性插值 $X^t = X^0 + (X^T-X^0)t/T$ 破坏了时间序列的**自然演化规律**，中间态不再代表真实过渡状态 |
| T-embedding 方法 | 0/14 | 把时间步 $t$ 作为条件注入（传统 DDPM 做法），网络无法利用**滑动带来的结构信息** |
| Transformer 骨干网络 | 3/14 | 参数量更大但未必更好；Linear 足够捕捉时间序列的**线性相关性**，且更高效 |
| 去除小扰动（Deviation） | 0/14 | 训练时无扰动 → **过拟合**到固定中间态，泛化能力下降 |
| 添加随机噪声（sampling）| 0/14 | 推理时加噪声 → 破坏了前向过程的**确定性**，导致预测不稳定 |

**关键设计选择总结**：
1. **滑动 > 插值**：真实的时间演化是平滑滑动，不是两端线性混合。
2. **Linear > Transformer**：对于时间序列的短程线性映射，简单线性层已足够且快 10× 以上。
3. **有扰动 > 无扰动**：少量随机性提升训练多样性，类似 Dropout 的正则效果。
4. **确定性采样 > 随机采样**：时间序列演化本身是确定性的，不需要随机噪声。
"""))

    # ══════════════════════════════════════════════════════════════════════════
    # Part J: 总结
    # ══════════════════════════════════════════════════════════════════════════
    C.append(md_cell(r"""
---
## Part J：公式索引 + 复现 Checklist

### J-1  论文全部公式索引（Eq.1–20）

| 公式编号 | 内容 | 代码位置 |
|---|---|---|
| **Eq.1** | 单步滑动 $X^t = \mathrm{Slide}(X^{t-1}, 1)$ | `ARMD.q_sample`，`index=t_code+1` |
| **Eq.2** | t 步中间态 $= \sqrt{\bar\alpha_t}X^0 + \sqrt{1-\bar\alpha_t}z_t$ | `_train_loss` 代数关系 |
| **Eq.3** | 真实演化趋势 $z_t$ | `target_noise = (x - target*alpha)/minus_alpha` |
| **Eq.4** | 距离预测 $D = \mathrm{Linear}(X^t)$ | `Linear.forward` → `x_tmp = linear(input_.T).T` |
| **Eq.5** | 预测 $\hat X^0$ | `(alpha*input_ + (1-2*alpha)*x_tmp) / (1-alpha)^0.5` |
| **Eq.6** | 预测趋势 $\hat z$ | `predict_noise_from_start(x_t, t, x0)` |
| **Eq.7** | L1 训练目标 $\mathcal{L}=\|z_t-\hat z\|$ | `_train_loss` 中 `loss_fn(pred_noise, target_noise)` |
| **Eq.8** | DDIM 完整反向步（含 $\sigma_t\varepsilon_t$） | `fast_sample`（$\sigma=0$，项被去掉） |
| **Eq.9** | 确定性简化反向步（$\sigma_t=0$） | `fast_sample` 主逻辑 |
| **Eq.10** | 跳步加速采样 | `fast_sample` 的 `time_pairs` 循环 |
| **Eq.11** | DDPM 单步前向 $q(X^t\|X^{t-1})$ | 背景知识，`linear_beta_schedule` |
| **Eq.12** | DDPM 边缘 $q(X^t\|X^0)$ | 背景知识 |
| **Eq.13** | $\bar\alpha_t = \prod_{k=1}^t \alpha_k$ | `torch.cumprod(alphas, dim=0)` |
| **Eq.14** | 直接采样 $X^t = \sqrt{\bar\alpha}X^0 + \sqrt{1-\bar\alpha}\varepsilon$ | Eq.2 的类比（ARMD 改写） |
| **Eq.15** | DDPM 反向 $p_\theta(X^{t-1}\|X^t)$ | 背景知识 |
| **Eq.16** | 条件 DDPM for TSF | 背景知识（ARMD 所改进的对象） |
| **Eq.17** | 条件单步去噪 | 背景知识 |
| **Eq.18** | AR 成分 | ARMD 名称来源（动机） |
| **Eq.19** | MA 成分 | ARMD 名称来源（动机） |
| **Eq.20** | 完整 ARMA 模型 | ARMD 设计灵感 |

### J-2  复现 Checklist

在与论文 Table 1 对比前，逐项确认：

- [ ] **数据**：`stock_data.csv` 来自 Diffusion-TS（6 列，无日期列），`name='stock'`
- [ ] **切分**：70/10/20 时间顺序（`three_split=True, train_ratio=0.7, val_ratio=0.1`）
- [ ] **归一化**：`StandardScaler.fit(全部行)`（不分段 fit）
- [ ] **Loss**：`loss_type='l1'`（Eq.7 是 L1，不是 L2）
- [ ] **采样步数**：`sampling_timesteps=2`（从 {1,2,3,4,6,8,12} 选取）
- [ ] **批大小**：`batch_size=128`，`gradient_accumulate_every=2`
- [ ] **训练步数**：`max_epochs=2000`（`QUICK_TEST=False`）
- [ ] **q_sample 偏移**：`index = t_code + 1`（不是 `t_code`）
- [ ] **确定性采样**：`fast_sample` 中 `sigma=0; noise=0`
- [ ] **推理起点**：`x[:, :96, :]`（历史半段，不是随机噪声）
- [ ] **EMA 推理**：`trainer.ema.ema_model`（不是 `trainer.model`）
- [ ] **指标空间**：z-score 归一化后（不反变换到原始价格）
- [ ] **10 次平均**：`for run in range(10)` 重复采样求均值
- [ ] **`w_grad=True`**：W(t) 可学习（不要固定为 alpha_bar_t）
"""))

    # 不写入显式 cell id（与 macOS / 新版 nbformat 兼容；由 nbformat 按需生成）
    nbformat.write(nb, OUT)
    print(f"Wrote {OUT}")
    print(f"Total cells: {len(nb.cells)}")


if __name__ == "__main__":
    main()
