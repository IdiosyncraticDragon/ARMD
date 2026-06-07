"""Validate ARMD tutorial notebook structure.

This is a lightweight maintenance check for tutorials/armd_standalone_full.ipynb
and the companion stock tutorial. It does not execute training cells; it verifies
that the generated notebook still contains the formula/code breadcrumbs that make
it useful as a paper-to-code guide, and that tutorial protocol notes do not drift.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
STANDALONE_NOTEBOOK = ROOT / "tutorials" / "armd_standalone_full.ipynb"
STANDALONE_PREVIEW = ROOT / "tutorials" / "armd_standalone_full_preview.html"
HISTORICAL_EXECUTED_NOTEBOOK = ROOT / "tutorials" / "armd_standalone_full_executed.ipynb"
STOCK_NOTEBOOK = ROOT / "tutorials" / "armd_stock_tutorial.ipynb"
USAGE_DOC = ROOT / "tutorials" / "usage.md"
STANDALONE_AUDIT_DOC = ROOT / "tutorials" / "standalone_audit.md"
RESULTS_JSON = ROOT / "results" / "stock_paper_comparison.json"
README = ROOT / "README.md"
GENERATOR = ROOT / "tutorials" / "generate_armd_standalone_notebook.py"
STOCK_PAPER_CONFIG = ROOT / "Config" / "stock_paper.yaml"
STOCK_CONFIG = ROOT / "Config" / "stock.yaml"
PAPER_PDF = ROOT / "Auto-Regressive Moving Diffusion Models for Time Series Forecasting.pdf"
PAPER_TEXT = ROOT / "tutorials" / "paper_text.txt"

PAPER_FIGURE_PATHS = [
    ROOT / "tutorials" / "paper_figs" / "fig1.png",
    ROOT / "tutorials" / "paper_figs" / "fig1_cropped.png",
    ROOT / "tutorials" / "paper_figs" / "fig1_from_paper.png",
    ROOT / "tutorials" / "paper_figs" / "fig2.png",
    ROOT / "tutorials" / "paper_figs" / "fig2_from_paper.png",
]

CORE_SOURCE_PATHS = [
    "main.py",
    "Config/stock_paper.yaml",
    "Config/stock.yaml",
    "Data/build_dataloader.py",
    "Utils/io_utils.py",
    "Utils/Data_utils/real_datasets.py",
    "Models/autoregressive_diffusion/model_utils.py",
    "Models/autoregressive_diffusion/linear.py",
    "Models/autoregressive_diffusion/armd.py",
    "engine/lr_sch.py",
    "engine/solver.py",
]

GENERATOR_SOURCE_READS = [
    'read("Models/autoregressive_diffusion/model_utils.py")',
    'read("Models/autoregressive_diffusion/linear.py")',
    'read("Models/autoregressive_diffusion/armd.py")',
    'read("engine/lr_sch.py")',
    'read("engine/solver.py")',
    'read("Utils/Data_utils/real_datasets.py")',
    'read("Data/build_dataloader.py")',
    'read("main.py")',
]

SOURCE_BEHAVIOR_SNIPPETS = {
    "Models/autoregressive_diffusion/armd.py": [
        "register_buffer('loss_weight', torch.sqrt(alphas) * torch.sqrt(1. - alphas_cumprod) / betas / 100)",
        "#x_start = maybe_clip(x_start)",
        "img = x[:,:pred_len,:]",
        "sigma = 0",
        "noise = 0",
        "index = int(t[0])+1",
        "target_noise = (x - target*alpha)/minus_alpha",
        "pred_noise = (x - model_out*alpha)/minus_alpha",
        "train_loss = train_loss * extract(self.loss_weight, t, train_loss.shape)",
        "t = torch.randint(0, self.num_timesteps, (1,), device=device).repeat(b).long()",
    ],
    "Models/autoregressive_diffusion/linear.py": [
        "self.betas = linear_beta_schedule(96)",
        "self.betas_dev = cosine_beta_schedule(96)",
        "self.w = torch.nn.Parameter(torch.FloatTensor(self.alphas_cumprod.numpy()), requires_grad=w_grad)",
        "self.w_dev = torch.nn.Parameter(torch.FloatTensor(self.alphas_dev.numpy()), requires_grad=False)",
        "input_+= self.w_dev[t[0]]*noise",
        "alpha = self.w[t[0]]",
        "output = (alpha*input_ + (1-2*alpha)*x_tmp) / (1-1*alpha)**(1/2)",
    ],
    "engine/solver.py": [
        "for _ in range(self.gradient_accumulate_every):",
        "loss = self.model(data, target=data)",
        "clip_grad_norm_(self.model.parameters(), 1.0)",
        "self.ema.update()",
        "x, t_m = batch",
        "sample = self.ema.ema_model.generate_mts(x)",
        "reals = np.row_stack([reals, x[:,shape[0]:,:].detach().cpu().numpy()])",
    ],
    "Utils/Data_utils/real_datasets.py": [
        "three_split=False",
        "norm_on_train=False",
        "train_data, val_data, test_data = x[:t_end], x[t_end:v_end], x[v_end:]",
        "masks[:, -predict_length:, :] = 0",
        "return torch.from_numpy(x).float(), torch.from_numpy(m)",
    ],
    "Data/build_dataloader.py": [
        "config['dataloader']['test_dataset']['params']['predict_length'] = args.pred_len",
        "drop_last=jud",
        "drop_last=False",
    ],
}

PAPER_TEXT_REQUIRED_SNIPPETS = [
    "Auto-Regressive Moving Diffusion Models for Time Series Forecasting",
    "Forward Diffusion (Evolution) of ARMD",
    "Reverse Denoising (Devolution) of ARMD",
    "Sampling/Forecasting of ARMD",
    "Algorithm 1: Training.",
    "Algorithm 2: Sampling/Forecasting.",
    "Generate the diffused sample",
    "Equation (2)",
    "Equation (3)",
    "Equation (5)",
    "Equation (6)",
    "Equation (7)",
    "Equation (10)",
    "For all datasets, the historical length and prediction length",
    "mean squared error (MSE) and mean absolute error",
    "z-score normalized data",
    "Noise in the Sampling Process",
    "Equation (8) is set to 0 during sampling/forecasting",
]

PROGRESSIVE_SECTION_ORDER = [
    "# ARMD 从论文到代码：完全自包含入门教程",
    "## 项目源码地图：从 `main.py` 到公式实现",
    "## 符号与变量对照表",
    "## Part A：环境检测与数据加载",
    "## Part B：背景知识 —— DDPM 与 ARMD 动机（论文 Eq.11–20）",
    "## Part C：ARMD 前向扩散过程（Evolution）—— Eq.1–3",
    "## Part D：ARMD 反向去噪过程（Devolution）—— Eq.4–7",
    "## Part E：采样/预测过程（Algorithm 2）—— Eq.8–10",
    "## Part F：代码实现（Beta Schedules / model_utils / Linear / ARMD）",
    "## Part G：训练支持代码（LR 调度 / Trainer / DataLoader）",
    "## Part H：训练（Algorithm 1）与评估（Algorithm 2）",
    "## Part I：`main.py` 等价代码与消融实验分析",
    "## Part J：公式索引 + 复现 Checklist",
]

PROGRESSIVE_TEACHING_SNIPPETS = [
    "**阅读路径**",
    "先把环境跑通",
    "为什么窗口长度 192",
    "**PyTorch 基础提示**",
    "**关键方法说明**",
    "先用 Eq.11–17 理解",
    "从命令到指标的最短调用链",
    "符号与变量对照表",
    "数值验证：Eq.2 的两种形式逐元素相等",
    "交互可视化：`q_sample` 滑动过程",
    "手动展开 `Trainer.train()`",
    "复现 Checklist",
]


REQUIRED_HEADINGS = [
    "# ARMD 从论文到代码：完全自包含入门教程",
    "## 项目源码地图：从 `main.py` 到公式实现",
    "### A-3  `CustomDataset`（standalone replica）",
    "### B-2b  Eq.11–17 在仓库里到底是什么地位？",
    "### C-2  Eq.1 —— 单步滑动",
    "### C-3  Eq.2 —— t 步直接计算中间态",
    "### C-4  Eq.3 —— 真实演化趋势 $z_t$",
    "### C-5  数值验证：Eq.2 的两种形式逐元素相等",
    "### D-1  Eq.4 —— Linear 距离预测 D",
    "### D-2  Eq.5 —— W(t) 加权混合，预测 $\\hat X^0$",
    "### D-3  训练时小扰动（Supplemental: Deviation）—— 代码第 (1) 行，Eq.5 里没有",
    "### E-1  Eq.8 —— DDIM 风格的完整反向步",
    "### E-3  Eq.10 —— 跳步加速采样",
    "### E-4  `fast_sample` 里几处\"公式上看不出\"的实现细节",
    "### F-3  `Linear` 类（standalone 嵌入版）—— Eq.4–5 的 Devolution 网络",
    "### F-4  `ARMD` 类（standalone 嵌入版）—— 主逻辑",
    "### H-1  超参数配置（对应 `Config/stock_paper.yaml`）",
    "### H-4b  论文 Table 1、仓库配置与本 notebook 的比较口径",
    "### I-4  线性模型实验：同一数据窗口上的 OLS / Ridge 对照",
    "### J-1  论文全部公式索引（Eq.1–20）",
    "### J-2  公式外但必须核对的代码实现",
    "### J-3  Algorithm 1/2 源码审计表",
    "### J-4  复现 Checklist",
]

REQUIRED_SNIPPETS = [
    "Notebook 章节 | 原始文件 | standalone 中的处理",
    "`main.py` → `load_yaml_config(args.config_path)`",
    "`Utils/io_utils.py::instantiate_from_config`",
    "`Data/build_dataloader.py::build_dataloader_cond`",
    "`Models/autoregressive_diffusion/armd.py::ARMD`",
    "`Models/autoregressive_diffusion/linear.py::Linear`",
    "Trainer.sample_forecast(...)",
    "standalone 改写边界",
    "ARMD.q_sample` 不加高斯噪声",
    "target_noise = (x - target*alpha)/minus_alpha",
    "input_+= self.w_dev[t[0]]*noise",
    "代码实际为单步 $1-\\beta_t$，不是累积 $\\bar\\alpha_t$",
    "`randint(..., (1,)).repeat(b)`，整 batch 共享同一个 t",
    "Eq.18–20 不是仓库里单独执行的 ARMA 模块",
    "predict_noise_from_start",
    "fast_sample",
    "`sampling_timesteps` 表示**反向更新次数**",
    "times = [95, 47, -1]",
    "sigma = 0",
    "noise = 0",
    "`clip_denoised=True` 在 `fast_sample` 路径里实际没有裁剪",
    "`eta` 参数也被硬覆盖",
    "w_dev` 不是 $\\eta_{0:t}$ 的累积量",
    "sample_forecast 不读取 mask",
    "测试 mask 不参与预测评分",
    "USE_REVIN = True",
    "use_revin=USE_REVIN",
    "linear baseline RevIN",
    "论文步骤 → 原始源码 → standalone 章节 → 实现备注",
    "Algorithm 1 输入 $X^0_{1:T}$",
    "`ARMD.forward`: `torch.randint(0, self.num_timesteps, (1,)).repeat(b).long()`",
    "Algorithm 2 采样时间网格",
    "`Trainer.sample_forecast`: `self.ema.ema_model.generate_mts(x)`",
    "Stock paper-style RevIN",
    "results/stock_paper_comparison.json",
    "不能直接当作当前 `Config/stock_paper.yaml` 的结果",
]


STOCK_REQUIRED_SNIPPETS = [
    "Protocol note",
    "Config/stock.yaml",
    "original demo protocol",
    "80/20 window split",
    "L2 loss",
    "`sampling_timesteps=1`",
    "Config/stock_paper.yaml",
    "70/10/20 split",
    "L1 loss",
    "RevIN",
    "`sampling_timesteps=2`",
    "prefer the standalone tutorial",
]

USAGE_REQUIRED_SNIPPETS = [
    "Config/stock.yaml",
    "项目导入型轻量演示口径",
    "Config/stock_paper.yaml",
    "paper-style 口径",
    "逐公式学习",
    "python tutorials/generate_armd_standalone_notebook.py --check",
    "python scripts/validate_standalone_notebook.py",
    "该校验还会检查 `tutorials/paper_text.txt`",
    "uv run --no-sync python -m jupyter nbconvert --to html --output armd_standalone_full_preview.html tutorials/armd_standalone_full.ipynb",
    "`tutorials/armd_standalone_full_executed.ipynb` 是旧版历史运行快照",
    "不是当前 Eq.1-20 standalone 教程入口",
    "`tutorials/paper_text.txt` 是从论文 PDF 抽取的维护锚点",
    "`tutorials/paper_figs/` 是论文图像辅助资产",
    "`tutorials/standalone_audit.md` 记录当前 standalone 教程的需求到证据映射",
    "维护 checklist",
    "优先修改 `tutorials/generate_armd_standalone_notebook.py`",
    "检查论文 PDF/text/figs 资产、论文文本锚点、HTML 预览、项目源码地图、公式索引、Algorithm 审计表、代码面包屑、协议说明和配置字段",
    "`tutorials/standalone_audit.md`",
    "results/stock_paper_comparison.json",
]

README_REQUIRED_SNIPPETS = [
    "armd_standalone_full.ipynb",
    "self-contained paper-to-code tutorial",
    "Config/stock_paper.yaml",
    "70/10/20 split, L1, RevIN, `sampling_timesteps=2`",
    "armd_stock_tutorial.ipynb",
    "lightweight project-import tutorial",
    "Config/stock.yaml",
    "80/20 split, L2, `sampling_timesteps=1`",
    "armd_standalone_full_preview.html",
    "rendered preview synchronized from the current standalone notebook",
    "tutorials/standalone_audit.md",
    "requirement-to-evidence audit",
    "tutorials/usage.md",
    "python tutorials/generate_armd_standalone_notebook.py --check",
    "python scripts/validate_standalone_notebook.py",
]

STANDALONE_AUDIT_REQUIRED_SNIPPETS = [
    "# Standalone Tutorial Audit",
    "Requirement audit",
    "Current entry points",
    "Standard validation commands",
    "Project entry points are organized",
    "Paper PDF is represented",
    "Original source code is represented",
    "Standalone is self-contained",
    "Tutorial is progressive",
    "Eq.1-20 are traceable",
    "Algorithm 1/2 are traceable",
    "Paper-style Stock protocol is clear",
    "Lightweight stock tutorial is not confused with paper-style protocol",
    "Generated notebook is clean and fresh",
    "python tutorials/generate_armd_standalone_notebook.py --check",
    "python scripts/validate_standalone_notebook.py",
    "python -m py_compile scripts/validate_standalone_notebook.py tutorials/generate_armd_standalone_notebook.py",
    "git diff --check",
]

RESULTS_REQUIRED_SNIPPETS = [
    "Current generated standalone defaults to paper-style 70/10/20",
    "Current generated standalone uses LOSS_TYPE='l1'",
    "QUICK_TEST=False uses 2000 optimizer steps",
    "Current generated standalone uses sampling_timesteps=2",
    "Current generated standalone sets USE_REVIN=True",
    "stock_tutorial_armd_stock_tutorial",
    "Original demo protocol: 80/20 window split, L2 loss, sampling_timesteps=1",
    "Historical local run. This config did not explicitly set model.params.use_revin=True",
    "Historical local run; this config did not explicitly enable use_revin",
]

PREVIEW_REQUIRED_SNIPPETS = [
    "ARMD 从论文到代码：完全自包含入门教程",
    "论文全部公式索引（Eq.1–20）",
    "Algorithm 1/2 源码审计表",
    "<strong>Eq.20</strong>",
]

PREVIEW_STALE_SNIPPETS = [
    "公式 (1)–(10)",
    "的全部技术内容（公式 (1)–(10)、Algorithm 1/2",
]

FORMULA_INDEX_REQUIRED_LOCATION_SNIPPETS = {
    1: ["ARMD.q_sample", "index=t_code+1"],
    2: ["ARMD._train_loss", "target_noise = (x - target*alpha)/minus_alpha"],
    3: ["ARMD._train_loss", "target_noise = (x - target*alpha)/minus_alpha"],
    4: ["Linear.forward", "x_tmp"],
    5: ["Linear.forward", "output ="],
    6: ["predict_noise_from_start"],
    7: ["ARMD._train_loss", "loss_fn", "loss_weight"],
    8: ["ARMD.fast_sample", "sigma = 0", "noise = 0"],
    9: ["ARMD.fast_sample", "img = x_start"],
    10: ["ARMD.fast_sample", "time_pairs"],
    11: ["ARMD.q_sample", "滑动切片"],
    12: ["Eq.2/Eq.3", "确定性中间态"],
    13: ["ARMD.__init__", "torch.cumprod"],
    14: ["target_noise", "不是随机"],
    15: ["ARMD.p_sample", "ARMD.fast_sample"],
    16: ["没有条件编码器"],
    17: ["x[:, :96, :]"],
    18: ["没有单独 ARMA 模块"],
    19: ["target_noise", "pred_noise"],
    20: ["没有单独 ARMA 模块", "Eq.1–10"],
}

STANDALONE_FORBIDDEN_IMPORT_PREFIXES = ("Models", "Utils", "engine", "Data")

STANDALONE_REQUIRED_DEFINITIONS = {
    "CustomDataset",
    "linear_beta_schedule",
    "cosine_beta_schedule",
    "exists",
    "default",
    "identity",
    "extract",
    "Linear",
    "ARMD",
    "ReduceLROnPlateauWithWarmup",
    "Trainer",
    "build_dataloader",
    "build_dataloader_cond",
}

STANDALONE_CODE_REQUIRED_SNIPPETS = [
    "register_buffer('loss_weight', torch.sqrt(alphas) * torch.sqrt(1. - alphas_cumprod) / betas / 100)",
    "#x_start = maybe_clip(x_start)",
    "img = x[:,:pred_len,:]",
    "sigma = 0",
    "noise = 0",
    "index = int(t[0])+1",
    "target_noise = (x - target*alpha)/minus_alpha",
    "pred_noise = (x - model_out*alpha)/minus_alpha",
    "train_loss = train_loss * extract(self.loss_weight, t, train_loss.shape)",
    "t = torch.randint(0, self.num_timesteps, (1,), device=device).repeat(b).long()",
    "self.betas = linear_beta_schedule(96)",
    "self.betas_dev = cosine_beta_schedule(96)",
    "self.w = torch.nn.Parameter(torch.FloatTensor(self.alphas_cumprod.numpy()), requires_grad=w_grad)",
    "self.w_dev = torch.nn.Parameter(torch.FloatTensor(self.alphas_dev.numpy()), requires_grad=False)",
    "input_+= self.w_dev[t[0]]*noise",
    "alpha = self.w[t[0]]",
    "output = (alpha*input_ + (1-2*alpha)*x_tmp) / (1-1*alpha)**(1/2)",
    "for _ in range(self.gradient_accumulate_every):",
    "loss = self.model(data, target=data)",
    "clip_grad_norm_(self.model.parameters(), 1.0)",
    "self.ema.update()",
    "x, t_m = batch",
    "sample = self.ema.ema_model.generate_mts(x)",
    "reals = np.row_stack([reals, x[:,shape[0]:,:].detach().cpu().numpy()])",
    "three_split=False",
    "drop_last=False",
]

FORBIDDEN_TUTORIAL_ARTIFACT_PATTERNS = [
    "*render_check*",
    "*out.ipynb",
    "*.tmp",
]


def load_notebook(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def cell_source(cell: dict) -> str:
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else str(src)


def require_clean_code_cells(failures: list[str], label: str, cells: list[dict]) -> None:
    for idx, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("execution_count") is not None:
            failures.append(f"{label} code cell {idx} should not have execution_count")
        if cell.get("outputs"):
            failures.append(f"{label} code cell {idx} should not have saved outputs")


def read_text_with_bom(path: Path) -> str:
    data = path.read_bytes()
    if data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff"):
        return data.decode("utf-16")
    if data.startswith(b"\xef\xbb\xbf"):
        return data.decode("utf-8-sig")
    return data.decode("utf-8")


def pdf_text_variants(text: str) -> tuple[str, str]:
    dehyphenated = re.sub(r"-\s*\n\s*", "", text)
    collapsed = re.sub(r"\s+", " ", dehyphenated)
    return text, collapsed


def has_pdf_anchor(text_variants: tuple[str, str], snippet: str) -> bool:
    normalized_snippet = re.sub(r"\s+", " ", snippet)
    return any(snippet in text or normalized_snippet in text for text in text_variants)


def config_lines_without_comments(path: Path) -> list[str]:
    lines: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if line.strip():
            lines.append(line)
    return lines


def count_yaml_scalar(lines: list[str], key: str, expected_value: str) -> int:
    pattern = re.compile(rf"^\s*{re.escape(key)}:\s*{re.escape(expected_value)}\s*$")
    return sum(1 for line in lines if pattern.match(line))


def require_yaml_scalar(
    failures: list[str],
    config_label: str,
    lines: list[str],
    key: str,
    expected_value: str,
    min_count: int = 1,
) -> None:
    count = count_yaml_scalar(lines, key, expected_value)
    if count < min_count:
        failures.append(
            f"{config_label} expected at least {min_count} `{key}: {expected_value}` entry, found {count}"
        )


def main() -> int:
    nb = load_notebook(STANDALONE_NOTEBOOK)
    cells = nb.get("cells", [])
    text = "\n".join(cell_source(cell) for cell in cells)
    standalone_code_text = "\n".join(
        cell_source(cell) for cell in cells if cell.get("cell_type") == "code"
    )

    failures: list[str] = []
    require_clean_code_cells(failures, STANDALONE_NOTEBOOK.relative_to(ROOT).as_posix(), cells)

    tutorials_dir = ROOT / "tutorials"
    for pattern in FORBIDDEN_TUTORIAL_ARTIFACT_PATTERNS:
        for artifact in tutorials_dir.glob(pattern):
            failures.append(f"Temporary tutorial artifact should not be committed/left behind: {artifact.relative_to(ROOT)}")

    if not PAPER_PDF.exists():
        failures.append(f"Paper PDF missing: {PAPER_PDF.relative_to(ROOT)}")
    elif PAPER_PDF.stat().st_size <= 0:
        failures.append(f"Paper PDF is empty: {PAPER_PDF.relative_to(ROOT)}")

    if not PAPER_TEXT.exists():
        failures.append(f"Extracted paper text missing: {PAPER_TEXT.relative_to(ROOT)}")
    else:
        paper_text = read_text_with_bom(PAPER_TEXT)
        paper_texts = pdf_text_variants(paper_text)
        for snippet in PAPER_TEXT_REQUIRED_SNIPPETS:
            if not has_pdf_anchor(paper_texts, snippet):
                failures.append(f"Extracted paper text missing source anchor: {snippet}")

    for figure_path in PAPER_FIGURE_PATHS:
        if not figure_path.exists():
            failures.append(f"Paper figure asset missing: {figure_path.relative_to(ROOT)}")
        elif figure_path.stat().st_size <= 0:
            failures.append(f"Paper figure asset is empty: {figure_path.relative_to(ROOT)}")

    for rel_path in CORE_SOURCE_PATHS:
        if not (ROOT / rel_path).exists():
            failures.append(f"Core source path missing: {rel_path}")

    for rel_path, snippets in SOURCE_BEHAVIOR_SNIPPETS.items():
        source_path = ROOT / rel_path
        if not source_path.exists():
            continue
        source_text = source_path.read_text(encoding="utf-8")
        for snippet in snippets:
            if snippet not in source_text:
                failures.append(f"Core source behavior changed in {rel_path}: missing `{snippet}`")

    generator_text = GENERATOR.read_text(encoding="utf-8")
    for snippet in GENERATOR_SOURCE_READS:
        if snippet not in generator_text:
            failures.append(f"Generator no longer reads expected source path: {snippet}")

    conflict_marker = re.compile(r"^(<<<<<<<|=======|>>>>>>>)", re.MULTILINE)
    if conflict_marker.search(text):
        failures.append("Git conflict markers found in notebook")

    last_pos = -1
    for section in PROGRESSIVE_SECTION_ORDER:
        pos = text.find(section)
        if pos < 0:
            failures.append(f"Missing progressive section anchor: {section}")
        elif pos <= last_pos:
            failures.append(f"Progressive section order regressed near: {section}")
        else:
            last_pos = pos

    for snippet in PROGRESSIVE_TEACHING_SNIPPETS:
        if snippet not in text:
            failures.append(f"Missing progressive teaching breadcrumb: {snippet}")

    for eq in range(1, 21):
        if f"Eq.{eq}" not in text and f"Eq.({eq})" not in text:
            failures.append(f"Missing formula reference Eq.{eq}")

    formula_index_rows = re.findall(r"^\| \*\*Eq\.(\d+)\*\* \| .*? \| (.*?) \|$", text, flags=re.MULTILINE)
    formula_index = {int(eq): location.strip() for eq, location in formula_index_rows}
    for eq in range(1, 21):
        location = formula_index.get(eq)
        if not location:
            failures.append(f"Formula index table missing Eq.{eq} row")
        elif location in {"", "-", "N/A"}:
            failures.append(f"Formula index table Eq.{eq} has no code-location explanation")
        else:
            for snippet in FORMULA_INDEX_REQUIRED_LOCATION_SNIPPETS[eq]:
                if snippet not in location:
                    failures.append(
                        f"Formula index table Eq.{eq} location missing `{snippet}`: {location}"
                    )

    algorithm_audit_rows = re.findall(r"^\| Algorithm [12] .*? \| .*? \| .*? \| .*? \|$", text, flags=re.MULTILINE)
    if len(algorithm_audit_rows) < 12:
        failures.append(f"Algorithm source audit table is too sparse: found {len(algorithm_audit_rows)} Algorithm rows")

    for heading in REQUIRED_HEADINGS:
        if heading not in text:
            failures.append(f"Missing required heading: {heading}")

    for snippet in REQUIRED_SNIPPETS:
        if snippet not in text:
            failures.append(f"Missing required breadcrumb: {snippet}")

    for snippet in STANDALONE_CODE_REQUIRED_SNIPPETS:
        if snippet not in standalone_code_text:
            failures.append(f"Standalone embedded code missing behavior snippet: {snippet}")

    if not STANDALONE_PREVIEW.exists():
        failures.append(f"Standalone HTML preview missing: {STANDALONE_PREVIEW.relative_to(ROOT)}")
    else:
        preview_text = STANDALONE_PREVIEW.read_text(encoding="utf-8")
        for snippet in PREVIEW_REQUIRED_SNIPPETS:
            if snippet not in preview_text:
                failures.append(f"Standalone HTML preview missing current rendered anchor: {snippet}")
        for snippet in PREVIEW_STALE_SNIPPETS:
            if snippet in preview_text:
                failures.append(f"Standalone HTML preview still contains stale rendered text: {snippet}")

    historical_executed_is_stale = False
    if HISTORICAL_EXECUTED_NOTEBOOK.exists():
        historical_nb = load_notebook(HISTORICAL_EXECUTED_NOTEBOOK)
        historical_text = "\n".join(cell_source(cell) for cell in historical_nb.get("cells", []))
        historical_executed_is_stale = (
            "公式 (1)–(10)" in historical_text
            and "Algorithm 1/2 源码审计表" not in historical_text
            and "Eq.20" not in historical_text
        )

    code_cells = 0
    standalone_definitions: set[str] = set()
    for idx, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        code_cells += 1
        source = cell_source(cell)
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            failures.append(f"Code cell {idx} has SyntaxError at line {exc.lineno}: {exc.msg}")
            continue

        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                standalone_definitions.add(node.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    root_name = alias.name.split(".", 1)[0]
                    if root_name in STANDALONE_FORBIDDEN_IMPORT_PREFIXES:
                        failures.append(
                            f"Standalone code cell {idx} imports repository package `{alias.name}`"
                        )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                root_name = module.split(".", 1)[0]
                if root_name in STANDALONE_FORBIDDEN_IMPORT_PREFIXES:
                    failures.append(
                        f"Standalone code cell {idx} imports from repository package `{module}`"
                    )

    if code_cells < 20:
        failures.append(f"Expected at least 20 code cells, found {code_cells}")
    missing_definitions = sorted(STANDALONE_REQUIRED_DEFINITIONS - standalone_definitions)
    if missing_definitions:
        failures.append("Standalone notebook missing embedded definitions: " + ", ".join(missing_definitions))

    stock_nb = load_notebook(STOCK_NOTEBOOK)
    stock_cells = stock_nb.get("cells", [])
    stock_text = "\n".join(cell_source(cell) for cell in stock_cells)
    require_clean_code_cells(failures, STOCK_NOTEBOOK.relative_to(ROOT).as_posix(), stock_cells)
    for snippet in STOCK_REQUIRED_SNIPPETS:
        if snippet not in stock_text:
            failures.append(f"Stock tutorial missing protocol note snippet: {snippet}")

    stock_code_cells = 0
    for idx, cell in enumerate(stock_cells):
        if cell.get("cell_type") != "code":
            continue
        stock_code_cells += 1
        try:
            ast.parse(cell_source(cell))
        except SyntaxError as exc:
            failures.append(f"Stock tutorial code cell {idx} has SyntaxError at line {exc.lineno}: {exc.msg}")

    usage_text = USAGE_DOC.read_text(encoding="utf-8")
    for snippet in USAGE_REQUIRED_SNIPPETS:
        if snippet not in usage_text:
            failures.append(f"usage.md missing tutorial protocol snippet: {snippet}")
    if historical_executed_is_stale and "不是当前 Eq.1-20 standalone 教程入口" not in usage_text:
        failures.append("usage.md must label armd_standalone_full_executed.ipynb as a historical snapshot")

    readme_text = README.read_text(encoding="utf-8")
    for snippet in README_REQUIRED_SNIPPETS:
        if snippet not in readme_text:
            failures.append(f"README.md missing tutorial entrypoint snippet: {snippet}")

    audit_text = STANDALONE_AUDIT_DOC.read_text(encoding="utf-8")
    for snippet in STANDALONE_AUDIT_REQUIRED_SNIPPETS:
        if snippet not in audit_text:
            failures.append(f"standalone_audit.md missing requirement audit snippet: {snippet}")

    results_text = RESULTS_JSON.read_text(encoding="utf-8")
    for snippet in RESULTS_REQUIRED_SNIPPETS:
        if snippet not in results_text:
            failures.append(f"stock_paper_comparison.json missing current tutorial protocol snippet: {snippet}")

    stale_results_snippets = [
        "Usually 80/20",
        "Often L2",
        "Often 1",
    ]
    for snippet in stale_results_snippets:
        if snippet in results_text:
            failures.append(f"stock_paper_comparison.json still contains stale standalone protocol text: {snippet}")

    stock_paper_lines = config_lines_without_comments(STOCK_PAPER_CONFIG)
    for key, expected in [
        ("sampling_timesteps", "2"),
        ("loss_type", "'l1'"),
        ("use_revin", "True"),
        ("max_epochs", "2000"),
        ("gradient_accumulate_every", "2"),
        ("batch_size", "128"),
        ("train_ratio", "0.7"),
        ("val_ratio", "0.1"),
    ]:
        require_yaml_scalar(failures, "Config/stock_paper.yaml", stock_paper_lines, key, expected)
    for key, expected, min_count in [
        ("three_split", "True", 2),
        ("norm_on_train", "False", 2),
    ]:
        require_yaml_scalar(
            failures,
            "Config/stock_paper.yaml",
            stock_paper_lines,
            key,
            expected,
            min_count=min_count,
        )

    stock_lines = config_lines_without_comments(STOCK_CONFIG)
    for key, expected in [
        ("sampling_timesteps", "1"),
        ("loss_type", "'l2'"),
        ("batch_size", "128"),
    ]:
        require_yaml_scalar(failures, "Config/stock.yaml", stock_lines, key, expected)
    if count_yaml_scalar(stock_lines, "use_revin", "True"):
        failures.append("Config/stock.yaml should remain the lightweight demo protocol without `use_revin: True`")
    if count_yaml_scalar(stock_lines, "three_split", "True"):
        failures.append("Config/stock.yaml should remain the original 80/20-style demo protocol, not three_split")

    generator_check = subprocess.run(
        [sys.executable, str(GENERATOR), "--check"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if generator_check.returncode != 0:
        failures.append(
            "Generated standalone notebook is stale. "
            + generator_check.stdout.strip()
            + (" " + generator_check.stderr.strip() if generator_check.stderr.strip() else "")
        )

    if failures:
        print("Standalone notebook validation FAILED:")
        for item in failures:
            print(f" - {item}")
        return 1

    print("Tutorial validation passed:")
    print(f" - standalone notebook: {len(cells)} cells / {code_cells} clean code cells")
    print(f" - stock tutorial: {len(stock_cells)} cells / {stock_code_cells} clean code cells")
    print(
        " - paper evidence: PDF, extracted text anchors, "
        f"and {len(PAPER_FIGURE_PATHS)} figure assets"
    )
    print(
        " - source evidence: "
        f"{len(CORE_SOURCE_PATHS)} core paths and "
        f"{sum(len(v) for v in SOURCE_BEHAVIOR_SNIPPETS.values())} source behavior snippets"
    )
    print(
        " - standalone integrity: embedded definitions, forbidden repo imports, "
        f"and {len(STANDALONE_CODE_REQUIRED_SNIPPETS)} embedded behavior snippets"
    )
    print(
        " - tutorial structure: "
        f"{len(PROGRESSIVE_SECTION_ORDER)} ordered section anchors, "
        f"Eq.1-20 formula index, Algorithm audit table, and HTML preview"
    )
    print(
        " - protocol docs: README, usage.md, standalone audit, "
        "results JSON, stock configs, and generator freshness"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
