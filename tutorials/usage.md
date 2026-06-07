# 教程 Notebook 说明与运行方式

请在**仓库根目录**（`ARMD/`）下操作，以便 `Data/datasets/`、`Config/` 等相对路径正确。

## 环境准备

- **使用 [uv](https://docs.astral.sh/uv/)（与 `pyproject.toml` 中笔记本依赖搭配时推荐）：**  
  执行 `uv sync`  
  之后用 `uv run …` 启动 Jupyter，确保使用项目虚拟环境（`.venv`）中的解释器。

- **传统 pip 方式：** 根据 `requirements.txt` 或 `pyproject.toml` 安装依赖，再使用对应 Python 下的 Jupyter。

- **股票数据：** 将 `stock_data.csv` 放在 `Data/datasets/`，获取方式见仓库根目录 [`README.md`](../README.md)（从 Diffusion-TS 的 `dataset.zip` 中复制 `stock_data.csv` 等）。

---

## `armd_stock_tutorial.ipynb`

**内容概要：** 上手教程，会 **`import` 本仓库模块**（如 `CustomDataset`、`ARMD`、`Trainer`、YAML 配置等）。在较短训练与注释说明下，走与 `main.py` 风格相近的流程。

**口径提示：** 该 notebook 默认读取 `Config/stock.yaml`，是项目导入型轻量演示口径（80/20 窗口切分、L2、`sampling_timesteps=1` 等），不等同于 `armd_standalone_full.ipynb` / `Config/stock_paper.yaml` 的 paper-style 口径（70/10/20、L1、RevIN、`sampling_timesteps=2`）。逐公式学习和 paper-style 对照优先看 standalone 教程。

**适用场景：** 已克隆本仓库，希望在项目目录结构内实验（`Config/` 下配置、检查点目录由配置指定）。

**交互运行（Jupyter）**

```bash
uv run jupyter notebook tutorials/armd_stock_tutorial.ipynb
```

等价写法示例：

```bash
uv run python -m notebook tutorials/armd_stock_tutorial.ipynb
uv run jupyter-notebook tutorials/armd_stock_tutorial.ipynb
```

**无界面执行（可选）**

在仓库根目录执行；`--no-sync` 可减少每次调用时对 `torch` 等的重复安装：

```bash
uv run --no-sync python -m jupyter nbconvert --execute tutorials/armd_stock_tutorial.ipynb --inplace
```

---

## `armd_standalone_full.ipynb`

**内容概要：** **自包含**教程：理论说明 + **内嵌**的实现代码（模型、训练器、数据相关辅助），notebook **不** `import` 本仓库包，仅依赖第三方库（`torch`、`pandas` 等）。适合整份文件阅读或单独分享。

**适用场景：** 需要一篇顺读的「论文 + 代码」文档，而不想把仓库加入 `PYTHONPATH`；或希望所有逻辑都展现在单元格中。

**维护方式：** 仓库中的该文件由脚本生成。若修改了 `Models/`、`engine/` 等源码，需要**重新生成**时执行：

```bash
python tutorials/generate_armd_standalone_notebook.py
```

会覆盖写入 `tutorials/armd_standalone_full.ipynb`。

若只想确认生成物没有和脚本漂移，可执行：

```bash
python tutorials/generate_armd_standalone_notebook.py --check
```

生成后建议运行轻量教程校验，确认 Eq.1-20、关键章节、源码线索、两个教程的实验口径提示和代码单元语法仍然完整：

```bash
python scripts/validate_standalone_notebook.py
```

该校验还会检查 `tutorials/paper_text.txt`，确认论文章节锚点、Algorithm 1/2 锚点，以及 standalone 源码审计依赖的 Equation (2)/(3)/(5)/(6)/(7)/(8)/(10) 引用仍然可追溯。

论文资产说明：`tutorials/paper_text.txt` 是从论文 PDF 抽取的维护锚点，用于防止公式/算法说明脱离论文原文；`tutorials/paper_figs/` 是论文图像辅助资产，当前 standalone 主要靠文字、公式和代码审计表完成学习路径，不依赖这些图片作为主入口。

审计说明：`tutorials/standalone_audit.md` 记录当前 standalone 教程的需求到证据映射，包括论文锚点、原始源码锚点、Eq.1-20 公式索引、Algorithm 1/2 审计表、Stock 协议口径和标准验证命令。

若维护仓库内的 HTML 预览，请用当前 notebook 同步生成：

```bash
uv run --no-sync python -m jupyter nbconvert --to html --output armd_standalone_full_preview.html tutorials/armd_standalone_full.ipynb
```

`tutorials/armd_standalone_full_executed.ipynb` 是旧版历史运行快照，不是当前 Eq.1-20 standalone 教程入口。当前阅读和维护以 `armd_standalone_full.ipynb`、生成脚本和同步后的 `armd_standalone_full_preview.html` 为准。

维护 checklist：

1. 优先修改 `tutorials/generate_armd_standalone_notebook.py`，不要直接改生成出的 notebook。
2. 运行 `python tutorials/generate_armd_standalone_notebook.py` 重新生成 `armd_standalone_full.ipynb`。
3. 如需保留 `armd_standalone_full_preview.html`，用 `uv run --no-sync python -m jupyter nbconvert --to html --output armd_standalone_full_preview.html tutorials/armd_standalone_full.ipynb` 同步渲染。
4. 运行 `python tutorials/generate_armd_standalone_notebook.py --check` 确认生成物没有漂移。
5. 运行 `python scripts/validate_standalone_notebook.py` 检查论文 PDF/text/figs 资产、论文文本锚点、HTML 预览、项目源码地图、公式索引、Algorithm 审计表、代码面包屑、协议说明和配置字段。
6. 如果 Stock paper-style 口径变化，同步更新 `README.md`、`tutorials/usage.md`、`tutorials/standalone_audit.md` 和 `results/stock_paper_comparison.json`。

**交互运行（Jupyter）**

```bash
uv run jupyter notebook tutorials/armd_standalone_full.ipynb
```

**无界面执行（可选）**

```bash
uv run --no-sync python -m jupyter nbconvert --execute tutorials/armd_standalone_full.ipynb --to notebook --output tutorials/armd_standalone_out.ipynb
```

（可按需修改 `--output` 路径；若环境缺少依赖，首个代码单元可能会执行 `uv pip` 或 `ensurepip` 再安装包。）

---

## `generate_armd_standalone_notebook.py`

**内容概要：** 小型构建脚本（基于 `nbformat`），从 `Models/`、`engine/` 等读取片段、去掉仅项目内可用的 import，写出 **`armd_standalone_full.ipynb`**。

**适用场景：** 修改了 ARMD 源码后，希望自包含教程里的代码与仓库实现保持一致。

**命令：**

```bash
python tutorials/generate_armd_standalone_notebook.py
```

---

## 两个 Notebook 对照

| Notebook | 是否依赖本仓库 Python 包 | 典型用途 |
|----------|--------------------------|----------|
| `armd_stock_tutorial.ipynb` | 是（`Models`、`Utils`、`engine` 等） | 在项目配置下调试、训练 |
| `armd_standalone_full.ipynb` | 否（代码写在单元格内） | 教学、离线阅读、单文件分享 |

GPU 为可选；若已安装支持 CUDA 的 PyTorch 且驱动可用，会自动使用 GPU。
