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
