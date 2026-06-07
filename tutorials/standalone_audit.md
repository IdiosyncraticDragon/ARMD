# Standalone Tutorial Audit

This document records how the current standalone tutorial is checked against the project goal: keep the project organized, compare against the paper PDF and original source code, and keep the standalone tutorial progressive, detailed, and traceable formula by formula.

## Current entry points

- `tutorials/armd_standalone_full.ipynb`: generated, clean, self-contained paper-to-code notebook.
- `tutorials/armd_standalone_full_preview.html`: rendered preview synchronized from the current standalone notebook.
- `tutorials/generate_armd_standalone_notebook.py`: source of truth for regenerating the standalone notebook.
- `scripts/validate_standalone_notebook.py`: maintenance gate for paper anchors, source anchors, formula mapping, protocol notes, preview freshness, and notebook cleanliness.

`tutorials/armd_standalone_full_executed.ipynb` is a historical executed snapshot, not the current Eq.1-20 tutorial entry point.

## Requirement audit

| Requirement | Current evidence | Verification gate |
|---|---|---|
| Project entry points are organized | README points to standalone, stock tutorial, preview, usage, and validation commands | `README_REQUIRED_SNIPPETS` in `scripts/validate_standalone_notebook.py` |
| Paper PDF is represented | Root PDF exists; `tutorials/paper_text.txt` contains paper section, Algorithm 1/2, and Equation anchors; `tutorials/paper_figs/` assets exist | PDF/text/figure checks in `scripts/validate_standalone_notebook.py` |
| Original source code is represented | Core source paths are checked; source behavior snippets cover ARMD, Linear, Trainer, Dataset, and DataLoader behavior | `CORE_SOURCE_PATHS` and `SOURCE_BEHAVIOR_SNIPPETS` |
| Standalone is self-contained | Code cells define `CustomDataset`, `Linear`, `ARMD`, `Trainer`, scheduler, dataloader helpers; repo package imports are forbidden | `STANDALONE_REQUIRED_DEFINITIONS` and `STANDALONE_FORBIDDEN_IMPORT_PREFIXES` |
| Standalone embedded code preserves key behavior | Embedded code must retain q_sample offset, target/pred noise, loss weighting, deterministic fast_sample, Linear perturbation, EMA forecast, and key loader behavior | `STANDALONE_CODE_REQUIRED_SNIPPETS` |
| Tutorial is progressive | Sections must stay ordered from source map and symbols through Part A-J; beginner breadcrumbs and numerical/visual checks must remain | `PROGRESSIVE_SECTION_ORDER` and `PROGRESSIVE_TEACHING_SNIPPETS` |
| Eq.1-20 are traceable | Formula index has all Eq.1-20 rows, and each row must include required code-location or motivation-only breadcrumbs | `FORMULA_INDEX_REQUIRED_LOCATION_SNIPPETS` |
| Algorithm 1/2 are traceable | J-3 maps paper algorithm steps to original source evidence and standalone sections | Algorithm audit table row-count and snippet checks |
| Paper-style Stock protocol is clear | `Config/stock_paper.yaml`, standalone defaults, README, usage, and results JSON agree on 70/10/20, L1, RevIN, `sampling_timesteps=2` | config, README, usage, and results checks |
| Lightweight stock tutorial is not confused with paper-style protocol | Stock tutorial and README state `Config/stock.yaml`, 80/20, L2, `sampling_timesteps=1` | `STOCK_REQUIRED_SNIPPETS` and README checks |
| Generated notebook is clean and fresh | Formal notebooks must have no saved outputs/execution counts; generator `--check` must pass; temporary tutorial artifacts are forbidden | clean cell checks, artifact checks, generator freshness |

## Standard validation commands

Run from the repository root:

```bash
python tutorials/generate_armd_standalone_notebook.py --check
python scripts/validate_standalone_notebook.py
python -m py_compile scripts/validate_standalone_notebook.py tutorials/generate_armd_standalone_notebook.py
git diff --check
```

If the HTML preview is maintained, regenerate it after standalone changes:

```bash
uv run --no-sync python -m jupyter nbconvert --to html --output armd_standalone_full_preview.html tutorials/armd_standalone_full.ipynb
```
