# Advanced ML project — LoRA + knowledge distillation

Authors: Eric Perez, Yaira Brudenell Guedella, Louis-Marie Loe.

Python scripts only (no notebooks). CIFAR-10/100 is the **prototype** from the labs; the same interfaces will point at a larger dataset later without changing experiment code.

## Research questions

- **RQ1** — Dense vs LoRA teacher, output vs feature KD, full-rank vs LoRA student (pipelines A–E).
- **RQ2** — Conventional intermediate-feature KD vs LoRA-space KD.
- **RQ3** — Distill Ax vs BAx (optional Ax+BAx).

All pipelines start from the same SSL teacher checkpoint and the same frozen target splits.

## Layout

```
config/                 project_config.json, pipelines.json, frozen splits
checkpoints/ssl|teachers|students
experiments/RQ1|RQ2|RQ3
results/figures|metrics|tables
files/                  environment reports and shared notes
src/                    models, LoRA (P4), KD, plots, data
scripts/                setup, SSL, teachers, final plots
```

## Prototype vs larger dataset

`config/project_config.json` currently uses `CIFAR10` (SSL, unlabeled) and a five-class `CIFAR100` subset (fish superclass, same as the Colab bootstrap). When you switch datasets:

1. Change `source_dataset`, `target_dataset`, `image_size`, and `target_classes`.
2. Implement loaders in `src/data.py` (`load_source_dataset` / `load_target_dataset`).
3. Re-run `scripts/prepare_splits.py` (do not overwrite v1; bump `version`).
4. Retrain SSL and teachers. Never overwrite frozen `teacher_ssl_v1.ckpt`.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/init_layout.py
python scripts/prepare_splits.py
```

## Run order

```bash
python scripts/train_ssl.py --epochs 5
python scripts/train_teachers.py --epochs 5
python experiments/RQ1/run.py --pipelines A B --epochs 5
python experiments/RQ1/run.py --pipelines C D E --epochs 5
python experiments/RQ2/run.py --epochs 5
python experiments/RQ3/run.py --epochs 5
python scripts/plot_final.py
python scripts/report_environment.py --member "Your Name"
```

`--epochs 5` is for smoke tests. Full runs use the values in `config/project_config.json`.

`scripts/plot_final.py` rebuilds Practical 4 training curves (loss/accuracy vs epoch) and comparison bar charts from `results/metrics/`.

## Code reused from the labs

- Practical 3: `ConvBackbone`, `ClassifierHead`, rotation SSL dataset and transforms.
- Practical 4: `LoraLayer` (A/B adapters, Ax and BAx) and `TrainingMetrics` plots.
- Shared Colab bootstrap: folder layout, frozen JSON splits, checkpoint hashing.

## Work split (provisional)

- Member 1: SSL + dense/LoRA teachers + RQ1 A,B
- Member 2: RQ1 C,D,E
- Member 3: RQ2 and RQ3
