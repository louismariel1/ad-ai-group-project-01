# Advanced ML project — LoRA + knowledge distillation

Authors: Eric Perez, Yaira Brudenell Guedella, Louis-Marie Loe.

CIFAR-10/100 is the **prototype** from the labs; the same interfaces will point at a larger dataset later without changing experiment code.

## Research questions

- **RQ1** — How do teacher adaptation (Dense FT vs LoRA), distillation objective (output vs feature KD), and student adaptation (full-rank vs LoRA) affect accuracy and training resources?
- **RQ2** — With LoRA teacher and student held constant, does distilling LoRA adapter representations help compared with conventional intermediate-feature KD?
- **RQ3** — Which LoRA representation transfers knowledge most effectively: Ax, BAx, or their combination?

All pipelines start from the same SSL-pretrained backbone, target dataset splits, preprocessing, configuration, and random seed. This keeps comparisons controlled: each named pair changes only the factor described below.

## Experiment purpose and design

### RQ1 — Teacher, KD, and student choices

RQ1 measures the effect of three choices on the final student: how the teacher is fine-tuned, what knowledge it transfers, and whether the student is full-rank or LoRA. The five pipelines are:

| Pipeline | Teacher fine-tuning | Student KD objective | Student |
| --- | --- | --- | --- |
| A | Dense | Output/logit KD | Full-rank |
| B | LoRA | Output/logit KD | Full-rank |
| C | Dense | Intermediate-feature KD | Full-rank |
| D | LoRA | Intermediate-feature KD | Full-rank |
| E | LoRA | Output/logit KD | LoRA |

Use the planned pairwise comparisons; do not interpret these as a complete factorial experiment:

- **A vs B:** change teacher fine-tuning from Dense to LoRA.
- **A vs C:** change KD from output to feature KD with a Dense teacher.
- **B vs D:** change KD from output to feature KD with a LoRA teacher.
- **B vs E:** change the student from full-rank to LoRA.

### RQ2 — Conventional features vs LoRA-space KD

Both RQ2 arms use a LoRA-fine-tuned teacher and a LoRA student. `baseline` matches intermediate backbone features; `lora_space` matches the adapter representations Ax and BAx. The comparison asks whether adapter-space alignment transfers knowledge better than conventional feature alignment.

### RQ3 — Which adapter representation to distill?

All arms use the same LoRA teacher/student setup. `ax` distills the low-rank bottleneck Ax; `bax` distills the adapter output BAx; optional `ax_bax` combines both losses. Compare these to identify which representation is the more useful KD target. The combined arm is optional; run it when time and compute allow.

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

## Step-by-step: setup and run

Run commands from the repository root in an activated Conda Prompt or terminal. The configured device is CUDA when PyTorch can see a GPU, otherwise CPU. Dataset preparation downloads CIFAR-10 and CIFAR-100 if they are not already present.

### 1. Create the environment and folders

```bash
conda env create -f environment.yml
conda activate AML
python scripts/init_layout.py
```

If the `AML` environment already exists, activate it rather than recreating it.

### 2. Create and freeze the data splits

```bash
python scripts/prepare_splits.py
```

This downloads/loads the configured datasets and writes the frozen SSL and target train/validation/test manifests under `config/`. Run this before training. Do not edit or replace these manifests between experiment arms: they are necessary for fair comparisons. If the dataset, seed, or split settings change, coordinate the change and create a new split/config version.

### 3. Train the shared SSL backbone

```bash
python scripts/train_ssl.py
```

This trains the rotation-prediction model on the source CIFAR-10 data and saves the reusable backbone as `checkpoints/ssl/teacher_ssl_v1.ckpt`. Every teacher and student starts from this checkpoint.

### 4. Fine-tune the shared teachers

```bash
python scripts/train_teachers.py
```

This creates one Dense teacher and one LoRA teacher in `checkpoints/teachers/`. RQ pipelines reuse the matching teacher rather than fine-tuning a separate copy for every arm. Run this step before RQ1–RQ3 so teacher adaptation is explicit and shared.

### 5. Run RQ1

```bash
python experiments/RQ1/run.py --pipelines A B C D E
```

Each pipeline trains a student and writes its metrics/checkpoint. You can run subsets (for example `--pipelines A B`) and resume the remaining pipeline command later.

### 6. Run RQ2

```bash
python experiments/RQ2/run.py --pipelines baseline lora_space
```

### 7. Run RQ3

```bash
python experiments/RQ3/run.py --pipelines ax bax
```

Optionally run the combined arm too:

```bash
python experiments/RQ3/run.py --pipelines ax_bax
```

### 8. Build plots, comparison tables, and the environment report

```bash
python scripts/plot_final.py
python scripts/report_environment.py --member "Your Name"
```

`plot_final.py` reads completed run metrics, rebuilds Practical 4-style loss/accuracy curves, writes comparison bars under `results/figures/`, and writes the consolidated comparison table under `results/tables/`. Run it again whenever new experiment metrics are added.

## Quick smoke run vs full experiments

For a quick end-to-end check, use a small epoch count consistently across the prerequisite stages and all RQs:

```bash
python scripts/train_ssl.py --epochs 5
python scripts/train_teachers.py --epochs 5
python experiments/RQ1/run.py --pipelines A B C D E --epochs 5
python experiments/RQ2/run.py --epochs 5
python experiments/RQ3/run.py --pipelines ax bax --epochs 5
python scripts/plot_final.py
```

Five epochs are only a smoke test, not final results. For full runs, omit `--epochs`; each script then uses its corresponding value from `config/project_config.json`. Run the full sequence from the start when changing the dataset or shared training configuration.

## Outputs, evaluation, and safe reruns

- **Checkpoints:** SSL, teacher, and student `*.ckpt` files live in `checkpoints/`. They store CPU PyTorch `state_dict`s and compact configuration/training metadata; optimizer state and per-batch history are not stored in these lightweight checkpoints.
- **Metrics:** `results/metrics/` contains each run's config/seed, validation and test scores, training history, parameter counts, training time, CUDA peak memory when available, checkpoint sizes, and checkpoint references.
- **Figures/tables:** training curves and comparison plots are written under `results/figures/`; the comparison CSV is written under `results/tables/`.
- **Model selection:** the student checkpoint saved for each run uses the epoch with the highest validation accuracy. The held-out test set is evaluated after that selection; do not use test accuracy to choose epochs or settings.
- **Cost interpretation:** teacher adaptation time/parameters are reported separately from student training time/parameters. Compare the same resource fields across arms; CUDA peak memory is unavailable on CPU runs.
- **Overwrite protection:** existing outputs are not replaced by default. To rerun a student arm, add `--overwrite` to its RQ command. RQ runners reuse existing teacher checkpoints; deliberately retrain teachers with `python scripts/train_teachers.py --overwrite`. The SSL checkpoint is protected too; replace it only with `python scripts/train_ssl.py --overwrite`, then retrain teachers and students that depend on it.
- **Smoke-to-full transition:** because output names are stable, replacing smoke-test outputs with full runs requires `--overwrite` for the relevant stage(s). Preserve the frozen splits and only overwrite model/result artifacts when you intend to replace those runs.

## Code reused from the labs

- Practical 3: `ConvBackbone`, `ClassifierHead`, rotation SSL dataset and transforms.
- Practical 4: `LoraLayer` (A/B adapters, Ax and BAx) and `TrainingMetrics` plots.
- Shared Colab bootstrap: folder layout, frozen JSON splits, checkpoint hashing.

## Work split (provisional)

- Member 1: SSL + dense/LoRA teachers + RQ1 A,B
- Member 2: RQ1 C,D,E
- Member 3: RQ2 and RQ3
