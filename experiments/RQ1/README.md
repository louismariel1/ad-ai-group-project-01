# RQ1 — Teacher adaptation, KD strategy, student adaptation

One shared set of five pipelines. Train each pipeline once; reuse the metrics in comparisons 1a–1d.

| Pipeline | Teacher | KD | Student |
| --- | --- | --- | --- |
| A | Dense FT | Output | Full-rank |
| B | LoRA FT | Output | Full-rank |
| C | Dense FT | Feature | Full-rank |
| D | LoRA FT | Feature | Full-rank |
| E | LoRA FT | Output | LoRA |

Comparisons: **1a** A vs B · **1b** A vs C · **1c** B vs D · **1d** B vs E.

```bash
python experiments/RQ1/run.py --pipelines A B
python experiments/RQ1/run.py --pipelines C D E
```
