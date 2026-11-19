# RQ2 — Conventional feature KD vs LoRA-space KD

Both arms use a LoRA teacher and a LoRA student. Only the feature location changes.

- `baseline`: MSE on intermediate backbone features
- `lora_space`: MSE on adapter representations (Ax / BAx)

```bash
python experiments/RQ2/run.py
```
