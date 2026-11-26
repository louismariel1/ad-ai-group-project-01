# RQ3 — Ax vs BAx (optional Ax+BAx)

Same LoRA teacher/student pair as RQ2. Only the distilled LoRA representation changes.

- `ax`: low-rank bottleneck
- `bax`: adapter output
- `ax_bax`: optional combined arm

```bash
python experiments/RQ3/run.py --pipelines ax bax
python experiments/RQ3/run.py --pipelines ax_bax
```

All arms use the shared fixed seed, split manifests, config, and metrics schema. Existing outputs are not replaced unless `--overwrite` is passed.
