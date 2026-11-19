from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_pipelines
from src.paths import FIGURES_DIR, METRICS_DIR, TABLES_DIR, ensure_layout
from src.plots import TrainingMetrics, plot_comparison_bars


def _load_runs(rq: str) -> dict[str, dict]:
    folder = METRICS_DIR / rq
    if not folder.exists():
        return {}
    runs = {}
    for path in folder.glob("*.json"):
        runs[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    return runs


def _curves(rq: str, runs: dict[str, dict]) -> None:
    for name, payload in runs.items():
        history = payload.get("history")
        if not history:
            continue
        metrics = TrainingMetrics.from_dict(history)
        metrics.plot(title=f"{rq} {name}", save_path=FIGURES_DIR / rq / f"{name}_curve.png")


def main() -> None:
    ensure_layout()
    pipelines = load_pipelines()
    rows = []

    for rq, spec in pipelines.items():
        runs = _load_runs(rq)
        _curves(rq, runs)
        for comparison, names in spec["comparisons"].items():
            scores = {}
            for name in names:
                if name in runs and "test_acc" in runs[name]:
                    scores[name] = runs[name]["test_acc"]
                    rows.append(
                        {
                            "comparison": comparison,
                            "pipeline": name,
                            "test_acc": runs[name]["test_acc"],
                            "best_val_acc": runs[name].get("best_val_acc"),
                        }
                    )
            if scores:
                plot_comparison_bars(
                    scores,
                    title=comparison,
                    save_path=FIGURES_DIR / rq / f"{comparison}_bars.png",
                    ylabel="Test accuracy",
                )
                print("Wrote", FIGURES_DIR / rq / f"{comparison}_bars.png")
            else:
                print(f"No metrics yet for {comparison}. Run the experiment scripts first.")

    if rows:
        import csv

        out = TABLES_DIR / "comparison_table.csv"
        with out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print("Wrote", out)


if __name__ == "__main__":
    main()
