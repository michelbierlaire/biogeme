#!/usr/bin/env python3
"""Compare Apollo timings with the existing JED Biogeme benchmark."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_reference(path: Path) -> dict[str, float]:
    with path.open(newline="") as stream:
        rows = csv.DictReader(stream)
        result: dict[str, float] = {}
        for row in rows:
            result[row["model"]] = float(row["3.3.4_wall_seconds"])
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("docs/biogeme-benchmark-results/66162235/timings.csv"),
    )
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()

    cases = {
        "b05a_normal_mixture": args.results_root / "b05a_normal_mixture.json",
        "b12_panel": args.results_root / "b12_panel.json",
    }
    reference = read_reference(args.reference)
    records: dict[str, dict[str, object]] = {}
    for model, path in cases.items():
        if not path.is_file():
            raise FileNotFoundError(f"Apollo result not found: {path}")
        records[model] = json.loads(path.read_text())

    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "model",
                "apollo_version",
                "apollo_wall_seconds",
                "biogeme_3.3.4_wall_seconds",
                "apollo_over_biogeme_3.3.4",
                "final_log_likelihood",
                "successful_estimation",
                "draws",
                "draw_type",
            ]
        )
        for model, record in records.items():
            apollo_time = float(record["wall_time_seconds"])
            biogeme_time = reference[model]
            writer.writerow(
                [
                    model,
                    record.get("apollo_version"),
                    apollo_time,
                    biogeme_time,
                    apollo_time / biogeme_time,
                    record.get("final_log_likelihood"),
                    record.get("successful_estimation"),
                    record.get("draws"),
                    record.get("draw_type"),
                ]
            )

    lines = [
        "# Apollo versus Biogeme benchmark",
        "",
        "The reference times are the Biogeme 3.3.4 runs from JED job 66162235.",
        "Apollo was run separately with the same filtered Swissmetro data and draw counts.",
        "The draw realizations are intentionally not required to match.",
        "",
        "| Model | Apollo version | Apollo wall time (s) | Biogeme 3.3.4 wall time (s) | Apollo / Biogeme | LL | Converged |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for model, record in records.items():
        apollo_time = float(record["wall_time_seconds"])
        biogeme_time = reference[model]
        ratio = apollo_time / biogeme_time
        ll = record.get("final_log_likelihood")
        ll_text = "n/a" if ll is None else f"{float(ll):.8f}"
        lines.append(
            f"| `{model}` | {record.get('apollo_version', 'unknown')} | "
            f"{apollo_time:.3f} | {biogeme_time:.3f} | {ratio:.3f} | "
            f"{ll_text} | {record.get('successful_estimation', False)} |"
        )
    lines.extend(
        [
            "",
            "The optimizers and derivative implementations are different, so this is a runtime comparison, not a claim of identical numerical paths.",
        ]
    )
    args.markdown.write_text("\n".join(lines) + "\n")
    print(f"Wrote {args.markdown}")
    print(f"Wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
