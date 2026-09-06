"""MCAR/MAR/MNAR/block missingness using all six SHA-256-pinned checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from diagnostic_common import (
    append_row,
    apply_missingness,
    evaluate,
    load_model,
    load_pins,
    read_completed,
    test_loader,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "tables" / "missingness_mechanisms.csv"
KEY_FIELDS = ["Dataset", "Model", "Mechanism", "Missing_Rate", "Corruption_Seed"]
FIELDS = KEY_FIELDS + ["MSE", "MAE", "Realized_Rate", "Elements", "Block_Size", "Checkpoint_ID", "Checkpoint_SHA256"]


def items(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mechanisms", default="mcar,mar,mnar,block")
    parser.add_argument("--rates", default="0.1,0.2,0.3")
    parser.add_argument("--corruption-seeds", default="11,22,33")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=24)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--max-jobs", type=int, default=0)
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    mechanisms = items(args.mechanisms)
    invalid = set(mechanisms) - {"mcar", "mar", "mnar", "block"}
    if invalid:
        raise SystemExit(f"Unsupported mechanisms: {sorted(invalid)}")
    rates = [float(value) for value in items(args.rates)]
    seeds = [int(value) for value in items(args.corruption_seeds)]
    pins = load_pins()
    jobs = []
    for pin in pins:
        jobs.append((pin, "clean", 0.0, 0))
        for mechanism in mechanisms:
            for rate in rates:
                for seed in seeds:
                    jobs.append((pin, mechanism, rate, seed))

    print(f"Missingness evaluations planned: {len(jobs)}")
    if args.dry_run:
        return 0
    completed = read_completed(args.output, KEY_FIELDS) if args.skip_completed else set()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    executed = 0
    for pin in pins:
        model_args, loader = test_loader(pin, args.batch_size)
        model = load_model(pin, model_args, device)
        for job_pin, mechanism, rate, seed in (job for job in jobs if job[0] is pin):
            key = tuple(str(value) for value in (pin["dataset"], pin["public_model"], mechanism, rate, seed))
            if key in completed:
                continue
            if args.max_jobs and executed >= args.max_jobs:
                return 0
            transform = lambda x, batch, m=mechanism, r=rate, s=seed: apply_missingness(
                x, m, r, s, batch, args.block_size
            )
            metrics = evaluate(model, loader, model_args, device, transform, args.max_batches)
            row = {
                "Dataset": pin["dataset"], "Model": pin["public_model"], "Mechanism": mechanism,
                "Missing_Rate": rate, "Corruption_Seed": seed, **metrics,
                "Block_Size": args.block_size, "Checkpoint_ID": pin["folder"],
                "Checkpoint_SHA256": pin["sha256"],
            }
            append_row(args.output, row, FIELDS)
            executed += 1
            print(
                f"[{executed}] {pin['public_model']} {pin['dataset']} {mechanism} "
                f"{rate:g}: MSE={metrics['MSE']:.6f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
