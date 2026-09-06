"""Branch-specific corruption analysis using the four pinned PFB checkpoints."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path

import torch

from diagnostic_common import (
    append_row,
    apply_gaussian,
    apply_missingness,
    evaluate,
    load_model,
    load_pins,
    read_completed,
    test_loader,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "tables" / "branch_corruption.csv"
KEY_FIELDS = ["Dataset", "Model", "Branch_Mode", "Corruption", "Severity", "Corruption_Seed"]
FIELDS = KEY_FIELDS + ["MSE", "MAE", "Realized_Rate", "Elements", "Checkpoint_ID", "Checkpoint_SHA256"]


def csv_items(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


@contextmanager
def branch_mode(model: torch.nn.Module, mode: str):
    hooks = []

    def zero_output(_module, _inputs, output):
        if isinstance(output, tuple):
            return (torch.zeros_like(output[0]), *output[1:])
        return torch.zeros_like(output)

    if mode == "primary-only":
        hooks.append(model.secondary_encoder.register_forward_hook(zero_output))
    elif mode == "secondary-only":
        hooks.append(model.patch_encoder.register_forward_hook(zero_output))
    elif mode != "both":
        raise ValueError(f"Unknown branch mode: {mode}")
    try:
        yield
    finally:
        for hook in hooks:
            hook.remove()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corruptions", default="gaussian,mcar,block")
    parser.add_argument("--severities", default="0.1,0.2,0.3")
    parser.add_argument("--corruption-seeds", default="11,22,33")
    parser.add_argument("--branch-modes", default="both,primary-only,secondary-only")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=24)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--max-jobs", type=int, default=0)
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    corruptions = csv_items(args.corruptions)
    severities = [float(value) for value in csv_items(args.severities)]
    corruption_seeds = [int(value) for value in csv_items(args.corruption_seeds)]
    modes = csv_items(args.branch_modes)
    invalid = set(corruptions) - {"gaussian", "mcar", "block"}
    if invalid:
        raise SystemExit(f"Unsupported corruptions: {sorted(invalid)}")

    pins = load_pins(["PFB-Direct", "PFB-Projected"])
    jobs = []
    for pin in pins:
        for mode in modes:
            jobs.append((pin, mode, "clean", 0.0, 0))
            for corruption in corruptions:
                for severity in severities:
                    for seed in corruption_seeds:
                        jobs.append((pin, mode, corruption, severity, seed))

    print(f"Branch-corruption evaluations planned: {len(jobs)}")
    if args.dry_run:
        for pin, mode, corruption, severity, seed in jobs:
            print(f"{pin['public_model']} {pin['dataset']} {mode} {corruption} {severity:g} seed={seed}")
        return 0

    completed = read_completed(args.output, KEY_FIELDS) if args.skip_completed else set()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    executed = 0
    for pin in pins:
        model_jobs = [job for job in jobs if job[0] is pin]
        if all(
            tuple(str(value) for value in (pin["dataset"], pin["public_model"], mode, corruption, severity, seed))
            in completed
            for _, mode, corruption, severity, seed in model_jobs
        ):
            continue
        model_args, loader = test_loader(pin, args.batch_size)
        model = load_model(pin, model_args, device)
        for _, mode, corruption, severity, seed in model_jobs:
            key = tuple(str(value) for value in (pin["dataset"], pin["public_model"], mode, corruption, severity, seed))
            if key in completed:
                continue
            if args.max_jobs and executed >= args.max_jobs:
                return 0
            if corruption == "clean":
                transform = lambda x, _batch: x
            elif corruption == "gaussian":
                transform = lambda x, batch, s=severity, r=seed: apply_gaussian(x, s, r, batch)
            else:
                transform = lambda x, batch, c=corruption, s=severity, r=seed: apply_missingness(
                    x, c, s, r, batch, args.block_size
                )
            with branch_mode(model, mode):
                metrics = evaluate(model, loader, model_args, device, transform, args.max_batches)
            row = {
                "Dataset": pin["dataset"], "Model": pin["public_model"], "Branch_Mode": mode,
                "Corruption": corruption, "Severity": severity, "Corruption_Seed": seed,
                **metrics, "Checkpoint_ID": pin["folder"], "Checkpoint_SHA256": pin["sha256"],
            }
            append_row(args.output, row, FIELDS)
            executed += 1
            print(
                f"[{executed}] {pin['public_model']} {pin['dataset']} {mode} "
                f"{corruption} {severity:g}: MSE={metrics['MSE']:.6f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
