from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_PY = ROOT / "run.py"
DATA_ROOT = ROOT / "data"

STANDARD_DATASETS = {
    "ETTm1": {"data": "ETTm1", "data_path": "ETTm1.csv", "enc_in": 7, "batch_size": 16, "freq": "t"},
    "ETTm2": {"data": "ETTm2", "data_path": "ETTm2.csv", "enc_in": 7, "batch_size": 16, "freq": "t"},
    "ETTh1": {"data": "ETTh1", "data_path": "ETTh1.csv", "enc_in": 7, "batch_size": 16, "freq": "h"},
    "ETTh2": {"data": "ETTh2", "data_path": "ETTh2.csv", "enc_in": 7, "batch_size": 16, "freq": "h"},
    "Exchange": {"data": "custom", "data_path": "exchange_rate.csv", "enc_in": 8, "batch_size": 16, "freq": "d"},
    "Weather": {"data": "custom", "data_path": "weather.csv", "enc_in": 21, "batch_size": 8, "freq": "10min"},
}

ILLNESS = {
    "Illness": {"data": "custom", "data_path": "national_illness.csv", "enc_in": 7, "batch_size": 16, "freq": "w"}
}

MODEL_CONFIGS = {
    "PatchTST": {"d_model": 128, "n_heads": 8, "e_layers": 3, "d_layers": 1, "d_ff": 512, "dropout": 0.1, "patch_len": 16, "stride": 8},
    "DLinear": {"d_model": 512, "n_heads": 8, "e_layers": 2, "d_layers": 1, "d_ff": 2048, "dropout": 0.1},
    "TiDE": {"d_model": 128, "n_heads": 8, "e_layers": 2, "d_layers": 1, "d_ff": 512, "dropout": 0.1},
    "TimeXer": {"d_model": 128, "n_heads": 8, "e_layers": 2, "d_layers": 1, "d_ff": 512, "dropout": 0.1},
    "iTransformer": {"d_model": 128, "n_heads": 8, "e_layers": 2, "d_layers": 1, "d_ff": 512, "dropout": 0.1},
    "PatchFusionBERT_v0": {"d_model": 128, "n_heads": 8, "e_layers": 3, "d_layers": 1, "d_ff": 512, "dropout": 0.1, "patch_len": 16, "stride": 8},
    "PatchFusionBERT_v2": {"d_model": 128, "n_heads": 8, "e_layers": 3, "d_layers": 1, "d_ff": 512, "dropout": 0.1, "patch_len": 16, "stride": 8},
    "PatchFusionBERT_BERTOnly": {"d_model": 128, "n_heads": 8, "e_layers": 3, "d_layers": 1, "d_ff": 512, "dropout": 0.1, "patch_len": 16, "stride": 8},
}

PHASES = {
    "h96": {"pred_len": 96, "label_len": 48, "suffix": "100epoch"},
    "h192": {"pred_len": 192, "label_len": 96, "suffix": "H192"},
    "h336": {"pred_len": 336, "label_len": 168, "suffix": "H336"},
}

ILLNESS_PHASES = {
    "h96": {"pred_len": 24, "label_len": 18, "suffix": "Illness24"},
    "h192": {"pred_len": 48, "label_len": 18, "suffix": "Illness48"},
    "h336": {"pred_len": 60, "label_len": 18, "suffix": "Illness60"},
}


def build_command(model: str, dataset: str, phase: str) -> list[str]:
    if dataset in STANDARD_DATASETS:
        ds = STANDARD_DATASETS[dataset]
    else:
        ds = ILLNESS[dataset]
    cfg = MODEL_CONFIGS[model]
    if dataset == "Illness":
        phase_cfg = ILLNESS_PHASES[phase]
        seq_len = 104
    else:
        phase_cfg = PHASES[phase]
        seq_len = 336

    model_id = f"{model}_{phase_cfg['suffix']}_{dataset}_{phase_cfg['pred_len']}"
    cmd = [
        sys.executable,
        str(RUN_PY),
        "--task_name",
        "long_term_forecast",
        "--is_training",
        "1",
        "--root_path",
        str(DATA_ROOT),
        "--data_path",
        ds["data_path"],
        "--model_id",
        model_id,
        "--model",
        model,
        "--data",
        ds["data"],
        "--features",
        "M",
        "--seq_len",
        str(seq_len),
        "--label_len",
        str(phase_cfg["label_len"]),
        "--pred_len",
        str(phase_cfg["pred_len"]),
        "--enc_in",
        str(ds["enc_in"]),
        "--dec_in",
        str(ds["enc_in"]),
        "--c_out",
        str(ds["enc_in"]),
        "--d_model",
        str(cfg["d_model"]),
        "--n_heads",
        str(cfg["n_heads"]),
        "--e_layers",
        str(cfg["e_layers"]),
        "--d_layers",
        str(cfg["d_layers"]),
        "--d_ff",
        str(cfg["d_ff"]),
        "--dropout",
        str(cfg["dropout"]),
        "--factor",
        "3",
        "--freq",
        ds["freq"],
        "--batch_size",
        str(ds["batch_size"]),
        "--learning_rate",
        "0.0001",
        "--train_epochs",
        "100",
        "--patience",
        "10",
        "--itr",
        "1",
        "--num_workers",
        "0",
        "--des",
        "publish_main",
    ]
    if "patch_len" in cfg:
        cmd.extend(["--patch_len", str(cfg["patch_len"]), "--stride", str(cfg["stride"])])
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PatchFusionBERT publication benchmark grid.")
    parser.add_argument("--phase", choices=["h96", "h192", "h336", "all"], default="all")
    parser.add_argument("--models", default="all", help="Comma-separated subset of models")
    parser.add_argument("--datasets", default="all", help="Comma-separated subset of datasets")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    phases = ["h96", "h192", "h336"] if args.phase == "all" else [args.phase]
    models = list(MODEL_CONFIGS.keys()) if args.models == "all" else [m.strip() for m in args.models.split(",") if m.strip()]
    datasets = list(STANDARD_DATASETS.keys()) + ["Illness"] if args.datasets == "all" else [d.strip() for d in args.datasets.split(",") if d.strip()]

    total = 0
    for phase in phases:
        for dataset in datasets:
            for model in models:
                total += 1
                cmd = build_command(model, dataset, phase)
                print(f"[{total}] phase={phase} dataset={dataset} model={model}")
                print(" ".join(cmd))
                if not args.dry_run:
                    subprocess.run(cmd, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
