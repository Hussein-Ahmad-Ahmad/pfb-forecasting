"""Declared candidate spaces, contrasts, and content-addressed run identity."""
from __future__ import annotations

import hashlib
import json
import copy
import platform
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results_analysis" / "controlled_addons_v1"
DATASETS = {
    "ETTh1": dict(loader="ETTh1", file="ETTh1.csv", freq="h", channels=7, batch=32),
    "ETTh2": dict(loader="ETTh2", file="ETTh2.csv", freq="h", channels=7, batch=32),
    "ETTm1": dict(loader="ETTm1", file="ETTm1.csv", freq="t", channels=7, batch=32),
    "ETTm2": dict(loader="ETTm2", file="ETTm2.csv", freq="t", channels=7, batch=32),
    "Exchange": dict(loader="custom", file="exchange_rate.csv", freq="d", channels=8, batch=32),
    "Weather": dict(loader="custom", file="weather.csv", freq="h", channels=21, batch=16),
}
FAMILIES = ("PatchTST", "PFB-Direct", "PFB-Projected", "DLinear", "iTransformer", "TiDE", "TimeXer")
SEEDS = (2021, 2022, 2023, 2024, 2025)
PROTOCOL = dict(
    version="controlled_addons_v1", seq_len=336, label_len=96, horizon=192,
    train_epochs=100, patience=10, optimizer="Adam", loss="MSE",
    lr_schedule="type1_epoch_end", selection_metric="element_weighted_validation_MSE",
    test_during_training=False, selection_seed=2020, seeds=list(SEEDS),
    scale="training_split_StandardScaler_no_inverse", features="M",
    validation_shuffle=False, train_shuffle="independent_seeded_generator",
    gpu="NVIDIA RTX 4090", precision="float32_no_AMP",
    metrics_reduction="float64_element_weighted; MAPE/MSPE denominator abs(true).clamp_min(1e-8)",
)
# lhs - rhs: negative relative MSE favors lhs. These are not interchangeable
# tests of a single universal mechanism. Multiplicity spans the declared set.
CONTRASTS = (
    ("projected34", "linear_d_reg", "GELU in the final module; same head/parameters", "primary"),
    ("projected34", "PatchTST_budget", "Budget-controlled complete configurations", "primary"),
    ("gelu_d", "linear_d", "GELU without fusion Dropout/LayerNorm", "secondary"),
    ("gelu_d_lift", "linear_d_lift", "GELU with common 2d head", "secondary"),
    ("gelu_2d", "linear_2d", "GELU at uncompressed width", "secondary"),
    ("linear_d_lift", "linear_2d", "Linear bottleneck, common head; capacity differs", "secondary"),
    ("gelu_d_lift", "gelu_2d", "Nonlinear bottleneck, common head; capacity differs", "secondary"),
    ("linear_d_lift", "linear_d", "Head interface under fixed linear bottleneck", "secondary"),
    ("gelu_d_lift", "gelu_d", "Head interface under fixed nonlinear bottleneck", "secondary"),
    ("projected34", "gelu_d", "Fusion Dropout+LayerNorm bundle", "secondary"),
    ("projected34", "concat34", "Projected vs unequal-depth direct concat; bundled", "secondary"),
    ("projected34", "PatchTST_base", "Complete configuration vs base PatchTST", "secondary"),
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def source_hashes():
    files = set()
    for directory in ("models", "layers", "utils", "data_provider"):
        files.update((ROOT / directory).rglob("*.py"))
    files.update(Path(__file__).parent / name for name in ("design.py", "models.py", "worker.py"))
    return {f.relative_to(ROOT).as_posix(): file_hash(f) for f in sorted(files)}


def software_versions():
    import torch
    return dict(python=platform.python_version(), torch=str(torch.__version__),
                numpy=version("numpy"), pandas=version("pandas"),
                scipy=version("scipy"), sklearn=version("scikit-learn"),
                compiled_cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version())


def effective_params(family, overrides=None):
    base = dict(learning_rate=0.0001, dropout=0.1, d_model=128, n_heads=8,
                e_layers=3, d_ff=512, patch_len=16, stride=8)
    if family == "DLinear":
        base = dict(learning_rate=0.001, moving_avg=25, individual=False)
    elif family == "TiDE":
        base["d_layers"] = 1
    base.update(overrides or {})
    active = {
        "DLinear": {"learning_rate", "moving_avg", "individual"},
        "TiDE": {"learning_rate", "dropout", "d_model", "e_layers", "d_layers", "d_ff"},
        "iTransformer": {"learning_rate", "dropout", "d_model", "n_heads", "e_layers", "d_ff"},
        "TimeXer": {"learning_rate", "dropout", "d_model", "n_heads", "e_layers", "d_ff", "patch_len"},
    }.get(family, set(base))
    return {k: (float(v) if k in {"learning_rate", "dropout"} else bool(int(v)) if k == "individual" else int(v))
            for k, v in base.items() if k in active}


def candidates(family):
    """Twelve unique effective configurations per family, not equal GPU-hours."""
    from scripts.paper.run_review_campaign import grid_candidate
    result = []
    def add(overrides):
        value = effective_params(family, overrides)
        if value not in result:
            result.append(value)
    for trial in range(6):
        add(grid_candidate(family, trial))
    if family == "DLinear":
        for moving in (7, 49, 25):
            for lr in (0.0003, 0.001, 0.003, 0.01):
                add(dict(learning_rate=lr, moving_avg=moving))
    elif family == "TiDE":
        for width, depth, lr in ((64, 1, 0.0001), (256, 2, 0.0001), (128, 3, 0.0003),
                                 (256, 3, 0.0003), (64, 2, 0.001), (256, 1, 0.001)):
            add(dict(d_model=width, e_layers=depth, d_layers=1, d_ff=4*width,
                     learning_rate=lr, dropout=0.1))
    else:
        for width, depth, lr, patch in ((128, 3, 0.0001, 64), (256, 3, 0.0001, 16),
                                        (128, 4, 0.0003, 32), (64, 4, 0.0003, 16),
                                        (256, 2, 0.001, 32), (256, 4, 0.0003, 64),
                                        (192, 2, 0.0001, 16)):
            add(dict(d_model=width, e_layers=depth, d_ff=4*width, n_heads=8,
                     learning_rate=lr, dropout=0.1, patch_len=patch, stride=patch//2))
    result = result[:12]
    if len(result) != 12 or len({canonical(v) for v in result}) != 12:
        raise ValueError(f"Invalid effective candidate diversity: {family}")
    return result


def environment(datasets, seeds=SEEDS):
    protocol = copy.deepcopy(PROTOCOL)
    protocol["seeds"] = list(seeds)
    return dict(protocol=protocol, software=software_versions(), source_sha256=source_hashes(),
                data_sha256={ds: file_hash(ROOT / "data" / DATASETS[ds]["file"]) for ds in datasets})


def run_spec(env, study, dataset, family, variant, seed, params, split="test"):
    spec = dict(study=study, dataset=dataset, family=family, variant=variant,
                seed=seed, params=params, split=split,
                protocol_sha256=digest(env["protocol"]),
                software_sha256=digest(env.get("software", {})),
                source_sha256=digest(env["source_sha256"]),
                data_sha256=env["data_sha256"][dataset])
    spec["run_id"] = f"{study}_{dataset}_{variant}_S{seed}_{digest(spec)[:20]}"
    return spec


def validate_spec(spec):
    required = {"study", "dataset", "family", "variant", "seed", "params", "split",
                "protocol_sha256", "software_sha256", "source_sha256", "data_sha256", "run_id"}
    if set(spec) != required or spec["dataset"] not in DATASETS or spec["family"] not in FAMILIES:
        raise ValueError("Invalid run specification")
    body = {k: v for k, v in spec.items() if k != "run_id"}
    expected = f"{spec['study']}_{spec['dataset']}_{spec['variant']}_S{spec['seed']}_{digest(body)[:20]}"
    if spec["run_id"] != expected or any(c in spec["run_id"] for c in ("/", "\\", ":")):
        raise ValueError("Run identity does not match its complete configuration")
    if spec["study"] == "hpo":
        if spec["split"] != "val" or spec["seed"] != 2020:
            raise ValueError("Selection must use validation and seed 2020")
    elif spec["study"] not in {"projected", "preprocessing", "hpo_test"} or spec["split"] != "test" or spec["seed"] == 2020:
        raise ValueError("Invalid test study/seed/split")


def projected_plan(env, datasets, seeds, tolerance=0.01):
    from .models import VARIANTS, profile_model, match_patch_budget, patch_budget
    target_spec = run_spec(env, "projected", "ETTh2", "PFB-Projected", "projected34", 2020, effective_params("PFB-Projected"))
    target = profile_model(target_spec)
    unit = {**target, "macs_per_sample": target["macs_per_sample"] // DATASETS["ETTh2"]["channels"]}
    match = match_patch_budget(unit, tolerance)
    comparison = patch_budget(match["params"])
    specs, profiles = [], []
    for ds in datasets:
        for variant in (*VARIANTS, "PatchTST_base", "PatchTST_budget"):
            family = "PatchTST" if variant.startswith("PatchTST") else "PFB-Projected"
            params = effective_params(family, match["params"] if variant == "PatchTST_budget" else {})
            template = run_spec(env, "projected", ds, family, variant, seeds[0], params)
            # The family is channel independent; rescale the verified ETTh2 CPU
            # profile instead of repeating expensive forwards for all 3 datasets.
            profile_template = {**template, "dataset": "ETTh2"}
            profile = profile_model(profile_template)
            profile["macs_per_sample"] = profile["macs_per_sample"] // 7 * DATASETS[ds]["channels"]
            if variant == "PatchTST_budget":
                if profile["parameters"] != comparison["parameters"] or profile["macs_per_sample"] != comparison["macs_per_sample"]*DATASETS[ds]["channels"]:
                    raise ValueError("Analytic match did not agree with executed-module profile")
            profiles.append(dict(dataset=ds, variant=variant, **profile))
            specs.extend(run_spec(env, "projected", ds, family, variant, seed, params) for seed in seeds)
    return specs, dict(match=match, profiles=profiles, contrasts=CONTRASTS,
                       note="Within 1% (or declared tolerance), not exact integer equality. Only final Projected is budget-matched; other ablations report their costs.")


def preprocessing_plan(env, datasets, seeds):
    return [run_spec(env, "preprocessing", ds, family, variant, seed, effective_params(family))
            for ds in datasets for family in FAMILIES[:4] for variant in ("norm_off", "norm_on") for seed in seeds]


def hpo_plan(env, datasets):
    return [run_spec(env, "hpo", ds, family, family+"_"+digest(params)[:10], 2020, params, "val")
            for ds in datasets for family in FAMILIES for params in candidates(family)]
