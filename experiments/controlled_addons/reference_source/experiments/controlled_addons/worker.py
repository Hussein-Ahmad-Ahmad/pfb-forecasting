"""One isolated RTX4090 training job. Called by the sequential runner only."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from .design import DATASETS, ROOT, digest, file_hash, source_hashes, validate_spec, software_versions
from .models import build_model, configuration
from .storage import write_csv, write_json
from utils.result_provenance import write_result_provenance


def require_gpu():
    if not torch.cuda.is_available() or "RTX 4090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("These experiments require RTX4090 CUDA. No CPU/other-GPU fallback.")
    torch.cuda.set_device(0)
    return torch.device("cuda:0")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def loader_for(spec, split):
    from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
    info = DATASETS[spec["dataset"]]
    cls = {"ETTh1": Dataset_ETT_hour, "ETTh2": Dataset_ETT_hour,
           "ETTm1": Dataset_ETT_minute, "ETTm2": Dataset_ETT_minute}.get(info["loader"], Dataset_Custom)
    data = cls(args=SimpleNamespace(augmentation_ratio=0), root_path=str(ROOT / "data"),
               flag=split, size=[336, 96, 192], features="M", data_path=info["file"],
               target="OT", scale=True, timeenc=1, freq=info["freq"])
    if len(data) <= 0 or data.data_x.shape[-1] != info["channels"]:
        raise ValueError("Dataset shape/window count mismatch")
    generator = torch.Generator().manual_seed(spec["seed"])
    return DataLoader(data, batch_size=info["batch"], shuffle=split == "train",
                      num_workers=0, drop_last=False, generator=generator)


def predict_batch(model, batch, device):
    x, y, x_mark, y_mark = [tensor.float().to(device) for tensor in batch]
    decoder = torch.cat([y[:, :96], torch.zeros_like(y[:, -192:])], dim=1)
    prediction = model(x, x_mark, decoder, y_mark)
    target = y[:, -192:]
    if prediction.shape != target.shape:
        raise ValueError(f"Forecast/target shape mismatch: {prediction.shape} vs {target.shape}")
    return prediction, target


def evaluate(model, loader, device):
    """Streaming element-weighted metrics; no giant prediction arrays or PDFs."""
    model.eval()
    sums = np.zeros(4, dtype=np.float64)
    count = 0
    with torch.inference_mode():
        for batch in loader:
            prediction, target = predict_batch(model, batch, device)
            if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
                raise ValueError("Non-finite forecast/target")
            difference = prediction.double() - target.double()
            ratio = difference / target.double().abs().clamp_min(1e-8)
            sums += np.array([difference.abs().sum().item(), difference.square().sum().item(),
                              ratio.abs().sum().item(), ratio.square().sum().item()])
            count += target.numel()
    if not count:
        raise ValueError("Empty evaluation loader")
    mae, mse, mape, mspe = sums / count
    return np.array([mae, mse, math.sqrt(mse), mape, mspe], dtype=np.float64)


def verify_environment(plan, spec):
    validate_spec(spec)
    if digest(plan["environment"]["protocol"]) != spec["protocol_sha256"]:
        raise ValueError("Protocol fingerprint mismatch")
    if digest(software_versions()) != spec["software_sha256"]:
        raise ValueError("Python/package/CUDA versions differ from the frozen plan")
    if digest(source_hashes()) != spec["source_sha256"]:
        raise ValueError("Training source changed since plan creation; use a new campaign directory")
    if file_hash(ROOT / "data" / DATASETS[spec["dataset"]]["file"]) != spec["data_sha256"]:
        raise ValueError("Dataset changed since plan creation")


def train_one(plan, spec, folder):
    device = require_gpu()  # Enforced in BOTH parent and worker.
    verify_environment(plan, spec)
    folder = Path(folder).resolve()
    expected_parent = Path(plan["output"]).resolve() / "runs" / spec["run_id"]
    if folder.parent != expected_parent or not folder.name.startswith("attempt_") or any(folder.iterdir()):
        raise ValueError("Worker requires a new empty attempt directory under this run")
    started = time.perf_counter()
    seed_everything(spec["seed"])
    torch.set_num_threads(1)
    model = build_model(spec).float().to(device)
    cfg = configuration(spec["dataset"], spec["params"])
    train_loader, val_loader = loader_for(spec, "train"), loader_for(spec, "val")
    # Test data are NEVER constructed here for HPO selection.
    protocol = plan["environment"]["protocol"]
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    best, stale, history, best_epoch = float("inf"), 0, [], None
    torch.cuda.synchronize()
    training_started = time.perf_counter()
    for epoch in range(1, protocol["train_epochs"]+1):
        model.train()
        total_loss, count = 0.0, 0
        epoch_started = time.perf_counter()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction, target = predict_batch(model, batch, device)
            loss = torch.nn.functional.mse_loss(prediction, target)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            optimizer.step()
            total_loss += loss.item()*target.numel()
            count += target.numel()
        validation_mse = float(evaluate(model, val_loader, device)[1])
        improved = validation_mse < best
        if improved:
            best, stale, best_epoch = validation_mse, 0, epoch
            temp = folder / "checkpoint.pth.tmp"
            torch.save(model.state_dict(), temp)
            temp.replace(folder / "checkpoint.pth")
        else:
            stale += 1
        history.append(dict(epoch=epoch, train_mse=total_loss/count, validation_mse=validation_mse,
                            learning_rate=optimizer.param_groups[0]["lr"],
                            seconds=time.perf_counter()-epoch_started, best_epoch=best_epoch))
        write_csv(folder / "epoch_history.csv", history)
        print(f"epoch={epoch} train={total_loss/count:.8f} val={validation_mse:.8f} best={best_epoch}; test=withheld", flush=True)
        if stale >= protocol["patience"]:
            break
        for group in optimizer.param_groups:
            group["lr"] = cfg.learning_rate * 0.5**(epoch-1)
    torch.cuda.synchronize()
    training_seconds = time.perf_counter()-training_started
    model.load_state_dict(torch.load(folder / "checkpoint.pth", map_location=device, weights_only=True))
    metric_loader = val_loader if spec["split"] == "val" else loader_for(spec, "test")
    metrics = evaluate(model, metric_loader, device)
    with (folder / "metrics.npy.tmp").open("wb") as stream:
        np.save(stream, metrics, allow_pickle=False)
    (folder / "metrics.npy.tmp").replace(folder / "metrics.npy")
    args = SimpleNamespace(device=device, use_gpu=True, gpu_type="cuda", model_id=spec["run_id"],
                           model=spec["family"], data=DATASETS[spec["dataset"]]["loader"],
                           pred_len=192, seq_len=336, label_len=96, seed=spec["seed"])
    write_result_provenance(folder, args, spec["split"])
    torch.cuda.synchronize()
    summary = dict(best_epoch=best_epoch, epochs_completed=len(history), best_validation_mse=best,
                   train_samples=len(train_loader.dataset), validation_samples=len(val_loader.dataset),
                   training_validation_wall_seconds=training_seconds,
                   process_wall_seconds=time.perf_counter()-started,
                   allocated_device_wall_hours=(time.perf_counter()-started)/3600,
                   compute_note="Occupied RTX4090 wall time, not active-kernel time; no equal-GPU-hour claim",
                   parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                   test_ever_loaded=spec["split"] == "test", torch_version=torch.__version__,
                   cuda_version=torch.version.cuda, cudnn_version=torch.backends.cudnn.version())
    write_json(folder / "training_summary.json", summary)
    filenames = ("checkpoint.pth", "metrics.npy", "run_provenance.json", "epoch_history.csv", "training_summary.json")
    write_json(folder / "complete.json", dict(spec=spec, spec_sha256=digest(spec),
               sha256={name: file_hash(folder / name) for name in filenames}, summary=summary))
    print(f"COMPLETED {spec['run_id']} {spec['split']} MSE={metrics[1]:.9f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--attempt", required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    specs = [spec for spec in plan["runs"] if spec["run_id"] == args.run_id]
    if len(specs) != 1:
        raise ValueError("Worker run ID must occur exactly once in frozen plan")
    train_one(plan, specs[0], args.attempt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
