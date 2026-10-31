import json
import platform
import time
from pathlib import Path

import torch


RTX4090_WATERMARK = "EXP_RUN_BY_RTX4090_GPU"


def build_result_provenance(args, split):
    cuda_available = torch.cuda.is_available()
    requested_device = getattr(getattr(args, "device", None), "type", None)
    uses_cuda = cuda_available and bool(getattr(args, "use_gpu", False)) and (
        requested_device == "cuda" or (
            requested_device is None and getattr(args, "gpu_type", "cuda") == "cuda"
        )
    )
    cuda_device = None
    if uses_cuda:
        try:
            cuda_device = torch.cuda.get_device_name(torch.cuda.current_device())
        except Exception:
            cuda_device = "unknown CUDA device"

    if uses_cuda and cuda_device and "RTX 4090" in cuda_device:
        watermark = RTX4090_WATERMARK
    elif uses_cuda:
        watermark = "EXP_RUN_BY_OTHER_CUDA_GPU"
    elif requested_device == "mps":
        watermark = "EXP_RUN_BY_MPS"
    else:
        watermark = "EXP_RUN_BY_CPU"
    return {
        "run_by_gpu_watermark": watermark,
        "hardware_note": "Hardware label reflects the selected execution device; historical results are not relabelled.",
        "actual_cuda_available": cuda_available,
        "actual_cuda_used": uses_cuda,
        "actual_cuda_device": cuda_device,
        "split": split,
        "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "model_id": getattr(args, "model_id", None),
        "model": getattr(args, "model", None),
        "dataset": getattr(args, "data", None),
        "horizon": getattr(args, "pred_len", None),
        "seed": getattr(args, "seed", None),
        "seq_len": getattr(args, "seq_len", None),
        "label_len": getattr(args, "label_len", None),
    }


def write_result_provenance(folder_path, args, split):
    path = Path(folder_path) / "run_provenance.json"
    payload = build_result_provenance(args, split)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
