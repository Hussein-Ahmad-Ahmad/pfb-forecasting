"""CPU-only safety checks for the original-grid continuation queue."""

import argparse
import errno
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from contextlib import redirect_stdout

import numpy as np

import run_full_p1_queue as queue
import run_review_gap_campaigns as gap

sys.path.insert(0, str(gap.ROOT))
from utils import result_provenance


class FullGridTests(unittest.TestCase):
    def test_original_scope_and_targeted_overlap(self):
        plans = queue.build_plans()
        self.assertEqual([len(runs) for runs in plans.values()], [180, 1080, 810])
        self.assertEqual(sum(map(len, plans.values())), 2070)
        all_ids = [run.model_id for runs in plans.values() for run in runs]
        self.assertEqual(len(set(all_ids)), 2070)
        datasets = ["ETTh2", "ETTm2", "Exchange", "Weather"]
        targets = {
            "ensemble-members": gap.ensemble_members(datasets, [192], queue.SEEDS),
            "lookback": gap.lookback(datasets, [720], queue.SEEDS[:3], [96, 336]),
            "rolling-origin": gap.rolling_origin(datasets, [192], queue.SEEDS[:3], queue.PREFIXES, prepare_files=False),
        }
        overlaps = [len({r.model_id for r in plans[name]} & {r.model_id for r in runs}) for name, runs in targets.items()]
        self.assertEqual(overlaps, [40, 0, 108])
        self.assertEqual(2070 - sum(overlaps), 1922)

    def test_planning_does_not_write_prefix_files(self):
        with patch.object(gap, "prepare_rolling_files") as prepare:
            queue.build_plans()
            prepare.assert_not_called()

    def test_lock_prevents_duplicate_queue(self):
        with tempfile.TemporaryDirectory(dir=gap.ROOT) as directory:
            lock = Path(directory) / "queue.lock"
            with queue.queue_lock(lock):
                with self.assertRaises(queue.QueueAlreadyRunning):
                    with queue.queue_lock(lock):
                        pass
            with queue.queue_lock(lock):
                pass

    def test_duplicate_launch_exits_cleanly_without_inspecting_or_training(self):
        with tempfile.TemporaryDirectory(dir=gap.ROOT) as directory:
            output = Path(directory)
            state_path = output / "queue.json"
            saved = json.dumps({"pid": 123, "active_campaign": "ensemble-members", "status": "running"})
            state_path.write_text(saved, encoding="utf-8")
            stdout = io.StringIO()
            with queue.queue_lock(output / "queue.lock"), patch.object(queue, "OUT", output), \
                    patch.object(sys, "argv", ["run_full_p1_queue.py"]), \
                    patch.object(queue, "run_locked_queue") as run, \
                    patch.object(queue, "inventory") as inspect, redirect_stdout(stdout):
                self.assertEqual(queue.main(), 0)
                run.assert_not_called()
                inspect.assert_not_called()
            self.assertIn("Already running (PID 123", stdout.getvalue())
            self.assertNotIn("Traceback", stdout.getvalue())
            self.assertEqual(state_path.read_text(encoding="utf-8"), saved)

    def test_genuine_lock_failure_is_not_misreported_as_duplicate(self):
        if queue.os.name != "nt":
            self.skipTest("Windows-specific lock failure check")
        with tempfile.TemporaryDirectory(dir=gap.ROOT) as directory:
            with patch("msvcrt.locking", side_effect=OSError(errno.EINVAL, "Invalid lock operation")):
                with self.assertRaises(OSError) as error:
                    with queue.queue_lock(Path(directory) / "queue.lock"):
                        pass
            self.assertNotIsInstance(error.exception, queue.QueueAlreadyRunning)

    def test_unlocked_launch_enters_queue_once(self):
        with tempfile.TemporaryDirectory(dir=gap.ROOT) as directory:
            with patch.object(queue, "OUT", Path(directory)), \
                    patch.object(sys, "argv", ["run_full_p1_queue.py"]), \
                    patch.object(queue, "run_locked_queue", return_value=0) as run:
                self.assertEqual(queue.main(), 0)
                run.assert_called_once()

    def test_plan_only_is_read_only_and_allowed_during_training(self):
        with tempfile.TemporaryDirectory(dir=gap.ROOT) as directory:
            output = Path(directory)
            with queue.queue_lock(output / "queue.lock"), patch.object(queue, "OUT", output), \
                    patch.object(sys, "argv", ["run_full_p1_queue.py", "--plan-only"]), \
                    patch.object(queue, "print_inventory") as inspect, \
                    patch.object(queue, "run_locked_queue") as run:
                self.assertEqual(queue.main(), 0)
                inspect.assert_called_once()
                run.assert_not_called()
            self.assertFalse((output / "queue.json").exists())


class ResultValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=gap.ROOT)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.root_patch = patch.object(gap, "ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.run_spec = gap.ensemble_members(["ETTh2"], [192], [2021])[0]
        folder = self.root / "results" / f"long_term_forecast_{self.run_spec.model_id}_test"
        folder.mkdir(parents=True)
        self.metric = folder / "metrics.npy"
        np.save(self.metric, [0.1, 0.2, 0.3, 0.4, 0.5])
        for name in ("pred.npy", "true.npy"):
            np.save(folder / name, np.zeros((2, 192, 7), dtype=np.float32))
        self.provenance = {
            "model_id": self.run_spec.model_id, "model": "PatchTST", "horizon": 192,
            "seed": 2021, "seq_len": 336, "label_len": 96, "split": "test",
            "run_by_gpu_watermark": gap.WATERMARK, "actual_cuda_available": True,
            "actual_cuda_device": "NVIDIA GeForce RTX 4090",
        }
        self.save_provenance()

    def save_provenance(self):
        self.metric.with_name("run_provenance.json").write_text(json.dumps(self.provenance), encoding="utf-8")

    def test_valid_legacy_rtx4090_result_is_reused(self):
        self.assertEqual(gap.metrics_path(self.run_spec, strict=True), self.metric)

    def test_bad_metrics_are_rejected_and_preserved(self):
        np.save(self.metric, [0.1, np.nan, 0.3, 0.4, 0.5])
        with self.assertRaises(ValueError):
            gap.metrics_path(self.run_spec, strict=True)
        self.assertTrue(self.metric.exists())

    def test_other_gpu_is_not_a_paired_reference(self):
        self.provenance["actual_cuda_device"] = "NVIDIA GeForce RTX 2080 Ti"
        self.save_provenance()
        with self.assertRaises(ValueError):
            gap.metrics_path(self.run_spec, strict=True)

    def test_wrong_seed_is_rejected(self):
        self.provenance["seed"] = 2022
        self.save_provenance()
        with self.assertRaises(ValueError):
            gap.metrics_path(self.run_spec, strict=True)

    def test_missing_ensemble_arrays_are_not_skipped(self):
        self.metric.with_name("pred.npy").unlink()
        with self.assertRaises(FileNotFoundError):
            gap.metrics_path(self.run_spec, strict=True)

    def test_numeric_export_preserves_scope_and_tags(self):
        out = self.root / "export.csv"
        queue.export_metrics(out, {"ensemble-members": [self.run_spec]})
        import csv
        with out.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["tag"], "member_a")
        self.assertEqual(rows[0]["model_id"], self.run_spec.model_id)
        self.assertEqual(float(rows[0]["mse"]), 0.2)
        self.assertEqual(rows[0]["run_by_gpu_watermark"], gap.WATERMARK)


class ProvenanceTests(unittest.TestCase):
    def test_available_cuda_does_not_mislabel_cpu_execution(self):
        args = argparse.Namespace(device=result_provenance.torch.device("cpu"), use_gpu=False)
        with patch.object(result_provenance.torch.cuda, "is_available", return_value=True):
            result = result_provenance.build_result_provenance(args, "test")
        self.assertEqual(result["run_by_gpu_watermark"], "EXP_RUN_BY_CPU")
        self.assertFalse(result["actual_cuda_used"])

    def test_rtx4090_execution_is_watermarked(self):
        args = argparse.Namespace(device=result_provenance.torch.device("cuda:0"), use_gpu=True)
        with patch.object(result_provenance.torch.cuda, "is_available", return_value=True), \
                patch.object(result_provenance.torch.cuda, "current_device", return_value=0), \
                patch.object(result_provenance.torch.cuda, "get_device_name", return_value="NVIDIA GeForce RTX 4090"):
            result = result_provenance.build_result_provenance(args, "test")
        self.assertEqual(result["run_by_gpu_watermark"], gap.WATERMARK)
        self.assertTrue(result["actual_cuda_used"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
