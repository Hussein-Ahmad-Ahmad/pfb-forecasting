"""CPU-only synthetic tests. No dataset training and no experiment GPU use."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from . import design
from .analyze import holm, interval, paired, summarize_pairs
from .models import (VARIANTS, FAMILIES, MechanismControl, NormalizationControl,
                     build_model, configuration, profile_model, patch_budget, match_patch_budget)
from .storage import completed, validate_receipt, write_json
from .worker import evaluate, require_gpu, predict_batch, train_one


def small_config():
    return configuration("ETTh2", dict(seq_len=32, label_len=8, pred_len=8,
                                       d_model=16, n_heads=4, d_ff=32,
                                       patch_len=8, stride=4, dropout=0.0))


def environment():
    return dict(protocol=copy.deepcopy(design.PROTOCOL), source_sha256={"fixture": "test"},
                data_sha256={ds: "fixture_data_"+ds for ds in design.DATASETS})


def spec(**kwargs):
    args = dict(study="projected", dataset="ETTh2", family="PFB-Projected",
                variant="projected34", seed=2021, params=design.effective_params("PFB-Projected"))
    args.update(kwargs)
    return design.run_spec(environment(), **args)


def receipt_fixture(folder, run):
    folder.mkdir(parents=True)
    np.save(folder / "metrics.npy", np.array([0.2, 0.09, 0.3, 0.1, 0.2]))
    torch.save({"fixture": torch.zeros(1)}, folder / "checkpoint.pth")
    provenance = dict(run_by_gpu_watermark="EXP_RUN_BY_RTX4090_GPU", actual_cuda_used=True,
                      actual_cuda_device="RTX 4090", split=run["split"], seed=run["seed"], model_id=run["run_id"])
    write_json(folder / "run_provenance.json", provenance)
    (folder / "epoch_history.csv").write_text("epoch,validation_mse\n1,0.09\n", encoding="utf-8")
    summary = dict(parameters=10, allocated_device_wall_hours=0.01, test_ever_loaded=run["split"] == "test")
    write_json(folder / "training_summary.json", summary)
    files = ("metrics.npy", "checkpoint.pth", "run_provenance.json", "epoch_history.csv", "training_summary.json")
    value = dict(spec=run, spec_sha256=design.digest(run), summary=summary,
                 sha256={name: design.file_hash(folder / name) for name in files})
    write_json(folder / "complete.json", value)
    return value


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_all_mechanism_shapes_and_gradients(self):
        for name in VARIANTS:
            with self.subTest(name=name):
                model = MechanismControl(small_config(), name)
                output = model(torch.randn(2, 32, 7))
                self.assertEqual(output.shape, (2, 8, 7))
                output.square().mean().backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_projected_control_reproduces_original(self):
        cfg = small_config()
        torch.manual_seed(7)
        control = MechanismControl(cfg, "projected34").eval()
        original = FAMILIES["PFB-Projected"].Model(cfg).eval()
        original.load_state_dict(control.base.state_dict())
        x = torch.randn(2, 32, 7)
        torch.testing.assert_close(control(x), original(x, None, None, None), rtol=0, atol=0)

    def test_on_reproduces_original_native_models(self):
        for family in ("PatchTST", "PFB-Direct", "PFB-Projected"):
            with self.subTest(family=family):
                control = NormalizationControl(small_config(), family, True).eval()
                x = torch.randn(2, 32, 7)
                torch.testing.assert_close(control(x), control.base(x, None, None, None), rtol=0, atol=0)

    def test_dlinear_off_reproduces_original(self):
        control = NormalizationControl(small_config(), "DLinear", False).eval()
        x = torch.randn(2, 32, 7)
        torch.testing.assert_close(control(x), control.base(x, None, None, None), rtol=0, atol=0)

    def test_normalization_pairs_have_identical_initial_parameters(self):
        for family in ("DLinear", "PatchTST", "PFB-Direct", "PFB-Projected"):
            torch.manual_seed(2021)
            off = NormalizationControl(small_config(), family, False)
            torch.manual_seed(2021)
            on = NormalizationControl(small_config(), family, True)
            for a, b in zip(off.parameters(), on.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            x = torch.randn(2, 32, 7)+3
            self.assertFalse(torch.equal(off(x), on(x)))

    def test_gelu_changes_no_parameters_or_head_geometry(self):
        for lhs, rhs in (("projected34", "linear_d_reg"), ("gelu_d", "linear_d"),
                         ("gelu_d_lift", "linear_d_lift"), ("gelu_2d", "linear_2d")):
            torch.manual_seed(10)
            a = MechanismControl(small_config(), lhs)
            torch.manual_seed(10)
            b = MechanismControl(small_config(), rhs)
            self.assertEqual(a.head_width, b.head_width)
            self.assertEqual(sum(p.numel() for p in a.parameters()), sum(p.numel() for p in b.parameters()))
            for first, second in zip(a.parameters(), b.parameters()):
                torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_same_wide_head_and_encoders_across_compression_conditions(self):
        torch.manual_seed(10)
        a = MechanismControl(small_config(), "gelu_d_lift")
        torch.manual_seed(10)
        b = MechanismControl(small_config(), "gelu_2d")
        for name in ("head", "patch_embedding", "patch_encoder", "secondary_encoder"):
            for pa, pb in zip(getattr(a.base, name).parameters(), getattr(b.base, name).parameters()):
                torch.testing.assert_close(pa, pb, rtol=0, atol=0)

    def test_macro_profile_matches_executed_modules(self):
        run = spec(family="PatchTST", variant="PatchTST_base", params=design.effective_params("PatchTST"))
        measured = profile_model(run)
        expected = patch_budget(run["params"], 7)
        self.assertEqual(measured["parameters"], expected["parameters"])
        self.assertEqual(measured["macs_per_sample"], expected["macs_per_sample"])

    def test_simultaneous_budget_match(self):
        target = profile_model(spec())
        match = match_patch_budget(dict(parameters=target["parameters"], macs_per_sample=target["macs_per_sample"]//7))
        for key in ("parameter_difference_pct", "mac_difference_pct"):
            self.assertLessEqual(abs(match[key]), 1)
        run = spec(family="PatchTST", variant="PatchTST_budget", params=design.effective_params("PatchTST", match["params"]))
        measured = profile_model(run)
        self.assertEqual(measured["parameters"], patch_budget(run["params"])["parameters"])

    def test_no_false_budget_match(self):
        with self.assertRaises(ValueError):
            match_patch_budget(dict(parameters=1, macs_per_sample=1))


class DesignTests(unittest.TestCase):
    def test_twelve_unique_active_candidates_each_family(self):
        for family in design.FAMILIES:
            values = design.candidates(family)
            self.assertEqual(len(values), 12)
            self.assertEqual(len({design.canonical(v) for v in values}), 12)
            if family in {"DLinear", "TiDE", "iTransformer"}:
                self.assertTrue(all("patch_len" not in v and "stride" not in v for v in values))

    def test_all_hpo_candidates_build_and_forecast(self):
        torch.set_num_threads(1)
        for family in design.FAMILIES:
            for params in design.candidates(family):
                with self.subTest(family=family, params=params):
                    model = build_model(spec(study="hpo", family=family, params=params)).eval()
                    with torch.inference_mode():
                        output = model(torch.randn(1, 336, 7), torch.zeros(1, 336, 4),
                                       torch.zeros(1, 288, 7), torch.zeros(1, 288, 4))
                    self.assertEqual(output.shape, (1, 192, 7))
                    self.assertTrue(torch.isfinite(output).all())

    def test_planned_counts_and_disjoint_hpo_test_seeds(self):
        env = environment()
        runs = design.hpo_plan(env, list(design.DATASETS))
        self.assertEqual(len(runs), 504)
        self.assertTrue(all(s["seed"] == 2020 and s["split"] == "val" for s in runs))
        self.assertEqual(len({s["run_id"] for s in runs}), 504)
        self.assertEqual(len(design.preprocessing_plan(env, ["ETTh2", "ETTm2", "Exchange", "Weather"], design.SEEDS)), 160)
        self.assertNotIn(2020, design.SEEDS)

    def test_identity_changes_when_training_configuration_changes(self):
        a, b = spec(), spec(params=design.effective_params("PFB-Projected", {"dropout": 0.2}))
        self.assertNotEqual(a["run_id"], b["run_id"])
        design.validate_spec(a)
        a["seed"] = 2022
        with self.assertRaises(ValueError):
            design.validate_spec(a)

    def test_selection_never_uses_test_split(self):
        run = spec(study="hpo", seed=2020, split="test")
        with self.assertRaises(ValueError):
            design.validate_spec(run)

    def test_no_cpu_fallback(self):
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaises(RuntimeError):
                require_gpu()


class ReceiptTests(unittest.TestCase):
    def test_resume_accepts_only_complete_intact_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = spec()
            folder = Path(tmp) / "runs" / run["run_id"] / "attempt_001"
            receipt_fixture(folder, run)
            self.assertAlmostEqual(completed(tmp, run)["mse"], 0.09)
            np.save(folder / "metrics.npy", np.array([0.2, 0.16, 0.4, 0.1, 0.2]))
            with self.assertRaises(ValueError):
                completed(tmp, run)

    def test_incomplete_attempt_is_not_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(completed(tmp, spec()))

    def test_wrong_gpu_is_rejected_even_with_matching_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "fixture"
            run = spec()
            receipt = receipt_fixture(folder, run)
            provenance = json.loads((folder / "run_provenance.json").read_text())
            provenance["actual_cuda_device"] = "RTX 2080 Ti"
            write_json(folder / "run_provenance.json", provenance)
            receipt["sha256"]["run_provenance.json"] = design.file_hash(folder / "run_provenance.json")
            write_json(folder / "complete.json", receipt)
            with self.assertRaises(ValueError):
                validate_receipt(folder, run)

    def test_duplicate_completions_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = spec()
            for i in (1, 2):
                receipt_fixture(Path(tmp) / "runs" / run["run_id"] / f"attempt_{i:03d}", run)
            with self.assertRaises(ValueError):
                completed(tmp, run)

    def test_frozen_files_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan.json"
            write_json(path, {"a": 1}, immutable=True)
            with self.assertRaises(ValueError):
                write_json(path, {"a": 2}, immutable=True)


class WorkflowTests(unittest.TestCase):
    def test_hpo_lock_requires_every_candidate(self):
        from .runner import lock_hpo
        with tempfile.TemporaryDirectory() as tmp:
            runs = design.hpo_plan(environment(), ["ETTh2"])
            runs = [s for s in runs if s["family"] == "PatchTST"]
            write_json(Path(tmp)/"plan.json", dict(action="hpo-select", runs=runs))
            with patch("experiments.controlled_addons.worker.verify_environment"):
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    lock_hpo(tmp)

    def test_hpo_lock_and_test_plan_use_validation_winners_only(self):
        from .runner import lock_hpo, test_hpo_plan
        with tempfile.TemporaryDirectory() as tmp:
            env = environment()
            runs = [s for s in design.hpo_plan(env, ["ETTh2", "Weather"]) if s["family"] == "PatchTST"]
            plan = dict(action="hpo-select", output=str(Path(tmp).resolve()), environment=env, runs=runs,
                        scope=dict(datasets=["ETTh2", "Weather"], seeds=list(design.SEEDS), families=["PatchTST"], tolerance=0.01))
            write_json(Path(tmp)/"plan.json", plan)
            for run in runs:
                receipt_fixture(Path(tmp)/"runs"/run["run_id"]/"attempt_001", run)
            with patch("experiments.controlled_addons.worker.verify_environment"):
                locked = lock_hpo(tmp)
                self.assertEqual(len(locked["winners"]), 2)
                tests = test_hpo_plan(tmp, list(design.SEEDS))
                self.assertEqual(len(tests["runs"]), 10)
                self.assertTrue(all(s["split"] == "test" and s["seed"] != 2020 for s in tests["runs"]))
                for item in locked["winners"]:
                    expected = min(s["run_id"] for s in runs if s["dataset"] == item["dataset"])
                    self.assertEqual(item["selection_run_id"], expected)
                with self.assertRaisesRegex(ValueError, "prospectively"):
                    test_hpo_plan(tmp, [2021])

    def test_synthetic_training_selection_never_constructs_test_loader(self):
        # Unit fixture only: patches explicitly bypass GPU/environment guards.
        # Its provenance remains CPU and can never pass the RTX4090 receipt check.
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(7, 7)
            def forward(self, x, *args):
                return self.linear(x[:, -192:])
        torch.set_num_threads(1)
        tensors = (torch.randn(2, 336, 7), torch.randn(2, 288, 7),
                   torch.zeros(2, 336, 4), torch.zeros(2, 288, 4))
        dataset = torch.utils.data.TensorDataset(*tensors)
        loader = torch.utils.data.DataLoader(dataset, batch_size=2)
        run = spec(study="hpo", family="PatchTST", variant="fixture", seed=2020, split="val")
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)/"runs"/run["run_id"]/"attempt_001"
            folder.mkdir(parents=True)
            env = environment()
            env["protocol"].update(train_epochs=1, patience=1)
            plan = dict(output=str(Path(tmp).resolve()), environment=env)
            with patch("experiments.controlled_addons.worker.require_gpu", return_value=torch.device("cpu")), \
                 patch("experiments.controlled_addons.worker.verify_environment"), \
                 patch("experiments.controlled_addons.worker.build_model", return_value=Tiny()), \
                 patch("experiments.controlled_addons.worker.loader_for", return_value=loader) as get_loader, \
                 patch("torch.cuda.is_available", return_value=False), patch("torch.cuda.synchronize"):
                train_one(plan, run, folder)
            self.assertEqual([call.args[1] for call in get_loader.call_args_list], ["train", "val"])
            receipt = json.loads((folder/"complete.json").read_text())
            self.assertFalse(receipt["summary"]["test_ever_loaded"])
            self.assertEqual(receipt["summary"]["best_epoch"], 1)
            with self.assertRaises(ValueError):
                validate_receipt(folder, run)


class AnalysisTests(unittest.TestCase):
    def test_complete_and_partial_analysis_workflows(self):
        from .analyze import analyze
        with tempfile.TemporaryDirectory() as tmp:
            names = (*VARIANTS, "PatchTST_base", "PatchTST_budget")
            runs = [spec(dataset=ds, variant=name, family="PatchTST" if name.startswith("PatchTST") else "PFB-Projected", seed=seed)
                    for ds in ("ETTh2", "Weather") for name in names for seed in (2021, 2022)]
            write_json(Path(tmp)/"plan.json", dict(action="projected", runs=runs))
            for run in runs:
                receipt_fixture(Path(tmp)/"runs"/run["run_id"]/"attempt_001", run)
            self.assertEqual(analyze(tmp), 0)
            table = pd.read_csv(Path(tmp)/"analysis/dataset_level_effects.csv")
            self.assertEqual(len(table), len(design.CONTRASTS))
            self.assertTrue(table.n_datasets.eq(2).all())
            # Remove a SPEC from the fixture plan rather than deleting files.
            # An added missing planned seed must invalidate complete-grid claims.
            runs.append(spec(dataset="ETTh2", seed=2023))
            write_json(Path(tmp)/"plan.json", dict(action="projected", runs=runs))
            self.assertEqual(analyze(tmp, allow_partial=True), 0)
            status = json.loads((Path(tmp)/"analysis/status.json").read_text())
            self.assertFalse(status["inferential_results_generated"])
            self.assertEqual(len(pd.read_csv(Path(tmp)/"analysis/dataset_level_effects.csv")), 0)

    def test_dataset_interval_does_not_treat_repeated_seeds_as_datasets(self):
        pairs = pd.DataFrame(dict(dataset=np.repeat(["a", "b", "c"], 5),
                                  effect_pct=np.repeat([-2., -1., 1.], 5), lhs="x", rhs="y"))
        first = summarize_pairs(pairs, "fixture", "primary")
        doubled = summarize_pairs(pd.concat([pairs, pairs]), "fixture", "primary")
        self.assertEqual(first["ci95_low_pct"], doubled["ci95_low_pct"])
        self.assertEqual(first["n_datasets"], 3)

    def test_holm(self):
        np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])

    def test_unbalanced_pairing_is_rejected(self):
        rows = [dict(dataset="a", seed=2021, variant="x"), dict(dataset="a", seed=2022, variant="y")]
        with self.assertRaises(ValueError):
            paired(pd.DataFrame(rows), "x", "y")

    def test_cross_software_pairing_is_rejected(self):
        rows = []
        for name, software in (("x", "version_a"), ("y", "version_b")):
            rows.append(dict(dataset="ETTh2", seed=2021, variant=name, mse=0.1, mae=0.2,
                             protocol_sha256="p", source_sha256="s", data_sha256="d", software_sha256=software))
        with self.assertRaisesRegex(ValueError, "pairing"):
            paired(pd.DataFrame(rows), "x", "y")

    def test_streaming_metric_weights_elements_not_batches(self):
        class Zero(torch.nn.Module):
            def forward(self, x, *args):
                return torch.zeros(x.shape[0], 192, x.shape[-1])
        batches = []
        for count, value in ((2, 1.0), (1, 3.0)):
            batches.append((torch.zeros(count, 336, 1), torch.full((count, 288, 1), value),
                            torch.zeros(count, 336, 4), torch.zeros(count, 288, 4)))
        result = evaluate(Zero(), batches, torch.device("cpu"))
        self.assertAlmostEqual(result[1], 11/3)
        self.assertAlmostEqual(result[0], 5/3)


if __name__ == "__main__":
    unittest.main()
