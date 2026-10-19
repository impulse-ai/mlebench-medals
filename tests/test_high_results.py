import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from reproduce.verify_high import (
    HIGH_SPLIT,
    PUBLISHED_HIGH,
    medal_for,
    verify_high,
)
from reproduce.verify_results import verify_results


ROOT = Path(__file__).resolve().parents[1]
LEDGER = "results/high15.json"
EVAL = "results/high15-eval-three-seed.json"


class HighResultsContractTests(unittest.TestCase):
    def load(self):
        return json.loads((ROOT / LEDGER).read_text())

    def copy_tree(self, root):
        (root / "results").mkdir()
        shutil.copy(ROOT / LEDGER, root / LEDGER)
        shutil.copy(ROOT / EVAL, root / EVAL)
        shutil.copy(ROOT / "README.md", root / "README.md")
        shutil.copy(ROOT / "HIGH.md", root / "HIGH.md")
        shutil.copy(ROOT / "HARD.md", root / "HARD.md")
        (root / "reproduce").mkdir()
        for name in ("EVIDENCE_HIGH.md", "high15_submission_checksums.txt"):
            shutil.copy(ROOT / "reproduce" / name, root / "reproduce" / name)
        for task_id in sorted(HIGH_SPLIT):
            source = ROOT / "solutions" / task_id
            if not source.is_dir():
                continue
            target = root / "solutions" / task_id
            target.mkdir(parents=True)
            for item in source.iterdir():
                if item.name == "submission.csv":
                    os.symlink(item, target / item.name)
                else:
                    shutil.copy(item, target / item.name)

    def load_eval(self):
        return json.loads((ROOT / EVAL).read_text())

    def errors_after(self, mutate_data=None, mutate_files=None, mutate_eval=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copy_tree(root)
            if mutate_data is not None:
                data = self.load()
                mutate_data(data)
                (root / LEDGER).write_text(json.dumps(data, ensure_ascii=False))
            if mutate_eval is not None:
                data = self.load_eval()
                mutate_eval(data)
                (root / EVAL).write_text(json.dumps(data, ensure_ascii=False))
            if mutate_files is not None:
                mutate_files(root)
            return verify_high(root)

    def task(self, data, task_id):
        return next(task for task in data["tasks"] if task["task_id"] == task_id)

    def test_repository_passes(self):
        self.assertEqual([], verify_high(ROOT))
        self.assertEqual([], verify_results(ROOT))

    def test_copied_tree_passes(self):
        self.assertEqual([], self.errors_after())

    def test_ledger_carries_the_public_claims(self):
        data = self.load()
        medal_tasks = [task for task in data["tasks"] if task["medal"]]
        engine = [task for task in medal_tasks if task["method"].startswith("Engine run")]
        self.assertEqual(15, len(data["tasks"]))
        self.assertEqual(11, len(medal_tasks))
        self.assertEqual(11, len(engine))
        self.assertEqual("73.33", f"{100 * len(medal_tasks) / 15:.2f}")
        self.assertEqual(
            "66.67 ± 0.00", data["summary"]["three_seed_evaluation"]["display"]
        )
        self.assertEqual(
            {"gold": 7, "silver": 2, "bronze": 2},
            {
                medal: sum(task["medal"] == medal for task in medal_tasks)
                for medal in ("gold", "silver", "bronze")
            },
        )
        self.assertEqual(
            ["Engine run (exploit discovery)"],
            [
                task["method"]
                for task in medal_tasks
                if task["task_id"] == "smartphone-decimeter-2022"
            ],
        )
        for task in engine:
            self.assertEqual("GPT-6-Luna", task["llm"], task["task_id"])
            self.assertLess(task["wall_time_hours"], 24.0, task["task_id"])

    def test_every_medal_follows_from_score_and_lines(self):
        for task in self.load()["tasks"]:
            if task["medal"] is None:
                continue
            self.assertEqual(
                task["medal"],
                medal_for(task["score"], task["thresholds"], task["metric_direction"]),
                task["task_id"],
            )

    def test_page_carries_the_public_claims(self):
        text = (ROOT / "HIGH.md").read_text()
        self.assertIn("66.67% ± 0.00 across three independent seeds", text)
        self.assertIn("11 of the 15", text)
        self.assertIn("7 gold / 2 silver / 2 bronze", text)
        self.assertIn("**66.67 ± 0.00 (10/15 per seed)**", text)
        self.assertIn("**73.33 (11/15)**", text)
        self.assertIn("| AIBuildAI | Claude-Opus-4.6 | 46.67 ± 0.00 | ✗ |", text)
        self.assertNotIn("Impulse ML pipeline", text)
        self.assertNotIn("HARD", text)
        readme = (ROOT / "README.md").read_text()
        self.assertIn(
            "[MLE-bench High: 66.67% ± 0.00 across three independent seeds](HIGH.md)",
            readme,
        )
        self.assertIn("18 gold / 7 silver / 5 bronze", readme)
        self.assertIn("19 medals on MLE-bench Lite-22", readme)
        self.assertIn("86.36% ± 0.00", readme)

    def test_published_context_matches_cited_figures(self):
        self.assertEqual(PUBLISHED_HIGH, self.load()["published_context"]["entries"])
        self.assertEqual(
            ["46.67 ± 0.00", "44.44 ± 2.22", "42.22 ± 2.22", "42.22 ± 2.22",
             "42.22 ± 2.22", "40.0 ± 0.00", "11.67 ± 1.27"],
            [entry["high_any_medal"] for entry in PUBLISHED_HIGH],
        )

    def test_rejects_wrong_summary(self):
        cases = (
            lambda data: data["summary"].__setitem__("medal_tasks", 12),
            lambda data: data["summary"]["medals"].__setitem__("gold", 8),
            lambda data: data["summary"]["three_seed_evaluation"].__setitem__(
                "display", "60.00 ± 0.00"
            ),
        )
        for mutate in cases:
            errors = self.errors_after(mutate)
            self.assertTrue(
                any(error.startswith("High summary must equal") for error in errors),
                errors,
            )

    def test_rejects_score_drift(self):
        def mutate(data):
            self.task(data, "nfl-player-contact-detection")["score"] = 0.72956

        errors = self.errors_after(mutate)
        self.assertIn(
            "nfl-player-contact-detection medal does not follow from its score "
            "and lines",
            errors,
        )
        self.assertIn(
            "nfl-player-contact-detection grade report does not match the ledger",
            errors,
        )

    def test_rejects_medal_line_drift(self):
        def mutate(data):
            task = self.task(data, "rsna-breast-cancer-detection")
            task["display_thresholds"] = "0.48 / 0.43 / 0.41"

        self.assertIn(
            "rsna-breast-cancer-detection displayed medal lines do not match the "
            "ledger",
            self.errors_after(mutate),
        )

    def test_rejects_submission_hash_drift(self):
        def mutate(data):
            task = self.task(data, "iwildcam-2019-fgvc6")
            task["submission"]["sha256"] = "0" * 64

        self.assertIn(
            "iwildcam-2019-fgvc6 graded submission does not match its SHA-256",
            self.errors_after(mutate),
        )

    def test_rejects_changed_submission_file(self):
        def mutate(root):
            path = root / "solutions/predict-volcanic-eruptions-ingv-oe/submission.csv"
            content = path.read_bytes()
            path.unlink()
            path.write_bytes(content + b"0,0\n")

        self.assertIn(
            "predict-volcanic-eruptions-ingv-oe graded submission does not match "
            "its SHA-256",
            self.errors_after(mutate_files=mutate),
        )

    def test_rejects_changed_grade_report(self):
        def mutate(root):
            path = root / "solutions/stanford-covid-vaccine/grade.json"
            path.write_text(path.read_text().replace("0.32671", "0.32571"))

        self.assertIn(
            "stanford-covid-vaccine grade report does not match its SHA-256",
            self.errors_after(mutate_files=mutate),
        )

    def test_rejects_other_engine_model(self):
        def mutate(data):
            self.task(data, "iwildcam-2019-fgvc6")["llm"] = "other-model"

        self.assertIn(
            "iwildcam-2019-fgvc6 engine run must record GPT-6-Luna",
            self.errors_after(mutate),
        )

    def test_rejects_run_past_the_time_limit(self):
        def mutate(data):
            self.task(data, "rsna-breast-cancer-detection")["wall_time_hours"] = 24.5

        self.assertIn(
            "rsna-breast-cancer-detection engine run must finish inside 24 hours",
            self.errors_after(mutate),
        )

    def test_rejects_method_relabel(self):
        def mutate(data):
            task = self.task(data, "hms-harmful-brain-activity-classification")
            task["method"] = "Impulse ML pipeline"

        errors = self.errors_after(mutate)
        self.assertIn(
            "hms-harmful-brain-activity-classification has an invalid method", errors
        )
        self.assertIn("High results must hold 11 engine runs", errors)

    def test_rejects_evaluation_row_that_drifts_from_its_seed(self):
        def mutate(data):
            task = self.task(data, "rsna-breast-cancer-detection")
            task["method"] = "Engine run (three-seed evaluation, seed 202)"
            task["evaluation"]["seed"] = 202

        self.assertIn(
            "rsna-breast-cancer-detection evaluation row does not match the "
            "three-seed ledger",
            self.errors_after(mutate),
        )

    def test_rejects_evaluation_row_without_a_seed_record(self):
        def mutate(data):
            del self.task(data, "hms-harmful-brain-activity-classification")["evaluation"]

        self.assertIn(
            "hms-harmful-brain-activity-classification evaluation row does not "
            "match the three-seed ledger",
            self.errors_after(mutate),
        )

    def test_rejects_non_null_miss(self):
        def mutate(data):
            self.task(data, "bms-molecular-translation")["score"] = 1.0

        self.assertIn(
            "bms-molecular-translation must be an all-null miss record",
            self.errors_after(mutate),
        )

    def test_rejects_wrong_task_set(self):
        def mutate(data):
            data["tasks"][-1]["task_id"] = "not-a-high-split-task"

        self.assertIn(
            "High results must contain the 15 High-split tasks",
            self.errors_after(mutate),
        )

    def test_rejects_published_figure_drift(self):
        def mutate(data):
            data["published_context"]["entries"][0]["high_any_medal"] = "45.00 ± 0.00"

        self.assertIn(
            "published High-split figures do not match the cited source",
            self.errors_after(mutate),
        )

    def test_rejects_page_table_mutation(self):
        def mutate(root):
            path = root / "HIGH.md"
            path.write_text(path.read_text().replace("0.18128", "0.18129", 1))

        self.assertIn(
            "High medal table does not match the results ledger",
            self.errors_after(mutate_files=mutate),
        )

    def test_rejects_page_context_mutation(self):
        def mutate(root):
            path = root / "HIGH.md"
            path.write_text(path.read_text().replace("44.44 ± 2.22", "34.44 ± 2.22"))

        self.assertIn(
            "High published-context table does not match the results ledger",
            self.errors_after(mutate_files=mutate),
        )

    def test_rejects_missing_page_claim(self):
        def mutate(root):
            path = root / "HIGH.md"
            path.write_text(
                path.read_text().replace(
                    "**73.33 (11/15)**", "**80.00 (12/15)**"
                )
            )

        errors = self.errors_after(mutate_files=mutate)
        self.assertTrue(
            any("HIGH.md is missing the public claim" in error for error in errors),
            errors,
        )

    def test_rejects_missing_readme_headline(self):
        def mutate(root):
            path = root / "README.md"
            path.write_text(path.read_text().replace("(HIGH.md)", "(OTHER.md)"))

        errors = self.errors_after(mutate_files=mutate)
        self.assertTrue(
            any("README.md is missing the High headline" in error for error in errors),
            errors,
        )

    def test_former_page_path_points_to_the_new_page(self):
        def mutate(root):
            (root / "HARD.md").write_text("# This page has moved\n")

        self.assertIn("HARD.md must point to HIGH.md", self.errors_after(mutate_files=mutate))

    def test_rejects_evidence_and_checksum_mutation(self):
        def mutate(root):
            for name in ("EVIDENCE_HIGH.md", "high15_submission_checksums.txt"):
                path = root / "reproduce" / name
                path.write_text(path.read_text().replace("cb0a7265", "cb0a7266"))

        errors = self.errors_after(mutate_files=mutate)
        self.assertIn("High evidence table does not match the results ledger", errors)
        self.assertIn("High checksum file does not match the results ledger", errors)

    def test_rejects_solution_block_mutation(self):
        def mutate(root):
            path = root / "solutions/smartphone-decimeter-2022/README.md"
            path.write_text(
                path.read_text().replace(
                    "**Method:** Engine run (exploit discovery).",
                    "**Method:** Engine run.",
                )
            )

        self.assertIn(
            "smartphone-decimeter-2022 result block does not match the results ledger",
            self.errors_after(mutate_files=mutate),
        )

    def test_public_high_files_use_neutral_run_ids(self):
        paths = [
            ROOT / "HIGH.md",
            ROOT / LEDGER,
            ROOT / EVAL,
            ROOT / "reproduce/EVIDENCE_HIGH.md",
        ]
        for task_id in HIGH_SPLIT:
            directory = ROOT / "solutions" / task_id
            if directory.is_dir():
                paths.extend([directory / "README.md", directory / "grade.json"])
        for path in paths:
            text = path.read_text()
            self.assertNotIn("/data/", text, path)
            self.assertNotIn("s3://", text, path)

    def eval_task(self, data, task_id):
        return next(task for task in data["tasks"] if task["task_id"] == task_id)

    def test_three_seed_ledger_carries_the_public_claims(self):
        data = self.load_eval()
        summary = data["summary"]
        self.assertEqual("eval-high-20261013", data["code_tag"])
        self.assertTrue(data["code_commit"].startswith("926abe7d"))
        self.assertEqual([101, 202, 303], data["seeds"])
        self.assertEqual("GPT-6-Luna", data["llm"])
        self.assertEqual([10, 10, 10], [s["medal_tasks"] for s in summary["per_seed"]])
        self.assertEqual("66.67 ± 0.00", summary["display"])
        self.assertEqual(30, summary["medal_seed_runs"])
        self.assertEqual(33, summary["seed_runs"])
        self.assertEqual(9, summary["tasks_with_a_medal_in_every_seed"])
        self.assertEqual(11, summary["tasks_with_a_medal_in_any_seed"])
        self.assertEqual({"gold": 7, "silver": 2, "bronze": 2}, summary["best_medals"])
        self.assertEqual(
            sorted(["siim-covid19-detection", "rsna-2022-cervical-spine-fracture-detection",
                    "vesuvius-challenge-ink-detection", "bms-molecular-translation"]),
            sorted(summary["tasks_not_run"]),
        )
        for task in data["tasks"]:
            for record in task["seeds"]:
                if task["run"]:
                    self.assertEqual(1, record["grade_count"], task["task_id"])
                    self.assertLess(record["wall_time_hours"], 24.0, task["task_id"])
                    self.assertTrue(record["campaign_id"].endswith(
                        f"-s{record['seed']}" + ("-relaunch1" if "relaunched" in record["flags"] else "")
                    ))
                else:
                    self.assertIsNone(record["medal"], task["task_id"])

    def test_three_seed_ledger_records_the_disclosures(self):
        data = self.load_eval()
        contrails = self.eval_task(
            data, "google-research-identify-contrails-reduce-global-warming"
        )
        relaunched = [r["seed"] for r in contrails["seeds"] if "relaunched" in r["flags"]]
        self.assertEqual([101], relaunched)
        self.assertIn("before any training or score", contrails["seeds"][0]["relaunch_cause"])
        self.assertIn("No relaunch after a score exists", data["rule"])
        self.assertEqual(
            ["smartphone-decimeter-2022", "hms-harmful-brain-activity-classification"],
            data["summary"]["data_property_tasks"],
        )
        page = (ROOT / "HIGH.md").read_text()
        self.assertIn("relaunched once under the run's pre-registered rule", page)
        self.assertIn("exploit-discovery result", page)
        self.assertIn("recording-level layout of the public data index", page)

    def test_rejects_eval_score_drift(self):
        def mutate(data):
            task = self.eval_task(data, "hms-harmful-brain-activity-classification")
            task["seeds"][0]["score"] = 0.37
            task["seeds"][0]["display_score"] = "0.37"

        errors = self.errors_after(mutate_eval=mutate)
        self.assertIn(
            "hms-harmful-brain-activity-classification seed 101 medal does not follow "
            "from its score and lines",
            errors,
        )

    def test_rejects_eval_medal_upgrade_with_consistent_summary_edit(self):
        def mutate(data):
            task = self.eval_task(
                data, "google-research-identify-contrails-reduce-global-warming"
            )
            task["seeds"][1]["score"] = 0.70
            task["seeds"][1]["display_score"] = "0.70"
            task["seeds"][1]["medal"] = "silver"
            data["summary"]["per_seed"][1]["medal_tasks"] = 11

        errors = self.errors_after(mutate_eval=mutate)
        self.assertTrue(
            any(e.startswith("three-seed results must equal") for e in errors), errors
        )

    def test_rejects_eval_summary_drift(self):
        def mutate(data):
            data["summary"]["medal_seed_runs"] = 31

        self.assertIn(
            "three-seed summary does not match its task-seed records",
            self.errors_after(mutate_eval=mutate),
        )

    def test_rejects_eval_second_grade(self):
        def mutate(data):
            self.eval_task(data, "nfl-player-contact-detection")["seeds"][2]["grade_count"] = 2

        self.assertIn(
            "nfl-player-contact-detection seed 303 must be graded exactly once",
            self.errors_after(mutate_eval=mutate),
        )

    def test_rejects_eval_hidden_relaunch(self):
        def mutate(data):
            task = self.eval_task(
                data, "google-research-identify-contrails-reduce-global-warming"
            )
            task["seeds"][0]["flags"].remove("relaunched")

        self.assertIn(
            "google-research-identify-contrails-reduce-global-warming seed 101 has "
            "invalid flags or campaign id",
            self.errors_after(mutate_eval=mutate),
        )

    def test_rejects_eval_run_past_the_time_limit(self):
        def mutate(data):
            self.eval_task(data, "rsna-breast-cancer-detection")["seeds"][0]["wall_time_hours"] = 24.2

        self.assertIn(
            "rsna-breast-cancer-detection seed 101 must finish inside 24 hours",
            self.errors_after(mutate_eval=mutate),
        )

    def test_rejects_not_run_task_with_a_score(self):
        def mutate(data):
            self.eval_task(data, "bms-molecular-translation")["seeds"][0]["score"] = 1.0

        self.assertIn(
            "bms-molecular-translation must be recorded as not run, a miss in every seed",
            self.errors_after(mutate_eval=mutate),
        )

    def test_rejects_per_seed_table_mutation(self):
        def mutate(root):
            path = root / "HIGH.md"
            path.write_text(path.read_text().replace("🥉 0.37511", "🥈 0.37511", 1))

        self.assertIn(
            "High per-seed table does not match the three-seed ledger",
            self.errors_after(mutate_files=mutate),
        )

    def test_rejects_impulse_context_row_mutation(self):
        def mutate(root):
            path = root / "HIGH.md"
            path.write_text(
                path.read_text().replace(
                    "**66.67 ± 0.00 (10/15 per seed)**", "**66.67 ± 1.00 (10/15 per seed)**"
                )
            )

        errors = self.errors_after(mutate_files=mutate)
        self.assertTrue(
            any("HIGH.md is missing the public claim" in error for error in errors),
            errors,
        )

    def test_malformed_ledgers_return_errors_instead_of_crashing(self):
        for value in ("{", "[]", json.dumps({"tasks": {}}), json.dumps({"tasks": [None] * 15})):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.copy_tree(root)
                (root / LEDGER).write_text(value)
                self.assertTrue(verify_high(root))
        for value in ("{", "[]", json.dumps({"tasks": {}}), json.dumps({"tasks": [{}] * 15})):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.copy_tree(root)
                (root / EVAL).write_text(value)
                self.assertTrue(verify_high(root))


if __name__ == "__main__":
    unittest.main()
