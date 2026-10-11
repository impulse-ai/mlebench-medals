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


class HighResultsContractTests(unittest.TestCase):
    def load(self):
        return json.loads((ROOT / LEDGER).read_text())

    def copy_tree(self, root):
        (root / "results").mkdir()
        shutil.copy(ROOT / LEDGER, root / LEDGER)
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

    def errors_after(self, mutate_data=None, mutate_files=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copy_tree(root)
            if mutate_data is not None:
                data = self.load()
                mutate_data(data)
                (root / LEDGER).write_text(json.dumps(data, ensure_ascii=False))
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
        self.assertEqual(10, len(engine))
        self.assertEqual("66.67", f"{100 * len(engine) / 15:.2f}")
        self.assertEqual("73.33", f"{100 * len(medal_tasks) / 15:.2f}")
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
        self.assertIn("11 medals on MLE-bench High", text)
        self.assertIn("11 of the 15", text)
        self.assertIn("7 gold / 2 silver / 2 bronze", text)
        self.assertIn("**66.67 (10/15)**", text)
        self.assertIn("**73.33 (11/15)**", text)
        self.assertIn("| AIBuildAI | Claude-Opus-4.6 | 46.67 ± 0.00 | ✗ |", text)
        readme = (ROOT / "README.md").read_text()
        self.assertIn("[11 medals on MLE-bench High](HIGH.md)", readme)
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
            lambda data: data["summary"].__setitem__("engine_run_percentage", 60.0),
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
            task["method"] = "Engine run"

        errors = self.errors_after(mutate)
        self.assertIn(
            "High results must hold 10 engine runs and 1 pipeline result", errors
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
            path.write_text(path.read_text().replace("**66.67 (10/15)**", "**60.00 (9/15)**"))

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
        paths = [ROOT / "HIGH.md", ROOT / LEDGER, ROOT / "reproduce/EVIDENCE_HIGH.md"]
        for task_id in HIGH_SPLIT:
            directory = ROOT / "solutions" / task_id
            if directory.is_dir():
                paths.extend([directory / "README.md", directory / "grade.json"])
        for path in paths:
            text = path.read_text()
            self.assertNotIn("/data/", text, path)
            self.assertNotIn("s3://", text, path)

    def test_malformed_ledgers_return_errors_instead_of_crashing(self):
        for value in ("{", "[]", json.dumps({"tasks": {}}), json.dumps({"tasks": [None] * 15})):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.copy_tree(root)
                (root / LEDGER).write_text(value)
                self.assertTrue(verify_high(root))


if __name__ == "__main__":
    unittest.main()
