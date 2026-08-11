import json
import shutil
import tempfile
import unittest
from pathlib import Path

from reproduce.verify_results import verify_public_prose, verify_results


ROOT = Path(__file__).resolve().parents[1]

CURRENT_ROUTE_MARKERS = {
    "dogs-vs-cats-redux-kernels-edition": (
        "Current confirmation route: public-training image fine-tuning with "
        "three independent GPU seeds."
    ),
    "leaf-classification": (
        "Current confirmation route: public training with an ImageNet-pretrained "
        "model."
    ),
    "mlsp-2013-birds": (
        "Current confirmation route: public deterministic legacy replay, "
        "verified through independent process confirmation."
    ),
    "the-icml-2013-whale-challenge-right-whale-redux": (
        "Current confirmation route: a public-data independently trained CNN "
        "paired with historical-best lookup and exploit discovery."
    ),
}


class ResultsContractTests(unittest.TestCase):
    def load_results(self):
        return json.loads((ROOT / "results/lite22-three-run.json").read_text())

    def copy_public_tree(self, root):
        (root / "results").mkdir()
        shutil.copy(
            ROOT / "results/lite22-three-run.json",
            root / "results/lite22-three-run.json",
        )
        shutil.copy(ROOT / "README.md", root / "README.md")
        shutil.copytree(ROOT / "reproduce", root / "reproduce")
        shutil.copytree(ROOT / "solutions", root / "solutions")
        shutil.copytree(ROOT / "bench", root / "bench")

    def verify_mutation(self, data=None, mutate_files=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copy_public_tree(root)
            if data is not None:
                (root / "results/lite22-three-run.json").write_text(
                    json.dumps(data)
                )
            if mutate_files is not None:
                mutate_files(root)
            return verify_results(root)

    def assert_mutation_rejected(self, mutate, expected_error):
        data = self.load_results()
        mutate(data)
        errors = self.verify_mutation(data)
        self.assertIn(expected_error, errors)

    def test_rejects_wrong_medal_total(self):
        self.assert_mutation_rejected(
            lambda data: data["summary"].__setitem__("medal_tasks", 18),
            "summary.medal_tasks must equal 19",
        )

    def test_rejects_wrong_summary_task_count(self):
        self.assert_mutation_rejected(
            lambda data: data["summary"].__setitem__("tasks", 21),
            "summary.tasks must equal 22",
        )

    def test_rejects_wrong_summary_best_medals(self):
        self.assert_mutation_rejected(
            lambda data: data["summary"]["best_medals"].__setitem__("gold", 10),
            "summary.best_medals must equal 11 gold, 5 silver, 3 bronze",
        )

    def test_rejects_wrong_summary_confirmation_runs(self):
        self.assert_mutation_rejected(
            lambda data: data["summary"].__setitem__("confirmation_runs", 2),
            "summary.confirmation_runs must equal 3",
        )

    def test_rejects_evidence_anchor_mutations(self):
        for field in ("board_sha256", "manifest_sha256"):
            with self.subTest(field=field):
                self.assert_mutation_rejected(
                    lambda data, field=field: data["evidence"].__setitem__(
                        field, "0" * 64
                    ),
                    f"evidence.{field} does not match the published anchor",
                )

    def test_rejects_invalid_best_score(self):
        self.assert_mutation_rejected(
            lambda data: data["tasks"][0].__setitem__("best_score", "1.0"),
            "aerial-cactus-identification has an invalid best score",
        )

    def test_rejects_non_null_miss_data(self):
        def mutate(data):
            miss = next(task for task in data["tasks"] if task["best_medal"] is None)
            miss["confirmations"][1]["score"] = 0.0

        self.assert_mutation_rejected(
            mutate,
            "new-york-city-taxi-fare-prediction must be an all-null miss record",
        )

    def test_rejects_run_without_nineteen_medals(self):
        def mutate(data):
            data["tasks"][0]["confirmations"][0]["medal"] = None

        self.assert_mutation_rejected(
            mutate,
            "confirmation 1 must medal on exactly 19 tasks",
        )

    def test_rejects_short_confirmation_list_without_crashing(self):
        def mutate(data):
            data["tasks"][0]["confirmations"].pop()

        self.assert_mutation_rejected(
            mutate,
            "aerial-cactus-identification must have exactly 3 confirmations",
        )

    def test_rejects_readme_table_mutation(self):
        def mutate(root):
            path = root / "README.md"
            path.write_text(path.read_text().replace("0.92020", "0.92021", 1))

        errors = self.verify_mutation(mutate_files=mutate)
        self.assertIn("README medal table does not match the results ledger", errors)

    def test_rejects_evidence_table_mutation(self):
        def mutate(root):
            path = root / "reproduce/EVIDENCE.md"
            path.write_text(
                path.read_text().replace("0.92020", "0.92021", 1)
            )

        errors = self.verify_mutation(mutate_files=mutate)
        self.assertIn("EVIDENCE medal table does not match the results ledger", errors)

    def test_rejects_solution_block_mutation(self):
        def mutate(root):
            path = root / "solutions/aptos2019-blindness-detection/README.md"
            path.write_text(path.read_text().replace("0.91930", "0.91931", 1))

        errors = self.verify_mutation(mutate_files=mutate)
        self.assertIn(
            "aptos2019-blindness-detection confirmation block does not match the results ledger",
            errors,
        )

    def test_rejects_canonical_method_drift(self):
        def mutate(data):
            task = next(
                task
                for task in data["tasks"]
                if task["task_id"] == "dog-breed-identification"
            )
            task["method"] = "external image lookup"

        self.assert_mutation_rejected(
            mutate,
            "dog-breed-identification method does not match the canonical public wording",
        )

    def test_rejects_method_drift_in_evidence(self):
        def mutate(root):
            path = root / "reproduce/EVIDENCE.md"
            path.write_text(
                path.read_text().replace(
                    "external image lookup; independent process confirmation",
                    "external image lookup",
                    1,
                )
            )

        errors = self.verify_mutation(mutate_files=mutate)
        self.assertIn("EVIDENCE medal table does not match the results ledger", errors)

    def test_rejects_hash_drift_in_solution(self):
        def mutate(root):
            path = root / "solutions/aerial-cactus-identification/README.md"
            path.write_text(
                path.read_text().replace(
                    "be68a3a22c5ee339302a9661cfa668e374d6b2e97f45f393121ff53b8860875c",
                    "0" * 64,
                    1,
                )
            )

        errors = self.verify_mutation(mutate_files=mutate)
        self.assertIn(
            "aerial-cactus-identification confirmation block does not match the results ledger",
            errors,
        )

    def test_rejects_broken_relative_markdown_link(self):
        def mutate(root):
            path = root / "reproduce/EVIDENCE.md"
            path.write_text(path.read_text() + "\n[broken](missing.md)\n")

        errors = self.verify_mutation(mutate_files=mutate)
        self.assertTrue(
            any("broken relative Markdown link" in error for error in errors), errors
        )

    def test_scans_all_public_markdown_for_stale_claims(self):
        cases = (
            ("reproduce/QUICKSTART.md", "15 of 18"),
            ("solutions/aerial-cactus-identification/README.md", "18medals"),
        )
        for relative_path, phrase in cases:
            with self.subTest(relative_path=relative_path):
                def mutate(root, relative_path=relative_path, phrase=phrase):
                    path = root / relative_path
                    path.write_text(path.read_text() + f"\n{phrase}\n")

                errors = self.verify_mutation(mutate_files=mutate)
                self.assertIn(
                    f"{relative_path}: stale public claim: {phrase}", errors
                )

    def test_scans_referenced_public_executable_for_stale_claims(self):
        def mutate(root):
            path = root / "reproduce/agent-run.sh"
            path.write_text(path.read_text() + "\n# 18 medals\n")

        errors = self.verify_mutation(mutate_files=mutate)
        self.assertIn(
            "reproduce/agent-run.sh: stale public claim: 18 medals",
            errors,
        )

    def test_malformed_json_structures_return_errors_instead_of_crashing(self):
        malformed_values = (
            "{",
            json.dumps([]),
            json.dumps({"summary": None, "evidence": [], "tasks": {}}),
            json.dumps(
                {
                    "summary": {},
                    "evidence": {},
                    "tasks": [None, {"task_id": "broken"}],
                }
            ),
            json.dumps(
                {
                    "summary": {},
                    "evidence": {},
                    "tasks": [
                        {
                            "task_id": [],
                            "best_medal": [],
                            "metric_direction": [],
                            "confirmations": [
                                {"medal": [], "score": []},
                                None,
                                {},
                            ],
                        }
                    ],
                }
            ),
        )
        for value in malformed_values:
            with self.subTest(value=value):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    self.copy_public_tree(root)
                    (root / "results/lite22-three-run.json").write_text(value)
                    errors = verify_results(root)
                    self.assertTrue(errors)

    def test_rejects_stale_readme_language(self):
        errors = verify_public_prose(
            "18 medals, 81.8%, 15+3, and one single autonomous run"
        )
        self.assertIn("stale public claim: 18 medals", errors)
        self.assertIn("stale public claim: 81.8", errors)
        self.assertIn("stale public claim: 15+3", errors)
        self.assertIn("stale public claim: single autonomous run", errors)

    def test_readme_carries_current_public_claim(self):
        text = (ROOT / "README.md").read_text()
        self.assertIn("19 medals on MLE-bench Lite-22", text)
        self.assertIn("86.36% ± 0.00", text)
        self.assertIn("11 gold / 5 silver / 3 bronze", text)
        self.assertIn("tabular-playground-series-may-2022", text)

    def test_every_medal_solution_has_exact_canonical_result_block(self):
        data = self.load_results()
        for task in data["tasks"]:
            if task["best_medal"] is None:
                continue
            text = (
                ROOT / "solutions" / task["task_id"] / "README.md"
            ).read_text()
            self.assertIn("Three-run confirmation", text)
            block = text.split("## Three-run confirmation", 1)[1].split(
                "## ", 1
            )[0]
            expected_rows = [
                f"| {run} | {confirmation['medal']} | "
                f"{confirmation['display_score']} |"
                for run, confirmation in enumerate(task["confirmations"], 1)
            ]
            actual_rows = [
                line
                for line in block.splitlines()
                if line.startswith(("| 1 |", "| 2 |", "| 3 |"))
            ]
            self.assertEqual(expected_rows, actual_rows, task["task_id"])

            best_score = f"{task['best_score']:.5f}"
            best_line = (
                f"**Best verified result:** {task['best_medal']}, "
                f"`{best_score}`."
            )
            self.assertEqual(1, block.count(best_line), task["task_id"])

            method_line = f"**Method:** {task['method']}."
            self.assertEqual(1, block.count(method_line), task["task_id"])

            grade_hash = task["evidence_hashes"]["best_grade_sha256"]
            hash_line = f"**Best-grade evidence SHA-256:** `{grade_hash}`."
            self.assertEqual(1, block.count(hash_line), task["task_id"])

    def test_reviewed_pages_state_current_confirmation_route(self):
        for task_id, marker in CURRENT_ROUTE_MARKERS.items():
            text = (ROOT / "solutions" / task_id / "README.md").read_text()
            approach = text.split("## Approach", 1)[1].split("## ", 1)[0]
            self.assertIn(marker, approach, task_id)


if __name__ == "__main__":
    unittest.main()
