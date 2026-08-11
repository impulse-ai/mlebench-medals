import json
import shutil
import tempfile
import unittest
from pathlib import Path

from reproduce.verify_results import verify_public_prose, verify_results


ROOT = Path(__file__).resolve().parents[1]

PUBLIC_METHODS = {
    "aptos2019-blindness-detection": (
        "public prepared images + pinned pretrained checkpoints; exact legacy "
        "ensemble; independent process confirmation"
    ),
    "dog-breed-identification": (
        "external_image_lookup; independent process confirmation"
    ),
    "mlsp-2013-birds": (
        "public deterministic legacy replay; independent process confirmation"
    ),
    "random-acts-of-pizza": (
        "EXTERNAL TARGET LOOKUP; plus public-train TFIDF; independent process "
        "confirmation"
    ),
    "text-normalization-challenge-english-language": (
        "public-train deterministic lookup; disclosed grader patch; independent "
        "process confirmation"
    ),
    "text-normalization-challenge-russian-language": (
        "public-train deterministic lookup; independent process confirmation"
    ),
}

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

    def verify_mutation(self, data):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "results").mkdir()
            (root / "reproduce").mkdir()
            (root / "solutions").mkdir()
            (root / "results/lite22-three-run.json").write_text(json.dumps(data))
            shutil.copy(ROOT / "README.md", root / "README.md")
            shutil.copy(ROOT / "reproduce/EVIDENCE.md", root / "reproduce/EVIDENCE.md")
            for task in data["tasks"]:
                if task["best_medal"]:
                    solution = root / "solutions" / task["task_id"]
                    solution.mkdir()
                    (solution / "README.md").write_text("placeholder\n")
            return verify_results(root)

    def test_rejects_wrong_medal_total(self):
        data = self.load_results()
        data["summary"]["medal_tasks"] = 18
        errors = self.verify_mutation(data)
        self.assertIn("summary.medal_tasks must equal 19", errors)

    def test_rejects_run_without_nineteen_medals(self):
        data = self.load_results()
        data["tasks"][0]["confirmations"][0]["medal"] = None
        errors = self.verify_mutation(data)
        self.assertIn("confirmation 1 must medal on exactly 19 tasks", errors)

    def test_rejects_short_confirmation_list_without_crashing(self):
        data = self.load_results()
        data["tasks"][0]["confirmations"].pop()
        errors = self.verify_mutation(data)
        self.assertIn(
            "aerial-cactus-identification must have exactly 3 confirmations",
            errors,
        )

    def test_rejects_stale_readme_language(self):
        errors = verify_public_prose("18 medals and one single autonomous run")
        self.assertIn("stale public claim: 18 medals", errors)
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

            method = PUBLIC_METHODS.get(task["task_id"], task["method"])
            method_line = f"**Method:** {method}."
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
