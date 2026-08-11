import json
import shutil
import tempfile
import unittest
from pathlib import Path

from reproduce.verify_results import verify_public_prose, verify_results


ROOT = Path(__file__).resolve().parents[1]


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


if __name__ == "__main__":
    unittest.main()
