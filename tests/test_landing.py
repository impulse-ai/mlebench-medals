import json
import shutil
import tempfile
import unittest
from pathlib import Path

from reproduce import make_charts
from reproduce.verify_landing import verify_landing
from reproduce.verify_results import verify_results


ROOT = Path(__file__).resolve().parents[1]
RESULTS = (
    "high15.json",
    "high15-eval-three-seed.json",
    "lite22-three-run.json",
    "published-lite-context.json",
)


class LandingContractTests(unittest.TestCase):
    def errors_after(self, mutate):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "results").mkdir()
            for name in RESULTS:
                shutil.copy(ROOT / "results" / name, root / "results" / name)
            for name in ("README.md", "HIGH.md"):
                shutil.copy(ROOT / name, root / name)
            shutil.copytree(ROOT / "assets", root / "assets")
            mutate(root)
            return verify_landing(root)

    def edit(self, relative, old, new):
        def mutate(root):
            path = root / relative
            text = path.read_text()
            self.assertIn(old, text)
            path.write_text(text.replace(old, new))

        return mutate

    def edit_json(self, name, change):
        def mutate(root):
            path = root / "results" / name
            data = json.loads(path.read_text())
            change(data)
            path.write_text(json.dumps(data, ensure_ascii=False))

        return mutate

    def test_repository_passes(self):
        self.assertEqual([], verify_landing(ROOT))
        self.assertEqual([], make_charts.stale(ROOT))

    def test_copied_tree_passes(self):
        self.assertEqual([], self.errors_after(lambda root: None))

    def test_charts_are_regenerated_from_the_ledgers(self):
        files = make_charts.build(ROOT)
        self.assertEqual(4, len(files))
        for relative, text in files.items():
            self.assertEqual(text, (ROOT / relative).read_text(), relative)
            self.assertNotIn("<image", text)
            self.assertNotIn("href", text)
            self.assertNotIn("<style", text)

    def test_chart_values_come_from_the_ledgers(self):
        high = json.loads((ROOT / "results/high15.json").read_text())
        lite = json.loads((ROOT / "results/published-lite-context.json").read_text())
        high_svg = (ROOT / "assets/high-any-medal-light.svg").read_text()
        lite_svg = (ROOT / "assets/lite-any-medal-light.svg").read_text()
        self.assertIn(">66.67 ± 0.00 (10/15 per seed)<", high_svg)
        self.assertIn(">Impulse (three seeds)<", high_svg)
        self.assertIn(">86.36% (19/22)<", lite_svg)
        for entry in high["published_context"]["entries"]:
            self.assertIn(f">{entry['high_any_medal']}<", high_svg)
        for entry in lite["entries"]:
            self.assertIn(f">{entry['lite_any_medal']}<", lite_svg)
        self.assertIn(
            "Impulse: three independent seeds, mean ± standard error. Others: "
            "MLE-bench main leaderboard, means over seeds.",
            high_svg,
        )
        self.assertIn(
            "Impulse: three confirmed runs. Others: MLE-bench main leaderboard, "
            "means over seeds.",
            lite_svg,
        )

    def test_rejects_chart_that_differs_from_the_ledger(self):
        errors = self.errors_after(
            self.edit("assets/high-any-medal-dark.svg", "66.67 ± 0.00", "76.67 ± 0.00")
        )
        self.assertIn(
            "chart is out of date with the ledgers: assets/high-any-medal-dark.svg",
            errors,
        )

    def test_rejects_ledger_change_without_new_charts(self):
        def change(data):
            data["published_context"]["entries"][0]["high_any_medal"] = "36.67 ± 0.00"

        errors = self.errors_after(self.edit_json("high15.json", change))
        self.assertIn(
            "chart is out of date with the ledgers: assets/high-any-medal-light.svg",
            errors,
        )

    def test_rejects_missing_chart(self):
        errors = self.errors_after(
            lambda root: (root / "assets/lite-any-medal-light.svg").unlink()
        )
        self.assertIn(
            "chart is out of date with the ledgers: assets/lite-any-medal-light.svg",
            errors,
        )

    def test_rejects_lite_context_drift(self):
        def change(data):
            data["entries"][0]["lite_any_medal"] = "70.3 ± 1.52"

        errors = self.errors_after(self.edit_json("published-lite-context.json", change))
        self.assertIn(
            "published Lite context file does not match the README table", errors
        )

    def test_rejects_at_a_glance_drift(self):
        errors = self.errors_after(
            self.edit(
                "README.md",
                "| 30 medals | High: 66.67% ± 0.00 |",
                "| 31 medals | High: 66.67% ± 0.00 |",
            )
        )
        self.assertTrue(
            any("README.md landing section is missing: | 30 medals" in e for e in errors),
            errors,
        )

    def test_rejects_seed_count_change_without_new_charts(self):
        def change(data):
            data["tasks"][0]["seeds"][0]["medal"] = None
            data["summary"]["per_seed"][0]["medal_tasks"] = 9

        errors = self.errors_after(self.edit_json("high15-eval-three-seed.json", change))
        self.assertIn(
            "chart is out of date with the ledgers: assets/high-any-medal-light.svg",
            errors,
        )

    def test_rejects_at_a_glance_seed_wording(self):
        errors = self.errors_after(
            self.edit(
                "README.md",
                "| on 37 competitions | three independent seeds, against",
                "| on 37 competitions | engine runs, against",
            )
        )
        self.assertTrue(
            any("README.md landing section is missing" in e for e in errors), errors
        )

    def test_rejects_missing_walkthrough_link(self):
        for name in ("README.md", "HIGH.md"):
            errors = self.errors_after(
                self.edit(name, "https://mlebench.impulselabs.ai", "https://example.invalid")
            )
            self.assertTrue(any("Interactive walkthrough" in e for e in errors), errors)

    def test_rejects_missing_chart_image(self):
        errors = self.errors_after(
            self.edit("HIGH.md", "high-any-medal-dark.svg#gh-dark-mode-only", "x.svg")
        )
        self.assertTrue(any(e.startswith("HIGH.md is missing") for e in errors), errors)

    def test_link_check_covers_chart_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "results").mkdir()
            for name in RESULTS:
                shutil.copy(ROOT / "results" / name, root / "results" / name)
            for name in ("README.md", "HIGH.md"):
                shutil.copy(ROOT / name, root / name)
            shutil.copytree(ROOT / "assets", root / "assets")
            shutil.copytree(ROOT / "reproduce", root / "reproduce")
            (root / "assets/high-any-medal-dark.svg").unlink()
            errors = verify_results(root)
        broken = [e for e in errors if "broken relative Markdown link" in e]
        self.assertTrue(
            any(e.startswith("README.md:") and "high-any-medal-dark.svg" in e for e in broken),
            errors,
        )
        self.assertTrue(
            any(e.startswith("HIGH.md:") and "high-any-medal-dark.svg" in e for e in broken),
            errors,
        )


if __name__ == "__main__":
    unittest.main()
