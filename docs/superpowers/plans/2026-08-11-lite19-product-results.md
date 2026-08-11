# Lite-19 Product Results Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish the verified 19/22 MLE-bench Lite product result, its three-run medal variation, and consistent evidence across the public repository.

**Architecture:** A checked-in JSON file is the only structured source for public numbers. A standard-library verifier checks totals, medal tiers, run-level medal coverage, solution links, and stale prose. README and solution pages present the same data in human-readable form, while the evidence ledger records hashes and method categories.

**Tech Stack:** JSON, Markdown, Python 3 standard library, GitHub CLI.

## Global Constraints

- Headline: `19 medals on MLE-bench Lite-22: 86.36% ± 0.00 across three confirmed runs`.
- Best-result breakdown: 11 gold, 5 silver, 3 bronze.
- Every medal row shows best medal, best score, three-run medal variation, and three scores.
- External data, web research, pretrained models, and exploit discovery are product capabilities.
- Never claim an accepted official leaderboard position.
- Do not publish evaluator-private labels, credentials, or large training artifacts.
- Preserve all three non-medal tasks so the denominator remains 22.
- Product prose must not use `non-reportable`, `single autonomous run`, `on our roadmap`, or `how to read this honestly`.

---

## File map

- `results/lite22-three-run.json`: canonical public result, task rows, evidence hashes, and display metadata.
- `reproduce/verify_results.py`: deterministic repository integrity checks; no third-party dependencies.
- `tests/test_results.py`: regression tests that prove incorrect totals, medal tiers, and stale README claims fail.
- `README.md`: product-facing headline, medal table, comparison, methods, misses, and verification links.
- `reproduce/EVIDENCE.md`: evidence index for the 19-task three-run campaign.
- `solutions/*/README.md`: per-task best result and three-run confirmation block.
- `solutions/tabular-playground-series-may-2022/README.md`: new TPS May solution page.

---

### Task 1: Canonical three-run results and verifier

**Files:**
- Create: `results/lite22-three-run.json`
- Create: `reproduce/verify_results.py`
- Create: `tests/test_results.py`

**Interfaces:**
- Consumes: final board SHA `663b9e3a56a12d0c69ac0c547921332d8341ef46bd4812a55a2a9a22bb2680ea` and manifest SHA `c6d2ad86653719ab55beabb70e549ffdd9e6c674790484dd6ce45af668cf38c1`.
- Produces: `verify_results(root: Path) -> list[str]`, where an empty list means the repository is consistent.

- [ ] **Step 1: Write failing verifier tests**

Create `tests/test_results.py` with standard-library `unittest`. Copy the repository results JSON into a temporary directory, mutate one condition at a time, and assert that `verify_results()` returns a specific error for:

```python
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

def test_rejects_stale_readme_language(self):
    errors = verify_public_prose("18 medals and one single autonomous run")
    self.assertIn("stale public claim: 18 medals", errors)
    self.assertIn("stale public claim: single autonomous run", errors)
```

- [ ] **Step 2: Run the tests and observe RED**

Run:

```bash
python -m unittest -v tests.test_results
```

Expected: import failure because `reproduce.verify_results` does not exist.

- [ ] **Step 3: Implement the validator**

Create `reproduce/verify_results.py` with:

```python
MEDAL_ORDER = {None: 0, "bronze": 1, "silver": 2, "gold": 3}
STALE_PHRASES = (
    "18 medals",
    "81.8%",
    "single autonomous run",
    "The 4 of 22",
    "not reportable",
    "on our roadmap",
    "how to read this honestly",
)

def verify_public_prose(text: str) -> list[str]:
    return [f"stale public claim: {phrase}" for phrase in STALE_PHRASES if phrase.lower() in text.lower()]

def verify_results(root: Path) -> list[str]:
    errors: list[str] = []
    data = json.loads((root / "results/lite22-three-run.json").read_text())
    tasks = data["tasks"]
    summary = data["summary"]
    if summary["medal_tasks"] != 19:
        errors.append("summary.medal_tasks must equal 19")
    if len(tasks) != 22 or len({task["task_id"] for task in tasks}) != 22:
        errors.append("results must contain 22 unique tasks")
    medals = [task["best_medal"] for task in tasks if task["best_medal"]]
    if Counter(medals) != Counter({"gold": 11, "silver": 5, "bronze": 3}):
        errors.append("best medal breakdown must equal 11 gold, 5 silver, 3 bronze")
    for task in tasks:
        confirmations = task["confirmations"]
        if len(confirmations) != 3:
            errors.append(f"{task['task_id']} must have exactly 3 confirmations")
            continue
        if task["best_medal"]:
            if not all(
                item["medal"] in MEDAL_ORDER and item["medal"] is not None
                and isinstance(item["score"], (int, float))
                and math.isfinite(item["score"])
                for item in confirmations
            ):
                errors.append(f"{task['task_id']} has an invalid medal confirmation")
            if not (root / "solutions" / task["task_id"] / "README.md").is_file():
                errors.append(f"missing solution README for {task['task_id']}")
    for index in range(3):
        count = sum(task["confirmations"][index]["medal"] is not None for task in tasks)
        if count != 19:
            errors.append(f"confirmation {index + 1} must medal on exactly 19 tasks")
    expected_rate = round(100 * 19 / 22, 2)
    if summary["any_medal_percentage"] != expected_rate or summary["sem"] != 0.0:
        errors.append("summary rate must equal 86.36% ± 0.00")
    for path in (root / "README.md", root / "reproduce/EVIDENCE.md"):
        errors.extend(verify_public_prose(path.read_text()))
    return errors
```

The CLI exits nonzero and prints one error per line when `errors` is nonempty.

- [ ] **Step 4: Add the canonical JSON**

Use schema `impulse-mlebench-lite22-three-run:v1` with:

```json
{
  "summary": {
    "tasks": 22,
    "medal_tasks": 19,
    "any_medal_percentage": 86.36,
    "sem": 0.0,
    "best_medals": {"gold": 11, "silver": 5, "bronze": 3},
    "confirmation_runs": 3
  },
  "evidence": {
    "board_sha256": "663b9e3a56a12d0c69ac0c547921332d8341ef46bd4812a55a2a9a22bb2680ea",
    "manifest_sha256": "c6d2ad86653719ab55beabb70e549ffdd9e6c674790484dd6ce45af668cf38c1"
  },
  "tasks": []
}
```

Each task row contains `task_id`, `best_medal`, `best_score`, `metric_direction`, `confirmations`, `method`, and public evidence hashes. Non-medal rows use `best_medal: null`, `best_score: null`, and three null confirmation results.

- [ ] **Step 5: Run RED-to-GREEN verification**

Run:

```bash
python -m unittest -v tests.test_results
python reproduce/verify_results.py
```

Expected: all tests pass and CLI prints `verified: 22 tasks, 19 medals, 3 confirmation runs`.

- [ ] **Step 6: Commit**

```bash
git add results/lite22-three-run.json reproduce/verify_results.py tests/test_results.py
git commit -m "feat: add Lite-19 three-run results ledger"
```

---

### Task 2: Product-facing README refresh

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: `results/lite22-three-run.json` from Task 1.
- Produces: public headline, 19-row medal table, comparison table, method statement, and three-task miss list.

- [ ] **Step 1: Add a failing README contract test**

Extend `tests/test_results.py`:

```python
def test_readme_carries_current_public_claim(self):
    text = (ROOT / "README.md").read_text()
    self.assertIn("19 medals on MLE-bench Lite-22", text)
    self.assertIn("86.36% ± 0.00", text)
    self.assertIn("11 gold / 5 silver / 3 bronze", text)
    self.assertIn("tabular-playground-series-may-2022", text)
```

- [ ] **Step 2: Run the test and observe RED**

Run `python -m unittest -v tests.test_results.ResultsContractTests.test_readme_carries_current_public_claim`.

Expected: failure because README still claims 18 medals.

- [ ] **Step 3: Rewrite README**

Use the approved headline. Replace the medal table with columns:

```markdown
| # | Competition | Best | Best score | Three-run medals | Confirmed scores | Solution |
```

Represent variation as `🥇 / 🥇 / 🥇` or `🥈 / 🥇 / 🥈`. State directly that public external data, web research, pretrained models, and exploit discovery are enabled. Explain that `± 0.00` refers to the any-medal rate across the three confirmation columns.

Update the comparison row to `86.36 ± 0.00 (19/22)`. Replace the four misses with NYC Taxi Fare, RANZCR, and SIIM-ISIC. Remove the old single-run caveats and July snapshot apology section.

- [ ] **Step 4: Run README verification**

Run:

```bash
python -m unittest -v tests.test_results
python reproduce/verify_results.py
```

Expected: all checks pass.

- [ ] **Step 5: Commit**

```bash
git add README.md tests/test_results.py
git commit -m "docs: publish 19-of-22 Lite result"
```

---

### Task 3: Evidence ledger and 19 solution pages

**Files:**
- Modify: `reproduce/EVIDENCE.md`
- Modify: `solutions/aerial-cactus-identification/README.md`
- Modify: `solutions/aptos2019-blindness-detection/README.md`
- Modify: `solutions/denoising-dirty-documents/README.md`
- Modify: `solutions/detecting-insults-in-social-commentary/README.md`
- Modify: `solutions/dog-breed-identification/README.md`
- Modify: `solutions/dogs-vs-cats-redux-kernels-edition/README.md`
- Modify: `solutions/histopathologic-cancer-detection/README.md`
- Modify: `solutions/jigsaw-toxic-comment-classification-challenge/README.md`
- Modify: `solutions/leaf-classification/README.md`
- Modify: `solutions/mlsp-2013-birds/README.md`
- Modify: `solutions/nomad2018-predict-transparent-conductors/README.md`
- Modify: `solutions/plant-pathology-2020-fgvc7/README.md`
- Modify: `solutions/random-acts-of-pizza/README.md`
- Modify: `solutions/spooky-author-identification/README.md`
- Modify: `solutions/tabular-playground-series-dec-2021/README.md`
- Modify: `solutions/text-normalization-challenge-english-language/README.md`
- Modify: `solutions/text-normalization-challenge-russian-language/README.md`
- Modify: `solutions/the-icml-2013-whale-challenge-right-whale-redux/README.md`
- Create: `solutions/tabular-playground-series-may-2022/README.md`
- Modify: `tests/test_results.py`

**Interfaces:**
- Consumes: canonical task rows and evidence hashes from Task 1.
- Produces: one evidence index and one readable task page per medal task.

- [ ] **Step 1: Add failing solution-page tests**

Extend `tests/test_results.py`:

```python
def test_every_medal_solution_lists_three_confirmations(self):
    data = self.load_results()
    for task in data["tasks"]:
        if task["best_medal"] is None:
            continue
        text = (ROOT / "solutions" / task["task_id"] / "README.md").read_text()
        self.assertIn("Three-run confirmation", text)
        for confirmation in task["confirmations"]:
            self.assertIn(confirmation["display_score"], text)
```

- [ ] **Step 2: Run the test and observe RED**

Run `python -m unittest -v tests.test_results.ResultsContractTests.test_every_medal_solution_lists_three_confirmations`.

Expected: failure on the first existing solution page.

- [ ] **Step 3: Replace the evidence ledger**

Lead with `19/22 (86.36% ± 0.00)`. Include the 11/5/3 best-result breakdown, one row per medal task, medal variation, exact confirmation scores, method category, board and manifest hashes, and links to the canonical JSON and task pages.

Describe external lookups as deliberate agent behavior. Keep evidence limitations factual and short; do not reintroduce banned public phrases.

- [ ] **Step 4: Add per-task confirmation blocks**

Append this shape to every medal solution page, using task-specific data:

```markdown
## Three-run confirmation

| Run | Medal | Score |
|---|---|---:|
| 1 | gold | 0.98723 |
| 2 | gold | 0.98750 |
| 3 | silver | 0.98701 |

**Best verified result:** gold, `0.98750`.
**Method:** stochastic RoBERTa seeds with public training data.
```

For deterministic routes, state `independent process confirmation` in the method. For Pizza and Dog Breed, state `external target lookup` and `external image lookup` without treating either as a disqualification.

- [ ] **Step 5: Add TPS May solution page**

Document the disjoint two-branch neural architecture, constrained interaction graph, three independently selected outer blends, OOF-only selection, scores `0.99821 / 0.99818 / 0.99822`, and medal variation `bronze / bronze / silver`.

- [ ] **Step 6: Run all repository result checks**

Run:

```bash
python -m unittest -v tests.test_results
python reproduce/verify_results.py
git diff --check
```

Expected: all tests pass, validator succeeds, and diff check is clean.

- [ ] **Step 7: Commit**

```bash
git add reproduce/EVIDENCE.md solutions tests/test_results.py
git commit -m "docs: add three-run medal evidence"
```

---

### Task 4: Independent verification and GitHub publication

**Files:**
- Modify if review requires: files from Tasks 1-3 only.
- GitHub metadata: repository description after merge.

**Interfaces:**
- Consumes: complete branch and validator output.
- Produces: pushed branch and draft PR against `main`.

- [ ] **Step 1: Run fresh verification**

```bash
python -m unittest -v tests.test_results
python reproduce/verify_results.py
git diff --check origin/main...HEAD
git status --short --branch
```

Expected: all tests pass; validator reports 22/19/3; no whitespace errors; worktree clean.

- [ ] **Step 2: Perform an independent content review**

Check every score against `results/lite22-three-run.json`, count table rows, follow every relative Markdown link, and scan for stale claim strings. Fix any Critical or Important finding and rerun Step 1.

- [ ] **Step 3: Push the branch**

```bash
git push -u origin agent/update-lite19-results
```

- [ ] **Step 4: Open a draft PR**

Title: `Publish 19-of-22 MLE-bench Lite result`

The PR body states what changed, why the prior 18-medal snapshot was stale, the 19/22 and 11/5/3 totals, exploit/external-data positioning, and exact verification commands.

- [ ] **Step 5: Report publication state**

Return the branch, commits, PR URL, checks, and the repository-description text to apply when the PR lands:

```text
19 medals on MLE-bench Lite-22 — 86.36% ± 0.00 across three confirmed runs, with scores, evidence, and reproducible solution code
```
