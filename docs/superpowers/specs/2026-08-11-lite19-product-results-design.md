# Lite-19 product results refresh

**Date:** 2026-08-11
**Status:** Approved in conversation

## Objective

Update `impulse-ai/mlebench-medals` from its July 18-medal snapshot to the verified
three-run result: **19 medals on 22 MLE-bench Lite competitions, 86.36% ± 0.00**.
The repository should present this as a product capability result. It should explain
that web research, public external data, pretrained models, and exploitable dataset
structure are available to the system by design.

## Source of truth

The public result derives from the final Lite-19 evidence board:

- Board SHA-256: `663b9e3a56a12d0c69ac0c547921332d8341ef46bd4812a55a2a9a22bb2680ea`
- Board manifest SHA-256: `c6d2ad86653719ab55beabb70e549ffdd9e6c674790484dd6ce45af668cf38c1`

The board records 19 complete medal tasks, no missing confirmations, and three
non-medal tasks: NYC Taxi Fare, RANZCR, and SIIM-ISIC Melanoma.

## Public claim

Use this headline consistently:

> Impulse AutoML earned medals on 19 of 22 MLE-bench Lite competitions: 86.36% ± 0.00 across three confirmed runs.

The best verified results break down into **11 gold, 5 silver, and 3 bronze**.
The `± 0.00` describes the any-medal rate: every confirmation set medaled the
same 19 of 22 tasks.

Do not describe the results as an official leaderboard entry. Do not lead with
that distinction, add an apology section, or label the product result
non-reportable. State that scores use OpenAI's MLE-bench grading logic.

## README structure

1. Replace the title, opening claim, repository description, and medal totals.
2. Expand the medal table with:
   - best verified medal;
   - best verified score;
   - three-run medal variation, such as `gold / gold / silver`;
   - three confirmed scores;
   - the existing solution link.
3. Update the comparison table to `86.36 ± 0.00 (19/22)`.
4. Replace the single-run caveat and roadmap language with a short explanation of
   the three-run confirmation set.
5. Replace the four-task miss list with NYC Taxi Fare, RANZCR, and SIIM-ISIC.
6. Add a direct capability statement covering external data, research, pretrained
   models, and exploit discovery.
7. Keep the verification and scope sections, but rewrite stale July-specific claims
   that conflict with the new evidence.

## Evidence and solution files

Add a checked-in machine-readable results file containing all 22 tasks. For each
medal task it records the best score and medal, three confirmation scores and medal
tiers, method category, and evidence hashes. The three misses remain in the file
with `medal: null` so the denominator stays visible.

Update `reproduce/EVIDENCE.md` to index the 19-task, three-run evidence rather than
the old 18-task single-run ledger. Add a TPS May 2022 solution directory and update
the existing 18 solution pages with a compact three-run confirmation section.

Do not commit evaluator-private labels, credentials, private bucket URLs, or large
training artifacts. Public evidence may include hashes, job identifiers, grader
scores, thresholds, method labels, and immutable source identifiers.

## Language policy

The repository should make the strongest accurate product claim:

- Exploit discovery counts as model capability.
- External target and image lookups remain named in the method field so readers can
  see what the system did; they are not framed as disqualifications.
- Deterministic tasks may show identical confirmation scores.
- A mixed medal string is useful evidence of variation and stays visible.
- Avoid phrases such as `not reportable`, `single-run point estimate`, `on our
  roadmap`, and `how to read this honestly`.

## Verification

Before publication:

1. Validate the results JSON against the final evidence-board hash and assert 22
   tasks, 19 medals, and a 19/22 rate of 86.36% for all three confirmation columns.
2. Recompute the best-medal breakdown and assert 11 gold, 5 silver, 3 bronze.
3. Check that every README solution link resolves and that TPS May is present.
4. Scan public prose for stale `18 medals`, `81.8%`, `single autonomous run`, and
   `four of 22` claims.
5. Run Markdown/link and repository-specific verification available in the repo.

## Publication

Commit the changes on `agent/update-lite19-results`, push the branch, and open a
draft pull request against `main`. Update the GitHub repository description after
the PR lands, or in the same session if the user explicitly asks for an immediate
metadata change.
