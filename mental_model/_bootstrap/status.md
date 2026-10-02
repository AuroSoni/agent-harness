# Bootstrap status

Working state for the mental-model bootstrap of this repo. This folder is deleted before the PR is marked ready.

## Where things stand

| | |
|---|---|
| Phase | 2: Checkpoint 1 (characters and map) written, waiting on Auro |
| Branch | `AuroSoni/mental-model` (the Conductor workspace branch; the handoff's default name `mental-model-bootstrap` is not used) |
| PR base | `dev` |
| Kit | `mental-model-kit/` (`HANDOFF.md` plus `templates/`), kept outside the repo and never committed |
| Last updated | 2026-10-02 |

## Waiting on Auro

- Answers to [1-characters-and-map.md](1-characters-and-map.md), questions Q1 to Q15.
- Q1 (which repo is the product's home) blocks the cross-repo parts. Q2 (where the turn loop lives, and what to call it) and Q3 (how the model sits beside `interface_plan/` and `AMENDMENTS.md`) block the file list.

## Next

1. Record Auro's answers in the checkpoint file.
2. Fix the cast and layout from those answers.
3. Write `2-plot.md` (Checkpoint 2): decisions and anomalies as guess-first questions. The raw material is in [survey.md](survey.md) under "Anomalies" and in [sources.md](sources.md).
4. Checkpoint 3: pilot one subsystem file and one feature file from the areas Auro names in Q8.

## Multi-repo run

- The three repos in the `mental-model` feature group run in parallel: `agent-harness` (this run), `nova_backend`, `nova_excel_addin`. All three are on branch `AuroSoni/mental-model`.
- No product map is settled anywhere yet, and the home repo is not confirmed (Q1). This run therefore drafts only where this repo sits, from the code.
- This run edits only this repo.

## Phase 1 asks (still open, folded into Q15)

- A spoken walkthrough of the library, if one exists or can be recorded.
- GitHub handle: assumed `AuroSoni`.
- Git author emails: all three identities in `git log` are assumed to be Auro's.
- Any notes or docs to use beyond what is in the repo.

## Files in this folder

- [survey.md](survey.md): phase 1 findings, condensed.
- [sources.md](sources.md): candidate whys and where each was found. None is confirmed yet.
- [1-characters-and-map.md](1-characters-and-map.md): Checkpoint 1.
