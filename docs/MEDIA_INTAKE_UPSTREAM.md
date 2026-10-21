# Acquisition upstream and research integration

The acquisition source of truth is now
[Code4me2/media-intake](https://github.com/Code4me2/media-intake), initially a
private repository. It was extracted with `git subtree split` from this project's
`packages/media-intake` directory, preserving the package's history. No media,
transcripts, private acquisition metadata, or research datasets were transferred.

## Why a copy remains here

`packages/media-intake/` is now a **vendored, commit-pinned dependency**, not a
second independently maintained implementation. The exact upstream revision and
Git tree are recorded in [`packages/media-intake.upstream.json`](../packages/media-intake.upstream.json).

Keeping this snapshot lets public CI, ordinary clones and self-contained parent
wheels work without credentials to the private upstream repository. No Git
submodule initialization or network access to upstream is required at runtime or
build time. Python dependency installation still uses the usual package indexes.

The initial snapshot is the extracted package commit, before standalone CI and
repository documentation were added upstream. Its acquisition implementation
matches the new repository's initial main branch. This application's existing
CLIs, receipt schemas and previously downloaded files remain unchanged.

## Where changes belong

- **Upstream:** metadata/media/caption adapters, format and caption policies,
  provenance, verification, generic orchestration, storage and generic CLI.
- **Here:** permission confirmation, dataset governance, channel auditing, causal
  clip generation, sampling, annotation, agreement checks and model experiments.

Make reusable acquisition changes upstream first. Use the standalone package's
own tests and wheel smoke checks. The repository name is `media-intake`; the
Python distribution stays `media-intake-local`, import `media_intake`, and CLI
`media-intake`. No package-index publication is required.

## Updating the pinned dependency

Updates are deliberate maintainer changes, not automatic pulls of a moving branch:

1. Obtain access to upstream and choose a reviewed full commit SHA.
2. Replace the vendored directory with **only tracked files** from that commit's
   Git tree, handling deleted files as well as additions. Do not copy `.git`,
   virtual environments, caches, build outputs or private acquisition artifacts.
3. Update both `revision` and `git_tree` in the pin manifest. The Git tree should
   equal `git rev-parse <upstream-sha>^{tree}` in the upstream checkout. There
   should be no local patches hidden inside the vendored snapshot.
4. Reconcile any dependency changes with the root workspace lock using `uv lock`.
   Check compatibility wrappers and explicitly handle receipt schema migrations.
5. Run the host and generic tests, build/install both wheels outside the checkout,
   and commit the snapshot and pin together on a review branch:

   ```bash
   uv sync --locked --extra dev
   uv run python -m pytest tests packages/media-intake/tests
   git diff --check
   ```

CI compares the committed vendored tree against the pin without contacting the
private repository. That detects drift from the recorded snapshot, not the
trustworthiness of upstream or proof that a revision was reviewed. Reviewers
with upstream access should additionally check the recorded revision/tree pair.

Do not replace this snapshot with an unpinned private Git dependency in the
public package: that would break clean public installs and CI.
