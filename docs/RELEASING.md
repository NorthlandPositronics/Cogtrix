# Releasing Cogtrix

The complete, step-by-step playbook for cutting a Cogtrix release. This is the
**operational how-to**; the **versioning policy** (semver rules, what counts as
breaking, the pre-1.0 bump flags) lives in [VERSIONING.md](VERSIONING.md) §6/§6a.
Read both before your first cut.

> Every rule below exists because it broke a real release. The v0.5.0 and v0.5.1
> cuts each stalled or failed several times; the failure modes and their fixes are
> captured in §7–§8 so they don't happen again.

---

## 1. Mental model

### Branches
| Branch | Role | Who pushes |
|---|---|---|
| `release/next` | Default branch. Integration line — every feature/fix PR targets it. | feature PRs (`feature/…`, `fix`/`bugfix/…`, `hotfix/…`) |
| `production` | The released line. Tags `vX.Y.Z` live here. | **only** PRs from `release/next` (the promotion) and from `release-please--*` branches |

Direct pushes to `release/next` and `production` are **forbidden by ruleset**. Both
require **2 approvals + a code-owner review + the CI Summary check**, and all commits
must be **GPG-signed**. Nothing self-merges.

### What release-please does
[release-please](https://github.com/googleapis/release-please) watches `production`.
When a release-triggering commit lands there, it opens a **`chore(production): release
X.Y.Z`** PR that bumps the version files and writes `CHANGELOG.md`. Merging *that* PR
tags `vX.Y.Z` and fires the publish pipeline. Config + state:
- `.github/release/release-please-config.json` — bump rules (`bump-minor-pre-major: true`, `bump-patch-for-minor-pre-major: true`).
- `.github/release/release-please-manifest.json` — the last released version.
- `extra-files` keeps `src/_version.py` in sync.

### Workflows involved
| Workflow | Trigger | Purpose |
|---|---|---|
| `release-title-guard.yml` | PR → production (opened/edited) | Title must be `feat\|fix\|deps\|docs` so the squash cuts a release (`release-please--*` exempt). |
| `branch-source-guard.yml` | PR → production | Source must be `release/next` (or `release-please--*`). |
| `openapi-diff.yml` | PR → production / release/next | Fails on a breaking REST contract change. |
| `ci.yml` | every PR | Quality (black/ruff/pyright/bandit) + unit/api test shards. |
| `markdownlint.yml` (Docs CI) | PR touching `*.md`/`docs/**`/… | Lints the **full** `**/*.md` set. |
| `release-please.yml` | push to `production` (+ `workflow_dispatch`) | Runs release-please; posts status checks on the release PR; dispatches the publish pipeline; **guards against a stalled cut**. |
| `release.yml` | dispatched by `release-please.yml` after a tag | The actual publish pipeline. |
| `release-back-merge.yml` | release PR closed/merged | Opens a `production → release/next` back-merge PR. |

---

## 2. Version number — increment the PATCH

Cogtrix is pre-1.0 and follows an **increment-patch convention** (#2324). Pre-1.0:
- `fix:` → **patch**
- `feat:` → **patch** (not minor — this is deliberate)
- `feat!:` / `BREAKING CHANGE:` → **minor** (capped; never forces 1.0.0)

So the release after `v0.5.0` is **`v0.5.1`**, **never `v0.6.0`**. To cut a specific
number (e.g. crossing to `1.0.0`), use a `Release-As: X.Y.Z` trailer (§4.2). Full
table: [VERSIONING.md §6a](VERSIONING.md).

---

## 3. One-time repo setting (required — the pipeline stalls without it)

**Settings → General → Pull Requests → "Default commit message for squash merging" →
`Pull request title and description`.**

Why it is mandatory: the promotion PR squashes 700+ commits into one production
commit. With GitHub's *default* squash message, the body becomes the **concatenation
of every commit** — a 100 KB+ blob that **breaks the release-please parser** (it cuts
nothing) and **buries any `Release-As:` footer**. With this setting, the squash body is
just the short PR description. This single toggle is what makes a clean cut possible;
it stalled both v0.5.0 and v0.5.1 before it was set (#2330).

---

## 4. Cutting a release — step by step

### 4.1 Pre-flight
- All intended PRs are **merged into `release/next` and green**. Do not cut with red or
  `CHANGES_REQUESTED` PRs open.
- `release/next` is even with the last release (the previous **back-merge** landed — §4.5).
  If not, expect metadata conflicts; resolve the back-merge first.

### 4.2 Open the promotion PR (`release/next` → `production`)
- **Title:** `fix: vX.Y.Z release` — a release-triggering type (`feat`/`fix`/`deps`/
  `docs`); `fix:` cuts a patch under our config. A `chore:`/`refactor:`/`ci:` title
  produces **no** release (the title-guard blocks it).
- **Body:** keep it **short**, and make the **last line** a standalone trailer:

  ```text
  Integration cut for vX.Y.Z (release/next → production). Patch bump. Human-gated.

  Release-As: X.Y.Z
  ```

  The `Release-As:` line makes the version deterministic. It **must** be a real
  trailer — the last paragraph, at the start of its own line — **in the PR
  description** (the squash uses title+description; see §3). Never write it mid-sentence
  or inside backticks: that is not a trailer and release-please ignores it (this is the
  #2329 failure).
- **Do not enable auto-merge.** Release/promotion PRs are human-gated.
- Confirm CI is green (note the markdownlint caveat in §6).

### 4.3 Merge the promotion PR
A maintainer (not you) squash-merges after 2 approvals + code-owner + CI. **Keep the
default squash message** (title + your description with the `Release-As:` trailer). Do
not hand-edit it to strip the footer.

### 4.4 Merge the release-please version PR
Within a minute or two, release-please opens **`chore(production): release X.Y.Z`**
(label `autorelease: pending`). `release-please.yml` auto-formats `CHANGELOG.md`,
regenerates `uv.lock`, and posts the required status checks on it (release-please's bot
can't trigger them itself). Review and merge it → this **tags `vX.Y.Z`** and dispatches
`release.yml` (`trigger-release` runs only when `release_created == 'true'`).

> On the *promotion* push, `trigger-release` shows **Skipped** — that is **expected**
> (`release_created=false` until the version PR merges), not a failure. See §6.

### 4.5 Merge the back-merge PR
When the version PR merges, `release-back-merge.yml` opens **`chore: back-merge
production → release/next post vX.Y.Z`**. This carries the version bumps back to
`release/next` so the next cut doesn't conflict on them. Review and merge it. If it
**aborts** on a non-metadata conflict, do the manual back-merge in §7.4.

---

## 5. Verification checklist

After **4.3** (promotion merged):
- [ ] A `chore(production): release X.Y.Z` PR exists (label `autorelease: pending`).
- [ ] If not within ~2 min, the `verify-release-pr-opened` job has gone **red** — go to §7.

After **4.4** (version PR merged):
- [ ] Tag `vX.Y.Z` exists: `git fetch --tags && git tag | grep vX.Y.Z`.
- [ ] `.github/release/release-please-manifest.json` on `production` reads `X.Y.Z`.
- [ ] `pyproject.toml` / `src/_version.py` read `X.Y.Z`.
- [ ] `release.yml` / image publish ran for the tag.

After **4.5** (back-merge merged):
- [ ] `release/next` manifest/version read `X.Y.Z` (no drift vs `production`).

---

## 6. The "skipped / stalled" signals, decoded

| What you see | Meaning | Action |
|---|---|---|
| `trigger-release` **Skipped** on the promotion push | Normal. `release_created=false` until the version PR merges. | None — it fires after §4.4. |
| `verify-release-pr-opened` **red** | A promotion landed but no version PR appeared — a real stall. | §7.1 / §7.2 per the job's printed cause. |
| `markdownlint` red on a production PR | Docs CI lints the **full** `**/*.md` set; a file its `paths:` trigger never saw on a `tests/` PR now fails. Verbatim prompts (`tests/**/system_prompt.md`) are excluded (#2327). | Fix or exclude the file. Note: CI does **not** check out the `docs/optional` submodule or gitignored `.idea`/`.claude`, so local lint shows extra noise that isn't in CI. |
| Back-merge workflow red | A non-metadata file conflicted (squash-divergence). | §7.4. |

---

## 7. Recovery runbooks

### 7.1 Stall — transient miss (normal-sized promotion commit)
Re-run the **Release Please** workflow (Actions tab → Release Please → Run workflow,
`workflow_dispatch`). It re-evaluates and opens the version PR. Direct pushes to
`production` are forbidden, so re-running is the only push-free retry. Root cause: #2283.

### 7.2 Stall — oversized squash body (100 KB+ commit message)
A re-run will **not** help — it re-parses the same poisoned commit. Force the version:

```bash
git checkout -B release-please--force-vX.Y.Z origin/production
git commit --allow-empty -S -m "fix: vX.Y.Z release

Release-As: X.Y.Z"
git push -u origin release-please--force-vX.Y.Z
```

Then open a PR → `production`, titled `fix: vX.Y.Z release`, whose **PR description's
last line** is `Release-As: X.Y.Z` (the `release-please--*` prefix is exempt from the
branch/title guards). Human-gated. The clean trailer overrides the broken parse.
Proven: #2286 (v0.5.0), #2332 (v0.5.1).

> **The trailer must live in the PR description, as a trailer.** A `Release-As` only in
> the commit message is dropped by the title+description squash; written as inline prose
> it isn't a trailer. That exact mistake (#2329) re-stalled v0.5.1.

### 7.3 Permanent fix for §7.1/§7.2
Set the repo squash setting in §3. With it, the promotion body is the short PR
description and neither stall can occur.

### 7.4 Back-merge aborted (non-metadata conflict)
The workflow auto-resolves only the metadata set (`CHANGELOG.md`, the manifest,
`pyproject.toml`, `src/_version.py`, `uv.lock`). If any **other** file conflicts (e.g.
a workflow edited on `release/next` after the promotion snapshot), it aborts for a
human. Do it manually:

```bash
git fetch origin
git checkout -b hotfix/<issue>-backmerge-vX.Y.Z origin/release/next
git merge origin/production
# Metadata files auto-merge to X.Y.Z. For each remaining conflict, release/next is
# normally the superset (production only ever gets these files FROM release/next):
git checkout --ours <conflicted-non-metadata-file>   # keep release/next's version
git add <conflicted-non-metadata-file>
git commit --no-edit
git push -u origin hotfix/<issue>-backmerge-vX.Y.Z
```

Open a PR → `release/next`. Verify the metadata files read `X.Y.Z` and the resolved
files match release/next HEAD. Example: #2335 (v0.5.1, `release-please.yml` conflict);
structural discussion in #2334.

---

## 8. What to avoid

- ❌ A **minor** bump / `0.6.0` — pre-1.0 it's a **patch** (`0.5.1`).
- ❌ A **long** promotion PR body — it degrades the release-please parser.
- ❌ `Release-As` only in the commit message, or as inline prose — it must be the **last
  line of the PR description**, as a trailer.
- ❌ A `chore:`/`refactor:`/`ci:` **title** on the promotion PR — it cuts no release.
- ❌ **Re-running** the workflow to fix an *oversized-body* stall — force with
  `Release-As` instead (§7.2).
- ❌ **Hand-editing** the squash message to drop the `Release-As` footer.
- ❌ **Self-merging** any release/promotion/back-merge PR — rulesets require 2 approvals
  + code-owner. Force/back-merge branches use `release-please--*` or
  `hotfix/<issue>-…` names (the only pushable prefixes).
- ❌ Proposing `gh` for stuck steps where the UI/MCP is the path.

---

## 9. Reference

- Policy: [VERSIONING.md](VERSIONING.md) §6/§6a (bump rules, breaking-change definitions).
- Config: `.github/release/release-please-{config,manifest}.json`, `src/_version.py`.
- Workflows: `.github/workflows/release-please.yml`, `release.yml`, `release-back-merge.yml`, `release-title-guard.yml`, `branch-source-guard.yml`.
- History / rationale: #1940 (back-merge automation), #2283 (stall 5-whys), #2324
  (increment-patch convention), #2330 (squash-body root cause), #2334 (back-merge
  divergence).
