# Contributing to Cogtrix

Thank you for your interest in Cogtrix! We welcome bug reports, feature suggestions, and code contributions under the terms described below.

## Important: Licensing

Cogtrix is released under the **Cogtrix Source-Available License 1.0**. By submitting any contribution (bug report, suggestion, patch, pull request, or other material), you agree that:

1. Your contribution may be used, modified, and incorporated into the Software by Northland Positronics (FZE) without restriction or obligation to you.
2. You have the right to submit the contribution and it does not violate any third-party rights.
3. Your contribution is provided under the same license terms as the Software.

See the [LICENSE](LICENSE) file for full terms.

## Getting Started

1. **Fork** the repository and clone your fork.
2. **Install dependencies:** `uv sync`. (No `uv`? Generate a pip file with `uv export --no-dev --no-hashes -o requirements.txt`, then `pip install -r requirements.txt`.)
3. **Run the test suite** to make sure everything works: `uv run pytest tests/ -v`.
4. **Read the docs** — [Architecture](docs/ARCHITECTURE.md) for system design, [Development](docs/DEVELOPMENT.md) for practical extension guides.

> **Optional submodule** — `docs/optional/` is a git submodule pointing at the
> private `NorthlandPositronics/cogtrix-docs` repository. It is **not required**
> to build, test, or run Cogtrix. Authorised contributors with access to that
> repository can fetch its contents with `git submodule update --init docs/optional`
> (or by cloning the parent repo with `--recurse-submodules`). Public
> contributors and CI runners without access can ignore it; the directory will
> appear empty and no build steps depend on it.

## How to Contribute

### Reporting Bugs

1. Check existing [issues](../../issues) to avoid duplicates.
2. Open a new issue with:
   - A clear, descriptive title
   - Steps to reproduce the problem
   - Expected vs. actual behavior
   - Your environment (OS, Python version, provider, model)
   - Relevant log output (run with `--debug` for verbose logs)

### Suggesting Features

1. Open an issue with the **feature request** label.
2. Describe the use case — what problem does it solve?
3. If possible, outline how you envision it working.

### Submitting Code

1. Create a feature branch from `release/next` (the integration branch — not `production`):
   ```bash
   git checkout release/next
   git checkout -b feat/my-feature
   ```
   Use the naming convention `feat/<short-description>` or `fix/<short-description>`.
2. Make your changes following the code style guidelines below.
3. Add or update tests as appropriate.
4. Run all quality checks:

```bash
uv run black cogtrix.py src/ tests/
uv run ruff check cogtrix.py src/ tests/
uv run pyright cogtrix.py src/
uv run pytest tests/ -v
```

5. Submit a pull request targeting `release/next` with:
   - A clear description of what the change does and why
   - Reference to any related issues

> **Branch policy:** `production` is release-only. All pull requests must target `release/next`. The CI enforces this — PRs directly to `production` from non-release branches are blocked.

### Your First Contribution

Good places to start:

- **Add a new tool** — see [Development Guide: Adding Custom Tools](docs/DEVELOPMENT.md#adding-custom-tools). Drop a `.py` file in `src/tools/` with a `TOOL_CONFIG` dict, and it's auto-discovered.
- **Improve test coverage** — pick a tool or memory mode and add edge-case tests.
- **Fix a typo or clarify docs** — documentation improvements are always welcome.

## Code Style

- Python 3.13.x
- Follow existing code patterns and conventions
- Use type hints where practical
- Format with [Black](https://github.com/psf/black) (line length 100)
- Lint with [Ruff](https://docs.astral.sh/ruff/)
- Type-check with [Pyright](https://github.com/microsoft/pyright)
- Keep commits focused — one logical change per commit
- Use [Conventional Commits](https://www.conventionalcommits.org/) for commit messages (`feat:`, `fix:`, `docs:`, etc.)

## Touching the Gate 2 Eval Framework?

If your PR modifies `tests/evaluation/runner.py`,
`tests/evaluation/stub_tool_registry.py`,
`tests/evaluation/ci_gate2.py`, `tests/evaluation/models.yaml`, or any
file under `tests/evaluation/scenarios/`, please follow the
strict-schema canary policy. The detailed policy doc lives in the
private documentation submodule at `docs/optional/testing/eval-canary.md`
(authorised contributors only). In short — run the two canary models
(`deepseek-v3` + `kimi-k2-5`) against the smoke matrix locally and paste
the summary into the PR body.  It's a 3-minute, $0.05 check that
catches the description-only-stubs class of regression before Gate 2
has to.

## Code of Conduct

All participants are expected to follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Questions?

If you have questions about contributing, open a discussion or issue.
