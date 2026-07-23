# Contributing to Cogtrix

Thank you for your interest in Cogtrix! We welcome bug reports, feature suggestions, and contributions under the terms described below.

## Important: Licensing

Cogtrix is released under the **Cogtrix Source-Available License 1.0**. By submitting any contribution (bug report, suggestion, patch, pull request, or other material), you agree that:

1. Your contribution may be used, modified, and incorporated into the Software by Northland Positronics (FZE) without restriction or obligation to you.
2. You have the right to submit the contribution and it does not violate any third-party rights.
3. Your contribution is provided under the same license terms as the Software.

See the [LICENSE](LICENSE) file for full terms.

## How to Contribute

### Reporting Bugs

1. Check existing [issues](../../issues) to avoid duplicates.
2. Open a new issue with:
   - A clear, descriptive title
   - Steps to reproduce the problem
   - Expected vs. actual behavior
   - Your environment (OS, Python version, provider, model)
   - Relevant log output (run with `--log -v` for verbose logs)

### Suggesting Features

1. Open an issue with the **feature request** label.
2. Describe the use case — what problem does it solve?
3. If possible, outline how you envision it working.

### Submitting Code

1. Fork the repository.
2. Create a feature branch from `main`.
3. Make your changes following the code style guidelines below.
4. Add or update tests as appropriate.
5. Run the test suite:

```bash
uv run pytest
```

6. Submit a pull request with:
   - A clear description of what the change does and why
   - Reference to any related issues

### Code Style

- Python 3.13+
- Follow existing code patterns and conventions
- Use type hints where practical
- Run linting before submitting:

```bash
uv run flake8
uv run mypy src/
```

- Keep commits focused — one logical change per commit
- Write clear commit messages

### Adding Tools

Cogtrix uses auto-discovery for tools. To add a new tool:

1. Create a new file in `src/tools/`.
2. Define your tool function(s) with proper type hints and docstrings.
3. Define an input schema using Pydantic.
4. Add a `TOOL_CONFIG` dictionary for auto-registration.
5. See `src/tools/calculator.py` for a minimal example.
6. Add tests in `tests/`.

## Code of Conduct

All participants are expected to follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Questions?

If you have questions about contributing, open a discussion or issue.
