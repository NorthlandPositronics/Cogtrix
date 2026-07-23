# Versioning and Stability Policy

## 1. Version Scheme

Cogtrix follows [Semantic Versioning 2.0.0](https://semver.org/).

```
MAJOR.MINOR.PATCH
```

| Component | Incremented when |
|-----------|-----------------|
| `MAJOR`   | Incompatible, breaking changes are introduced |
| `MINOR`   | New features added in a backwards-compatible way |
| `PATCH`   | Backwards-compatible bug fixes and security patches |

**v0.2.0** is the first stable release. All stability guarantees below apply from v0.2.0 onward.

Pre-release suffixes follow semver conventions: `-alpha.1`, `-beta.2`, `-rc.1`.

---

## 2. What Is a Breaking Change

A breaking change is any modification that requires users or integrators to update their
code, config, scripts, or tooling without a fallback. Each layer has its own definition.

### REST API (`/api/v1/`)

Breaking:
- Removing an endpoint or HTTP method
- Removing a field from any response body (required or optional)
- Renaming a field in a request or response body
- Changing the type of an existing field (e.g. `string` → `integer`)
- Changing a successful HTTP status code (e.g. `200` → `201`, `204` → `200`)
- Changing an error status code to a different 4xx/5xx class
- Making a previously optional request field required
- Removing or renaming a query parameter that was previously accepted

Not breaking:
- Adding a new endpoint
- Adding a new optional field to a response body
- Adding a new optional query parameter
- Adding a new enum value to an existing enum (consumers must handle unknown values)
- Changing internal implementation with no observable contract change

### WebSocket Protocol (`/ws/v1/`)

Breaking:
- Removing a server message type
- Removing a field from an existing server message payload
- Changing the meaning of an existing message type
- Removing a client message type that the server previously accepted

Not breaking:
- Adding a new server message type (clients must ignore unknown types)
- Adding a new optional field to an existing message payload

### CLI

Breaking:
- Removing a slash command or CLI flag
- Changing the output format of a command in a machine-readable way (e.g. `/info` JSON output)
- Changing the name of a config file path searched by default
- Removing a previously supported environment variable

Not breaking:
- Adding new slash commands or flags
- Changing human-readable output formatting (colors, spacing, wording)
- Adding new environment variables

### Configuration File (`.cogtrix.yaml`)

Breaking:
- Removing a configuration key that was previously accepted
- Renaming a key without providing an auto-migration path
- Changing the type or allowed values of a key in a backwards-incompatible way
- Removing an environment variable override without replacement

Not breaking:
- Adding a new optional configuration key with a documented default
- Adding a new environment variable override

### Plugin / Tool API

Breaking:
- Changing the signature of `TOOL_SETUP(config)` in a backwards-incompatible way
- Changing the required fields in `TOOL_CONFIGS` entries
- Removing or renaming a field in `TOOL_CONFIG` / `TOOL_CONFIGS`
- Changing the `cogtrix.tools` entry-point contract

Not breaking:
- Adding optional fields to `TOOL_CONFIGS`
- Adding new hook methods to the pluggy hookspec with default implementations
- Adding new optional parameters to `TOOL_SETUP`

---

## 3. Stability Guarantees

### After v0.2.0

- **No breaking change** (as defined above) will be introduced without a `MAJOR` version bump.
- A `MAJOR` bump is the only mechanism for removing or incompatibly changing existing behaviour.
- `MINOR` releases add new features and new optional fields/endpoints. Existing integrations
  continue to work without any modification.
- `PATCH` releases contain bug fixes, security patches, and performance improvements only.
  No new public surface is added.

### Deprecation Policy

Before any breaking change is introduced in a `MAJOR` release:

1. The item is marked deprecated in the release notes and, where possible, with a runtime warning.
2. The deprecated item remains functional for **at least one `MINOR` release cycle** after the
   deprecation notice.
3. The final removal is documented in the `CHANGELOG.md` with a migration path.

Example timeline:
```
v1.2.0  — endpoint /foo deprecated, warning added
v1.3.0  — /foo still works, deprecation warning in docs
v2.0.0  — /foo removed (MAJOR bump required)
```

### Security Exceptions

A security vulnerability may require an incompatible change in a `PATCH` or `MINOR` release
if delaying the fix to the next `MAJOR` release would leave users at unacceptable risk.
Such exceptions are clearly documented in the release notes with the CVE reference.

---

## 4. Pre-Release Versions (v0.x)

Versions prior to v0.2.0 carry **no stability guarantee**.

Any release in the `v0.x` series may introduce breaking changes at any version component.
Users running v0.x should:

- Pin to an exact version in production.
- Consult `docs/MIGRATION.md` before upgrading across minor versions.
- Expect breaking changes to be documented in `CHANGELOG.md` under a `BREAKING` header.

The v0.x → v0.2.0 migration is fully documented in [docs/MIGRATION.md](MIGRATION.md).

---

## 5. Checking for Breaking Changes

### In CI (automatic)

Every pull request targeting `main` runs the **API Breaking Change Check** workflow
(`.github/workflows/api-breaking-change.yml`). The workflow:

1. Generates the OpenAPI spec from the current branch.
2. Retrieves the OpenAPI spec from the last stable git tag.
3. Diffs the two specs and fails the check if any endpoints have been removed.

A failed check blocks merge. The PR author must either:
- Revert the breaking change, or
- Obtain explicit sign-off from a maintainer acknowledging the MAJOR bump requirement.

### Locally

Generate the current OpenAPI spec:

```bash
uv run python -c "
import json
from src.api.app import create_app
app = create_app()
with open('openapi.json', 'w') as f:
    json.dump(app.openapi(), f, indent=2)
"
```

Compare two specs with `openapi-diff`:

```bash
uvx openapi-diff openapi-baseline.json openapi-current.json
```

Or install and run directly:

```bash
pip install openapi-spec-validator
python3 -c "
import json, sys
old = json.load(open('openapi-baseline.json'))
new = json.load(open('openapi-current.json'))
old_paths = set(old.get('paths', {}).keys())
new_paths = set(new.get('paths', {}).keys())
removed = old_paths - new_paths
if removed:
    print('BREAKING: removed endpoints:', removed)
    sys.exit(1)
print('No removed endpoints detected.')
"
```

---

## 6. Changelog

All changes are documented in `CHANGELOG.md` (generated by
[Release Please](https://github.com/googleapis/release-please)).

- Breaking changes appear under a `### Breaking Changes` header and include a migration note.
- Deprecations appear under a `### Deprecated` header.
- The git tag for every release matches the version string exactly (e.g. `v1.2.3`).

Conventional commit prefixes used in this project:

| Prefix | Version bump |
|--------|-------------|
| `fix:` | PATCH |
| `feat:` | MINOR |
| `feat!:` or `BREAKING CHANGE:` footer | MAJOR |
| `refactor:`, `test:`, `docs:`, `chore:` | no bump (PATCH on release) |

---

## 7. Supported Versions

| Version | Status | Support ends |
|---------|--------|-------------|
| v0.2.x (current) | Active — security + bug fixes | Until v0.3.0 + 6 months |
| v0.1.x | End-of-life | No further patches |

Only the latest `MINOR` release of the current `MAJOR` line receives security patches.
Users are encouraged to upgrade to the latest `MINOR` version before the next `MAJOR` release.
