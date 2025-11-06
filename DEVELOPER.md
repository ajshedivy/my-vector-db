# Developer Guide

This guide covers development workflows, tooling, and release processes for my-vector-db.

## Table of Contents

- [Developer Guide](#developer-guide)
  - [Table of Contents](#table-of-contents)
  - [Development Setup](#development-setup)
    - [Prerequisites](#prerequisites)
    - [Initial Setup](#initial-setup)
    - [Running the Server Locally](#running-the-server-locally)
  - [Code Quality Tools](#code-quality-tools)
    - [Type Checking](#type-checking)
    - [Linting and Formatting](#linting-and-formatting)
    - [Testing](#testing)
  - [Release Process](#release-process)
    - [Overview](#overview)
    - [Workflow](#workflow)
    - [Versioning Guidelines](#versioning-guidelines)
    - [Pre-Release Testing](#pre-release-testing)
    - [Dry Run (Preview Changes)](#dry-run-preview-changes)
  - [CI/CD Workflows](#cicd-workflows)
    - [Publish Python Package and Docker Image](#publish-python-package-and-docker-image)
    - [Build and Push Docker Image](#build-and-push-docker-image)
  - [Repository Protection](#repository-protection)
    - [Branch Rules](#branch-rules)
    - [Release Security](#release-security)
  - [Troubleshooting](#troubleshooting)
    - [Version Bump Issues](#version-bump-issues)
    - [Workflow Failures](#workflow-failures)
  - [Additional Resources](#additional-resources)


## Development Setup

### Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) for dependency management
- Git

### Initial Setup

```bash
# Clone the repository
git clone https://github.com/ajshedivy/my-vector-db.git
cd my-vector-db

# Install dependencies
uv sync --all-groups

# Install pre-commit hooks (optional)
uv run pre-commit install
```

### Running the Server Locally

```bash
# Start the API server with auto-reload
uv run uvicorn my_vector_db.main:app --reload

# Or use Docker Compose
docker compose up -d
```

The API will be available at:
- API Endpoint: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

## Code Quality Tools

### Type Checking

```bash
# Run mypy type checker
uv run mypy src
```

### Linting and Formatting

```bash
# Check code style
uv run ruff check

# Auto-fix issues
uv run ruff check --fix

# Format code
uv run ruff format
```

### Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=my_vector_db --cov-report=html

# Run specific test file
uv run pytest tests/test_sdk.py
```

## Release Process

### Overview

Releases are managed using semantic versioning (MAJOR.MINOR.PATCH) with `bump-my-version` for version management and GitHub Actions for publishing.

### Workflow

1. **Development Phase**
   - Make changes on `main` branch or feature branches
   - Update `CHANGELOG.md` under the `[Unreleased]` section as you develop
   - Commit and push changes

2. **Prepare for Release**

   Ensure all changes for the release are merged and committed:
   ```bash
   # Make sure working directory is clean
   git status

   # Pull latest changes
   git checkout main
   git pull
   ```

3. **Update Changelog**

   Review `CHANGELOG.md` and ensure the `[Unreleased]` section contains all changes:
   ```markdown
   ## [Unreleased]

   ### Added
   - New feature X
   - New API endpoint Y

   ### Fixed
   - Bug fix Z
   ```

4. **Bump Version**

   Use `bump-my-version` to increment the version:
   ```bash
   # Patch release (0.1.0 → 0.1.1) - Bug fixes
   uv run bump-my-version bump patch

   # Minor release (0.1.0 → 0.2.0) - New features, backward compatible
   uv run bump-my-version bump minor

   # Major release (0.1.0 → 1.0.0) - Breaking changes
   uv run bump-my-version bump major
   ```

   **What this does:**
   - Updates version in `pyproject.toml`
   - Updates `__version__` in `src/my_vector_db/__init__.py`
   - Converts `[Unreleased]` to `[X.Y.Z] - YYYY-MM-DD` in `CHANGELOG.md`
   - Creates a git commit: `"chore: bump version X.Y.Z → X.Y.Z"`
   - Creates a git tag: `vX.Y.Z`

5. **Review Changes**

   ```bash
   # Review the version bump commit
   git show HEAD

   # Check the tag was created
   git tag
   ```

6. **Push Version Tag**

   ```bash
   # Push the commit and tag to GitHub
   git push --follow-tags
   ```

7. **Trigger Release Workflow**

   Go to GitHub Actions and manually trigger the appropriate workflow:

   **For Full Release (PyPI + Docker):**
   1. Navigate to: https://github.com/ajshedivy/my-vector-db/actions
   2. Select: "Publish Python Package and Docker Image"
   3. Click: "Run workflow"
   4. Confirm: Select `main` branch
   5. Click: "Run workflow"

   **For Docker-Only Release:**
   1. Navigate to: https://github.com/ajshedivy/my-vector-db/actions
   2. Select: "Build and Push Docker Image"
   3. Click: "Run workflow"
   4. Confirm: Select `main` branch
   5. Click: "Run workflow"

8. **Verify Release**

   After the workflow completes:

   **PyPI:**
   - Check https://pypi.org/project/my-vector-db/
   - Verify new version is published

   **Docker:**
   - Check https://github.com/ajshedivy/my-vector-db/pkgs/container/my-vector-db
   - Verify new tag is published

   **GitHub Release:**
   - Consider creating a GitHub release from the tag
   - Copy changelog entries for release notes

### Versioning Guidelines

Follow [Semantic Versioning](https://semver.org/):

- **PATCH** (0.1.0 → 0.1.1): Bug fixes, documentation updates, internal changes
- **MINOR** (0.1.0 → 0.2.0): New features, backward-compatible changes
- **MAJOR** (0.1.0 → 1.0.0): Breaking changes, incompatible API changes

### Pre-Release Testing

Before bumping version:

```bash
# Run all quality checks
uv run mypy src
uv run ruff check
uv run pytest

# Build package locally to test
uv build

# Test Docker build locally
docker build -t my-vector-db:test .
docker run -p 8000:8000 my-vector-db:test
```

### Dry Run (Preview Changes)

To see what `bump-my-version` would do without making changes:

```bash
uv run bump-my-version bump --dry-run --verbose patch
```

## CI/CD Workflows

### Publish Python Package and Docker Image

**Trigger:** Manual (`workflow_dispatch`)
**Purpose:** Full release - builds and publishes to both PyPI and GitHub Container Registry

**Jobs:**
1. `build` - Runs mypy, builds Python package, extracts version
2. `publish-pypi` - Publishes to PyPI using trusted publishing
3. `publish-docker` - Builds multi-platform Docker image (amd64, arm64)

**Environment:** Requires `release` environment approval

### Build and Push Docker Image

**Trigger:** Manual (`workflow_dispatch`)
**Purpose:** Docker-only release - rebuilds and publishes Docker image without PyPI

**Jobs:**
1. `build-docker` - Builds and pushes multi-platform Docker image

**Use Case:** Quick Docker updates without changing Python package version

## Repository Protection

### Branch Rules

The `main` branch is protected with:
- ✅ Restrict deletions
- ✅ Block force pushes
- 🔓 Repository admins can bypass (for version bumps)

### Release Security

- **PyPI Trusted Publishing**: Only accepts releases from GitHub Actions workflow
- **Environment Protection**: `release` environment requires manual approval
- **Tag Protection**: Version tags (`v*`) are protected from unauthorized creation

## Troubleshooting

### Version Bump Issues

**Problem:** `bump-my-version` fails with "Git working directory is not clean"

**Solution:**
```bash
# Commit or stash your changes first
git status
git add .
git commit -m "your message"
```

**Problem:** Tag already exists

**Solution:**
```bash
# Delete local tag
git tag -d v0.1.0

# Delete remote tag (careful!)
git push --delete origin v0.1.0

# Then bump again
```

### Workflow Failures

**Problem:** PyPI publish fails

**Solution:**
- Verify PyPI trusted publishing is configured correctly
- Check that version doesn't already exist on PyPI
- Review workflow logs for specific error

**Problem:** Docker build fails

**Solution:**
- Check Dockerfile syntax
- Verify base images are available
- Review build logs for specific error

## Additional Resources

- [Changelog](CHANGELOG.md) - Version history
- [SDK Documentation](docs/README.md) - Complete API reference
- [Examples](examples/README.md) - Usage examples
- [Main README](README.md) - Project overview
