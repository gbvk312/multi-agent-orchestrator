# Contributing to Multi-Agent Orchestrator

Thank you for your interest in contributing! This document outlines the process for contributing to this project.

## Getting Started

1. **Fork** the repository and clone your fork:
   ```bash
   git clone https://github.com/<your-username>/multi-agent-orchestrator.git
   cd multi-agent-orchestrator
   ```

2. **Set up the development environment** (using [uv](https://docs.astral.sh/uv/) is recommended):
   ```bash
   uv venv
   source .venv/bin/activate
   uv pip install -e ".[dev]"
   ```

3. **Create a branch** for your changes:
   ```bash
   git checkout -b feat/my-feature
   ```

## Development Workflow

### Running Tests

```bash
pytest tests/ -v
```

### Linting & Formatting

This project uses [Ruff](https://docs.astral.sh/ruff/) for both linting and formatting:

```bash
# Check for lint issues
ruff check .

# Auto-format code
ruff format .
```

### Code Style

- **Line length:** 120 characters max.
- **Python version:** 3.11+ features are welcome.
- **Type hints:** Use them for all public function signatures.
- **Docstrings:** Required for all public classes and methods.
- **Logging:** Use lazy `%`-style formatting (e.g., `logger.info("msg %s", val)`), not f-strings.

## Submitting Changes

1. Ensure all tests pass and linting is clean.
2. Write clear, descriptive commit messages.
3. Open a Pull Request against the `master` branch.
4. Describe **what** changed and **why** in the PR description.

## Reporting Issues

Open an issue on GitHub with:
- A clear title and description.
- Steps to reproduce (if applicable).
- Expected vs. actual behaviour.
