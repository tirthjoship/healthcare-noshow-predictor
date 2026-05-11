# CLAUDE.md

This file provides guidance to Claude Code when working with this repository. Read `AGENTS.md` for coding standards.

## Project Context

Hospital readmission risk prediction — predicting 30-day readmission using pre-discharge features only. Domain-specific details to be added after brainstorming.

## Architecture

Hexagonal (Ports & Adapters) with inward-pointing dependencies.
- domain/ — pure business logic, zero external imports
- adapters/ — data sources, ML models, visualization
- application/ — orchestration

## Commands

```bash
make check    # lint + typecheck + test with coverage
make test     # pytest -v --tb=short
make lint     # pre-commit run --all-files
make typecheck # mypy strict
```

## Phase Status

Domain-specific phase status to be added after brainstorming.
