# CLAUDE.md

This file provides guidance to Claude Code when working with this repository. Read `AGENTS.md` for coding standards.

## Project Context

**PIVOT (2026-05-30):** Healthcare **appointment no-show** prediction — not MIMIC readmission.

Read **`CONTEXT.md`** for locked dataset (Kaggle Medical Appointments), EDA gate, and pivot map from readmission domain models.

Public benchmark only — not VGH/BCCNM production data.

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
