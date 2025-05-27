# Development Environment Setup

## Overview
This document outlines the current state and requirements for the CGN project's development environment, with a focus on reproducibility and efficiency.

## Environment Setup Requirements
The project uses a **devcontainer** for reproducible development environments.

### Devcontainer Validation Checklist:
- [ ] Container builds without errors (Needs verification)
- [ ] All neuroimaging dependencies install correctly (nilearn, nibabel) (Needs verification)
- [ ] PyTorch and PyTorch Geometric work properly (Needs verification)
- [ ] Data loading scripts execute successfully (Needs verification)
- [ ] File mounting between host/container works (Needs verification)
- [ ] Extension integration (especially for development tools) (Needs verification)

## Technical Dependencies & Environment
- **Package configuration** (`pyproject.toml`, `requirements.txt`, `environment.yml`): Appears to be well-defined, but requires verification of all dependencies.
- **Docker setup** for reproducible environments: Devcontainer setup is present, but functionality needs to be confirmed.
- **Optional dependency handling** (nilearn, nibabel graceful degradation): Assumed to be handled, but requires explicit testing.
- **Development tooling** (testing, linting, formatting): Basic setup is present, but may need enhancement for comprehensive code quality.

## Issues and Next Steps
- [ ] **Critical:** Fully validate devcontainer setup and resolve any build or runtime issues.
- [ ] Ensure all dependencies are correctly managed and installed across environments.
- [ ] Review and optimize development workflow, including testing, linting, and formatting.
