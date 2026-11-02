---
name: Python package resolver
description: Replit package additions can rewrite root uv metadata during dependency resolution.
---

After a Python package-add attempt, inspect `pyproject.toml` and `uv.lock` even if resolution fails. Keep only intended dependencies and package-manager settings, then regenerate the lock file.

**Why:** A failed ML evaluation dependency install added a large set of unrelated uv package-source mappings to the root project.

**How to apply:** Check the root Python files after every Replit package addition, especially for packages with PyTorch dependencies.
