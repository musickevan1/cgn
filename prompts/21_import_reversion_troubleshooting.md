# CGN Import Reversion Troubleshooting Guide

## Step 1: Check for Pre-commit Hooks

Look for these files in your project root:

```bash
# Check if these files exist:
ls -la .pre-commit-config.yaml
ls -la .git/hooks/pre-commit
ls -la pyproject.toml  # May contain tool configurations
ls -la setup.cfg       # May contain isort/other tool configs
ls -la .isort.cfg
ls -la .flake8
```

**If `.pre-commit-config.yaml` exists**, examine its contents:

```yaml
# Example of what might be causing the issue:
repos:
  - repo: https://github.com/PyCQA/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black"]
  
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.0
    hooks:
      - id: ruff
      - id: ruff-format
```

**Solution**: Temporarily disable pre-commit hooks:
```bash
# Disable pre-commit hooks temporarily
mv .pre-commit-config.yaml .pre-commit-config.yaml.backup

# Or skip them for specific commits
git commit --no-verify -m "Update import paths for cgn restructure"
```

## Step 2: Check IDE/Editor Settings

### VS Code
Check `.vscode/settings.json` for:
```json
{
    "python.linting.enabled": true,
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
    "python.sortImports.enabled": true,
    "python.sortImports.args": [...]
}
```

### PyCharm/IntelliJ
- Check Settings > Tools > Python Integrated Tools > Import sorting
- Check Settings > Editor > Code Style > Python > Imports

### General Solution
Temporarily disable auto-formatting:
- VS Code: Set `"editor.formatOnSave": false`
- PyCharm: Disable auto-import optimization

## Step 3: Check Python Tool Configurations

### Check pyproject.toml
```toml
[tool.isort]
profile = "black"
src_paths = ["src", "tests"]  # This might be the issue!

[tool.ruff]
src = ["src"]  # This defines where ruff looks for imports

[tool.black]
line-length = 88
```

### Check setup.cfg
```ini
[isort]
src_paths = src,tests
profile = black

[flake8]
max-line-length = 88
```

**Solution**: Update configurations to recognize new `cgn` structure:

```toml
# In pyproject.toml
[tool.isort]
profile = "black"
src_paths = ["cgn", "tests"]  # Changed from "src" to "cgn"

[tool.ruff]
src = ["cgn"]  # Changed from "src" to "cgn"
```

## Step 4: Check for Running Background Processes

### File Watchers
Some IDEs or tools run background file watchers that auto-format on file changes:

```bash
# Check for running processes that might be formatting files
ps aux | grep -E "(black|isort|ruff|autopep8|yapf)"

# Check if any file watchers are running
ps aux | grep -E "(watchdog|inotify|fswatch)"
```

### Language Servers
Python language servers (pylsp, pyright, etc.) might be auto-formatting:

```bash
# Check for language servers
ps aux | grep -E "(pylsp|pyright|jedi)"
```

## Step 5: Test Import Changes Systematically

### Method 1: Atomic Testing
1. Make a small test change to one import
2. Save the file immediately
3. Watch if it reverts within seconds
4. Note the exact timing and what reverts it

### Method 2: Disable All Automation
```bash
# Temporarily move/rename config files
mv .pre-commit-config.yaml .pre-commit-config.yaml.disabled
mv pyproject.toml pyproject.toml.disabled
mv setup.cfg setup.cfg.disabled
mv .isort.cfg .isort.cfg.disabled

# Try making changes now
```

### Method 3: Manual Git Operations
```bash
# Stage files manually to bypass hooks
git add -A
git commit --no-verify -m "WIP: Import path updates"
```

## Step 6: Systematic Fix Strategy

### Option A: Update Tool Configurations First
Before making import changes, update all tool configurations:

```bash
# Update pyproject.toml
sed -i 's/src_paths = \["src"/src_paths = ["cgn"/g' pyproject.toml
sed -i 's/src = \["src"\]/src = ["cgn"]/g' pyproject.toml

# Update any setup.cfg
sed -i 's/src_paths = src/src_paths = cgn/g' setup.cfg
```

### Option B: Bulk Import Replacement
Use a script to update all imports at once:

```python
#!/usr/bin/env python3
"""
Script to update all import statements from src.connectome_analysis to cgn.core
"""
import os
import re

def update_imports_in_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Replace import patterns
    patterns = [
        (r'from src\.connectome_analysis', 'from cgn.core'),
        (r'import src\.connectome_analysis', 'import cgn.core'),
        (r'src\.connectome_analysis', 'cgn.core'),
    ]
    
    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)
    
    with open(filepath, 'w') as f:
        f.write(content)

def update_all_python_files():
    for root, dirs, files in os.walk('cgn'):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                print(f"Updating {filepath}")
                update_imports_in_file(filepath)

if __name__ == "__main__":
    update_all_python_files()
    print("All imports updated!")
```

### Option C: Disable Tools During Transition
```bash
# Create a script to temporarily disable all formatting
cat > disable_formatting.sh << 'EOF'
#!/bin/bash
echo "Disabling formatting tools..."
mv .pre-commit-config.yaml .pre-commit-config.yaml.backup 2>/dev/null || true
mv pyproject.toml pyproject.toml.backup 2>/dev/null || true
mv setup.cfg setup.cfg.backup 2>/dev/null || true
echo "Formatting tools disabled. Remember to re-enable after import updates!"
EOF

chmod +x disable_formatting.sh
./disable_formatting.sh
```

## Step 7: Re-enable Tools After Import Updates

After successfully updating all imports:

```bash
# Re-enable configurations with updated settings
mv .pre-commit-config.yaml.backup .pre-commit-config.yaml
mv pyproject.toml.backup pyproject.toml  # After updating src paths
mv setup.cfg.backup setup.cfg  # After updating src paths

# Update pre-commit hooks if needed
pre-commit install
pre-commit autoupdate
```

## Quick Diagnostic Commands

Run these to quickly identify the culprit:

```bash
# Check what's installed that could affect imports
pip list | grep -E "(isort|black|ruff|autopep8|yapf|pre-commit)"

# Check for git hooks
ls -la .git/hooks/

# Check all configuration files
find . -name "*.toml" -o -name "*.cfg" -o -name "*.yaml" -o -name "*.yml" | grep -E "(pre-commit|isort|black|ruff|flake8)"

# Check for VSCode settings
ls -la .vscode/settings.json

# Check for active Python processes
ps aux | grep python
```

## Immediate Action Plan

1. **Identify the tool**: Run the diagnostic commands above
2. **Disable temporarily**: Move config files to `.backup` extensions  
3. **Update imports**: Make all your import changes
4. **Update configs**: Modify tool configurations to use `cgn` instead of `src`
5. **Re-enable**: Restore configurations with updated settings
6. **Test**: Verify everything works with the new structure

The most likely culprit is `isort` or `ruff` with a configuration that still references the old `src` path structure.