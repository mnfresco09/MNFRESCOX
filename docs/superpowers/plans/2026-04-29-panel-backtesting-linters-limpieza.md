# Panel Backtesting Quality Tooling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reproducible Python and Rust code review tooling for `PANEL BACKTESTING` and `run.py`, then clean low-risk quality issues without changing trading behavior.

**Architecture:** Root-level Python tooling config drives Ruff and pytest for `run.py` plus `PANEL BACKTESTING`. Rust checks stay inside `PANEL BACKTESTING/MOTOR` and are orchestrated by small root scripts that quote paths safely. Cleanup is incremental because the workspace already contains user changes in panel files.

**Tech Stack:** Python 3.12 virtualenv, Ruff, pytest, Cargo, rustfmt, Clippy, Bash.

---

## File Structure

- Create `pyproject.toml`: central Ruff and pytest configuration for the panel scope.
- Create `requirements-dev.txt`: development-only dependencies for code review.
- Create `scripts/revisar_codigo.sh`: single command for lint, format checks, tests and Rust review.
- Create `scripts/formatear_codigo.sh`: single command for safe automatic formatting.
- Modify `.gitignore`: ignore `.ruff_cache/`.
- Modify `README.md`: document how to install and run the reviewer.
- Potentially modify `run.py`, `PANEL BACKTESTING/**/*.py` and `PANEL BACKTESTING/MOTOR/**/*.rs`: only for formatter output or concrete lint findings.

## Dirty Worktree Rule

Before implementation, record the current dirty files:

```bash
git status --short
```

Existing dirty files at planning time include panel configuration, strategy, risk, optimization and report files. If a cleanup touches one of those files, do not commit it unless the user explicitly asks to include their existing edits. New tooling files can be committed independently.

### Task 1: Add Tooling Configuration And Review Scripts

**Files:**
- Create: `pyproject.toml`
- Create: `requirements-dev.txt`
- Create: `scripts/revisar_codigo.sh`
- Create: `scripts/formatear_codigo.sh`
- Modify: `.gitignore`

- [ ] **Step 1: Create `pyproject.toml`**

Use this exact content:

```toml
[tool.pytest.ini_options]
testpaths = ["PANEL BACKTESTING/TESTS"]
python_files = ["test_*.py"]
addopts = "-q"

[tool.ruff]
target-version = "py312"
line-length = 120
force-exclude = true
extend-exclude = [
    ".git",
    ".venv",
    "__pycache__",
    "PANEL BACKTESTING/HISTORICO",
    "PANEL BACKTESTING/RESULTADOS",
    "PANEL BACKTESTING/MOTOR/target",
]

[tool.ruff.lint]
select = [
    "E4",
    "E7",
    "E9",
    "F",
    "I",
    "B",
    "UP",
    "RUF",
]
ignore = [
    "E501",
]

[tool.ruff.lint.per-file-ignores]
"**/__init__.py" = ["F401"]
"PANEL BACKTESTING/TESTS/*.py" = ["E402"]
"run.py" = ["E402"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
line-ending = "lf"
```

- [ ] **Step 2: Create `requirements-dev.txt`**

Use this exact content:

```txt
-r requirements.txt
pytest
ruff
```

- [ ] **Step 3: Create `scripts/revisar_codigo.sh`**

Use this exact content:

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

cd "$ROOT"

"$PYTHON_BIN" -m ruff check run.py "PANEL BACKTESTING"
"$PYTHON_BIN" -m ruff format --check run.py "PANEL BACKTESTING"
"$PYTHON_BIN" -m pytest "PANEL BACKTESTING/TESTS"

(
  cd "PANEL BACKTESTING/MOTOR"
  cargo fmt --check
  cargo clippy --all-targets -- -D warnings
  cargo test
)
```

- [ ] **Step 4: Create `scripts/formatear_codigo.sh`**

Use this exact content:

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

cd "$ROOT"

"$PYTHON_BIN" -m ruff check --fix --select I run.py "PANEL BACKTESTING"
"$PYTHON_BIN" -m ruff format run.py "PANEL BACKTESTING"

(
  cd "PANEL BACKTESTING/MOTOR"
  cargo fmt
)
```

- [ ] **Step 5: Make scripts executable**

Run:

```bash
chmod +x scripts/revisar_codigo.sh scripts/formatear_codigo.sh
```

Expected: command exits with code 0.

- [ ] **Step 6: Add Ruff cache to `.gitignore`**

Append this exact block under the Python cache section:

```gitignore
.ruff_cache/
```

- [ ] **Step 7: Install development dependencies**

Run:

```bash
.venv/bin/pip install -r requirements-dev.txt
```

Expected: `pytest` and `ruff` are importable from `.venv/bin/python`.

- [ ] **Step 8: Verify tooling commands are discoverable**

Run:

```bash
.venv/bin/python -m ruff --version
.venv/bin/python -m pytest --version
cargo fmt --version
cargo clippy -V
```

Expected: each command prints a version and exits with code 0.

- [ ] **Step 9: Commit only tooling files**

Run:

```bash
git add pyproject.toml requirements-dev.txt scripts/revisar_codigo.sh scripts/formatear_codigo.sh .gitignore
git commit -m "chore: add panel code quality tooling"
```

Expected: commit contains only tooling configuration and scripts, not dirty panel files.

### Task 2: Establish Baseline Before Cleanup

**Files:**
- Read: `run.py`
- Read: `PANEL BACKTESTING/**/*.py`
- Read: `PANEL BACKTESTING/MOTOR/**/*.rs`

- [ ] **Step 1: Record current workspace status**

Run:

```bash
git status --short
```

Expected: output still shows user dirty files plus any new cleanup changes from subsequent tasks.

- [ ] **Step 2: Run Python tests before formatting**

Run:

```bash
.venv/bin/python -m pytest "PANEL BACKTESTING/TESTS" -q
```

Expected: tests either pass, or failures are recorded as baseline before cleanup. If tests fail, do not change production behavior to hide the failure; inspect the failure first.

- [ ] **Step 3: Run Rust checks before formatting**

Run:

```bash
cd "PANEL BACKTESTING/MOTOR"
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

Expected: commands either pass, or failures are recorded as baseline before cleanup.

- [ ] **Step 4: Run the full reviewer once**

Run:

```bash
scripts/revisar_codigo.sh
```

Expected: the command may fail on existing lint or format findings. Save the first failing command and its findings for the cleanup tasks.

### Task 3: Apply Safe Automatic Formatting

**Files:**
- Potentially modify: `run.py`
- Potentially modify: `PANEL BACKTESTING/**/*.py`
- Potentially modify: `PANEL BACKTESTING/MOTOR/**/*.rs`

- [ ] **Step 1: Run the formatter script**

Run:

```bash
scripts/formatear_codigo.sh
```

Expected: Ruff formats Python files, sorts imports, and `cargo fmt` formats Rust files.

- [ ] **Step 2: Inspect changed files**

Run:

```bash
git diff --stat -- run.py "PANEL BACKTESTING"
git diff --name-only -- run.py "PANEL BACKTESTING"
```

Expected: changed files are limited to `run.py`, panel Python files and Rust motor files.

- [ ] **Step 3: Verify formatting is now clean**

Run:

```bash
.venv/bin/python -m ruff format --check run.py "PANEL BACKTESTING"
cd "PANEL BACKTESTING/MOTOR"
cargo fmt --check
```

Expected: both commands exit with code 0.

### Task 4: Fix Python Lint Findings Conservatively

**Files:**
- Potentially modify: `run.py`
- Potentially modify: `PANEL BACKTESTING/**/*.py`

- [ ] **Step 1: Run Ruff lint**

Run:

```bash
.venv/bin/python -m ruff check run.py "PANEL BACKTESTING"
```

Expected: either exits with code 0 or prints concrete Ruff rule findings.

- [ ] **Step 2: Apply only safe automatic import fixes**

Run:

```bash
.venv/bin/python -m ruff check --fix --select I,F401 run.py "PANEL BACKTESTING"
```

Expected: import ordering and unused imports are fixed where Ruff marks them safe.

- [ ] **Step 3: Manually fix remaining `F841` unused local findings**

For each `F841` finding, use this rule:

```python
# Before
unused_value = expensive_call()
return result

# After, only when expensive_call has no required side effect
return result
```

If the right-hand side may have a side effect, keep the call and discard explicitly:

```python
_ = expensive_call()
return result
```

Expected: no behavior change; tests still exercise the same paths.

- [ ] **Step 4: Manually fix remaining `B007` unused loop variable findings**

For each `B007` finding, replace the unused variable with `_`:

```python
# Before
for indice, valor in enumerate(valores):
    total += valor

# After
for _, valor in enumerate(valores):
    total += valor
```

Expected: loop semantics remain identical.

- [ ] **Step 5: Manually fix remaining `UP` modernization findings only when syntax stays Python 3.12-compatible**

Use these transformations:

```python
# Before
from typing import Optional
valor: Optional[float]

# After
valor: float | None
```

```python
# Before
from typing import Dict, List
items: List[Dict[str, float]]

# After
items: list[dict[str, float]]
```

Expected: no runtime behavior change.

- [ ] **Step 6: Re-run Ruff lint**

Run:

```bash
.venv/bin/python -m ruff check run.py "PANEL BACKTESTING"
```

Expected: command exits with code 0. If a remaining rule could require a behavior change, stop and document the exact file, line and rule before changing it.

### Task 5: Fix Rust Formatting And Clippy Findings

**Files:**
- Potentially modify: `PANEL BACKTESTING/MOTOR/build.rs`
- Potentially modify: `PANEL BACKTESTING/MOTOR/src/capital.rs`
- Potentially modify: `PANEL BACKTESTING/MOTOR/src/lib.rs`
- Potentially modify: `PANEL BACKTESTING/MOTOR/src/simulador.rs`
- Potentially modify: `PANEL BACKTESTING/MOTOR/src/tipos.rs`

- [ ] **Step 1: Run Rust formatter**

Run:

```bash
cd "PANEL BACKTESTING/MOTOR"
cargo fmt
```

Expected: Rust files are formatted.

- [ ] **Step 2: Run Clippy with warnings as errors**

Run:

```bash
cd "PANEL BACKTESTING/MOTOR"
cargo clippy --all-targets -- -D warnings
```

Expected: either exits with code 0 or prints concrete Clippy findings.

- [ ] **Step 3: Fix simple Clippy findings without changing simulation semantics**

Use these transformations when they match a reported finding:

```rust
// Before
if value == true {
    return 1;
}

// After
if value {
    return 1;
}
```

```rust
// Before
let size = values.len() as usize;

// After
let size = values.len();
```

```rust
// Before
match maybe_value {
    Some(value) => value,
    None => return None,
}

// After
let value = maybe_value?;
```

Expected: no change to trade entry, exit, capital or fee calculations.

- [ ] **Step 4: Re-run Rust checks**

Run:

```bash
cd "PANEL BACKTESTING/MOTOR"
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

Expected: all commands exit with code 0.

### Task 6: Document The Reviewer

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add this section to `README.md` after the root structure section**

Use this exact Markdown:

````markdown
---

## Revision de codigo

Instalar dependencias de desarrollo:

```bash
.venv/bin/pip install -r requirements-dev.txt
```

Ejecutar la revision completa del panel de backtesting:

```bash
scripts/revisar_codigo.sh
```

Aplicar formato automatico seguro:

```bash
scripts/formatear_codigo.sh
```

La revision cubre `run.py`, `PANEL BACKTESTING` y el motor Rust en `PANEL BACKTESTING/MOTOR`. El proyecto no tiene codigo C/C++ propio dentro de este alcance; si se agrega en el futuro, se configurara `clang-format` y `clang-tidy` en una fase separada.
````

- [ ] **Step 2: Verify README renders as Markdown**

Run:

```bash
sed -n '1,120p' README.md
```

Expected: the new section appears once, with fenced code blocks correctly closed.

### Task 7: Final Verification And Handoff

**Files:**
- Read: all files changed in previous tasks

- [ ] **Step 1: Run the full reviewer**

Run:

```bash
scripts/revisar_codigo.sh
```

Expected: command exits with code 0.

- [ ] **Step 2: Inspect final diff**

Run:

```bash
git diff --stat
git diff --name-only
```

Expected: diff contains only approved tooling, documentation, formatting and lint cleanup within `run.py` and `PANEL BACKTESTING`.

- [ ] **Step 3: Separate commit-safe files from dirty user files**

Run:

```bash
git status --short
```

Expected: files that were dirty before implementation are still identifiable. Do not commit those files unless the user explicitly approves including their previous edits.

- [ ] **Step 4: Commit safe final files if they do not include pre-existing user edits**

Run only for files created or modified solely by this task:

```bash
git add pyproject.toml requirements-dev.txt scripts/revisar_codigo.sh scripts/formatear_codigo.sh .gitignore README.md
git commit -m "docs: document panel code reviewer"
```

Expected: commit excludes any panel file that already had user edits before this work.

- [ ] **Step 5: Report residual risk**

Final response must include the real verification command results, the exact files that remain dirty because they had pre-existing user edits, and either `Riesgo residual: ninguno conocido` or the exact command and reason for any check that did not pass.
