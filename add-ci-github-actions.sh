#!/usr/bin/env bash
set -euo pipefail

BRANCH="add-ci-github-actions"
PR_TITLE="Add CI: GitHub Actions"
PR_BODY="Adds CI: Black, isort, Flake8 (bugbear), Mypy, Pytest (coverage), notebook execution, pip caching. Runs on Python 3.10/3.11/3.12; notebooks executed on 3.11."

# create branch
git checkout -b "$BRANCH"

# create directories
mkdir -p .github/workflows

# write workflow
cat > .github/workflows/ci.yml <<'YAML'
# GitHub Actions CI workflow: lint, type-check, test, notebooks, upload artifacts
# - Python matrix: 3.10, 3.11, 3.12
# - Uses pip cache to speed up installs
# - Runs isort (check), Black (check), Flake8 (with bugbear), Mypy, Pytest (with coverage)
# - Executes notebooks (if any) using jupyter nbconvert
name: CI

on:
  push:
    branches:
      - main
      - develop
      - 'releases/*'
  pull_request:
    branches:
      - main
      - develop
  schedule:
    - cron: '0 3 * * *'
  workflow_dispatch:

env:
  PIP_CACHE_DIR: ${{ runner.cache }}/pip

jobs:
  lint-type-test:
    name: Lint / Type-check / Test (Python ${{ matrix.python-version }})
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.10', '3.11', '3.12']
    steps:
      - name: Checkout repo
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}

      - name: Cache pip
        uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ matrix.python-version }}-${{ hashFiles('**/requirements.txt') }}
          restore-keys: |
            ${{ runner.os }}-pip-${{ matrix.python.version }}-

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
          pip install "black==24.*" "isort==5.*" "flake8==7.*" "flake8-bugbear==23.*" "mypy==1.*" "pytest==8.*" "pytest-cov" "nbval" "jupyter" "coverage"

      - name: Run isort (check)
        run: |
          isort --version
          isort --check-only .

      - name: Run Black (check)
        run: |
          black --version
          black --check .

      - name: Run Flake8 (including bugbear checks)
        run: |
          flake8 .

      - name: Run Mypy
        run: |
          mypy .

      - name: Run Pytest (with coverage)
        continue-on-error: false
        run: |
          pytest --maxfail=1 --disable-warnings -q --cov=. --cov-report=xml

      - name: Upload test/coverage artifacts
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: coverage-and-tests-${{ matrix.python-version }}
          path: |
            coverage.xml
            .pytest_cache || true

      - name: Upload coverage to Codecov (optional)
        if: secrets.CODECOV_TOKEN != ''
        uses: codecov/codecov-action@v4
        with:
          files: coverage.xml
          token: ${{ secrets.CODECOV_TOKEN }}

  notebooks:
    name: Execute Jupyter notebooks (single runner)
    runs-on: ubuntu-latest
    needs: lint-type-test
    strategy:
      matrix:
        python-version: ['3.11']
    steps:
      - name: Checkout repo
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}

      - name: Cache pip for notebooks
        uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: notebooks-pip-${{ matrix.python-version }}-${{ hashFiles('**/requirements.txt') }}
          restore-keys: |
            notebooks-pip-${{ matrix.python.version }}-

      - name: Install jupyter & nbconvert
        run: |
          python -m pip install --upgrade pip
          if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
          pip install "jupyter" "nbconvert" "nbformat"

      - name: Find and execute notebooks
        run: |
          set -euo pipefail
          mapfile -t NOTEBOOKS < <(git ls-files '*.ipynb' || true)
          if [ ${#NOTEBOOKS[@]} -eq 0 ]; then
            echo "No notebooks found in repository — skipping execution."
            exit 0
          fi
          echo "Found notebooks:"
          printf '%s\n' "${NOTEBOOKS[@]}"
          for nb in "${NOTEBOOKS[@]}"; do
            echo "Executing $nb"
            out="/tmp/$(basename "$nb" .ipynb)-executed.ipynb"
            jupyter nbconvert --to notebook --execute "$nb" --ExecutePreprocessor.timeout=600 --output "$out"
            echo "$out"
          done

      - name: Upload executed notebooks (artifact)
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: executed-notebooks
          path: /tmp/*-executed.ipynb
YAML

# write .flake8
cat > .flake8 <<'INI'
[flake8]
max-line-length = 88
extend-ignore = E203, W503
extend-select = B
exclude = .git,__pycache__,build,dist,.venv,.eggs
INI

# write mypy.ini
cat > mypy.ini <<'INI'
[mypy]
python_version = 3.11
ignore_missing_imports = True
warn_unused_configs = True
disallow_untyped_defs = False
disallow_untyped_calls = False
check_untyped_defs = False
INI

# write pyproject.toml
cat > pyproject.toml <<'TOML'
[tool.black]
line-length = 88
target-version = ["py310"]
include = '\.pyi?$'
exclude = '''
/(
  \.eggs
 | \.git
 | \.hg
 | \.mypy_cache
 | \.tox
 | \.venv
 | build
 | dist
)/
'''

[tool.isort]
profile = "black"
line_length = 88
combine_as_imports = true
known_first_party = "OptionsLab"
TOML

# commit & push
git add .github/workflows/ci.yml .flake8 mypy.ini pyproject.toml
git commit -m "Add CI: GitHub Actions — isort, black, flake8 (bugbear), mypy, pytest, notebooks, caching, coverage"
git push --set-upstream origin "$BRANCH"

# create PR using gh if available
if command -v gh >/dev/null 2>&1; then
  gh pr create --title "$PR_TITLE" --body "$PR_BODY" --base main --head "$BRANCH"
  echo "PR created. Attempting to merge..."
  # Attempt to merge (this will respect branch protection; it will fail if merge not allowed)
  gh pr merge --auto --merge --subject "$PR_TITLE" --body "$PR_BODY" || {
    echo "Auto-merge failed (branch protections or required checks). Please merge manually via GitHub UI or gh."
  }
else
  echo "Branch pushed. Install GitHub CLI (gh) to auto-create PR; otherwise create the PR via the GitHub web UI."
fi

echo "Done. If a PR was created, CI will run on the branch and on the PR."
