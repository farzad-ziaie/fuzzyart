#!/usr/bin/env bash
# =============================================================================
# scripts/ci.sh — Full CI/CD pipeline for fuzzyart
#
# Usage:
#   ./scripts/ci.sh              # run all stages
#   ./scripts/ci.sh test         # tests only
#   ./scripts/ci.sh lint         # lint only
#   ./scripts/ci.sh docs         # build docs only
#   ./scripts/ci.sh build        # build wheel/sdist only
#   ./scripts/ci.sh publish      # publish to PyPI (requires PYPI_TOKEN)
#   ./scripts/ci.sh publish-test # publish to TestPyPI
# =============================================================================

set -euo pipefail

STAGE="${1:-all}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; NC='\033[0m'

info()    { echo -e "${BLUE}[CI]${NC} $*"; }
success() { echo -e "${GREEN}[CI ✓]${NC} $*"; }
warn()    { echo -e "${YELLOW}[CI ⚠]${NC} $*"; }
error()   { echo -e "${RED}[CI ✗]${NC} $*"; exit 1; }

# ── Helpers ───────────────────────────────────────────────────────────────────
check_tool() {
    command -v "$1" >/dev/null 2>&1 || error "'$1' not found. Install it first."
}

stage_lint() {
    info "Running linters..."
    check_tool ruff
    check_tool black

    info "  ruff check..."
    ruff check fuzzyart tests || error "ruff found issues."

    info "  black format check..."
    black --check fuzzyart tests || error "black found formatting issues. Run: black fuzzyart tests"

    info "  mypy type check..."
    if command -v mypy >/dev/null 2>&1; then
        mypy fuzzyart || warn "mypy reported issues (non-blocking)."
    else
        warn "mypy not installed, skipping type check."
    fi

    success "Lint passed."
}

stage_test() {
    info "Running test suite..."
    check_tool pytest

    pytest \
        --cov=fuzzyart \
        --cov-report=term-missing \
        --cov-report=xml:coverage.xml \
        --cov-report=html:htmlcov \
        -v \
        -x \
        tests/

    success "All tests passed."
}

stage_test_fast() {
    info "Running tests (parallel, no coverage)..."
    check_tool pytest
    pytest -n auto -q tests/
    success "Fast tests passed."
}

stage_docs() {
    info "Building Sphinx documentation..."
    check_tool sphinx-build

    # Clean previous build
    rm -rf docs/_build

    sphinx-build -b html docs docs/_build/html -W --keep-going 2>&1 \
        | tee docs/_build/build.log \
        || error "Sphinx build failed. See docs/_build/build.log"

    success "Docs built → docs/_build/html/index.html"
}

stage_build() {
    info "Building distribution packages..."
    check_tool poetry

    # Validate pyproject.toml
    poetry check || error "pyproject.toml validation failed."

    # Clean old builds
    rm -rf dist/

    poetry build

    ls -lh dist/
    success "Build complete → dist/"
}

stage_publish_test() {
    info "Publishing to TestPyPI..."
    stage_build

    check_tool poetry

    if [[ -z "${PYPI_TEST_TOKEN:-}" ]]; then
        error "PYPI_TEST_TOKEN environment variable is not set."
    fi

    poetry config repositories.testpypi https://test.pypi.org/legacy/
    poetry config pypi-token.testpypi "$PYPI_TEST_TOKEN"
    poetry publish --repository testpypi

    success "Published to TestPyPI."
    info "Install with: pip install -i https://test.pypi.org/simple/ fuzzyart"
}

stage_publish() {
    info "Publishing to PyPI..."
    stage_build

    check_tool poetry

    if [[ -z "${PYPI_TOKEN:-}" ]]; then
        error "PYPI_TOKEN environment variable is not set."
    fi

    poetry config pypi-token.pypi "$PYPI_TOKEN"
    poetry publish

    success "Published to PyPI."
}

stage_all() {
    info "Running full CI pipeline..."
    stage_lint
    stage_test
    stage_docs
    stage_build
    success "Full CI pipeline completed successfully."
}

# ── Dispatch ─────────────────────────────────────────────────────────────────
case "$STAGE" in
    all)           stage_all ;;
    lint)          stage_lint ;;
    test)          stage_test ;;
    test-fast)     stage_test_fast ;;
    docs)          stage_docs ;;
    build)         stage_build ;;
    publish)       stage_publish ;;
    publish-test)  stage_publish_test ;;
    *)             error "Unknown stage: '$STAGE'. Valid: all lint test test-fast docs build publish publish-test" ;;
esac
