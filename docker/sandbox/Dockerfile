# =============================================================================
# Advanced GitHub Code Reviewer — Sandbox Execution Image
# =============================================================================
#
# PURPOSE
# -------
# This image is the isolated execution environment used by docker_runner.py
# to lint and test PR code safely. It is NOT the application image.
#
# WHAT THIS IS NOT
# ----------------
# This is not the FastAPI app image. That lives at the project root
# Dockerfile and is built in P6 for AWS ECS deployment.
#
# HOW IT IS USED
# --------------
# docker_runner.py spins up a fresh container from this image for every
# PR review. The container:
#   - receives the PR diff as a file patch
#   - runs ruff check (lint) or pytest (test)
#   - returns stdout/stderr/exit code
#   - is destroyed immediately after
#
# SECURITY MODEL
# --------------
# Containers launched from this image are always started with:
#   --cpus=0.5        CPU limit — reviewed code cannot starve the host
#   --memory=256m     RAM limit — reviewed code cannot exhaust memory
#   --network=none    No network — reviewed code cannot make HTTP calls
#   --read-only       Root filesystem read-only (writable /sandbox only)
#   --rm              Auto-remove on exit — no state persists
#
# BUILD COMMAND (run once, from project root)
# -------------------------------------------
#   docker build -f docker/sandbox/Dockerfile -t code-reviewer-sandbox:latest .
#
# DO NOT rebuild unless upgrading ruff or pytest versions.
# Pin versions explicitly — reproducibility is the point.
# =============================================================================

# Base: official Python 3.11 slim — minimal OS, no extras
# slim is used over alpine because ruff has native binary deps
# that compile more reliably on Debian-based images
FROM python:3.11-slim

# ---------------------------------------------------------------------------
# Install linting and testing tools ONLY
# No application code. No app dependencies. Nothing else.
# Pinned versions — every PR review uses the exact same tool versions.
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir \
    ruff==0.9.0 \
    pytest==8.3.0 \
    pytest-timeout==2.3.1

# pytest-timeout: kills test runs that exceed the per-test timeout.
# Prevents malicious or infinite-loop test code from hanging the sandbox.

# ---------------------------------------------------------------------------
# Working directory inside the container
# docker_runner.py will write the patched PR files here before running tools.
# This is the only writable location when --read-only is applied.
# ---------------------------------------------------------------------------
WORKDIR /sandbox

# ---------------------------------------------------------------------------
# Default command — overridden by docker_runner.py at runtime
# This is a safety fallback; docker_runner.py always passes explicit commands.
# ---------------------------------------------------------------------------
CMD ["ruff", "check", "."]

# =============================================================================
# IMAGE SUMMARY
# =============================================================================
# Base image   : python:3.11-slim (Debian Bookworm)
# Tools        : ruff 0.9.0, pytest 8.3.0, pytest-timeout 2.3.1
# WORKDIR      : /sandbox
# Image size   : ~180MB (slim base + 3 pip packages)
# Rebuild when : upgrading ruff or pytest versions only
# Tag          : code-reviewer-sandbox:latest
# =============================================================================