"""Shared pytest fixtures for integration tests.

Usage: fixtures here are auto-discovered by pytest for every test under
``test/integration/`` — no import needed.
"""

import shutil

import pytest

from test.integration.harness import OUTPUT_DIR


@pytest.fixture(autouse=True)
def clean_output():
    """Remove pipeline output before each test so cache-dependent assertions
    start from a clean slate. Output is left in place afterwards for inspection."""
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    yield
