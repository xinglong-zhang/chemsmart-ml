"""Shared pytest configuration.

Fixtures live in ``tests/fixtures/`` and are registered here via
``pytest_plugins``. Plain builder functions live in ``tests/helpers.py``
and are imported directly.
"""

pytest_plugins = [
    "tests.fixtures.databases",
]
