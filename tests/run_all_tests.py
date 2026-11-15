"""
MineGuard AI - Comprehensive Test Suite Runner.
Executes all unit and integration tests across inspection engine,
database persistence, and Flask REST routes.
"""

import unittest
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_all_tests():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Discover and add test cases
    tests_dir = Path(__file__).parent
    suite.addTests(loader.discover(start_dir=str(tests_dir), pattern="test_*.py"))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 60)
    print(f"MINEGUARD AI TEST SUITE SUMMARY")
    print("=" * 60)
    print(f"Tests Run:     {result.testsRun}")
    print(f"Failures:      {len(result.failures)}")
    print(f"Errors:        {len(result.errors)}")
    print(f"Success:       {result.wasSuccessful()}")
    print("=" * 60)

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
