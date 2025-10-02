#!/usr/bin/env python3
"""
Test runner script for DecayShape package.

This script runs all tests and provides a summary of results.
"""

import subprocess
import sys


def run_tests():
    """Run all tests and return the result."""
    print("=" * 60)
    print("Running DecayShape Test Suite")
    print("=" * 60)

    # Run pytest with verbose output
    cmd = [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short", "--color=yes"]

    try:
        result = subprocess.run(cmd, check=False, capture_output=False)
        return result.returncode
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1


def run_specific_test_suite(test_file):
    """Run a specific test suite."""
    print(f"\nRunning {test_file}...")
    print("-" * 40)

    cmd = [sys.executable, "-m", "pytest", f"tests/{test_file}", "-v"]

    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running {test_file}: {e}")
        return False


def main():
    """Main test runner."""
    if len(sys.argv) > 1:
        # Run specific test file
        test_file = sys.argv[1]
        if not test_file.startswith("test_"):
            test_file = f"test_{test_file}"
        if not test_file.endswith(".py"):
            test_file = f"{test_file}.py"

        success = run_specific_test_suite(test_file)
        sys.exit(0 if success else 1)
    else:
        # Run all tests
        return_code = run_tests()

        print("\n" + "=" * 60)
        if return_code == 0:
            print("✅ All tests passed!")
        else:
            print("❌ Some tests failed!")
        print("=" * 60)

        sys.exit(return_code)


if __name__ == "__main__":
    main()
