"""
Test runner - runs all project tests

Author: Fabrizio Soppelsa <fabrizio.soppelsa [@] community.unipa.it>
University of Palermo - Natural Language Processing course
Academic Year 2025/2026
"""

import subprocess
import sys
import os

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def run_test(test_file):
    """Run a single test file."""
    print(f"\n{'='*60}")
    print(f"Running {test_file}")
    print('='*60)
    
    result = subprocess.run(
        [sys.executable, test_file],
        capture_output=False
    )
    
    return result.returncode == 0


if __name__ == "__main__":
    tests = [
        "test_tokenizer.py",
    ]
    
    print("PALMO Test Suite")
    print("=" * 60)
    
    results = {}
    for test in tests:
        results[test] = run_test(test)
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test:30} {status}")
    
    all_passed = all(results.values())
    print("\n" + "=" * 60)
    if all_passed:
        print("All tests passed! ✓")
    else:
        print("Some tests failed! ✗")
        sys.exit(1)
