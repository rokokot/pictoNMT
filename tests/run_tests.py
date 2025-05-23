#!/usr/bin/env python3
import subprocess
import sys
import os
from pathlib import Path

def run_test(test_file, description):
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            [sys.executable, test_file],
            capture_output=False,
            check=True,
            cwd=Path(__file__).parent.parent
        )
        print(f"{description} - PASS")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{description} - FAIL")
        return False

def main():
    """Run all tests"""
    print("PictoNMT Tests")
    print("Running all component tests...")
    
    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    tests_to_run = [
        ("tests/test_picto_encoder.py", "Visual Encoder Tests"),
        ("tests/test_hybrid_encoder.py", "Dual-Path Encoder Tests"),
        ("tests/test_complete_pipeline.py", "Complete Pipeline Tests"),
        ("scripts/test_data_pipeline.py", "Data Pipeline Check")
    ]
    
    passed = 0
    total = len(tests_to_run)
    
    for test_file, description in tests_to_run:
        if run_test(test_file, description):
            passed += 1
    
    print(f"\n{'='*60}")
    print(f"Test Results: {passed}/{total} tests passed")
    print(f"{'='*60}")
    
    if passed == total:
        print("All tests passed! System is ready for deployment.")
        return 0
    else:
        print(f"{total - passed} tests failed. Please fix issues before proceeding.")
        return 1

if __name__ == "__main__":
    sys.exit(main())