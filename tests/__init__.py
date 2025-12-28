import os
import sys

# Automatically add the project root to sys.path
# This allows tests to import modules from 'src' easily when running 
# as a module (e.g., python -m tests.test_benchmarks)

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(TESTS_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)