# conftest.py
# -----------
# Makes the project root importable so that "from src.xxx import ..."
# works correctly when pytest is run from the sign_language_analyzer/ directory.

import os
import sys

# Insert the project root (the folder containing src/) at the front of sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
