"""
Ensures the project root is on sys.path so `model`, `data`, and `utils`
can be imported as packages regardless of where pytest is invoked from
(e.g. Colab, CI, or a subdirectory).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))