"""Make the evals package importable from tests without installing it.

Also force offline sample datasets so the unit tests are deterministic and
never hit the network (no HuggingFace download during CI/CPU-side runs).
"""
import os
import sys

os.environ.setdefault("EVALS_NO_HF", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
