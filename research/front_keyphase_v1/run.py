#!/usr/bin/env python3
"""Single entry point so no install step is needed: py -3.13 run.py <command> ..."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from wis_lab.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
