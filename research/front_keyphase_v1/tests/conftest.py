"""Put the candidate's own src/ on sys.path. No project data, model or recording is read."""
import os
import sys

CANDIDATE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(CANDIDATE_ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
