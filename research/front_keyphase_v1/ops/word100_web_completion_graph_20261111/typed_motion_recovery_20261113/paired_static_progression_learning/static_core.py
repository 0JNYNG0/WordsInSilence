from pathlib import Path
import sys
S=Path(__file__).resolve().parent
sys.path.insert(0,str(S.parent/'progression_presence_learning'))
from presence_core import *
