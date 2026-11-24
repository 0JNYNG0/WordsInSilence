"""Identical prior arithmetic for the new saved candidate, no model replay."""
from pathlib import Path
import importlib.util
W=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('previous_stage_scoring',W.parent/'episode_stage/score.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.W=W
if __name__=='__main__':module.main()
