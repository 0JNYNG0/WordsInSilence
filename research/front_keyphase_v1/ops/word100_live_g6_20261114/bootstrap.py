from pathlib import Path
import sys
U=Path(__file__).resolve().parent;O=U.parent
sys.dont_write_bytecode=True
sys.path[:0]=[str(U),str(O/'word100_action_linkage_repair_20261113'),str(O/'word100_actionable_feedback_20261113'),str(O/'word100_action_feedback_demo_20261108')]
