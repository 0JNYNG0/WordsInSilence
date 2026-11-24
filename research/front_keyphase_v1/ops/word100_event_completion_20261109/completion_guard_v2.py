"""Respect the pre-existing guide's single-motion contract and optional endpoint.

An optional arrival posture of the same stroke is not a competing earlier/later
action. Relationships come from existing guide data, not a word-specific branch.
"""
from pathlib import Path
import importlib.util,copy
P=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('_completion_guard_v1',P/'completion_guard.py');v1=importlib.util.module_from_spec(s);s.loader.exec_module(v1)
def apply_guard(decision,prob,raw,ts,contracts):
 test=copy.deepcopy(decision)
 for c in test['cards']:
  contract=contracts.get(c['id']);linked=contract.get('phase_groups',[]) if contract else []
  for e in c['phase_evidence']:
   if set(e['group']).issubset(linked):e['group']=sorted(set(e['group'])|set(linked))
 out=v1.apply_guard(test,prob,raw,ts)
 for c,original in zip(out['cards'],decision['cards']):
  c['phase_evidence']=copy.deepcopy(original['phase_evidence'])
  c['completion_identity_motion_contract']=copy.deepcopy(contracts.get(c['id']))
 out['completion_guard_version']='positive_identity_conflict_v2_motion_contract';return out
