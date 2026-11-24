from pathlib import Path
import json,hashlib
U=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=U/'INTEGRATION_FROZEN.json';assert not p.exists()
p.write_text(json.dumps(dict(reason='New measurement+movement candidate recovers combined direction but one normal becomes undecided. A weak new observation must not erase existing supported evidence. Confident contradictions remain unknown.',
 finger='New per-finger model remains shadow only: individual known-cell regressions; no corrected hand labels or score thresholds tuned after evaluation',
 merge='Both confident and opposite -> conflict; new weak -> preserve baseline; all three new states strong and compatible may resolve old undecided direction; original action link/occlusion/conflict gates retained',
 scope='Isolated adapter and public replay; current web not changed in this task',
 public_regression='82 previous DEV scenarios and40 archived G4 public scenarios; all exposed earlier, not independent generalization',
 no_target_shape_selection=True,no_word_exception=True,no_expected_order_selection=True,
 inputs='Observed normalized coordinates, original timestamps, current automatically selected spans and structured descriptors; no condition/edit flags',
 gates=['No additional normal opposite-direction or conflict warning','All old normal supported directions retained','All old detected error evidence retained','Combined shape+direction detection increases without losing native source spans','Missing observations remain unknown','Order evidence/candidates/TCN/common100 unchanged'],
 pins={str(p):sha(p)for p in [U/'direction_adapter.py',U/'stable_geometry.py',U/'motion_core.py']+[U/'motion_models_v2'/f'{s}.pt'for s in[42,43,44]]}),ensure_ascii=False,indent=2),'utf-8')
print(p)
