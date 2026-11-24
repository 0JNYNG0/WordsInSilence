from pathlib import Path
import json,sys
import numpy as np
from localized_direction import refine,localize_cues
U=Path(__file__).resolve().parent;O=U.parent;D=O/'word100_positive_direction_20261109'
sys.path.insert(0,str(O/'word100_actionable_feedback_20261113'))
from coaching import compose
name='NIA_SL_WORD1133_REAL11_F_main_reversed';z=np.load(D/'predictions'/(name+'.npz'))
base=json.loads((D/'final_decisions'/(name+'.json')).read_text('utf-8'))['before']
r=refine(base,z['attributes'],z['x'],z['timestamp_us'],z['frame_id'],z['phase'])
r.update(word='WORD1133',label='잘하다',timestamp_us=z['timestamp_us'].tolist(),synthetic=True,source='public_saved_component_test',path=[],video='')
r['coaching']=localize_cues(r,compose)
with (U/'UI_CONTRACT_INPUT.json').open('x',encoding='utf-8') as f:json.dump(r,f,ensure_ascii=False)
source=(O/'word100_actionable_feedback_20261113/test_feedback_ui.cjs').read_text('utf-8')
source=source.replace("const cue=data.coaching.cues[0];", "const cue=data.coaching.cues[0];assert.deepEqual(cue.focus_indices,[54,63]);")
source=source.replace("'same-direction cue merged'", "'only measured contiguous direction span [54,63] highlighted','same-direction cue merged'")
with (U/'test_feedback_ui.cjs').open('x',encoding='utf-8') as f:f.write(source)
