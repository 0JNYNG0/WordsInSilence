from pathlib import Path
import sys,json,numpy as np,torch,hashlib
from finger_core import FingerHead
U=Path(__file__).resolve().parent;O=U.parent
sys.path.insert(0,str(O/'word100_guide_requirement_learning_20261111'))
import requirement_core as rc
from stable_geometry import geometry
torch.set_num_threads(2);z=np.load(U/'finger_arrays.npz');rows=rc.read(U/'FINGER_INPUTS.json')['rows'];ix=np.array([i for i,r in enumerate(rows)if r['role']=='fit']);permuted=z['x'][ix].copy();permuted[:,:,63:]=np.roll(permuted[:,:,63:],1,axis=0);context=[]
for seed in [42,43,44]:
 ck=torch.load(U/'finger_models'/f'{seed}.pt',weights_only=True);m=FingerHead();m.load_state_dict(ck['state_dict']);m.eval();mu=np.array(ck['mean']);sd=np.array(ck['std'])
 with torch.no_grad():a=m(torch.tensor((z['x'][ix]-mu)/sd,dtype=torch.float32)).sigmoid().numpy();b=m(torch.tensor((permuted-mu)/sd,dtype=torch.float32)).sigmoid().numpy()
 context.append(dict(seed=seed,fit_anchors=len(ix),same_local_hand_changed_global_context_crossed_half=int(((a>=.5)!=(b>=.5))[:,1:].sum()),max_probability_change=float(abs(a-b)[:,1:].max()),scope='Synthetic input ablation; not a real sign or accuracy test'))
rc.write(U/'CONTEXT_SENSITIVITY.json',dict(rows=context))
old=rc.read(U/'MOTION_INPUTS.json');rr={r['clip_id']:r for r in rc.public_rows()};gg=[];meta=[]
from motion_core import endpoint_pause
for r in old['rows']:
 src=rr[r['clip_id']];g,ok,info=geometry(rc.prior,src['raw_x'],src['timestamp_us'],*r['span'],r['axis'],r['moving'],r['anchor']);gg.append(endpoint_pause(g,r['pause_fraction'])if ok else g);meta.append(dict(**{k:v for k,v in r.items()if k not in['eligible','observed_delta']},eligible=bool(ok),observed_delta=info.get('signed_change'),frame_basis=info.get('axis_mode','unchanged')))
np.savez_compressed(U/'motion_arrays_v2.npz',geometry=np.array(gg),targets=np.array([r['target']for r in meta]));rc.write(U/'MOTION_INPUTS_V2.json',dict(rows=meta,source_pins=old['source_pins']))
rc.write(U/'REPAIR_FROZEN_V2.json',dict(reason=['Automatic span remains unchanged: changing forearm basis near foreshortening reversed the computed normal trajectory; median observed interval basis removes this coordinate-frame artifact','FIT-only context-permutation audit checks whether the same visible hand receives different finger outputs from unrelated whole-body latent'],
 motion='Same existing motion conv+GRU fine-tuning schedule, corrected common along-forearm measurement; other4axes identical; no word condition',
 finger='Same temporalConv+GRU readout and source labels; remove unrelated whole-body input from per-finger branch. Common100 network remains the action context model.',
 initial_models='Same current frozen baseline, not best DEV checkpoint; same42/43/44 seeds and final600steps',
 comparison='All earlier failures retained; same exposed development used for diagnosis, not fresh accuracy. Candidate must pass full regression before web.',
 primary_gates=rc.read(U/'PLAN.json')['gates'],pins={str(U/n):rc.sha(U/n)for n in['stable_geometry.py','CONTEXT_SENSITIVITY.json','motion_arrays_v2.npz','FINGER_RGB_LABELS.json']},web_promotion=False))
print(json.dumps(context))
