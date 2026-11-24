"""Independent guard verification; no candidate modules imported."""
import json,hashlib,math,time
from pathlib import Path
import numpy as np
U=Path(__file__).resolve().parent;G=U/'guarded';F=U.parent/'word100_forearm_contour_20261018'
def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
start=time.monotonic();ctx=read(G/'CONTEXT.json');basecheck=read(U/'INDEPENDENT.json');assert basecheck['numeric_status']=='PASS'
assert not (G/'INDEPENDENT.json').exists()
for p,h in ctx['pins'].items():assert sha(p)==h,p
rows={r['clip_id']:r for r in read(U/'CONTEXT.json')['rows']};v1=read(U/'CHANGES.json');expected={};vetoes=[]
for x in v1:
    key=(x['clip_id'],x['view'],x['index']);r=rows[key[0]];t=key[2]
    with np.load(r['views'][key[1]]['raw']) as z:
        valid=[]
        for k in [15,16]:
            a,b=map(float,z['pose_xy_px'][t,k]);valid.append(bool(z['pose_usable'][t,k]) and math.isfinite(a) and math.isfinite(b) and a>=0 and b>=0 and a<int(z['width'][t]) and b<int(z['height'][t]))
        both=all(valid)
    expected[key]=x['before'] if both else x['after']
    if both:vetoes.append(key)
events=read(G/'EVENTS.json')
assert len(events)==len(expected)
for x in events:
    key=(x['clip_id'],x['view'],x['index']);assert x['after']==expected[key] and x['guard']['veto']==(key in vetoes)
patches={(x['clip_id'],x['view'],x['index']):x for x in read(G/'SIDE_PATCHES.json')}
assert set(patches)=={k for k,v in expected.items() if v!=next(x['before'] for x in v1 if (x['clip_id'],x['view'],x['index'])==k)}
changed=[];frames=0;views=0
for r in rows.values():
    for view in ['masked','sensor']:
        with np.load(U/'results'/f"{r['clip_id']}_{view}.npz") as a:
            original=a['baseline_sides'];proposal=a['sides'];verified=original.copy();sparse=original.copy()
            # Exhaustively account for every V1 change; V2 may only veto it.
            for t,h in np.argwhere(original!=proposal):
                key=(r['clip_id'],view,int(t));assert key in expected
                verified[t,h]=expected[key][h]
            for key,x in patches.items():
                if key[:2]==(r['clip_id'],view):sparse[key[2],:len(x['after'])]=x['after']
            assert np.array_equal(verified,sparse)
            for t,h in np.argwhere(original!=verified):changed.append([r['clip_id'],view,int(t),int(h),int(original[t,h]),int(verified[t,h])])
            frames+=len(original);views+=1
assert frames==226362 and views==1798 and len(changed)==1
normal=[a for a in read(G/'ANNOTATED.json') if a['verdict']=='normal_identity'];assert len(normal)==441 and all(a['after']==a['actual_side']==a['before'] for a in normal)
cross=read(G/'CROSSING.json');other=[a for a in cross if not a['previous_known_error']];assert len(other)==314 and all(a['before']==a['after'] for a in other)
prior=read(F/'CHANGES.json');assert len(prior)==13
for x in prior:
    key=(x['clip_id'],x['view'],x['index']);assert key not in patches
assert len(vetoes)==1 and read(G/'RESULT.json')['vetoed_frames']==1
note=read(U/'VERIFICATION_NOTE.json')
assert sha(U/'candidate.py')==note['candidate_sha256'] and sha(U/'RESULTS.json')==note['results_sha256'] and sha(U/'SUMMARY.json')==note['summary_sha256']
for p,h in ctx['pins'].items():assert sha(p)==h,p
result={'status':'PASS','frames':frames,'views':views,'raw_hands':305602,'changed_hands':len(changed),'unchanged_hand_associations':305602-len(changed),'guard_vetoes':len(vetoes),'normal441_preserved':True,'non_known_error_crossing314_preserved':True,'prior13preserved':True,'points_preserved':6292471,'protected_pins':len(ctx['pins']),'first_candidate_unchanged':True,'changed':changed,'elapsed_seconds':time.monotonic()-start,'method':'Independent current-Pose usable/in-frame truth table plus exhaustive1798array sparse reconstruction. Full geometry independently verified by verify_numeric.py. SameCodex, no second human.'}
(G/'INDEPENDENT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(json.dumps(result),flush=True)
