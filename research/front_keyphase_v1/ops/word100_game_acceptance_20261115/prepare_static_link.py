from pathlib import Path
import json,hashlib
U=Path(__file__).resolve().parent;O=U.parent
source=O/'word100_live_g6_20261114/engine.py';text=source.read_text('utf-8-sig')
old="span=c['chosen']['span']if c['chosen']else None;es=[];ph=[]"
new="""span=c['chosen']['span']if c['chosen']else None;es=[];ph=[]
    # A stationary role has no movement-window candidate. Reuse its independently
    # detected native span to observe shape, rather than silently dropping it.
    # Never merge gaps or search for the required shape inside a larger region.
    if span is None and spec['axis'] is None:
     matches=c.get('role_confirmation',{}).get('matching',[])
     unique=sorted({tuple(e['span']) for e in matches if e['family'] in spec['families']})
     if len(unique)==1:span=list(unique[0])"""
assert text.count(old)==1;target=U/'engine_static_candidate.py';assert not target.exists();target.write_text(text.replace(old,new),'utf-8')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan=dict(change='Attach shape observations to exactly one independently detected stationary role interval; no new/changed windows, thresholds, model weights, expected-order search or word special cases.',scope='Existing shape heads only; this is not yet terminal completion or static-position validation.',gate='Existing candidate/order hashes unchanged, existing measured/action differences retained, no new normal correction in public raw controls. No release solely on adding positives.',pins={str(p):sha(p)for p in [source,target,U/'lesson_contract.py']})
(U/'STATIC_LINK_FROZEN.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),'utf-8');print('isolated common stationary-role link prepared')
