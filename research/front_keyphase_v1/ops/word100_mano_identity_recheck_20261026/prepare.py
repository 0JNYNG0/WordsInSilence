from pathlib import Path
from datetime import datetime,timezone,timedelta
import hashlib,json
from identity import PARAMS
U=Path(__file__).resolve().parent;O=U.parent;OLD=O/'word100_mano_duplicate_guard_20261025'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
f=json.loads((OLD/'FROZEN.json').read_text('utf-8'))
pins=dict(f['prior_pins'])
for p in OLD.rglob('*'):
 if p.is_file() and not any(x in p.parts for x in ['__pycache__','before_closeout','source_clips','source_frames','views']):pins[str(p)]=sha(p)
pins[str(O/'word100_fullframe_hand_merge_20261020/candidate.py')]=sha(O/'word100_fullframe_hand_merge_20261020/candidate.py')
contract={'at':datetime.now(timezone(timedelta(hours=9))).isoformat(),'task':'Identity and wrist recheck after same-hand duplicate guard; user Proceed','implementation':'Codex local','candidate_sha256':sha(U/'identity.py'),'params':PARAMS,
'parameter_origin':'Current arm .25/.10, optical flow640/21/3/FB1.5/error20,match.12/ambiguity.5 inherited from current image tracker. Bidirectional nearest-clean anchors within0.5s chosen as engineering diagnostic hypothesis, not fitted success threshold. min4/5tips inherited duplicate majority. No sweep or word exception.',
'protocol':'Primary all7public clips/812frames/57missing slots; resolve only priorV2 actual duplicate conflicts, diagnostic all243saved requests and protected anchors/crossings. Keep all old files, choose at most one hypothesis on current arm + forward/backward original RGB tracks. No reliable bracketing or ambiguous winner => explicit hold. Finger shape remains model estimate. Fixed ROI only after selection. No creation of hidden hand. Same-image flow supports identity/position not exact anatomy.',
'evaluation':'OldG/GT versus separated identity-support candidate. Source-review all changed frames; preserve nonconflict coordinates and all20crossframes/94normalcontrolframes. Additional stored300clip two-hand duplicate scan result reused; test new identity rule at two flagged broad scenes from original RGB. Not newHaMeR300 inference nor untouched evaluation.',
'acceptance':'No normal crossing or known distinct-hand contact lost. Any newly wrong identity/replacement fails adoption. Unresolved identity not a successful correction. Current and previous candidates preserved independently. Report whether actual resolution occurs, no requirement to force any.',
'raw_preservation':'Keep G/GT and all original input, provide effective comparison eligibility separate from raw. Suppressed old wrong slot remains in raw archive; corrected candidate may replace its effective slot, never mutate original source.',
'manifest':f['manifest'],'controls':f['controls'],'protected':f['protected'],'prior_pins':pins,'prior_unit':str(OLD),'training':False,'user_data_access':False,'deployment':False,'new_neural_inference':False}
with (U/'FROZEN.json').open('x',encoding='utf-8') as out:json.dump(contract,out,ensure_ascii=False,indent=2)
print('Frozen',len(pins),'prior pins')
