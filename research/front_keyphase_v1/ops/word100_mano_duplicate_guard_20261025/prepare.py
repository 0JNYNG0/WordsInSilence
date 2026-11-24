"""Freeze public replay, controls, and candidate before results; no tuning loop."""
from pathlib import Path
import json,hashlib,datetime
U=Path(__file__).resolve().parent;O=U.parent;OLD=O/'word100_mano_threeway_20261025';AUD=O/'word100_mano_missing_hand_accuracy_20261025'
def read(p):return json.loads(Path(p).read_text('utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
assert not (U/'FROZEN.json').exists()
old=read(OLD/'FROZEN_PROTOCOL.json');audit=read(AUD/'FROZEN.json')
groups={}
for p in sorted((O/'word100_body_boundary_20261017/data/records').glob('*.json')):
    r=read(p)
    if r['role']!='train_candidate':continue
    groups.setdefault(r['word_code'],[]).append((p,r))
controls=[]
for word,rows in sorted(groups.items()):
    for p,r in sorted(rows,key=lambda x:x[1]['signer_id'])[:3]:
        path=r['native_capture']['candidate'];h=r['native_capture']['pins'][path]
        assert sha(path)==h
        controls.append({'word_code':word,'clip_id':r['clip_id'],'signer':r['signer_id'],'path':path,'sha256':h,'record':str(p),'record_sha256':sha(p),'source':r['source'],'label_for_display_only':r['canonical_label']})
assert len(groups)==100 and len(controls)==300
pins={str(p):sha(p) for base in [OLD,AUD] for p in base.rglob('*') if p.is_file() and p.suffix in ['.json','.npz','.py','.html','.txt'] and not set(p.parts)&{'private_assets','external','runtime_env','downloads','cache','__pycache__'}}
f={'at':datetime.datetime.now().astimezone().isoformat(),'implementation':'Codex','candidate_sha256':sha(U/'candidate.py'),'authorization':'User requests same-hand duplicate guard, public verification, C/T3F and user data unchanged.',
'algorithm':'Before accepting a NEW HaMeR hand, compare same anatomical finger chains in absolute original image coordinates against existing hand. At least4of5 chains (>=3supported points each, median<=0.08shoulder AND fingertip<=0.08shoulder). Wrist/arm label cannot override this contradiction. Mark unresolved and hold ONLY new inferred use; retain raw candidate and original coordinates. No identity forced or baseline erased.',
'parameters_origin':'0.08 inherited unchanged from current frame point_agreement. 4of5 majority requires several fingers, not one overlapping contact; engineering hypothesis not calibrated probabilistic certainty. Frozen before replay, no score-driven changes.',
'conditions':['B unchanged original','H/HT unchanged saved candidate','G/GT H/HT plus finger occupancy conflict guard'],
'primary':'All57 missing-slot triggers in7public812frames. Diagnostic cohort, previously reviewed, not untouched heldout.',
'controls':'All243savedHaMeR requests for pair diagnostics; allbaseline normal/cross arrays preserved. Wider paired-hand proxy: first3 TRAIN signer IDs per100word,300clips, all complete visible hand pairs. Cached native hands are NOT image-ground-truth; flags need image inspection.',
'acceptance':'E05 duplicate no longer added; inspect every changed accepted addition and all newly flagged broad paired-hand controls; normal E01/E16 recoveries and20cross frames preserved. Any new visually valid hand held is reported, not silently excluded. Proxy flags are not error rates.',
'scope_limits':'No new inference/training/TCNscore, no user/mentor data, no production changes. Rules get no word ID or label. All100word scope is engineering portability, not proven allKSL performance.',
'manifest':old['manifest'],'controls':controls,'control_words':100,'protected':audit['protected'],'prior_pins':pins,'reference_audit':str(AUD),'forbidden':['modify_old_outputs','model_change','user_data','training','threshold_tuning_after_replay','git','external_upload']}
write(U/'FROZEN.json',f)
write(U/'AUTHORIZATION.json',{'at':f['at'],'user':'기존 손과 HaMeR가 찾은 손 중복 보완·공개검증, 정상손교차유지, C·T3-F유지, 사용자학습튜닝금지','implementation':'Codex local isolated research candidate','training':False,'deployment':False})
print('Frozen300controlclips100words; existing7clips57triggers; candidatehash',f['candidate_sha256'])

