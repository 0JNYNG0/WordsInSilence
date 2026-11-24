import json,hashlib
from pathlib import Path
from datetime import datetime
from region import PARAMS,BASE_PARAMS
U=Path(__file__).resolve().parent;O=U.parent;B=O/'word100_hand_visibility_20261017'
def read(p):return json.loads(Path(p).read_text('utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
base=read(B/'SCOPE.json');manifest=read(O/'word100_training_20261013/MANIFEST.json')
codes=sorted({r['word_code'] for r in manifest}-{r['word_code'] for r in base['records']})
selected=codes[::7][:12]
records=[{**r,'group':'previous48','source_video':r['source_video'] or str(B/'public_videos'/f"{r['clip_id']}.mp4")} for r in base['records']]
pins=dict(base['pins'])
for i,code in enumerate(selected):
    src=next(r for r in manifest if r['word_code']==code and r['signer_id']==4+i%2)
    p=O/'word100_body_boundary_20261017/data/records'/f"{src['clip_id']}.json";r=read(p)
    assert r['role']=='train_candidate'
    c=r['sensor_capture'];paths={'raw':c['raw'],'candidate':c['candidate'],'assignment':str(Path(c['candidate']).parent/'assignment.npz')}
    for q in [str(p)]+list(paths.values()):pins[q]=sha(q)
    records.append({'clip_id':r['clip_id'],'word_code':r['word_code'],'signer_id':r['signer_id'],'label':r['canonical_label'],
                    'role':r['role'],'group':'fresh12','paths':paths,'source':r['source'],'box':r['calibration']['box_source_pixels'],
                    'intervals':src['intervals'],'source_video':None})
scope={'at':datetime.now().astimezone().isoformat(timespec='seconds'), 'operator':'Codex root local implementation and separate verification code, no other reviewer',
 'authorization':'User approved region and movement uncertainty diagnostic and public validation, no training before result review.',
 'parameters':PARAMS,'base_parameters':BASE_PARAMS,'parameter_basis':'Engineering envelope fixed before public replay, not calibrated probability. Base0.05 shoulder for location tolerance, one last-hand diameter/sec unknown deformation, measured previous relative-shape speed added, history0.5s unchanged, abstain over half hand diameter. No sweep/post-result tuning.',
 'code_sha256':{'region.py':sha(U/'region.py'),'baseline':sha(B/'visibility.py')},'pins':pins,'records':records,
 'protocol':'Previous48 saved baseline and87 manual labels reused as regression (not new evidence). Disjoint12 word types, signer4/5 alternating, sorted unused word codes every7th frozen now. Fresh longest core missing gap first/middle/last each side + first both-observed control, visual labels before outputs. Replay once all60, previous baseline reuse; baseline fresh only. Review confident changed-state runs start/middle/end after result.',
 'acceptance':['Observed/partial/assignment states unchanged','No landmark/mask modification, future/native source signals, user data or word-specific branch',
 'Report every disagreement and abstention, do not count uncertainty as correct','Clear interior active hand labelled boundary/outside prevents adoption',
 'Region overlap means geometrical suspicion, not certified occlusion or recognition improvement'],
 'prohibited':['training','classifier inference','new detection','user/mentor recording read','DEV/heldout','production changes','camera','download','Git','post-result parameter sweeps']}
with (U/'SCOPE.json').open('x',encoding='utf-8') as f:json.dump(scope,f,ensure_ascii=False,indent=2)
print(json.dumps({'clips':len(records),'fresh':[(r['clip_id'],r['label']) for r in records if r['group']=='fresh12']},ensure_ascii=False))
