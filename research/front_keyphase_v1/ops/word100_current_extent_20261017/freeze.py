import json,hashlib
from pathlib import Path
from datetime import datetime
from spatial import PARAMS
U=Path(__file__).resolve().parent;O=U.parent;B=O/'word100_hand_evidence_split_20261017'
def read(p):return json.loads(Path(p).read_text('utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
s=read(B/'SCOPE.json');records=s['records'];pins=dict(s['source_pins'])
manifest=read(O/'word100_training_20261013/MANIFEST.json')
codes=sorted({r['word_code'] for r in manifest}-{r['word_code'] for r in records})[::9][:8]
for i,code in enumerate(codes):
 src=next(r for r in manifest if r['word_code']==code and r['signer_id']==6+i%2)
 path=O/'word100_body_boundary_20261017/data/records'/f"{src['clip_id']}.json";r=read(path)
 assert r['role']=='train_candidate'
 c=r['sensor_capture'];paths={'raw':c['raw'],'candidate':c['candidate'],'assignment':str(Path(c['candidate']).parent/'assignment.npz')}
 for p in [str(path)]+list(paths.values()):pins[p]=sha(p)
 records.append({'clip_id':r['clip_id'],'word_code':code,'signer_id':r['signer_id'],'label':r['canonical_label'],
 'role':r['role'],'group':'new8','paths':paths,'source':r['source'],'box':r['calibration']['box_source_pixels'],
 'intervals':src['intervals'],'source_video':None})
deps=[U/'spatial.py',B/'separate.py',B/'OUTPUT.json',O/'word100_hand_region_uncertainty_20261017/region.py',
 O/'word100_hand_visibility_20261017/visibility.py',O/'word100_hand_region_uncertainty_20261017/VISUAL_LABELS.json']
out={'at':datetime.now().astimezone().isoformat(timespec='seconds'),'parameters':PARAMS,
 'authorization':'User: further common improvement for two boundary cases and overlap-related missing hands; public validation, preserve prior good cases, no user data or training until joint review.',
 'implementation':'Codex local; separate scalar verification code by same operator, not expert/other-agent review.',
 'design':'Preserve all separated fields. Add current Pose4 hull extent and geometric intersection with current actually observed other Hand points. Confidence .5 reused; intersection epsilon numerical only, no sweep. Out-of-image Pose is extrapolation not unseen RGB. No confirmed cause or hand recovery.',
 'acceptance':['Preserve all previous location/shape/overlap fields and raw arrays','Two known edge cases get boundary evidence without claiming recovered shape',
 'Report ALL visual cue contradictions, abstentions, overlap false positives and misses; any clear interior-hand boundary false assertion precludes automatic adoption',
 'No cause certainty or TCN accuracy claim; finite geometry validation is separate from real-world reliability'],
 'protocol':'Old60/141labels known regression. New8 distinct words sorted unused codes every9th, signer6/7 alternating. Longest core hand gap first/middle/last + first both-observed controls visually labeled before candidate output. One frozen replay, no post-result tuning.',
 'prohibited':['user/mentor recording read','training','TCN inference','new detector','camera','DEV/heldout','production/web/DTW change','download','external transfer','Git','post-result parameter tuning'],
 'source_pins':pins,'protected_pins':s['protected_pins'],'dependencies':{str(p):sha(p) for p in deps},'records':records}
for p,h in {**pins,**s['protected_pins']}.items():assert sha(p)==h,p
with (U/'SCOPE.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps({'frozen':True,'videos':len(records),'new':[(r['clip_id'],r['label']) for r in records if r['group']=='new8']},ensure_ascii=False))
