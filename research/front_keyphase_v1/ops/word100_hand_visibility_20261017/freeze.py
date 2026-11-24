"""Freeze common rules and sample selection before candidate replay."""
import json, hashlib
from pathlib import Path
from datetime import datetime
from visibility import PARAMS
U=Path(__file__).resolve().parent;OPS=U.parent
def read(p):return json.loads(Path(p).read_text('utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
previous=read(OPS/'word100_overlap_sequence_20261017/AUDIT.json')
manifest=read(OPS/'word100_training_20261013/MANIFEST.json')
known=[r['clip_id'] for r in previous['records']]
codes=sorted({r['word_code'] for r in manifest}-{r['word_code'] for r in previous['records']})
selected_codes=[codes[i] for i in range(0,len(codes),8)][:12]
extra=[next(r for r in manifest if r['word_code']==code and r['signer_id']==2+i%2) for i,code in enumerate(selected_codes)]
rows=[];pins={}
for cid in known+[r['clip_id'] for r in extra]:
    mp=OPS/'word100_body_boundary_20261017/data/records'/f'{cid}.json';r=read(mp)
    assert r['role']=='train_candidate' and r['signer_id']<=9
    source=next(x for x in manifest if x['clip_id']==cid)
    cap=r['sensor_capture'];cp=Path(cap['candidate']).parent
    paths={'raw':cap['raw'],'candidate':cap['candidate'],'assignment':str(cp/'assignment.npz')}
    for p in [str(mp)]+list(paths.values()):pins[p]=sha(p)
    rows.append({'clip_id':cid,'word_code':r['word_code'],'signer_id':r['signer_id'],
                 'label':r['canonical_label'],'role':r['role'],'group':'known36' if cid in known else 'additional12',
                 'intervals':source['intervals'],'source':r['source'],'paths':paths,
                 'box':r['calibration']['box_source_pixels'],
                 'source_video':next((x['video'] for x in previous['records'] if x['clip_id']==cid),None)})
scope={'at':datetime.now().astimezone().isoformat(timespec='seconds'),
       'authorization':'User requests common outside-vs-overlap observation diagnosis implementation and public validation; no training until review.',
       'parameters':PARAMS,'code_sha256':sha(U/'visibility.py'),'records':rows,'pins':pins,
       'parameter_basis':'0.5 confidence/history,0.35 pose proximity,0.4 track link reuse earlier common assignment scales; boundary0.15 shoulder and broad arm length0.1..1.5 are conservative engineering gates fixed before replay, not fitted thresholds. No sweep.',
       'semantics':'Evidence labels only. boundary includes partly cut hands, not proof of entire hand outside or unused hand. overlap requires own recent observation and other hand proximity, not certified occlusion. Unknown retained.',
       'validation':'All48 sensor caches complete clips, core separate. Label-blind video review: additional12 each side longest missing-palm gap start/mid/end, plus first common-observed core frame. Known mind06 55..68, Gapyeong01 83..98, Gangjin03 85..98 and Sueo01 35 reuse previous visual interpretation. Compare labels after freezing manual review. Additional12 are challenge sample, not independent new people or clinical ground truth.',
       'acceptance':['No per-word inputs/branches','No native/full-source signals in algorithm','No future-frame lookahead or landmark/mask modification',
                     'Known overlap must never be called boundary/outside; known outside-rest must never be called overlap',
                     'Report coverage and every disagreement in additional public sample, do not tune after result',
                     'Any clear in-frame active hand labelled outside stops adoption; uncertainty not counted as correct classification'],
       'prohibited':['training','classifier inference','new detector','user/mentor recording reads','DEV/heldout use','production changes','camera','downloads','Git','post-result parameter tuning'],
       'operator':'Codex root; separate verification code, no external/subagent review'}
with (U/'SCOPE.json').open('x',encoding='utf-8') as f:json.dump(scope,f,ensure_ascii=False,indent=2)
print(json.dumps({'records':len(rows),'additional':[(r['clip_id'],r['label']) for r in rows if r['group']=='additional12']},ensure_ascii=False))
