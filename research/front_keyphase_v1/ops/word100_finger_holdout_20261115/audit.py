"""Separate saved-artifact check. Same Codex author, not independent human review."""
from holdout_common import *
import collections
checks=[]
def check(name,ok):
 assert ok,name
 checks.append(name)
pins=read(U/'PROTECTED.json')
for p,h in pins.items():check('protected:'+p,sha(p)==h)
cfg=read(U/'CANDIDATE_FROZEN.json')
for p,h in cfg['pins'].items():check('candidate frozen:'+p,sha(p)==h)
check('not admitted',cfg['selected']=='unchanged' and not cfg['allow_web_candidate'])
rows=read(U/'TRAIN_INPUTS.json')['rows'];fit=[r for r in rows if r['role']=='fit'];dev=[r for r in rows if r['role']=='development']
check('fit identities',all(1<=r['person']<=9 for r in fit))
check('identity split',not({r['person']for r in fit}&{r['person']for r in dev}))
check('17/18 excluded from optimization',not any(r['person']in[17,18]for r in rows))
extra=read(U/'FIT_LABELS.json')['rows'];check('32 extra source labels',len(extra)==32 and len({r['clip_id']for r in extra})==6)
for r in extra:check('new source hash:'+r['clip_id']+':'+str(r['frame']),sha(r['source_video'])==r['source_sha256'])
for s in read(U/'FROZEN.json')['sources']:
 check('public cached source:'+s['clip_id'],sha(s['path'])==s['sha256'] and 'word100_t3f_heldout_20261019' in s['path'])
 check('no fine-finger selection overlap:'+s['clip_id'],all(s['clip_id']not in v['clips'] for v in read(U/'EXPOSURE_AUDIT.json')['manifests'].values()))
 r=next(v for v in read(U/'EXTRACTED.json')['rows']if v['clip_id']==s['clip_id']);z=np.load(r['raw_path']);check('full contiguous sequence:'+s['clip_id'],np.array_equal(z['frame_id'],np.arange(len(z['frame_id']))) and np.all(np.diff(z['timestamp_us'])>0))
totals={};front={}
for role in ['diagnostic','confirmation_reserved']:
 data=read(U/f'{role}_old/RESULTS.json');rr=data['rows'];calc=dict(known=sum(sum(y>=0 for y in r['y'])for r in rr),correct=sum(sum(y>=0 and y==p for y,p in zip(r['y'],r['pred']))for r in rr),wrong=sum(sum(y>=0 and p>=0 and y!=p for y,p in zip(r['y'],r['pred']))for r in rr),unknown=sum(sum(y>=0 and p<0 for y,p in zip(r['y'],r['pred']))for r in rr));check('recount:'+role,calc==data['summary']['all']);totals[role]=calc
 ff=[r for r in rr if r['view']=='front_pointing_index'];front[role]=dict(index_decisions=len(ff),correct=sum(r['pred'][0]==r['y'][0]for r in ff),wrong=sum(r['pred'][0]>=0 and r['pred'][0]!=r['y'][0]for r in ff),unknown=sum(r['pred'][0]<0 for r in ff))
check('front-index denominator not all fingers',front['diagnostic']==dict(index_decisions=5,correct=0,wrong=0,unknown=5) and front['confirmation_reserved']==dict(index_decisions=5,correct=4,wrong=0,unknown=1))
replay=read(U/'SHADOW_REPLAY.json');check('122 condition replay',len(replay['rows'])==122);check('64 normal preserved',sum(r['normal']for r in replay['rows'])==64 and replay['summary']['normal_new_warnings']==0);check('baseline verdicts preserved',all(r['verdicts_unchanged']for r in replay['rows']));check('no source truth relabel',read(U/'diagnostic_old/RESULTS.json')['labels_sha256']==sha(U/'LABELS_FROZEN.json')==read(U/'confirmation_reserved_old/RESULTS.json')['labels_sha256'])
write(U/'VERIFIED.json',dict(passed=True,checks=len(checks),protected_files_unchanged=len(pins),front_index_only=front,totals=totals,implementation='Codex',verification='Codex separate artifact audit, not independent reviewer',independent_KSL_expert_review=False,new_models_connected=False,user_recordings_used=False,web_files_unchanged=True,checks_detail=checks))
print(json.dumps(dict(checks=len(checks),protected=len(pins),front=front,passed=True)))
