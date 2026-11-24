"""Separate arithmetic/integrity audit of saved outputs; no model forward or fitting."""
from pathlib import Path
import json,hashlib,collections
import numpy as np
E=Path(__file__).resolve().parent;O=E.parent;N=O/'word100_motion_event_candidates_20261109'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks=[]
def check(ok,label):
 checks.append(label)
 if not bool(ok):raise AssertionError(label)
def counts(cases,arm):
 cc=[r[arm] for r in cases]
 return dict(clips=len(cc),difference=sum(any(x=='difference' for x in c['display']) for c in cc),uncertain=sum(any(x=='uncertain' for x in c['display']) for c in cc),missing=sum(any(x=='missing_candidate' for x in c['status']) for c in cc),similar=sum(all(v=='similar' for v,req in zip(c['display'],c['required']) if req) for c in cc),required_present=sum(all(v for v,req in zip(c['phase_present'],c['required']) if req) for c in cc))
cfg=read(E/'FROZEN.json');check(sha(E/'REVIEW_LABELS.json')==cfg['labels_sha256'],'label freeze');check(sha(E/'INPUTS.json')==cfg['inputs_sha256'],'input freeze')
for p,h in cfg['protected'].items():check(sha(p)==h,'protected '+p)
for pins in ['TRAIN_CODE_PINS.json','EVAL_CODE_PINS.json']:
 for p,h in read(E/pins).items():check(sha(p)==h,'code pin '+p)
check(sha(E/'COMPARISON.json')==read(E/'FOLLOWUP_FROZEN.json')['first_comparison_sha256'],'first comparison preserved')
labels=read(E/'REVIEW_LABELS.json')['rows'];check(len(labels)==36,'36 source clips');check(sum(len(r['anchors']) for r in labels)==576,'576 sampled frames')
check({r['person'] for r in labels}==set(range(1,10)),'FIT01-09 only')
for r in labels:
 check(sha(r['sheet'])==r['sheet_sha256'],'RGB sheet '+r['clip_id']);check([a['frame'] for a in r['anchors']]==r['frames'],'source identity '+r['clip_id']);check(all(-1<=a['label']<=18 for a in r['anchors']),'labels range')
for r in read(E/'INPUTS.json')['rows']:
 check(r['person'] in range(1,10),'no DEV training');check(sha(r['path'])==r['sha256'],'input hash');z=np.load(r['path']);check(len(z['target'])==len(z['raw']),'frame denominator');check(np.all(np.diff(z['timestamp_us'])>0),'monotonic time');check(np.all((z['weight']==0)|(z['weight']==.2)|(z['weight']==1)),'weak weights explicit')
for name in ['heldout_1_2_3','heldout_4_5_6','heldout_7_8_9','final_42','final_43','final_44']:
 d=read(E/'runs'/name/'COMPLETE.json');check(sha(E/'runs'/name/'model.pt')==d['head_sha256'],'head pin');check(sha(N/'runs'/name/'model.pt')==d['encoder_sha256'],'encoder pin');check(not any(int(c.split('REAL')[1][:2]) in d['excluded_people'] for c in d['fit_clips']),'excluded people '+name)
reports={}
for stage,folder,manifest in [('first','scores','COMPARISON.json'),('repaired','followup_scores','FOLLOWUP.json')]:
 for name,s in read(E/manifest)['models'].items():
  rows=read(E/folder/f'{name}_cases.json');check(len(rows)==237,'237 conditions '+name);check(len({r['key'] for r in rows})==237,'unique conditions');check({r['clip_id'].split('REAL')[1][:2] for r in rows}=={'11','12','13'},'historical DEV IDs')
  for k,summary in s['by_kind'].items():
   inv,kind=k.split('/');rr=[r for r in rows if r['inventory']==inv and r['kind']==kind]
   for arm in ['deployed','prior','after']:check(counts(rr,arm)==summary[arm],'independent count '+stage+'/'+name+'/'+k+'/'+arm)
  for r in rows:
   check(sha(r['decision'])==r['sha256'],'decision integrity');d=read(r['decision']);check(len(d['cards'])==len(r['after']['display']),'cards preserved')
   for c,display in zip(d['cards'],r['after']['display']):
    check(c['display_status']==display,'display matches');check(not c['educational_wrong'],'no unverified correctness');check(display!='similar' or c['core_observed'],'no similar without core')
   if r['kind']=='all_hand_missing':
    check(all(x=='uncertain' for x,req in zip(r['after']['display'],r['after']['required']) if req),'all missing required remains unknown');check(not any(x in ['similar','difference'] for x in r['after']['display']),'missing never success or wrong')
  reports[stage+'/'+name]=dict(normal=counts([r for r in rows if r['inventory']=='main144' and r['kind']=='full_normal'],'after'),crossing=counts([r for r in rows if r['inventory']=='main144' and r['kind']=='full_normal' and 'WORD1567' in r['clip_id']],'after'))
# Wrong finger interventions leave geometry-only event predictions exactly unchanged.
for name in ['final_42','final_43','final_44']:
 for cid in sorted({r['clip_id'] for r in read(E/'scores'/f'{name}_cases.json')}):
  base=np.load(E/'predictions'/name/f'main144_{cid}_full_normal.npz')['prob']
  for kind in ['close_R','close_L','close_both']:
   check(np.array_equal(base,np.load(E/'predictions'/name/f'main144_{cid}_{kind}.npz')['prob']),'finger invariant '+name+'/'+cid+'/'+kind)
h=read(E/'HELDOUT.json');check(len(h['partial_controls'])==61,'61 heldout partial controls');check(sum(r['wrong_core_claim'] for r in h['partial_controls'])==h['partial_wrong_core'],'partial arithmetic')
for p in h['predictions']:check(sha(p['path'])==p['sha256'],'heldout prediction pin')
for kind,s in h['summary'].items():
 rr=[r for r in h['anchors'] if r['kind']==kind];core=[r for r in rr if r['label']>0 and r['label']%2==0];check(s==dict(anchors=len(rr),exact=sum(r['pred']==r['label'] for r in rr),event_identity=sum(((r['pred']-1)//2==(r['label']-1)//2) if r['label'] else r['pred']==0 for r in rr),core_anchors=len(core),core_above_threshold=sum(r['core_probability']>=.65 for r in core)),'heldout arithmetic '+kind)
out=dict(status='PASS',checks=len(checks),reports=reports,reviewer='Same Codex separate saved-output arithmetic, not independent person or expert truth',neural_forward=False,training=False,protected_preserved=True)
(E/'VERIFIED.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(status='PASS',checks=len(checks),reports=reports),ensure_ascii=False))
