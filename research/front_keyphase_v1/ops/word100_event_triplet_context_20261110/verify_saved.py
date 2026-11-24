"""Saved-output arithmetic only; no model/DTW inference or learned code imports."""
from pathlib import Path
import json,hashlib,statistics,traceback
U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
def compact(d):
 ds=[c['display_status'] for c in d['cards']];w='difference' in ds;s=not w and all(c['display_status']=='similar' for c in d['cards'] if c.get('required',True))
 return dict(display=ds,warning=w,similar=s,uncertain=not w and not s)
def main():
 checks=[]
 def ck(n,v):checks.append(dict(name=n,passed=bool(v)));assert v,n
 cfg=read(U/'FROZEN.json');rr=read(U/'FULLPATH_COMPLETE.json')['rows'];ck('234_outputs',len(rr)==234);ck('unique',len({(r['seed'],r['clip_id'],r['kind']) for r in rr})==234)
 stats={};changed=[];newwarnings=[];regressions=[]
 for r in rr:
  ck('hash',sha(r['baseline'])==r['baseline_sha256'] and sha(r['result'])==r['sha256']);b=read(r['baseline']);a=read(r['result']);bb,aa=compact(b),compact(a)
  ck('originals_preserved',all(a[k]==b[k] for k in ['path','tcn_top3','frame_id','phase_events','phase_partials']))
  ck('no_false_positive_completion',not aa['similar'] or bb['similar'])
  for c,p in zip(a['cards'],b['cards']):
   ck('shape_direction_unchanged',c.get('shape_evidence')==p.get('shape_evidence') and c.get('motion_evidence')==p.get('motion_evidence'))
   ck('old_warning_preserved',p['display_status']!='difference' or c['display_status']=='difference')
  for p in a['triplet_segments']:
   ck('probability',abs(sum(p['probabilities'])-1)<1e-5 and abs(sum(p['core_probabilities'])-1)<1e-5)
   ck('mask_is_not_core',p['observed'] or p['core_state']=='unobserved')
  sk=str(r['seed']);stats.setdefault(sk,{})
  for kind in [r['kind']]+(['crossing'] if r['kind']=='full_normal' and 'WORD1567' in r['clip_id'] else []):
   s=stats[sk].setdefault(kind,{z:dict(n=0,warning=0,similar=0,uncertain=0) for z in ['before','after']})
   for side,data in [('before',bb),('after',aa)]:
    s[side]['n']+=1
    for k in ['warning','similar','uncertain']:s[side][k]+=int(data[k])
  if bb!=aa:changed.append(dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],before=bb,after=aa,result=r['result']))
  if aa['warning'] and not bb['warning']:newwarnings.append(changed[-1])
  if r['kind'] in ['full_normal','speed075','speed125'] and bb!=aa:regressions.append(changed[-1])
 sc=read(U/'SEGMENT_SCORES.json');goldstats={}
 for seed in cfg['seeds']:
  ss=[r for r in sc if r['seed']==seed];ck('24_reviewed_windows',len(ss)==24)
  confusion=[[0]*10 for _ in range(10)];coreconf=[[0]*3 for _ in range(3)]
  for r in ss:
   g=r['gold'];p=r['new'];confusion[g['context']][p['context']]+=1;pred=max(range(3),key=lambda k:p['core_probabilities'][k]);coreconf[g['state']][pred]+=1
  goldstats[str(seed)]=dict(n=24,old_context_correct=sum(r['old_context']==r['gold']['context'] for r in ss),new_context_correct=sum(r['new']['context']==r['gold']['context'] for r in ss),core_appearance_correct=sum(coreconf[k][k] for k in range(3)),context_confusion=confusion,core_confusion=coreconf,errors=[r for r in ss if r['new']['context']!=r['gold']['context']])
 training=[]
 for seed in cfg['seeds']:
  t=read(U/'runs'/str(seed)/'COMPLETE.json');ck('trained100',len(t['seen'])==899 and len(t['history'])==12 and all(ep['common_seen']==899 for ep in t['history']));ck('encoder_trained',t['encoder_delta']>0);ck('model_hash',sha(U/'runs'/str(seed)/'model.pt')==t['sha256']);training.append({k:t[k] for k in ['seed','encoder_delta','sha256','seconds']})
 for p,h in {**cfg['pins'],**cfg['protections'],**read(U/'CODE_PINS.json'),**read(U/'EVAL_PINS.json')}.items():ck('pinned:'+p,sha(p)==h)
 result=dict(status='EVALUATED_PENDING_CAUSAL_REVIEW',data=read(U/'DATA_AUDIT.json'),training=training,automatic=stats,reviewed_segments=goldstats,changes=changed,new_warnings=newwarnings,normal_regressions=regressions,latency=dict(median_seconds=statistics.median(r['seconds'] for r in rr),max_seconds=max(r['seconds'] for r in rr),scope='New geometry/context and connection only, excludes RTMW, frozen latent, TCN, DTW, capture/UI.'),private_used=False,default_changed=False,implementation='Codex',verification='Same Codex separate saved arithmetic, not independent expert.',limitations=['Historical development reused, no new independent set.','100word auxiliary training, dense event labels only36FIT of4expressions.','Core appearance is not completed correct sign; no automatic missing declaration solely from absence.'])
 write(U/'RESULT.json',result);write(U/'VERIFICATION.json',dict(passed=True,count=len(checks),checks=checks,result_sha256=sha(U/'RESULT.json')))
 print(json.dumps(dict(passed=True,checks=len(checks),primary=stats['42'],gold=goldstats['42'],changes=len(changed),regressions=len(regressions)),ensure_ascii=False))
if __name__=='__main__':
 try:main()
 except BaseException:(U/'VERIFY_FAILURE.txt').write_text(traceback.format_exc(),'utf-8');raise
