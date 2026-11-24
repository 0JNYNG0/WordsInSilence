"""Separate saved-output arithmetic and protection checks; no neural inference."""
from pathlib import Path
import json,hashlib,statistics,traceback,time
U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def compact(o):
 ds=[c['display_status'] for c in o['cards']]
 return dict(display=ds,warning='difference' in ds,similar_no_warning=all(c['display_status']=='similar' for c in o['cards'] if c.get('required',True)) and 'difference' not in ds)
def count(rr,side):return dict(n=len(rr),warning=sum(r[side]['warning'] for r in rr),similar=sum(r[side]['similar_no_warning'] for r in rr),uncertain=sum(not r[side]['warning'] and not r[side]['similar_no_warning'] for r in rr))
def main():
 checks=[]
 def ck(n,v):checks.append(dict(name=n,passed=bool(v)));assert v,n
 cfg=read(U/'FROZEN.json')
 for p,h in {**cfg['pins'],**cfg['protections'],**read(U/'CODE_PINS.json'),**read(U/'EVAL_PINS.json')}.items():ck('pin:'+p,sha(p)==h)
 ck('precheck',read(U/'PRETRAIN_CHECK.json')['passed'] and read(U/'CONTRACT_CHECK.json')['passed'])
 training=[]
 for seed in cfg['seeds']:
  d=read(U/'runs'/str(seed)/'COMPLETE.json');ck('training:'+str(seed),len(d['seen'])==899 and len(d['history'])==12 and all(r['common_seen']==899 for r in d['history']) and d['encoder_delta']>0 and sha(U/'runs'/str(seed)/'model.pt')==d['sha256']);training.append({k:d[k] for k in ['seed','encoder_delta','sha256','seconds']})
 rr=read(U/'FULLPATH_COMPLETE.json')['rows'];ck('all_outputs',len(rr)==234);frames=read(U/'FRAME_CONTEXT_SCORES.json');gold={r['clip_id']:{a['frame']:a['label'] for a in r['anchors']} for r in read(U/'LINK_LABELS.json')['rows']};oldmap={0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:6,9:7,10:7,11:8,12:9,13:9};changed=[];summary={};lookup={}
 for r in rr:
  b=read(r['baseline']);a=read(r['result']);lookup[(r['seed'],r['clip_id'],r['kind'])]=(a,b)
  ck('hash:'+r['result'],sha(r['baseline'])==r['baseline_sha256'] and sha(r['result'])==r['sha256'])
  ck('compact:'+r['result'],compact(a)==r['after'] and compact(b)==r['before'])
  ck('protected_outputs:'+r['result'],all(a[k]==b[k] for k in ['path','tcn_top3','frame_id','phase_events','phase_partials']) and all(c.get('shape_evidence')==d.get('shape_evidence') and c.get('motion_evidence')==d.get('motion_evidence') for c,d in zip(a['cards'],b['cards'])))
  ck('coverage:'+r['result'],r['frames']==r['model_frames']==len(a['frame_id'])==len(a['guide_link_probabilities']))
  ck('no_promotion:'+r['result'],not r['after']['similar_no_warning'] or r['before']['similar_no_warning'])
  for c in a['guide_link_order_checks']:
   aa,bb=c['first'],c['second'];states={('forward' if x['b']<y['a'] else 'reverse' if y['b']<x['a'] else 'overlap') for x in aa for y in bb};expected='unknown' if not aa or not bb else 'supported' if states=={'forward'} else 'contradicted' if states=={'reverse'} else 'ambiguous';ck('order_logic',c['state']==expected)
   ck('no_optional_as_mandatory','reach_wrist' not in c['pair'])
  if r['before']!=r['after']:changed.append({k:r[k] for k in ['seed','clip_id','kind','before','after','result']})
 for f in frames:
  a,b=lookup[(f['seed'],f['clip_id'],'full_normal')];j=f['frame'];new=max(range(10),key=lambda k:a['guide_link_probabilities'][j][k]);bp=b['phase_probability'][j];old=oldmap[max(range(len(bp)),key=lambda k:bp[k])];ck('frame_prediction',new==f['new'] and old==f['old'] and f['target']==gold[f['clip_id']][j])
 for seed in cfg['seeds']:
  ss=[r for r in rr if r['seed']==seed];ff=[r for r in frames if r['seed']==seed];ck('78_conditions:'+str(seed),len(ss)==78);ck('all_gold_frames',len(ff)==sum(len(v) for v in gold.values()))
  cats={}
  for kind in dict.fromkeys(r['kind'] for r in ss):cats[kind]={side:count([r for r in ss if r['kind']==kind],side) for side in ['before','after']}
  cats['crossing']={side:count([r for r in ss if r['kind']=='full_normal' and 'WORD1567' in r['clip_id']],side) for side in ['before','after']}
  conf=[[0]*10 for _ in range(10)]
  for f in ff:conf[f['target']][f['new']]+=1
  summary[str(seed)]=dict(automatic=cats,frame_context=dict(total=len(ff),old_correct=sum(f['old']==f['target'] for f in ff),new_correct=sum(f['new']==f['target'] for f in ff),confusion=conf),event_links=[e for e in read(U/'EVENT_LINK_SCORES.json') if e['seed']==seed])
 primary=[r for r in rr if r['seed']==42];normal=[r for r in primary if r['kind'] in ['full_normal','speed075','speed125']];reg=[r for r in normal if r['after']['warning'] and not r['before']['warning']];gain=[r for r in primary if r['kind'] not in ['full_normal','speed075','speed125','all_hand_missing','mask_final_event'] and r['after']['warning'] and not r['before']['warning']]
 verdict='NO_DEFAULT_ADOPTION'
 if not reg and gain:verdict='LIMITED_RESEARCH_GAIN_NO_DEFAULT_ADOPTION'
 result=dict(decision=verdict,primary_seed=42,data=read(U/'DATA_AUDIT.json'),training=training,models=summary,changed=changed,normal_warning_regressions=len(reg),new_modified_fixture_warnings=len(gain),latency=dict(median=statistics.median(r['seconds'] for r in primary),max=max(r['seconds'] for r in primary),scope='new context neural forward + event binding + selected frozen direction heads; excludes RTMW,base latent preparation,TCN,DTW,UI'),implementation='Codex',verification='Same Codex separate saved arithmetic; not an independent human review',private_used=False,default_changed=False,limitations=['HistoricalDEV reused, no independent unseen-person evidence.','Fine context review4videos recordedID11 only; oldphase target-conditioned vs newglobal contexts is a diagnostic comparison, not controlled same-task accuracy.','Whole-main reversal is a synthetic stress fixture, not proof each transformed clip is linguistically wrong.','Missing a predicted context is not proof the person omitted the motion.'])
 write(U/'RESULT.json',result);write(U/'VERIFICATION.json',dict(passed=True,count=len(checks),checks=checks,result_sha256=sha(U/'RESULT.json')))
 print(json.dumps(dict(passed=True,checks=len(checks),decision=verdict,regressions=len(reg),new_fixture_warnings=len(gain),frames={k:v['frame_context'] for k,v in summary.items()},changed=len(changed)),ensure_ascii=False))
if __name__=='__main__':
 try:main()
 except BaseException:(U/f'VERIFY_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
