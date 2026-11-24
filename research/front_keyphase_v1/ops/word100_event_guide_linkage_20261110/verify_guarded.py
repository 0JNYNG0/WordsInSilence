"""Final saved-output verification, independent of the guard implementation."""
from pathlib import Path
import json,hashlib,statistics,time,traceback
U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n','utf-8')
def compact(d):
 vals=[c['display_status'] for c in d['cards']];warn='difference' in vals;similar=not warn and all(c['display_status']=='similar' for c in d['cards'] if c.get('required',True))
 return dict(display=vals,warning=warn,similar=similar,uncertain=not warn and not similar)
def main():
 checks=[]
 def ck(n,v):checks.append(dict(name=n,passed=bool(v)));assert v,n
 first=read(U/'RESULT.json');v=read(U/'VERIFICATION.json');ck('first_result_preserved',v['passed'] and sha(U/'RESULT.json')==v['result_sha256']);ck('shape_separation',read(U/'HANDSHAPE_IDENTITY_CHECK.json')['passed'])
 protocol=read(U/'GUARD_PROTOCOL.json');ck('guard_pin',sha(U/'semantic_guard.py')==protocol['code_sha256'])
 mapping={0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:6,9:7,10:7,11:8,12:9,13:9};rr=read(U/'GUARDED_COMPLETE.json')['rows'];ck('all234',len(rr)==234);stats={};changed=[];veto=[]
 for r in rr:
  b=read(r['before']);c=read(r['candidate']);a=read(r['result']);ck('hashes',all(sha(r[k])==r[k+'_sha256'] for k in ['before','candidate']) and sha(r['result'])==r['sha256'])
  ck('observations_preserved',all(a[k]==c[k] for k in ['guide_link_probabilities','guide_link_bindings','guide_link_order_checks','guide_link_direction_checks','phase_events','phase_partials']) and all(a[k]==b[k] for k in ['tcn_top3','path','frame_id']))
  for cc,bc in zip(a['cards'],b['cards']):ck('shape_direction_preserved',cc.get('shape_evidence')==bc.get('shape_evidence') and cc.get('motion_evidence')==bc.get('motion_evidence'))
  for card,au in a['guide_link_context_conflicts'].items():
   for x in au['checks']:
    lo,hi=x['span'];votes=0
    for p in b['phase_probability'][lo:hi+1]:
     k=max(range(len(p)),key=lambda k:p[k]);votes+=p[k]>=.65 and mapping[k]!=x['new_context']
    ck('conflict_arithmetic',votes==x['old_conflicting_frames'] and (votes>(hi-lo+1)/2)==x['conflict'])
  ba,aa=compact(b),compact(a);ck('no_false_completion_promotion',not aa['similar'] or ba['similar']);ck('no_new_warning_from_veto',not aa['warning'] or compact(c)['warning'])
  if ba!=aa:changed.append(dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],before=ba,after=aa,result=r['result']))
  if r['vetoes']:veto.append(dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],vetoes=r['vetoes']))
  sk=str(r['seed']);stats.setdefault(sk,{})
  categories=[r['kind']]+(['crossing'] if r['kind']=='full_normal' and 'WORD1567' in r['clip_id'] else [])
  for kind in categories:
   vv=stats[sk].setdefault(kind,{side:dict(n=0,warning=0,similar=0,uncertain=0) for side in ['before','after']})
   for side,d in [('before',ba),('after',aa)]:
    vv[side]['n']+=1
    for k in ['warning','similar','uncertain']:vv[side][k]+=int(d[k])
 for seed,cats in stats.items():
  for kind in ['full_normal','speed075','speed125','crossing','all_hand_missing','mask_final_event','omit_final_event','right_closed_hand_main','main_reversed','closed_hand_and_reversed']:ck('regression:'+seed+kind,cats[kind]['before']==cats[kind]['after'])
  ck('one_extra_orderfixture_warning:'+seed,cats['reversed_core_auto_order']['after']==dict(n=9,warning=4,similar=1,uncertain=4))
 for p,h in read(U/'FROZEN.json')['protections'].items():ck('protected:'+p,sha(p)==h)
 result=dict(status='COMPLETE_LIMITED_DIRECTION_LINK_GAIN_NO_DEFAULT_ADOPTION',primary_seed=42,first_result=str(U/'RESULT.json'),first_result_sha256=sha(U/'RESULT.json'),data=first['data'],training=first['training'],automatic=stats,frame_context={k:v['frame_context'] for k,v in first['models'].items()},event_links={k:v['event_links'] for k,v in first['models'].items()},changed=changed,vetoes=veto,latency_before_guard=first['latency'],findings=['One added opposite-direction warning in a reversed-core public fixture per seed; not proof pure order or omission comprehension.','False semantic association in omission fixture was vetoed; same threshold used, no retraining/parameter search.','Another reversed case is withheld because semantic observers conflict.','All prior normal/speed/crossing,shape/direction error,missing/omission results preserved.'],limitations=first['limitations']+['Final conflict veto was added after historicalDEV error review; final results are retrospective research evidence, not independent confirmation.','All added final warnings currently concern one expression; implementation is shared, broad gains are not established.'],private_used=False,default_changed=False,implementation='Codex local',verification='Same Codex separate saved-output arithmetic; no independent human expert review.')
 write(U/'FINAL_RESULT.json',result);write(U/'FINAL_VERIFICATION.json',dict(passed=True,count=len(checks),checks=checks,result_sha256=sha(U/'FINAL_RESULT.json')));print(json.dumps(dict(passed=True,checks=len(checks),changed=len(changed),vetoed=len(veto),primary=stats['42']),ensure_ascii=False))
if __name__=='__main__':
 try:main()
 except BaseException:(U/f'FINAL_VERIFY_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
