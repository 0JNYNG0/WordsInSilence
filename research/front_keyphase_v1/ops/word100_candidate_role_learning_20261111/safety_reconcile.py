"""Post-score conservative reconciliation; reuses saved decisions only."""
from pathlib import Path
import json,hashlib,copy
U=Path(__file__).resolve().parent;O=U.parent;A=O/'word100_autonomous_localization_20261111';G=U/'safety_reconcile';G.mkdir(exist_ok=True)
rd=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def wr(p,v):p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2),'utf-8')
def compact(cards):
 ds=[c['display_status'] for c in cards];return dict(display=ds,warning='difference' in ds,similar_no_warning=all(c['display_status']=='similar' for c in cards if c.get('required',True)) and 'difference' not in ds)
MAP={0:0,1:1,2:2,3:3,4:4,5:4,6:5,7:5,8:5,9:5,10:5,11:6,12:7,13:7}
def conflict(spans,prob):
 checks=[]
 for s in spans:
  votes=sum(max(p)>=.65 and MAP[max(range(len(p)),key=lambda j:p[j])]!=s['family'] for p in prob[s['a']:s['b']+1]);checks.append(dict(span=[s['a'],s['b']],family=s['family'],votes=votes,frames=s['b']-s['a']+1,conflict=votes>(s['b']-s['a']+1)/2))
 return checks
assert not (G/'COMPLETE.json').exists()
wr(G/'FROZEN.json',dict(reason='Post-score RGB audit found a removed action misexplained as order reversal: face-to-chest motion was assigned to a later forearm role. Existing earlier phase evidence strongly disagrees. Restore pre-existing conflicting-evidence guard for new order/direction warnings.',rule='Reuse unchanged0.65 confidence/majority disagreement test from prior episode integration; conflicting role identities cannot produce new direction/order correction. Preserve old warnings and all candidate evidence.',no_threshold_change=True,no_training=True,no_model_inference=True,post_score_safety_repair=True,not_fresh_validation=True))
original={r['key']:r for r in rd(A/'DECISION_CACHE.json')};contracts=rd(O/'word100_action_episode_supervision_20261110/GUIDE_FAMILIES.json')['words'];result=[];rejected=[]
for r in rd(U/'FULLPATH_COMPLETE.json')['rows']:
 if r['mode']!='combined':continue
 key=r['clip_id']+'_'+r['kind'];raw=rd(r['result']);base=rd(raw['baseline']);orig=rd(original[key]['baseline_file']);cfg=contracts[orig['word']];by={c['id']:s for c,s in zip(raw['cards'],cfg)};cards=copy.deepcopy(base['cards']);decisions=[]
 for card,proposed in zip(cards,raw['cards']):
  b=proposed['learned_role_binding'];spans=b['spans'];checks=conflict(spans,orig['phase_probability']);new=set(proposed.get('difference_components',[]))-set(card.get('difference_components',[]));allowed=[];highlight=[];message=None
  if 'role_direction' in new:
   ok=not any(c['conflict'] for c in checks);decisions.append(dict(card=card['id'],component='role_direction',allowed=ok,checks=checks))
   if ok:allowed.append('role_direction');message='표시된 구간의 이동 방향을 바꿔 보세요. '+by[card['id']]['instruction'];highlight.extend(d['span'] for d in b['directions'] if d['eligible'] and d['confidence']>=.65)
  if 'role_order' in new:
   relevant=[v for v in raw['order_checks'] if v['second']==card['id'] and v['state']=='contradicted'];cs=[v for t in relevant for v in conflict(t['first_spans']+t['second_spans'],orig['phase_probability'])];ok=bool(relevant) and not any(c['conflict'] for c in cs);decisions.append(dict(card=card['id'],component='role_order',allowed=ok,checks=cs))
   if ok:allowed.append('role_order');message='이 두 행동의 순서를 확인해 보세요. 먼저 '+by[relevant[0]['first']]['instruction'];highlight.extend([s['a'],s['b']] for v in relevant for s in v['first_spans']+v['second_spans'])
  if 'role_omission' in new:
   allowed.append('role_omission');message=proposed['message'];highlight.extend(proposed.get('highlight_spans',[]));decisions.append(dict(card=card['id'],component='role_omission',allowed=True,checks='Original observed-return/coverage/no-positive-target criteria retained; no new omission from disagreement.'))
  if allowed:
   card.update(status='difference_candidate',display_status='difference',educational_wrong=False,message=message,candidate_message=message,highlight_spans=highlight);card.setdefault('difference_components',[]).extend(x for x in allowed if x not in card.get('difference_components',[]))
  card['learned_role_binding']=copy.deepcopy(b)
 out=dict(raw,cards=cards,reconciliation=decisions,first_result=r['result'],first_sha256=r['sha256']);p=G/'feedback'/str(r['seed'])/f'{key}.json';wr(p,out);result.append(dict(r,after=compact(cards),result=str(p),sha256=sha(p),first_result=r['result'],first_sha256=r['sha256']));rejected.extend(dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],**d) for d in decisions if not d['allowed'])
wr(G/'FULLPATH_COMPLETE.json',dict(rows=result));wr(G/'REJECTED.json',dict(rows=rejected));wr(G/'COMPLETE.json',dict(rows=len(result),rejected_components=len(rejected),neural_inferences=0,post_score=True));print(json.dumps(dict(rows=len(result),rejected_components=len(rejected))))
