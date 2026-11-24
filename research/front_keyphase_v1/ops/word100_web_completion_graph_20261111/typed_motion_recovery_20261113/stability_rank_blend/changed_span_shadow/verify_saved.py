"""Independent saved-result arithmetic; deliberately imports no neural code."""
from pathlib import Path
from collections import Counter
import datetime, hashlib, json, traceback
import numpy as np

D=Path(__file__).resolve().parent; S=D.parent; H=S.parent; G=H.parent; O=G.parent
J=H/'splice_rank_learning/changed_span_shadow'; MODE='equal_rank_blend'
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rk(r): return r['seed'],r['clip_id'],r['kind']
def req(cid,kind,card,span): return cid,kind,card,None if span is None else tuple(span)
def rt(k): return '|'.join([k[0],k[1],k[2],'none' if k[3] is None else ':'.join(map(str,k[3]))])
checks=0
def ck(ok,label):
 global checks
 checks+=1
 if not ok: raise AssertionError(label)

def verdict(p,w,eligible):
 if not eligible:return 'observation_unknown'
 if w==1:return 'supported' if p[1]>=.65 and p[0]<.65 and p[2]<.65 else 'uncertain'
 if p[w]>=.65 and p[2-w]<.65 and p[1]<.65:return 'supported'
 if p[2-w]>=.65 and p[w]<.35 and p[1]<.65:return 'opposite_direction'
 if p[1]>=.65:return 'movement_not_established'
 return 'uncertain'

def geometry(x,ts,a,b,axis,side):
 # Independent body-wrist-only reconstruction for the two actually requested axes.
 p=np.asarray(x[:,:102],dtype=float).reshape(-1,51,2).copy(); valid=x[:,186:237]>.5
 shoulder=valid[:,1]&valid[:,2]
 scale=float(np.median(np.sqrt(((p[shoulder,1]-p[shoulder,2])**2).sum(1)))) if shoulder.any() else 0.
 if scale<1e-5:
  p[:]=0;valid=np.zeros_like(valid)
 else:
  p=(p-(p[:,1]+p[:,2])[:,None]/2)/scale;valid=valid&shoulder[:,None]
 p[:,:,0]*=(np.sign(p[:,2,0]-p[:,1,0])*(1 if side=='R' else -1))[:,None]
 wrist,elbow=(6,4) if side=='R' else (5,3)
 if axis==0:
  q=p[:,wrist]; values=np.stack([-q[:,1],q[:,0],-q[:,1],q[:,0]],1);good=valid[:,wrist]
 elif axis==1:
  arm=p[:,wrist]-p[:,elbow]; length=np.sqrt((arm**2).sum(1))
  values=np.stack([-arm[:,1]/np.maximum(length,1e-5),arm[:,0]/np.maximum(length,1e-5),length,-p[:,wrist,1]],1)
  good=valid[:,wrist]&valid[:,elbow]&(length>.05)
 else:raise AssertionError(('unexpected new direction axis',axis))
 good=good&np.isfinite(values).all(1);good=good[a:b+1];values=values[a:b+1];t=np.asarray(ts[a:b+1],float)
 observed=int(good.sum());coverage=float(good.mean());eligible=coverage>=.8 and observed>=4 and bool(good[0] and good[-1])
 if not eligible:return np.zeros(58,np.float32),False,dict(coverage=coverage,observed=observed)
 knots=np.linspace(t[good][0],t[good][-1],9);sample=np.column_stack([np.interp(knots,t[good],values[good,j])for j in range(4)])
 delta=sample[-1]-sample[0]
 geo=np.concatenate([(sample-sample[0]).reshape(-1),sample[0],delta,np.abs(np.diff(sample,axis=0)).mean(0),sample.max(0)-sample.min(0),np.eye(5)[axis],[coverage]]).astype(np.float32)
 info=dict(coverage=coverage,observed=observed,signed_change=float(delta[0]),trajectory=sample[:,0].tolist(),body_wrist_only=True,contact_verified=False)
 return geo,True,info

def status_signature(c):
 return (c['status'],[(e['task'],e.get('feature'),e.get('axis'),e.get('expected'),e.get('desired'),e['state']) for e in c['evidence']],c['completion'])

def main():
 dst=D/'VERIFICATION.json';ck(not dst.exists(),'one verification only')
 cfg=read(D/'FROZEN.json');res=read(D/'RESULTS.json');old=read(J/'RESULTS.json');rank=read(S/'RESULTS.json')
 ck(res['frozen_sha256']==sha(D/'FROZEN.json'),'result freeze hash')
 ck(cfg['at']<read(D/'RUN_STARTED.json')['at']<=res['at'],'freeze precedes execution')
 for p,h in cfg['pins'].items():ck(sha(p)==h,('frozen pin',p))
 protected=read(G/'PROTECTIONS.json');ck(len(protected)==477,'477 protected files')
 ck(all(cfg['pins'].get(p)==h for p,h in protected.items()),'protections preserved in freeze')
 maps=[{rk(r):r for r in data['rows']}for data in [res,old,rank]];rm,om,lm=maps
 ck(set(rm)==set(om)==set(lm) and len(rm)==246,'246 same scenario/seed rows')
 saved={}
 for r in old['rows']:
  ck(len(r['conditions'])==5,'five old conditions')
  for cards in r['conditions'].values():
   for c in cards:
    k=req(r['clip_id'],r['kind'],c['card'],c['span'])
    ck(k not in saved or saved[k]==c,('saved request consistency',k));saved[k]=c
 movements={r['request_id']:r for r in res['new_movements']};ck(len(movements)==8,'eight unique movement outputs')
 caps={(r['word'],r['card'],r['feature'],r['expected']):r['qualified']for r in read(O/'word100_guide_requirement_learning_20261111/GUIDE_CAPABILITY.json')['rows']}
 cardspec=read(O/'word100_action_feedback_demo_20261108/CARDS.json')['words'];families=read(O/'word100_action_episode_supervision_20261110/GUIDE_FAMILIES.json')['words']
 sources={(r['clip_id'],r['kind']):r for r in cfg['sources']}; arrays={};newcards={};max_geo_error=0.;shapecells=0
 newrequests=[r for r in cfg['requests']if r['action']=='new_direction_only'];ck(len(newrequests)==8,'eight frozen new requests')
 for q in newrequests:
  key=q['clip_id'],q['kind'];source=sources[key];m=movements[q['request_id']]
  ck(m['source_sha256']==source['sha256'],'new output source hash')
  if key not in arrays:
   with np.load(source['source'],allow_pickle=False)as z:arrays[key]={k:z[k].copy()for k in ['x','ts']}
  x,t=arrays[key]['x'],arrays[key]['ts'];word=q['clip_id'].split('_')[2];guide=next(c for c in cardspec[word]if c['id']==q['card']);spec=next(c for c in families[word]if c['id']==q['card'])
  a,b=q['span'];aa=a+(b-a)//2 if guide.get('shape_scope')=='late'else a;items=[]
  for j,want in guide.get('features',{}).items():
   j=int(j);pts=[9,30]if j in[5,6]else([9]+list(range(14,30))if j in[2,4]else[30]+list(range(35,51)))
   availability=np.all(x[aa:b+1,186:237][:,pts]>.5,axis=1)
   ck(not availability.any(),'new request has zero eligible shape frames');shapecells+=1
   items.append(dict(task='shape',feature=j,expected=want,span=[aa,b],probability=None,state='observation_unknown',qualified=caps.get((word,q['card'],j,want),False),observed_count=0))
  geo,eligible,info=geometry(x,t,a,b,spec['axis'],spec['hands'][0]);error=float(np.max(np.abs(geo-np.asarray(m['geometry']))));max_geo_error=max(max_geo_error,error)
  ck(error<=2e-7,'independent geometry values');ck(eligible==m['eligible'],'independent geometry eligibility')
  ck(info.keys()==m['info'].keys(),'geometry info keys')
  for k,v in info.items():ck(bool(np.allclose(v,m['info'][k],rtol=1e-7,atol=1e-8)),('geometry info',k))
  p=m['probabilities'];ck(len(p)==3 and np.isfinite(p).all() and min(p)>=0 and max(p)<=1 and abs(sum(p)-1)<2e-6,'stored movement probability validity')
  want=int(spec['sign']+1);items.append(dict(task='direction',axis=spec['axis'],desired=want,span=[a,b],probabilities=p,state=verdict(p,want,eligible)))
  states=[e['state']for e in items];status='difference_evidence'if any(s in['difference','opposite_direction']for s in states)else'supported_features_only'if states and all(s=='supported'for s in states)else'insufficient_evidence'
  newcards[q['request_id']]=dict(card=q['card'],span=[a,b],status=status,evidence=items,completion=None)
 plan=[];requests={};changed=0;unchanged=0;normal_count=0;normal_changes=[];allmissing=[];missing=[]
 for key,r in rm.items():
  prev=om[key];current=lm[key];ck(set(r['conditions'])==set(prev['conditions'])|{MODE},'one added condition')
  for mode,oldcs in prev['conditions'].items():ck(r['conditions'][mode]==oldcs,('old condition exact',mode,key))
  bc={c['card']:c for c in prev['conditions']['local_full']};cs={c['card']:c for c in r['conditions'][MODE]}
  ck(set(cs)=={c['card']for c in current['conditions'][MODE]['cards']},'new cards match selector')
  for c in current['conditions'][MODE]['cards']:
   s=c['chosen']['span']if c['chosen']else None;k=req(key[1],key[2],c['card'],s);rid=rt(k);before=bc[c['card']];after=cs[c['card']]
   if s==before['span']:action='copy_same_local_full';expected=before;unchanged+=1
   else:
    changed+=1;ck(c['family']in[2,3],'only family2/3 changed')
    action='reuse_saved_old_or_J'if k in saved else'new_direction_only';expected=saved[k]if k in saved else newcards[rid];requests[k]=action
   ck(after==expected,('entire new card matches independent source',key,c['card']))
   ck(after['completion']is None,'no semantic completion promotion')
   plan.append(dict(seed=key[0],clip_id=key[1],kind=key[2],card=c['card'],old_span=before['span'],new_span=s,action=action,request_id=rid))
   if key[2]in['full_normal','speed075','speed125']:
    normal_count+=1
    if status_signature(before)!=status_signature(after):normal_changes.append(dict(key=key,card=c['card'],before=status_signature(before),after=status_signature(after)))
   if key[2]=='all_hand_missing':allmissing.append(after)
   if key[2]in['omit_final_event','mask_final_event']:missing.append(after)
 ck(plan==cfg['plan']==res['execution'],'all execution mapping matches freeze')
 ck(changed==63 and unchanged==255 and len(plan)==318,'318cards63changed')
 ck(len(requests)==48 and Counter(requests.values())=={'reuse_saved_old_or_J':40,'new_direction_only':8},'48unique40cached8new')
 ck(not normal_changes,'normal and speed card/evidence states unchanged')
 shape_missing=[e for c in allmissing for e in c['evidence']if e['task']=='shape']
 ck(len(shape_missing)==72 and all(e['state']=='observation_unknown'for e in shape_missing),'72allmissing shape unknown')
 ck(all(c['completion']is None and c['status']!='supported_features_only'for c in allmissing),'no fully supported missing-hand card')
 ck(all(c['completion']is None for c in missing),'omission/mask no completion')
 # Recount all conditions, not merely copying existing summary fields.
 reconstructed=[]
 for seed in[42,43,44]:
  for mode in res['rows'][0]['conditions']:
   for kind in sorted({r['kind']for r in res['rows']}):
    cs=[c for r in res['rows']if r['seed']==seed and r['kind']==kind for c in r['conditions'][mode]];e=[v for c in cs for v in c['evidence']]
    reconstructed.append(dict(seed=seed,condition=mode,kind=kind,cards=len(cs),statuses=dict(Counter(c['status']for c in cs)),shape=dict(Counter(v['state']for v in e if v['task']=='shape')),direction=dict(Counter(v['state']for v in e if v['task']=='direction'))))
 sm=lambda rows:{(r['seed'],r['condition'],r['kind']):r for r in rows}
 ck(sm(reconstructed)==sm(res['summary']),'all summaries independently recounted')
 ck([r for r in res['summary']if r['condition']!=MODE]==old['summary'],'old summary exact')
 for k in['new_attribute_frames','attribute_layer_calls']:ck(res[k]==0,('zero attribute field',k))
 ck(res['G2_pass']is None and res['G3_pass']is None and res['semantic_completion']is None,'no gate or completion assertion')
 report=dict(at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),status='SAVED_ARITHMETIC_AND_PRESERVATION_PASS_NOT_G2_OR_G3_PASS',reviewer='Delegated Codex independent saved arithmetic; no human/KSL gold',checks=checks,rows=246,total_cards=318,unchanged_cards=255,changed_cards=63,unique_changed_requests=48,saved_reused_requests=40,new_direction_requests=8,new_attribute_frames=0,attribute_layer_calls=0,old_conditions=5,protected_files=477,frozen_pins_checked=len(cfg['pins']),normal_speed_cards=normal_count,normal_speed_state_changes=normal_changes,all_hand_missing_cards=len(allmissing),all_hand_missing_shape_unknown=len(shape_missing),omission_mask_cards_no_completion=len(missing),new_direction_shape_cells_checked=shapecells,independent_geometry_max_absolute_error=max_geo_error,all_summaries_recounted=len(reconstructed),old_condition_and_summary_exact=True,new_neural_calls=0,new_training=False,threshold_selection=False,old_artifacts_modified=False,G2_pass=None,G3_pass=None,limitations=['Saved movement probabilities were checked for arithmetic and eligibility; the neural model was not rerun.','Direction evidence may remain visible through body wrists when handshape is missing. That is not whole-action support.','Shape unknown remains unknown; omission/mask cards have completion=null. This does not independently validate all absence semantics.','Normal state preservation here is relative to local_full shadow, not a replacement of the original G2 baseline.','Selector seeds are42/43/44; the shared semantic observer isfixed42. These are not three independently trained semantic observers.','Only public exposed-development sources; no final-test or new-user accuracy claim.'],pins={str(p):sha(p)for p in[Path(__file__),D/'RESULTS.json',D/'FROZEN.json',S/'RESULTS.json',J/'RESULTS.json',G/'PROTECTIONS.json']})
 with dst.open('x',encoding='utf-8')as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
 print(json.dumps({k:v for k,v in report.items()if k not in['pins','limitations']},ensure_ascii=False))

if __name__=='__main__':
 try:main()
 except BaseException:
  stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
  with (D/f'VERIFICATION_FAILURE_{stamp}.txt').open('x',encoding='utf-8')as f:f.write(traceback.format_exc())
  raise
