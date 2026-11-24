"""Read-only public-data diagnosis; never trains or selects a new checkpoint."""
from pathlib import Path
import json, hashlib, sys, math, shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
import numpy as np
sys.dont_write_bytecode = True
HERE=Path(__file__).resolve().parent; OPS=HERE.parent; ROOT=OPS.parents[2]
PREV=OPS/'word100_assignment_training_20261018'; BASE=OPS/'word100_body_boundary_20261017'
SEEDS=[20261013,20261014,20261015]; VIEWS=['masked','sensor']
PINS={}; EXPECTED={}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):
 p=Path(p).resolve(); h=sha(p); k=str(p)
 if k in EXPECTED: assert h==EXPECTED[k],('prior_pin_changed',k)
 if k in PINS: assert h==PINS[k]
 PINS[k]=h; return p
def read(p): return json.loads(pin(p).read_text(encoding='utf-8-sig'))
def load(p):
 with np.load(pin(p),allow_pickle=False) as z:return {k:z[k] for k in z.files}
def write(name,obj):
 p=HERE/name; assert not p.exists(),str(p)
 p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def stamp():return datetime.now(timezone(timedelta(hours=9))).isoformat()
def clean(v):return float(v) if np.isfinite(v) else None
def quant(v):
 v=np.asarray(v,dtype=float)
 return {'n':len(v),'min':float(v.min()),'p50':float(np.median(v)),'p90':float(np.quantile(v,.9)),'max':float(v.max())} if len(v) else {'n':0}
def result_path(a,s):return (BASE/f'training_C_{s}_RESULT.json') if a=='C' else PREV/f'{a}_{s}_RESULT.json'
def history_path(a,s):return (BASE/f'training_C_{s}_history.json') if a=='C' else PREV/f'{a}_{s}_history.json'
def checkpoint(s):return BASE/f'training_C_{s}_best.pt'
def freeze():
 HERE.mkdir(exist_ok=True)
 for n in ['WordsInSilence_PROJECT_CONTEXT.md','NEW_CHAT_HANDOFF.md','research/front_keyphase_v1/ops/STATE.json']:
  dst=HERE/'before'/Path(n).name;dst.parent.mkdir(exist_ok=True);assert not dst.exists();shutil.copy2(ROOT/n,dst)
 paths=[PREV/n for n in ['PREP_CONTEXT.json','TRAIN_CONTEXT.json','TENSOR_CHANGE_EXTENT.json','COMPARISON_INDEPENDENT.json','FINAL_AUDIT.json','TRAIN_CHANGED_FRAME_AUDIT.json','DEV_INPUT_AUDIT.json']]
 paths += [result_path(a,s) for a in ['C','P','Q'] for s in SEEDS]
 paths += [history_path(a,s) for a in ['C','P','Q'] for s in SEEDS]
 paths += [checkpoint(s) for s in SEEDS]
 paths += [ROOT/'research/sign_model_pilot_v1/src/pilot/model_masked_v1.py',ROOT/'research/sign_model_pilot_v1/src/pilot/data.py',BASE/'training_core.py',Path(__file__)]
 for p in paths:pin(p)
 write('CONTRACT.json',{'at':stamp(),'user_request':'공개 자료로 손 배정 정정 후 성능 정체 및 손 크기 정규화 영향 분석, 원인과 다음 계획. 추가 학습은 사용자 확인 후.',
  'executor':'Codex local','training_allowed':False,'production_change_allowed':False,'user_or_mentor_data_allowed':False,
  'analyses':['Reuse all nine saved C/P/Q DEV predictions; paired gain/loss and saved selection histories, no checkpoint reselection.',
   'TRAIN899 public-only hand-scale support/distribution and whole/context/view variation.',
   'All P/Q TRAIN tensors changed from C: 2x2 observation(old,new) x hand scale(old,new). Exact original-corner reconstruction and scale-only propagation.',
   'Frozen three original C checkpoints: TRAIN changed-tensor 2x2 sensitivity only. No DEV inference, no gradients/optimizer, no performance/adoption claim from TRAIN sensitivity.'],
  'interpretation_limits':['No causal attribution of retrained DEV change to scale alone without a future controlled training experiment.','Changed-frame and tensor counts are distinct; repeated views/temporal slices are not independent samples.','DEV is repeatedly consulted development data, not blind or webcam testing.'],
  'future_plan_only':'Control normalization independently while preserving corrected assignments; define TRAIN-only stable-scale rule and paired comparison before any new training.',
  'source_pins':PINS.copy()})
def scale(xy,m,base):
 lengths=[]; frames=[]
 for t in range(len(xy)):
  if m[t,base]:
   for j in [5,9,13,17]:
    if m[t,base+j]:lengths.append(float(np.linalg.norm(xy[t,base+j]-xy[t,base])));frames.append(t)
 h=float(np.median(lengths)) if lengths else float('nan')
 if h<=1e-6:h=float('nan')
 return h,{'samples':len(lengths),'frames':len(set(frames)),'distance_summary_px':quant(lengths)}
def local_with_scale(template,xy,m,scales):
 out=template.copy();out[:,102:186]=0;out[:,237:279]=0
 for s,base in enumerate([33,54]):
  h=scales[s]
  if not np.isfinite(h):continue
  for j in range(21):
   ok=m[:,base]&m[:,base+j];k=21*s+j
   out[ok,102+2*k:104+2*k]=(xy[ok,base+j]-xy[ok,base])/h;out[ok,237+k]=1
 return out
def margin(z,y):return float(z[y]-np.max(np.delete(z,y)))
def saved_scores(rows,train_changed_words):
 labels=read(PREV/'TRAIN_CONTEXT.json')['labels'];results={(a,s):read(result_path(a,s)) for a in ['C','P','Q'] for s in SEEDS}
 paired=[];events=[];histories=[]
 for a,b in [('C','P'),('C','Q'),('P','Q')]:
  for v in VIEWS:
   counts=Counter();ev=[]
   for s in SEEDS:
    aa={x['clip_id']:x for x in results[a,s]['predictions'][v]};bb={x['clip_id']:x for x in results[b,s]['predictions'][v]};assert aa.keys()==bb.keys() and len(aa)==300
    for cid,x in aa.items():
     y=bb[cid];c0=bool(x['correct']);c1=bool(y['correct']);kind='gain' if c1 and not c0 else 'loss' if c0 and not c1 else 'both_correct' if c0 else 'both_wrong';counts[kind]+=1
     if kind in ['gain','loss'] or not c0 or not c1:
      e={'comparison':a+'->'+b,'view':v,'seed':s,'clip':cid,'word':labels[x['label_index']],'signer':x['signer_id'],'kind':kind,
       'before_prediction':labels[x['prediction']],'after_prediction':labels[y['prediction']],
       'before_margin':margin(np.asarray(x['logits']),x['label_index']),'after_margin':margin(np.asarray(y['logits']),y['label_index']),
       'before_ce':x['cross_entropy'],'after_ce':y['cross_entropy'],
       'dev_input_changed':cid=='NIA_SL_WORD2029_REAL11_F' and v=='sensor' and a=='C',
       'word_has_changed_TRAIN':x['label_index'] in train_changed_words[b]}
      events.append(e);ev.append(e)
   paired.append({'comparison':a+'->'+b,'view':v,'counts':dict(counts),'unique_gain_clips':len({e['clip'] for e in ev if e['kind']=='gain'}),'unique_loss_clips':len({e['clip'] for e in ev if e['kind']=='loss'}),'loss_events_with_identical_dev_input':sum(e['kind']=='loss' and not e['dev_input_changed'] for e in ev)})
 for a in ['C','P','Q']:
  for s in SEEDS:
   r=results[a,s];h=read(history_path(a,s));h=h['history'] if isinstance(h,dict) else h;sel=next(x for x in h if x['epoch']==r['selected_epoch'])
   histories.append({'arm':a,'seed':s,'selected_epoch':r['selected_epoch'],'selected_ce':r['selected_ce'],'selected_sensor_correct':r['metrics']['sensor']['top1_correct'],'selected_masked_correct':r['metrics']['masked']['top1_correct'],
    'last10_sensor':[(x['epoch'],x['sensor_top1']) for x in h[-10:]],'last10_masked':[(x['epoch'],x['masked_top1']) for x in h[-10:]],
    'first_epoch_train_loss':h[0]['train_ce'],'last_epoch_train_loss':h[-1]['train_ce'],'selection_rule':'mean DEV CE, no reselection'})
 write('SAVED_SCORE_DIAGNOSIS.json',{'paired':paired,'events':events,'histories':histories,'inference_performed':False})
 return results
def inputs(rows):
 summary={a:Counter() for a in ['P','Q']};details=[];scale_inventory=[];jobs=[];changed_words={a:set() for a in ['P','Q']};view_stats=[]
 for ri,r in enumerate(rows):
  if r['role']!='train_candidate':continue
  rec=read(PREV/'records'/f"{r['clip_id']}.json");EXPECTED.update(rec['outputs']);perclip={}
  for v in VIEWS:
   old=load(r['views'][v]['observation']);c=load(r['views'][v]['channels'])
   data={a:load(rec['views'][v][a]) for a in ['P','Q'] if not rec['feature_info'][v][a]['equals_C_tensor']}
   for t in ['whole','context']:
    ids=c[t+'_frame_id'];idx=np.searchsorted(old['frame_id'],ids);assert np.array_equal(old['frame_id'][idx],ids)
    oxy=old['xy_px'][idx];om=old['usable'][idx];x00=c['x_'+t];osc=[scale(oxy,om,b) for b in [33,54]];os=[q[0] for q in osc];perclip[v,t]=os
    for s in range(2):scale_inventory.append({'clip':r['clip_id'],'view':v,'temporal':t,'side':s,'scale_px':clean(os[s]),**osc[s][1]})
    for a,n in data.items():
     assert np.array_equal(ids,n[t+'_frame_id']);x11=n['x_'+t]
     if np.array_equal(x00,x11):continue
     ni=np.searchsorted(n['frame_id'],ids);nxy=n['xy_px'][ni];nm=n['usable'][ni];nsc=[scale(nxy,nm,b) for b in [33,54]];ns=[q[0] for q in nsc]
     assert np.array_equal(local_with_scale(x00,oxy,om,os),x00),'old corner reconstruction'
     assert np.array_equal(local_with_scale(x11,nxy,nm,ns),x11),'new corner reconstruction'
     x10=local_with_scale(x11,nxy,nm,os);x01=local_with_scale(x00,oxy,om,ns)
     raw_same=np.all(om==nm,1)&np.all((oxy==nxy)|(np.isnan(oxy)&np.isnan(nxy)),axis=(1,2))
     total=x11.astype(float)-x00;direct=x10.astype(float)-x00;scalepart=x11.astype(float)-x10
     assert np.allclose(total,direct+scalepart,atol=1e-12,rtol=0)
     changed=np.any(total!=0,1);spread=changed&raw_same
     # On unchanged observations all changed values must be local coordinates.
     assert not np.any(total[spread,:102]) and not np.any(total[spread,186:])
     ct=summary[a];ct['tensor_entries']+=1;ct['tensor_frames']+=len(ids);ct['direct_changed_frames']+=int((~raw_same).sum());ct['all_changed_frames']+=int(changed.sum());ct['scale_only_unchanged_frames']+=int(spread.sum());ct['changed_values']+=int(np.count_nonzero(total));ct['spread_values']+=int(np.count_nonzero(total[spread]))
     changed_words[a].add(r['label_index'])
     e={'clip':r['clip_id'],'word':r['label'],'label_index':r['label_index'],'view':v,'arm':a,'temporal':t,'frames':len(ids),
      'direct_changed_frames':ids[~raw_same].tolist(),'scale_propagated_frames':ids[spread].tolist(),'old_scales':[clean(z) for z in os],'new_scales':[clean(z) for z in ns],
      'old_support':[q[1] for q in osc],'new_support':[q[1] for q in nsc],
      'relative_scale_change_percent':[100*(y/x-1) if np.isfinite(x) and np.isfinite(y) else None for x,y in zip(os,ns)],
      'l1_total':float(np.abs(total).sum()),'l1_direct_then_scale':float(np.abs(direct).sum()),'l1_scale_after_direct':float(np.abs(scalepart).sum()),
      'l1_propagated_unchanged_frames':float(np.abs(total[spread]).sum()),
      'max_abs_propagated_value':float(np.abs(total[spread]).max()) if spread.any() else 0,
      'finite_scale_status_changed':[bool(np.isfinite(x)!=np.isfinite(y)) for x,y in zip(os,ns)]}
     details.append(e);jobs.append((e,[x00,x10,x01,x11]))
  for s in range(2):
   for v in VIEWS:
    x,y=perclip[v,'whole'][s],perclip[v,'context'][s]
    if np.isfinite(x) and np.isfinite(y):view_stats.append({'clip':r['clip_id'],'side':s,'type':'whole_to_context','view':v,'percent':100*(y/x-1)})
   for t in ['whole','context']:
    x,y=perclip['masked',t][s],perclip['sensor',t][s]
    if np.isfinite(x) and np.isfinite(y):view_stats.append({'clip':r['clip_id'],'side':s,'type':'masked_to_sensor','temporal':t,'percent':100*(y/x-1)})
  if (ri+1)%100==0:print('INPUT_SCALE_DIAGNOSIS',ri+1,flush=True)
 out={'summary':{a:dict(c) for a,c in summary.items()},'details':details,'changed_words':{a:sorted(v) for a,v in changed_words.items()},'channel_energy_is_not_accuracy_attribution':True}
 write('INPUT_DECOMPOSITION.json',out)
 write('TRAIN_SCALE_INVENTORY.json',{'hands':scale_inventory,'paired_variation':view_stats,
  'support_frames':quant([e['frames'] for e in scale_inventory if e['scale_px'] is not None]),
  'scale_unavailable':sum(e['scale_px'] is None for e in scale_inventory),
  'variation_summary':{k:quant([abs(e['percent']) for e in view_stats if e['type']==k]) for k in ['whole_to_context','masked_to_sensor']}})
 return jobs,changed_words
def sensitivity(jobs):
 import torch
 sys.path.insert(0,str(ROOT/'research/sign_model_pilot_v1/src'));sys.path.insert(0,str(BASE))
 from pilot.model_masked_v1 import MaskedSignTCN
 from training_core import MODEL
 torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 items=[]
 for s in SEEDS:
  ck=torch.load(pin(checkpoint(s)),map_location='cpu',weights_only=True);m=MaskedSignTCN(**MODEL);m.load_state_dict(ck['state_dict']);m.eval()
  with torch.inference_mode():
   for e,xs in jobs:
    x=torch.from_numpy(np.stack(xs));p=torch.ones((4,len(xs[0])),dtype=torch.float32);logits=m(x,p).numpy();assert np.isfinite(logits).all()
    y=e['label_index'];out={'clip':e['clip'],'word':e['word'],'arm':e['arm'],'view':e['view'],'temporal':e['temporal'],'seed':s,'label_index':y,'variants':{}}
    for name,z in zip(['old_old','new_old','old_new','new_new'],logits):
     prob=np.exp(z.astype(float)-z.max());prob/=prob.sum()
     out['variants'][name]={'logits':z.tolist(),'prediction':int(np.argmax(z)),'target_margin':margin(z,y),'target_probability':float(prob[y])}
    out['scale_after_direct_max_abs_logit']=float(np.abs(logits[3]-logits[1]).max());out['scale_after_direct_margin_delta']=margin(logits[3],y)-margin(logits[1],y)
    items.append(out)
  print('FIXED_C_TRAIN_SENSITIVITY',s,len(jobs)*4,flush=True)
 write('FROZEN_C_TRAIN_SENSITIVITY.json',{'items':items,'forward_samples':len(items)*4,'gradient_steps':0,'DEV_inference':0,'scope':'Changed TRAIN tensors only. This is sensitivity of already trained C, not accuracy of a new trained model or generalization evidence.'})
def main():
 contract=read(HERE/'CONTRACT.json');EXPECTED.update(contract['source_pins'])
 for p,h in contract['source_pins'].items():assert sha(p)==h
 c=read(PREV/'PREP_CONTEXT.json');EXPECTED.update(c['pins']);rows=c['rows'];assert Counter(r['role'] for r in rows)=={'train_candidate':899,'development_only':300}
 assert not (PREV/'TRAIN_RUNNING.json').exists() and not (PREV/'PREP_RUNNING.json').exists()
 jobs,words=inputs(rows);saved_scores(rows,words);sensitivity(jobs)
 for p,h in PINS.items():assert sha(p)==h,('changed_during_analysis',p)
 write('EXECUTION_AUDIT.json',{'at':stamp(),'status':'COMPLETE_DIAGNOSIS_PENDING_CHECK','source_pins':PINS,'new_training_runs':0,'user_data_used':False,'production_changed':False,'DEV_reinference':False,'code_sha256':sha(__file__)})
if __name__=='__main__':
 if '--freeze' in sys.argv:freeze()
 else:main()
