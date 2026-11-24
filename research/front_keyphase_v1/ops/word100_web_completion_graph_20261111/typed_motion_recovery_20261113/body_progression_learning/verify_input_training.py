"""Saved body-input/label/model-state audit. No inference, preparation or training."""
from pathlib import Path
import json,hashlib,datetime,ast
import numpy as np
import torch
B=Path(__file__).resolve().parent;H=B.parent;G=H.parent;N=H/'null_candidate_learning';S=H/'paired_static_progression_learning';P=H/'progression_presence_learning'
F=[2,3,4,5,7];AX={2:0,3:1,4:2,5:4,7:3};checks=[];cache={}
def rd(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):
 p=str(Path(p).resolve())
 if p not in cache:cache[p]=hashlib.sha256(Path(p).read_bytes()).hexdigest()
 return cache[p]
def ck(name,ok,detail=None):checks.append(dict(name=name,ok=bool(ok),detail=detail))
def pins(d,label):
 for p,h in d.items():ck(label+':'+p,Path(p).is_file() and sha(p)==h)
def check_nohands(x,name):
 ck(name+' finite1160',x.ndim==2 and x.shape[1]==1160 and np.isfinite(x).all())
 parts=x.reshape(-1,4,290);q=parts[:,:,64:164].reshape(-1,4,10,10)
 ck(name+' explicit hand absent',np.count_nonzero(q[:,:,:,4:8])==0 and np.count_nonzero(q[:,:,:,9])==0 and np.count_nonzero(parts[:,:,164:])==0)
def main():
 assert not(B/'INPUT_TRAINING_REVIEW.json').exists()
 cfg=rd(B/'FROZEN.json');prep=rd(B/'PREPARATION_CONTRACT.json');meta=rd(B/'INPUTS_COMPLETE.json');tc=rd(B/'TRAINING_COMPLETE.json');la=rd(B/'FIT_LABEL_AUDIT.json')
 rows=rd(N/'TRAINING_INPUT_INDEX.json')['rows'];masks=rd(N/'local_bag_evaluation/RULES_FROZEN.json')['partial_masks'];sm={r['bag_index']:r for r in rd(S/'INPUTS_COMPLETE.json')['bags']}
 for label,d in [('train',cfg['pins']),('prepare',prep['pins']),('inputs',meta['pins'])]:pins(d,label)
 ck('286 body and190static',len(rows)==len(masks)==len(meta['body_bags'])==286 and len(meta['static_bags'])==190)
 fit=[r for r in rows if r['split']=='train'];probe=[r for r in rows if r['split']=='probe']
 ck('source SHA split',not{r['source_sha256'] for r in fit}&{r['source_sha256'] for r in probe})
 ck('people FIT02..07 probe08..09',{r['person'] for r in fit}=={2,3,4,5,6,7} and {r['person'] for r in probe}=={8,9})
 ck('only body families',all(r['family'] in [3,4,5,7] for r in rows))
 bodyids=np.array([j*290+64+k*10+t for j in range(4) for k in range(10) for t in [0,1,2,3,8]])
 bags=[];first_errors=[];origins={};static={};eligibility=[]
 for i,(r,m,lab) in enumerate(zip(rows,meta['body_bags'],masks)):
  ck('row mapping:'+str(i),m['bag_index']==i and all(m[k]==r[k] for k in ['clip_id','person','split','family','view','reverse','origin','source_sha256']))
  ck('mask mapping:'+str(i),lab['bag_index']==i and all(lab[k]==r[k] for k in ['clip_id','family','split','view','reverse']))
  pins({m['path']:m['sha256'],r['path']:r['sha256']},'body prior and output');z=np.load(m['path']);old=np.load(r['path']);x=z['local_body'];check_nohands(x,'body'+str(i))
  ck('existing labels/spans:'+str(i),np.array_equal(z['spans'],old['spans']) and np.array_equal(z['y'],old['y']) and len(x)==len(z['y']))
  ck('body relations bit-identical:'+str(i),np.array_equal(x[:,bodyids],old['local_full'][:,bodyids]))
  ck('prior pin exact:'+str(i),m['prior_path']==r['path'] and m['prior_sha256']==r['sha256'])
  origins[r['origin']]=origins.get(r['origin'],0)+1
  if r['origin']=='positive_source_extent':
   ck('cached no-hands all candidates exact:'+str(i),np.array_equal(x,old['local_no_hands']))
   ck('first body reconstruction:'+str(i),m['first_cached_body_max_error']<1e-5 and m['feature_origin']=='reused_frozen_local_no_hands_cache')
   first_errors.append(m['first_cached_body_max_error'])
  else:
   a,b=r['source_span'];ck('hold source map:'+str(i),m['source_span']==[a,b] and not r['reverse'] and np.array_equal(old['source_frames'],np.arange(a,b+1)))
   ck('first full reconstruction:'+str(i),m['first_cached_full_max_error']<1e-5 and m['feature_origin']=='new_body_only_encoding_of_existing_reviewed_hold')
   first_errors.append(m['first_cached_full_max_error'])
  full=z['y'];partial=np.zeros(len(full),bool)
  for v in lab['candidates']:partial[v['candidate_index']]=True
  hold=r['origin']=='reviewed_local_hold';known=np.ones(len(full),bool) if hold else full|partial;y=np.zeros(len(full),bool) if hold else full|partial
  ck('partial/full/hold contract:'+str(i),not(full&partial).any() and (not hold or not(full|partial).any()))
  bags.append(dict(n=len(full),full=full,partial=partial,known=known,y=y,spans=z['spans']))
 ck('190reused96newhold',origins=={'positive_source_extent':190,'reviewed_local_hold':96})
 ck('190static covers all positives',[r['bag_index'] for r in meta['static_bags']]==[i for i,r in enumerate(rows) if r['origin']=='positive_source_extent'])
 for m in meta['static_bags']:
  i=m['bag_index'];r=rows[i];d=bags[i];pins({m['path']:m['sha256'],sm[i]['path']:sm[i]['sha256']},'static output/control');z=np.load(m['path']);old=np.load(sm[i]['path']);x=z['x'];check_nohands(x,'static'+str(i));mapping=z['mapping'];known=np.flatnonzero(d['known']);expected=[]
  for k in known:
   a,b=map(int,d['spans'][k]);expected.extend([[int(k),j,t] for j,t in enumerate([a,(a+b)//2,b])])
  ck('static meta:'+str(i),all(m[k]==r[k] for k in ['clip_id','person','split','family','view','reverse','origin','source_sha256']))
  ck('all known candidates3anchor exact:'+str(i),np.array_equal(mapping,expected) and np.array_equal(z['positive_indices'],known) and len(x)==len(known)*3)
  ck('static same source mapping as S:'+str(i),np.array_equal(mapping,old['mapping']) and np.array_equal(z['positive_indices'],old['positive_indices']))
  ck('static body bit identical to fullhand control:'+str(i),np.array_equal(x[:,bodyids],old['x'][:,bodyids]))
  parts=x.reshape(-1,4,290);vary=float(np.max(np.abs(parts[:,:3,64:]-parts[:,:1,64:])));std=float(np.max(np.abs(parts[:,3,64:])))
  ck('static explicit pose constant:'+str(i),vary<1e-4 and std<1e-4)
  cover=x[:,64+AX[r['family']]*2*10+8];ck('static visibilitybinary:'+str(i),np.isin(cover,[0,1]).all());eligible=cover>.5
  ck('invisible excluded:'+str(i),np.array_equal(z['eligible'],eligible) and m['eligible']==int(eligible.sum()) and m['excluded_unobserved']==int((~eligible).sum()))
  ck('eachpositive has visible anchor:'+str(i),all(eligible[mapping[:,0]==k].any() for k in known))
  eligibility.append(dict(bag_index=i,clip_id=r['clip_id'],family=r['family'],split=r['split'],view=r['view'],reverse=r['reverse'],generated=len(eligible),eligible=int(eligible.sum()),excluded_unobserved=int((~eligible).sum())))
  static[i]=dict(n=len(x),eligible=eligible)
 ck('anchor audit exact',eligibility==la['static_anchor_eligibility'])
 ck('matched duration pools preserved',la['matched_buckets']==rd(S/'FIT_LABEL_AUDIT.json')['matched_buckets'])
 ck('fixed3seed1200',cfg['seeds']==[42,43,44] and cfg['steps']==1200 and {r['seed'] for r in tc['rows']}=={42,43,44})
 original={r['seed']:r for r in rd(G/'candidate_description_supervision/local_rank_learning/TRAINING_COMPLETE.json')['rows'] if r['condition']=='local_no_hands'}
 model_rows=[]
 for r in tc['rows']:
  for k in ['model','predictions','static_predictions']:pins({r[k]:r[k+'_sha256']},'trained output')
  pins({r['initial_model']:r['initial_sha256']},'initial no-hands');old=original[r['seed']]
  ck('pretrained source identity:'+str(r['seed']),r['initial_model']==old['model'] and r['initial_sha256']==old['model_sha256'])
  state=torch.load(r['model'],weights_only=True,map_location='cpu');initial=torch.load(r['initial_model'],weights_only=True,map_location='cpu')
  ck('same architecture and normalization:'+str(r['seed']),set(state)==set(initial) and all(state[k].shape==initial[k].shape for k in state) and torch.equal(state['mean'],initial['mean']) and torch.equal(state['std'],initial['std']))
  ck('finite model weights:'+str(r['seed']),all(torch.isfinite(v).all() for v in state.values()))
  ck('1200 finalstep:'+str(r['seed']),r['log'][-1]['step']==1200)
  pred=np.load(r['predictions']);sp=np.load(r['static_predictions']);ck('all real/static prediction keys:'+str(r['seed']),set(pred.files)=={f'logits_{i}' for i in range(286)} and set(sp.files)=={f'logits_{i}' for i in static})
  for i,d in enumerate(bags):ck('real savedlogits:'+str((r['seed'],i)),pred[f'logits_{i}'].shape==(d['n'],5) and np.isfinite(pred[f'logits_{i}']).all())
  for i,d in static.items():ck('static savedlogits:'+str((r['seed'],i)),sp[f'logits_{i}'].shape==(d['n'],5) and np.isfinite(sp[f'logits_{i}']).all())
  model_rows.append(dict(seed=r['seed'],normalization_identical=True,initial_source_condition='local_no_hands',hidden_final_changed=not torch.equal(state['head.0.weight'],initial['head.0.weight']),lastlayer_final_changed=not torch.equal(state['head.2.weight'],initial['head.2.weight']),initialization_evidence='Frozen training code loads source state then resets head[2]; no pre-update hidden snapshot was saved, so no numerical initialization replay claimed.'))
 code=(B/'train.py').read_text('utf-8');ast.parse(code)
 ck('frozen code initialization sequence','net.load_state_dict(st);net.head[2].reset_parameters();net.train()' in code)
 ck('frozen code FIT sampling','posbags[f]=[i for i in tr' in code and 'for i in tr if rows[i]' in code and 'assert len(xx)==128' in code)
 protections=rd(G/'PROTECTIONS.json');pins(protections,'protected');ck('477 unchanged',len(protections)==477)
 ck('original models unchanged',all(tc[k] is False for k in ['common100_changed','ranker_changed','handshape_head_changed','web_changed','private_access','final8']))
 result=dict(at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),status='PASS' if all(c['ok'] for c in checks) else 'FAIL',checks=len(checks),failed=[c for c in checks if not c['ok']],body_bags=286,reused_positive_bags=190,new_reviewed_hold_bags=96,static_bags=190,max_first_candidate_reconstruction_error=max(first_errors),model_reviews=model_rows,protected_files=len(protections),no_model_forward=True,no_preparation_or_training_rerun=True,full_evaluation_pending=True,script_sha256=sha(__file__),reviewer='Codex motion_contract_review_1113 separate saved arithmetic from root execution; same agent authored B preparation/training code, not an independent implementation author or human truth reviewer.',limitations=['All cached positive arrays and explicit body channels verified; recorded first-candidate reconstruction only, no fresh encoder calls.','Progression presence does not certify action identity, direction, extent or completion.','Previously exposed probe08/09 is not a fresh final evaluation.'])
 (B/'INPUT_TRAINING_REVIEW.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
 print(json.dumps({k:result[k] for k in ['status','checks','failed','body_bags','static_bags','max_first_candidate_reconstruction_error','protected_files']},ensure_ascii=False))
 if result['failed']:raise SystemExit(1)
if __name__=='__main__':main()
