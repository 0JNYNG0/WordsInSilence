"""Separate NumPy reference and saved-score analysis. Does not import diagnostic/model code."""
from pathlib import Path
import json,hashlib,traceback
import numpy as np
D=Path(__file__).resolve().parent;U=D.parent/'word100_auxiliary_supervision_train_20261019'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 assert not p.exists(),p;p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def reference(h,w,head):
 hidden=np.zeros(16,dtype=np.float64)
 for frame in h:
  gi=w['weight_ih_l0'].astype('float64')@frame+w['bias_ih_l0']
  gh=w['weight_hh_l0'].astype('float64')@hidden+w['bias_hh_l0']
  r=1/(1+np.exp(-(gi[:16]+gh[:16])));z=1/(1+np.exp(-(gi[16:32]+gh[16:32])))
  n=np.tanh(gi[32:]+r*gh[32:]);hidden=n*(1-z)+hidden*z
 return head.astype('float64')@hidden
def main():
 s=read(D/'SCOPE.json');c=read(D/'COMPLETE.json')
 assert sha(D/'SCOPE.json')==c['scope_sha256'];assert sha(D/'PROBES.npz')==c['probes_sha256']
 assert sha(D/'NUMERIC_SAMPLES.npz')==c['numeric_samples_sha256']
 for p,h in s['pins'].items():assert sha(p)==h,p
 with np.load(D/'PROBES.npz',allow_pickle=False) as q:z={k:q[k] for k in q.files}
 with np.load(D/'NUMERIC_SAMPLES.npz',allow_pickle=False) as q:t={k:q[k] for k in q.files}
 a=z['aux'].astype('float64');b=z['base'].astype('float64');y=z['labels'];labels=s['labels']
 assert a.shape==(3,2,899,6,100) and np.isfinite(a).all() and np.isfinite(b).all()
 assert len(set(z['ids']))==899 and set(y)==set(range(100))
 errors=[];perm_checks=0
 for si in range(3):
  w={k:t[f'w{si}_{k}'] for k in ('weight_ih_l0','weight_hh_l0','bias_ih_l0','bias_hh_l0')}
  for vi in range(2):
   for ri in s['sample_indices']:
    h=t[f'h{si}_{vi}_{ri}'];n=len(h)
    # Independent Python block construction, matching np.array_split remainder allocation.
    q,r=divmod(n,3);cuts=[0,q+(r>0),2*q+(r>0)+(r>1),n]
    first=list(range(cuts[0],cuts[1]));middle=list(range(cuts[1],cuts[2]));last=list(range(cuts[2],cuts[3]))
    orders=[list(range(n)),middle+first+last,last+middle+first,first+middle,last,[n-1]]
    assert sorted(orders[1])==sorted(orders[2])==list(range(n));assert orders[1][-len(last):]==last
    for ci,ids in enumerate(orders):
     ref=reference(h[ids],w,t[f'head{si}']);err=float(np.max(np.abs(ref-a[si,vi,ri,ci])))
     assert err<5e-6,(si,vi,ri,ci,err);errors.append(err);perm_checks+=1
 rows=[];byword=[]
 for si,seed in enumerate(s['seeds']):
  for vi,v in enumerate(s['views']):
   original=a[si,vi,:,0];origpred=original.argmax(-1);oc=original-original.mean(-1,keepdims=True)
   norm=np.sqrt(np.mean(oc**2,axis=-1));assert np.all(norm>0)
   for ci,cond in enumerate(s['conditions']):
    current=a[si,vi,:,ci];cp=current.argmax(-1);cc=current-current.mean(-1,keepdims=True)
    rms=np.sqrt(np.mean((cc-oc)**2,axis=-1));rel=rms/norm
    exact=np.max(np.abs(current-original),axis=-1)
    allsame=sum(np.all(cp[y==li]==origpred[y==li]) for li in range(100))
    rows.append({'seed':seed,'view':v,'condition':cond,'n':899,'aux_label_hits':int((cp==y).sum()),
     'same_aux_class':int((cp==origpred).sum()),'same_aux_class_pct':float(100*np.mean(cp==origpred)),
     'centered_logit_rms_median':float(np.median(rms)),'relative_centered_rms_median_pct':float(100*np.median(rel)),
     'relative_centered_rms_p90_pct':float(100*np.quantile(rel,.9)),'all_unchanged_words':int(allsame),
     'max_absolute_logit_difference':float(exact.max())})
    for li in range(100):
     ids=y==li;byword.append({'seed':seed,'view':v,'condition':cond,'label':labels[li],'n':int(ids.sum()),'same_aux_class':int(np.sum(cp[ids]==origpred[ids]))})
 source=[];dev=[];errcases=[];hist=[]
 for si,seed in enumerate(s['seeds']):
  result=read(U/f'FG16AUX01_{seed}_RESULT.json');hh=read(U/f'FG16AUX01_{seed}_history.json')
  hist.append({'seed':seed,'selected_epoch':result['selected_epoch'],'last_epoch':hh[-1]['epoch'],
   'first':hh[0],'selected':hh[result['selected_epoch']-1],'last':hh[-1]})
  for vi,v in enumerate(s['views']):
   source.append({'seed':seed,'view':v,'n':899,'base_hits':int((b[si,vi].argmax(-1)==y).sum()),
    'aux_hits':int((a[si,vi,:,0].argmax(-1)==y).sum()),'final_hits':int(((b[si,vi]+a[si,vi,:,0]).argmax(-1)==y).sum())})
   saved=[sorted(result[k][v],key=lambda r:r['clip_id']) for k in ('base_predictions','aux_predictions','predictions')]
   bs,aa,ff=saved;assert all(r['status']=='OK' for r in bs+aa+ff)
   assert len(bs)==len(aa)==len(ff)==300
   hitbase=[];hitaux=[];hitfinal=[];correction=[];improve=[];auxranks=[]
   for br,ar,fr in zip(bs,aa,ff):
    assert br['clip_id']==ar['clip_id']==fr['clip_id'];target=br['label_index']
    bz=np.array(br['logits'],dtype='float32');az=np.array(ar['logits'],dtype='float32');fz=np.array(fr['logits'],dtype='float32')
    assert np.array_equal(bz+az,fz)
    bp=int(bz.argmax());ap=int(az.argmax());fp=int(fz.argmax())
    assert br['correct']==(bp==target) and ar['correct']==(ap==target) and fr['correct']==(fp==target)
    hitbase.append(bp==target);hitaux.append(ap==target);hitfinal.append(fp==target)
    if bp!=target:
     contribution=float(az[target]-az[bp]);gap=float(bz[bp]-bz[target]);rank=int(np.sum(az>az[target]))+1
     correction.append(contribution);improve.append(contribution>0);auxranks.append(rank)
     errcases.append({'seed':seed,'view':v,'clip_id':br['clip_id'],'target':labels[target],
      'base_prediction':labels[bp],'aux_prediction':labels[ap],'final_prediction':labels[fp],
      'aux_target_rank':rank,'old_rival_gap':gap,'aux_target_minus_old_rival':contribution,
      'covers_old_rival_gap':contribution>gap,'new_correct':fp==target})
   hb=np.array(hitbase);ha=np.array(hitaux);hf=np.array(hitfinal)
   dev.append({'seed':seed,'view':v,'n':300,'base_correct':int(hb.sum()),'aux_correct':int(ha.sum()),'final_correct':int(hf.sum()),
    'base_wrong':int((~hb).sum()),'aux_correct_when_base_wrong':int((ha&~hb).sum()),
    'aux_correct_when_base_correct':int((ha&hb).sum()),'final_correct_when_base_wrong':int((hf&~hb).sum()),
    'positive_correction_against_old_rival':sum(improve),'covers_old_rival_gap':sum(e['covers_old_rival_gap'] for e in errcases if e['seed']==seed and e['view']==v),
    'aux_ranks_on_base_errors':auxranks})
 lengthstats=[{'view':v,'min':int(z['lengths'][vi].min()),'median':float(np.median(z['lengths'][vi])),
  'max':int(z['lengths'][vi].max()),'within_61_frames':int((z['lengths'][vi]<=61).sum())} for vi,v in enumerate(s['views'])]
 write(D/'ANALYSIS.json',{'status':'ANALYZED_SAVED_DEV_AND_FIXED_TRAIN_FEATURE_DIAGNOSIS','source_scope_sha256':sha(D/'SCOPE.json'),
 'probe_rows':rows,'per_word':byword,'original_train':source,'saved_DEV':dev,'base_errors':errcases,
 'history':hist,'sequence_lengths':lengthstats,'receptive_field_input_frames_max':61,
 'limits':['No actual stage labels or sequence-completion ground truth.','Internal features already include temporal context.','Training labels retained under perturbations are not recognition accuracy.','Same DEV is reused and near ceiling.','No inferred numerical expected accuracy gain.']})
 write(D/'INDEPENDENT.json',{'status':'PASS','reference':'Separate NumPy GRU and saved-score arithmetic by same Codex, not a separate agent/expert.',
 'sampled_GRU_outputs_checked':perm_checks,'max_logit_absolute_error':max(errors),'samples':len(s['sample_indices']),
 'saved_DEV_component_triples':1800,'original_GRU_exact':c['original_feature_pass_exact_error']==0,
 'protected_files_unchanged':len(s['pins']),'diagnostic_code_sha256':sha(D/'diagnose.py'),'verifier_code_sha256':sha(Path(__file__)),
 'analysis_sha256':sha(D/'ANALYSIS.json'),'training_runs':0,'DEV_new_inference':0,'user_recordings_read':0})
 print(json.dumps({'independent_max_error':max(errors),'source_train':source,'dev':dev,'sensor_probes':[r for r in rows if r['view']=='sensor'],'lengths':lengthstats},ensure_ascii=False,indent=2))
if __name__=='__main__':
 try:main()
 except BaseException:
  p=D/'FIRST_VERIFY_ERROR.json'
  if not p.exists():write(p,{'traceback':traceback.format_exc()})
  raise
