"""Planning-only public TRAIN timing review and synthetic pooling algebra.
No torch model construction/forward, optimizer, or private/user recording access.
"""
from pathlib import Path
import json,hashlib,datetime,shutil,ast
from collections import Counter
import numpy as np
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2];BASE=O/'word100_body_boundary_20261017'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,x):
 p=U/name;assert not p.exists(),str(p);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()

def weights(ts):
 n=len(ts)
 u=(ts-ts[0])/(ts[-1]-ts[0]) if n>1 and ts[-1]>ts[0] else np.full(n,.5)
 return np.stack([np.maximum(1-2*u,0),1-np.abs(2*u-1),np.maximum(2*u-1,0)],1)

def summarize(h,ts,pad=None):
 keep=np.ones(len(h),bool) if pad is None else np.asarray(pad,bool)
 h=np.asarray(h,float)[keep];ts=np.asarray(ts,float)[keep]
 if not len(h):return np.zeros(np.asarray(h).shape[-1]),np.zeros((3,np.asarray(h).shape[-1]))
 g=h.mean(0);w=weights(ts);mass=w.sum(0)
 parts=np.stack([(w[:,k,None]*h).sum(0)/mass[k] if mass[k]>0 else g for k in range(3)])
 return g,parts-g

def main():
 assert not (U/'ANALYSIS.json').exists()
 for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:
  (U/'before').mkdir(exist_ok=True);assert not (U/'before'/p.name).exists();shutil.copyfile(p,U/'before'/p.name)
 save('SCOPE.json',{'at':now(),'request':'균등 배치 보류. 반복 혼동의 단계·순서 보완 방법 검토 및 구체 계획. 기존 C 유지, 사용자촬영 학습/튜닝 금지, 추가학습 전 보고.','allowed':['current code and stored public diagnosis review','public TRAIN timing coverage/count audit','synthetic mathematical pooling counterexamples','proposed protocol/report/status docs'],'training_authorized':False,'new_model_inference_authorized':False,'production_changes':False,'user_mentor_access':False})
 c=read(BASE/'training_CONTEXT.json');tr=[r for r in c['ordered_rows'] if r['role']=='train_candidate'];assert len(tr)==899 and set(r['signer_id'] for r in tr)==set(range(1,10))
 pins={str(BASE/'training_CONTEXT.json'):sha(BASE/'training_CONTEXT.json')}
 for p,h in read(O/'word100_balanced_batch_20261019/FINAL_AUDIT.json')['original_checkpoint_pins'].items():assert sha(p)==h;pins[p]=h
 for n in ['VISUAL_REVIEW.json','SAVED_ANALYSIS.json','NEXT_PLAN.json']:
  p=O/'word100_confusion_late_training_diagnosis_20261019'/n;pins[str(p)]=sha(p)
 model=ROOT/'research/sign_model_pilot_v1/src/pilot/model_masked_v1.py';pins[str(model)]=sha(model)
 assert sha(model)==c['pins'][str(model)]
 groups={};rows=[]
 for r in tr:
  for view in ['masked','sensor']:
   info=r['views'][view];p=info['channel_file'];assert Path(p).resolve().is_relative_to((BASE/'data/clips').resolve())
   assert sha(p)==info['sha256'];pins[p]=info['sha256']
   with np.load(p,allow_pickle=False) as z:
    for name,tk in [('x_whole','whole_ts'),('x_context','context_ts')]:
     x=z[name];ts=z[tk].astype(np.float64);assert len(x)==len(ts)>0
     if len(ts)>1:assert (np.diff(ts)>0).all()
     w=weights(ts);assert np.allclose(w.sum(1),1,atol=1e-12) and np.all(w>=0)
     # Existing dt channel is sufficient; no new observation channel required.
     recon=np.cumsum(x[:,-1].astype(np.float64));error=float(np.max(np.abs(weights(recon)-w)))
     assert error<1e-5
     active=x[:,186+9:186+51].sum(1)>0
     mass=w.sum(0);coverage=np.divide((w*active[:,None]).sum(0),mass,out=np.zeros(3),where=mass>0)
     row={'clip_id':r['clip_id'],'view':view,'temporal':name,'n':len(ts),'seconds':float((ts[-1]-ts[0])/1e6),'effective_phase_frames':mass.tolist(),'phase_with_any_hand_fraction':coverage.tolist(),'dt_reconstruction_max_weight_error':error}
     rows.append(row);groups.setdefault(view+'/'+name,[]).append(row)
 def stats(a):return {'min':float(np.min(a)),'median':float(np.median(a)),'p95':float(np.percentile(a,95)),'max':float(np.max(a))}
 summary={}
 for key,rr in groups.items():
  summary[key]={'sequences':len(rr),'length_frames':stats([r['n'] for r in rr]),'duration_seconds':stats([r['seconds'] for r in rr]),'longer_than_nominal_conv_RF61':sum(r['n']>61 for r in rr),'shorter_than3':sum(r['n']<3 for r in rr),'empty_phase_mass':sum(v<=0 for r in rr for v in r['effective_phase_frames']),'no_observed_hand_phases':sum(v==0 for r in rr for v in r['phase_with_any_hand_fraction']),'max_dt_reconstruction_error':max(r['dt_reconstruction_max_weight_error'] for r in rr)}
 tests={};t=np.arange(9,dtype=float);h=np.eye(3)[np.repeat(np.arange(3),3)]
 g,d=summarize(h,t);gr,dr=summarize(h[::-1],t)
 tests['A_B_C_vs_C_B_A_same_global']=bool(np.allclose(g,gr))
 tests['A_B_C_vs_C_B_A_different_ordered_parts']=bool(not np.allclose(d,dr))
 gp,dp=summarize(np.vstack([h,np.full((5,3),999)]),np.r_[t,np.zeros(5)],np.r_[np.ones(9),np.zeros(5)])
 tests['padding_values_ignored']=bool(np.allclose(g,gp) and np.allclose(d,dp))
 tests['uniform_time_scale_pooling_only']=bool(np.allclose(summarize(h,7*t)[1],d))
 tests['translation_of_timestamps']=bool(np.allclose(summarize(h,t+100)[1],d))
 tests['constant_features_no_phase_difference']=bool(np.allclose(summarize(np.ones((9,3)),t)[1],0))
 tests['all_padding_finite']=bool(all(np.isfinite(a).all() for a in summarize(h,t,np.zeros(9))))
 tests['one_two_frame_finite']=all(all(np.isfinite(a).all() for a in summarize(h[:n],t[:n])) for n in [1,2])
 missing=np.array([1,0,0,1],bool);wm=weights(np.arange(4));tests['missing_real_frames_keep_time_position']=bool(np.allclose(wm.sum(1),1) and len(wm)==len(missing))
 assert all(tests.values())
 # Hard split discontinuity vs continuous soft weights at a third-boundary.
 ua=np.array([0,1/3-1e-5,1]);ub=np.array([0,1/3+1e-5,1])
 hard=lambda x:np.eye(3)[np.minimum((x*3).astype(int),2)]
 sensitivity={'hard_boundary_weight_L1':float(abs(hard(ua)[1]-hard(ub)[1]).sum()),'soft_weight_L1':float(abs(weights(ua)[1]-weights(ub)[1]).sum()),'scope':'Mathematical weight continuity only, not trained model robustness.'}
 # Parameter arithmetic from actual state-independent layer configuration.
 width=c['model']['width'];classes=c['model']['n_classes'];original=37636;extra=3*width*classes
 out={'at':now(),'scope':'public TRAIN899 only, original C bytes; no new DEV statistics or scores','public_sequences':len(rows),'public_channel_files':1798,'summary':summary,'model_facts':{'width':width,'nominal_temporal_conv_receptive_field_frames':1+2*(c['model']['kernel_size']-1)*sum(c['model']['dilations']),'pooling':'valid-frame global mean before Linear(32,100)','already_order_sensitive_TCN':True,'training_BN_time_coupling_outside_nominal_conv_RF':True,'original_parameters':original,'proposed_extra_zero_bias_linear_weights':extra,'proposed_parameters':original+extra,'increase_percent':100*extra/original},'synthetic_tests':tests,'boundary_sensitivity':sensitivity,'synthetic_example':{'global':g.tolist(),'forward_order_contrasts':d.tolist(),'reverse_order_contrasts':dr.tolist()},'pins':pins,'training':0,'model_inference':0,'new_video_extraction':0,'user_mentor_data_used':False,'production_changed':False,'limits':['Synthetic pooling checks do not demonstrate TCN whole-model accuracy or speed invariance.','Early/middle/late are relative time locations, not language-labeled action phases.','A phase with no observed hand must not be treated as an observed resting hand or silently removed.']}
 save('ANALYSIS.json',out);save('TRAIN_TIMING_ROWS.json',rows)
 print(json.dumps({k:out[k] for k in ['summary','model_facts','synthetic_tests','boundary_sensitivity']},ensure_ascii=False))
if __name__=='__main__':main()
