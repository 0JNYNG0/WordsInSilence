"""Read-only public T3 diagnosis. No optimization, extraction or score replacement."""
from pathlib import Path
import json,hashlib,sys,datetime,shutil,traceback
from collections import Counter
import numpy as np
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2]
T=O/'word100_temporal_phase_training_20261019';B=O/'word100_body_boundary_20261017'
sys.path[:0]=[str(ROOT/'research/sign_model_pilot_v1/src'),str(T)]
import torch
from temporal import TemporalPhaseTCN,phase_weights,summarize
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
def save(n,x):
 p=U/n;assert not p.exists(),str(p);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def statehash(d):
 h=hashlib.sha256()
 for k,v in sorted(d.items()):h.update(k.encode());h.update(v.cpu().numpy().tobytes())
 return h.hexdigest()
def gaps(ts,seen,frame_ids):
 out=[];ix=np.flatnonzero(seen)
 for a,b in zip(ix[:-1],ix[1:]):
  if b>a+1:out.append({'before':int(frame_ids[a]),'after':int(frame_ids[b]),'missing_start':int(frame_ids[a+1]),'missing_end':int(frame_ids[b-1]),'missing_frames':int(b-a-1),'before_s':float(ts[a]/1e6),'after_s':float(ts[b]/1e6),'duration_s':float((ts[b]-ts[a])/1e6-float(np.median(np.diff(ts)))/1e6)})
 return out
def input_info(row,view,meta,pins):
 recp=B/'data/records'/f"{row['clip_id']}.json";rec=read(recp);pins[str(recp)]=sha(recp)
 p=Path(row['views'][view]['channel_file']);assert sha(p)==row['views'][view]['sha256'];pins[str(p)]=sha(p);z=load(p)
 op=Path(rec['views'][view]['observation']);assert sha(op)==rec['outputs'][str(op)];pins[str(op)]=sha(op);obs=load(op)
 ts=z['whole_ts'].astype(float);ids=z['whole_frame_id'];x=z['x_whole'];w=phase_weights(torch.from_numpy(x)[None],torch.ones(1,len(x)))[0].numpy()
 hand=x[:,195:237].reshape(-1,2,21)>0;seen=hand.any(2)
 fullseen=np.stack([obs['usable'][:,33:54].any(1),obs['usable'][:,54:75].any(1)],1)
 info={'clip_id':row['clip_id'],'label':meta['label'],'signer':row['signer_id'],'role':row['role'],'view':view,'frames':len(x),'frame_range':[int(ids[0]),int(ids[-1])],'seconds_range':(ts[[0,-1]]/1e6).tolist(),'intervals':meta['intervals'],'seen_fraction':seen.mean(0).tolist(),'no_hands_frames':int((~seen.any(1)).sum()),'phase_seen_fraction':[(w[:,k,None]*seen).sum(0).tolist() for k in range(3)],'phase_mass':w.sum(0).tolist(),'input_gaps':{},'full_clip_gaps':{}}
 info['phase_seen_fraction']=[(np.array(a)/mass).tolist() for a,mass in zip(info['phase_seen_fraction'],info['phase_mass'])]
 core=np.logical_or.reduce([(ts/1e6>=a)&(ts/1e6<=b) for a,b in meta['intervals']]);info['annotation_frames_in_input']=int(core.sum());info['annotation_seen_fraction']=seen[core].mean(0).tolist() if core.any() else None
 for side in range(2):
  name=['left','right'][side];info['input_gaps'][name]=gaps(ts,seen[:,side],ids);info['full_clip_gaps'][name]=gaps(obs['timestamp_us'],fullseen[:,side],obs['frame_id'])
 # Smooth relative-time exposure is positional weighting, not a semantic stage label.
 coords=x[:,:102].reshape(-1,51,2);m=x[:,186:237]>0
 motion=[]
 for side,(wrist,start) in enumerate([(9,9),(30,30)]):
  valid=m[1:,wrist]&m[:-1,wrist];d=np.linalg.norm(np.diff(coords[:,wrist],axis=0),axis=1);validd=np.where(valid,d,0)
  u=(ts-ts[0])/(ts[-1]-ts[0]);mass=float(validd.sum())
  motion.append({'side':['left','right'][side],'valid_motion_total':mass,'relative_time_motion_center':float(np.dot(validd,u[1:])/mass) if mass>0 else None,'naive_motion_total_including_missing_jumps':float(d.sum()),'invalid_adjacent_pairs':int((~valid).sum()),'valid_adjacent_pairs':int(valid.sum())})
 info['wrist_motion']=motion
 return info,z

def main():
 assert not (T/'TRAIN_RUNNING.json').exists()
 for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:
  (U/'before').mkdir(exist_ok=True);assert not (U/'before'/p.name).exists();shutil.copyfile(p,U/'before'/p.name)
 c=read(B/'training_CONTEXT.json');rows=c['ordered_rows'];idx={r['clip_id']:r for r in rows};labels=c['labels'];groups=c['missingness_groups']
 prior=read(T/'ALL_CLASS_ANALYSIS.json');comp=read(T/'COMPARISON_INDEPENDENT.json')
 focus=['NIA_SL_WORD1519_REAL12_F','NIA_SL_WORD1675_REAL13_F','NIA_SL_WORD1545_REAL12_F','NIA_SL_WORD2768_REAL12_F','NIA_SL_WORD2370_REAL13_F']
 targetlabels={idx[cid]['label_index'] for cid in focus}
 # Every frozen reentry-group DEV clip, every prior changed DEV clip, and all DEV of the five focus words.
 selected=sorted({cid for v in groups for cid,g in groups[v].items() if g=='internal_reentry'}|{r['clip_id'] for vv in prior['paired_events'].values() for r in vv}|{r['clip_id'] for r in rows if r['role']=='development_only' and r['label_index'] in targetlabels})
 scope={'at':now(),'request':'T3 상처/재등장 회귀 원인 공개자료 분석 및 다음 계획. 추가학습 전 결과검토. 기존C유지/사용자촬영미사용.','new_training_allowed':False,'allowed':['saved score and history analysis','public saved observation/timing audit','fixed T3 eval-mode hidden feature/head decomposition with stored-score binding','local public RGB sampled review','common improvement proposal'],'frozen_diagnostic_selection':selected,'focus_cases':focus,'selection_rule':'all frozen reentry DEV + all existing C/T3 correctness-change DEV + all DEV of five focus labels; no new best-model/weight selection','counterfactual':'global branch alone is an algebraic diagnosis of the already jointly trained T3, NOT the original C or a deployable ablation result. No weight search/crop retiming for better accuracy.','no_user_mentor_access':True,'no_production_change':True,'no_new_MediaPipe':True,'no_completed_score_replacement':True}
 save('SCOPE.json',scope)
 pins={str(T/n):sha(T/n) for n in ['ALL_CLASS_ANALYSIS.json','COMPARISON_INDEPENDENT.json','FROZEN_PROTOCOL.json','temporal.py','FINAL_AUDIT.json']};pins[str(B/'training_CONTEXT.json')]=sha(B/'training_CONTEXT.json')
 protected=read(T/'FINAL_AUDIT.json')['original_checkpoint_pins']
 for p,h in protected.items():assert sha(p)==h;pins[p]=h
 metap=O/'word100_assignment_scale_factorial_20261019/PREP_CONTEXT.json';meta={r['clip_id']:r for r in read(metap)['rows']};pins[str(metap)]=sha(metap)
 needed=sorted(set(selected)|{r['clip_id'] for r in rows if r['role']=='train_candidate' and r['label_index'] in targetlabels})
 inputs={};data={}
 torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 for cid in needed:
  for v in ['masked','sensor']:
   info,z=input_info(idx[cid],v,meta[cid],pins);inputs[cid+'/'+v]=info
   if cid in selected:data[cid,v]=z
 save('INPUT_ANALYSIS.json',{'at':now(),'clips':len(needed),'rows':inputs,'pins':pins,'scope':'saved public observations, no fresh detector','group_definition':'full saved observation has any-point hand visibility, >=0.3s internal gap, with one-hand-never priority; not proof of essential-hand occlusion or model-input gap'})
 decomp=[];maxerror=0.;stored={}
 for seed in [20261013,20261014,20261015]:
  rp=T/f'T3_{seed}_RESULT.json';rpC=B/f'training_C_{seed}_RESULT.json';sr=read(rp);cr=read(rpC)
  for p in [rp,rpC]:pins[str(p)]=sha(p)
  kp=T/f'T3_{seed}_best.pt';assert sha(kp)==sr['checkpoint_sha256'];pins[str(kp)]=sha(kp)
  ck=torch.load(kp,map_location='cpu',weights_only=True);m=TemporalPhaseTCN(**ck['model_config']);m.load_state_dict(ck['state_dict']);m.eval();before=statehash(m.state_dict())
  for v in ['masked','sensor']:
   ss={x['clip_id']:x for x in sr['predictions'][v]};cc={x['clip_id']:x for x in cr['predictions'][v]}
   for cid in selected:
    z=data[cid,v];x=torch.from_numpy(z['x_whole'])[None];pad=torch.ones(1,x.shape[1]);y=idx[cid]['label_index']
    with torch.no_grad():
     h=m.proj(x.transpose(1,2))
     for blk in m.blocks:h=blk(h,pad.unsqueeze(1))
     weights=phase_weights(x,pad);g,d=summarize(h,pad,weights)
     global_logits=m.head(g)[0].numpy();phase_logits=torch.einsum('bpc,jpc->bpj',d.reshape(1,3,32),m.phase_weight.reshape(100,3,32))[0].numpy()
     total=global_logits+phase_logits.sum(0)
    err=float(abs(total-np.array(ss[cid]['logits'])).max());maxerror=max(maxerror,err);assert err<2e-4,(cid,seed,v,err)
    # Rank/accuracy continue to come from saved predictions; decomposition only explains them.
    assert int(np.argmax(total))==ss[cid]['prediction']
    rival=max((j for j in range(100) if j!=y),key=lambda j:(ss[cid]['logits'][j],-j))
    gm=float(global_logits[y]-global_logits[rival]);pm=(phase_logits[:,y]-phase_logits[:,rival]).astype(float)
    # Independent NumPy phase means and dot products, not reuse torch summarize.
    hh=h[0].numpy().T;ww=weights[0].numpy();gg=hh.mean(0);parts=np.array([(hh*ww[:,k,None]).sum(0)/ww[:,k].sum() for k in range(3)])
    weight=m.phase_weight.detach().numpy().reshape(100,3,32);independent=np.einsum('pc,jpc->pj',parts-gg,weight)
    assert np.allclose(phase_logits,independent,atol=3e-5,rtol=3e-5)
    decomp.append({'clip_id':cid,'label':labels[y],'signer':idx[cid]['signer_id'],'view':v,'seed':seed,'group':groups[v][cid],'stored_C_correct':cc[cid]['correct'],'stored_T3_correct':ss[cid]['correct'],'stored_C_predicted':labels[cc[cid]['prediction']],'stored_T3_predicted':labels[ss[cid]['prediction']],'rival':labels[rival],'stored_total_margin':float(ss[cid]['logits'][y]-ss[cid]['logits'][rival]),'T3_global_only_prediction':labels[int(np.argmax(global_logits))],'T3_global_only_correct':int(np.argmax(global_logits))==y,'global_margin':gm,'phase_margin_by_early_middle_late':pm.tolist(),'phase_margin':float(pm.sum()),'reconstruction_error':err})
  assert before==statehash(m.state_dict())
 assert len(decomp)==len(selected)*6
 save('HEAD_DECOMPOSITION.json',{'at':now(),'target_clips':len(selected),'diagnostic_feature_passes':len(decomp),'max_saved_logit_reconstruction_error':maxerror,'rows':decomp,'pins':pins,'torch_numpy_contribution_check':True,'weights_and_BN_unchanged':True,'new_training':0,'no_candidate_selected_from_diagnostic':'No global-only deployment or score tuning. This is T3 co-trained global branch, not original C.'})
 # Attribution to the additional summary versus an error already present in T3's jointly trained global branch.
 attribution={}
 for v in ['sensor','masked']:
  rr=[r for r in decomp if r['view']==v and r['stored_C_correct'] and not r['stored_T3_correct']]
  attribution[v]={'regressions':len(rr),'global_already_wrong':sum(not r['T3_global_only_correct'] for r in rr),'phase_flips_global_correct_to_wrong':sum(r['T3_global_only_correct'] for r in rr),'cases':rr}
 save('SUMMARY.json',{'at':now(),'public_inputs':len(needed),'public_diagnostic_DEV_clips':len(selected),'focus':focus,'attribution':attribution,'original_group_counts':comp['metrics'],'max_reconstruction_error':maxerror,'limits':['Observed group association is not causal proof.','Same input bytes across C/T3: no new detector/crop defect introduced by T3.','Global-only T3 has jointly changed trunk and classifier, so it is not a C ablation.','Relative time position is not semantic phase; no value chosen using DEV or private recordings.']})
 print(json.dumps({'inputs':len(needed),'diagnostic_DEV':len(selected),'max_error':maxerror,'attribution':{v:{k:n for k,n in a.items() if k!='cases'} for v,a in attribution.items()},'focus_decomposition':[r for r in decomp if r['clip_id'] in focus]},ensure_ascii=False))

if __name__=='__main__':
 try:main()
 except BaseException:
  if not (U/'FIRST_ANALYSIS_ERROR.json').exists():save('FIRST_ANALYSIS_ERROR.json',{'at':now(),'traceback':traceback.format_exc()})
  raise
