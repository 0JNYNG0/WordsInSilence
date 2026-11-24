"""Independent saved-data arithmetic only; no torch modules/forward/training."""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
import ast, json, hashlib, sys, importlib.util
import numpy as np
import torch
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parents[1];O=U.parent;W=U/'local_motion_quality'
R=O/'word100_candidate_role_learning_20261111';S=O/'word100_shared_motion_learning_20261109';J=O/'word100_episode_joint_linkage_20261111'
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
geom=load('_previous_arithmetic_geometry',U/'reviews/verify_boundary_probe.py')

def assign(a,b,axis,side,annotations):
 if b<=a:return -1,[],'single_observation_no_temporal_extent'
 rows=[r for r in annotations if r['axis']==axis and r['moving']==side]
 if not rows:return -1,[],'axis_or_side_unannotated'
 es=[(e,f"{r['clip_id']}|axis{axis}|{side}|event{i}") for r in rows for i,e in enumerate(r['events'])]
 non=[v for r in rows for v in r['holds']+r['background']]
 full=[ident for e,ident in es if e['onset'][0]<=a<=e['onset'][1] and e['offset'][0]<=b<=e['offset'][1]]
 inside=[ident for e,ident in es if e['onset'][0]<=a<=b<=e['offset'][1]]
 contained=[ident for e,ident in es if a<=e['onset'][0] and b>=e['offset'][1]]
 hold=any(lo<=a<=b<=hi for lo,hi in non)
 if hold and (full or inside):return -1,sorted(set(full+inside)),'annotation_conflict'
 if hold:return 0,[],'reviewed_hold_or_background'
 if len(full)==1:return 2,full,'reviewed_endpoint_ranges'
 if len(full)>1:return -1,full,'overlapping_complete_annotations'
 if len(inside)==1:return 1,inside,'within_one_reviewed_event'
 if len(inside)>1:return -1,inside,'overlapping_event_annotations'
 if len(set(contained))>=2:return 3,sorted(set(contained)),'multiple_reviewed_event_identities'
 if contained and any(max(a,lo)<=min(b,hi) for lo,hi in non):return -1,contained,'movement_plus_nonmotion_mix_not_new_class3_gold'
 return -1,contained,'unreviewed_or_uncertain_extent'

def pack_sequence(values,good,ts,a,b):
 v=values[a:b+1].astype(np.float32);g=good[a:b+1].copy()&np.isfinite(v).all(1);t=ts[a:b+1]
 v[~g]=0;delta=np.zeros_like(v);pair=np.zeros(len(g),bool);pair[1:]=g[1:]&g[:-1]
 delta[1:]=np.where(pair[1:,None],v[1:]-v[:-1],0)
 dt=np.zeros(len(g),np.float32);dt[1:]=np.diff(t).astype(float)/1e6
 return np.c_[v,delta,g,pair,dt].astype(np.float32)

def summary(z,a,b,k):
 h=z[a:b+1,228:292];q=z[a:b+1,292:392].reshape(-1,10,10)[:,k];f=z[a:b+1,392:456]
 return np.r_[h.mean(0),h[0],h[-1],q.mean(0),q[0],q[-1],q[-1]-q[0],f.mean(0),np.eye(5)[k//2],np.eye(2)[k%2],(b-a+1)/100.].astype(np.float32)

def main():
 cfg=read(W/'FROZEN.json');manifest=read(W/'INPUTS_COMPLETE.json');unknown=read(W/'UNKNOWN_CANDIDATES.json')['rows'];ia=read(W/'SOURCE_INPUT_AUDIT.json')
 for p,h in cfg['pins'].items():assert sha(p)==h,p
 assert cfg['conditions']==['B','C'] and cfg['seeds']==[42,43,44] and cfg['epochs']==40
 assert cfg['aux_enabled'] is False and cfg['aux_admissions']==0 and manifest['aux_enabled'] is False and manifest['aux_positive']==0
 fit=[r for r in read(S/'READY.json')['rows'] if r['role']=='fit'];annotations=read(O/'word100_autonomous_localization_20261111/FIT_LABELS.json')['rows']
 wanted={a['clip_id'] for a in annotations};sources=[r for r in fit if r['clip_id'] in wanted];assert len(sources)==30
 caches={(c['clip_id'],c['view']):c for c in cfg['source_context_caches']};assert len(caches)==60
 normalization=O/'word100_rtmw_joint_learning_20261108/feedback_viewfix/FEEDBACK_NORMALIZATION.json';stats=read(normalization)
 ck=torch.load(J/'runs/42/model.pt',map_location='cpu',weights_only=True);qm=ck['qm'].numpy();qs=ck['qs'].numpy()
 cache_arrays={};maxq=0.;source_checks=[]
 for r in sources:
  assert sha(r['path'])==r['sha256'];aa=[a for a in annotations if a['clip_id']==r['clip_id']]
  assert all(a['source_sha256']==r['source_sha256'] for a in aa)
  with np.load(r['path'],allow_pickle=False) as z:
   ids=z['frame_id'].copy();ts=z['timestamp_us'].copy();assert np.array_equal(ids,np.arange(len(ids))) and np.all(np.diff(ts)>0)
   for view in ['full','upper']:
    x=z['x_'+view].copy();x[:,:186]=x[:,:186]*np.asarray(stats['std'])+np.asarray(stats['mean']);x=x.astype(np.float32)
    cm=caches[r['clip_id'],view];assert sha(cm['path'])==cm['sha256']
    with np.load(cm['path'],allow_pickle=False) as c:latent=c['z'].copy()
    assert latent.shape==(len(x),684)
    q=geom.relations(x);qn=(q-qm)/qs;qn=np.concatenate([qn[:,:,:4]*q[:,:,8,None],qn[:,:,4:8]*q[:,:,9,None],q[:,:,8:]],-1)
    err=float(np.max(np.abs(qn.reshape(len(x),100)-latent[:,292:392])));maxq=max(maxq,err);assert err<1e-6
    relation_values={}
    for a in aa:
     key=a['axis'],a['moving']
     if key not in relation_values:relation_values[key]=geom.relation(x,*key)
    cache_arrays[r['clip_id'],view]=dict(z=latent,x=x,ts=ts,series=relation_values,annotations=aa,row=r)
    source_checks.append(dict(clip_id=r['clip_id'],view=view,frames=len(x),source_sha256=r['source_sha256'],cache_sha256=cm['sha256'],qnorm_max_error=err))
 accepted_keys=set();unknown_keys=set();counts=Counter();seen_axes=Counter();maxsummary=0.;maxsequence=0.;samples=[]
 for r in manifest['rows']:
  cid,view=r['clip_id'],r['view'];data=cache_arrays[cid,view];a,b,axis,side=r['a'],r['b'],r['axis'],r['moving'];key=(cid,view,a,b,axis,side)
  assert key not in accepted_keys;accepted_keys.add(key)
  y,eids,reason=assign(a,b,axis,side,data['annotations'])
  assert y==r['extent_target']>=0 and eids==r['episode_ids'] and reason==r['label_reason']
  assert r['source_sha256']==data['row']['source_sha256'] and r['person']==data['row']['person']
  assert r['split']==cfg['split'][cid]==('head_probe' if r['person']==7 else 'train')
  assert r['kind']=='source_original' and r['extent_mask']==1 and r['aux_mask']==r['aux_target']==0
  v,g=data['series'][axis,side];assert g[a:b+1].all()
  expected_seq=pack_sequence(v,g,data['ts'],a,b);expected_summary=summary(data['z'],a,b,axis*2+(side=='L'))
  assert sha(r['path'])==r['sha256']
  with np.load(r['path'],allow_pickle=False) as z:
   assert set(z.files)=={'summary','sequence','timestamps_us','extent_target','extent_mask','aux_target','aux_mask'}
   for n in ['extent_target','extent_mask','aux_target','aux_mask']:assert z[n].item()==r[n]
   assert np.array_equal(z['timestamps_us'],data['ts'][a:b+1])
   se=float(np.max(np.abs(z['summary']-expected_summary)));qe=float(np.max(np.abs(z['sequence']-expected_seq)));maxsummary=max(maxsummary,se);maxsequence=max(maxsequence,qe)
   assert np.allclose(z['summary'],expected_summary,atol=3e-6,rtol=1e-6),(r['id'],se)
   assert np.array_equal(z['sequence'],expected_seq),(r['id'],qe)
   assert z['summary'].shape==(304,) and z['sequence'].shape==(b-a+1,11) and np.isfinite(z['summary']).all() and np.isfinite(z['sequence']).all()
  counts[f"{r['split']}/{y}"]+=1;seen_axes[axis,side]+=1
  samples.append(dict(id=r['id'],summary_error=se,sequence_error=qe,source_same=True,targets_same=True))
 unknown_reasons=Counter();missing_target=Counter();missing_by_view=Counter()
 for r in unknown:
  cid,view=r['clip_id'],r['view'];data=cache_arrays[cid,view];a,b=r['span'];axis,side=r['axis'],r['moving'];key=(cid,view,a,b,axis,side)
  assert key not in unknown_keys and key not in accepted_keys;unknown_keys.add(key)
  y,eids,reason=assign(a,b,axis,side,data['annotations'])
  if r['reason']=='extent_observation_incomplete':
   assert y>=0 and r['intended_extent_target']==y and r['episode_ids']==eids and r['interval_label_reason']==reason
   _,g=data['series'][axis,side];assert not g[a:b+1].all()
   assert r['observed_frames']==int(g[a:b+1].sum()) and r['total_frames']==b-a+1
   missing_target[y]+=1;missing_by_view[view]+=1
  else:assert y==-1 and r['reason']==reason,(r,reason)
  unknown_reasons[r['reason']]+=1
 assert len(accepted_keys)==853 and len(unknown_keys)==1746 and dict(counts)==manifest['extent_counts']
 dispositions=Counter((k[0],k[1]) for k in accepted_keys|unknown_keys)
 assert len(ia['rows'])==60 and ia['aux']==[]
 for row in ia['rows']:assert dispositions[row['clip_id'],row['view']]==row['unique_candidates']
 assert sum(dispositions.values())==2599
 splitsha={s:{r['source_sha256'] for r in manifest['rows'] if r['split']==s} for s in ['train','head_probe']}
 assert not splitsha['train']&splitsha['head_probe']
 amendment=read(W/'FROZEN_IMPORT_AMENDMENT.json');initial=read(W/'FROZEN_INITIAL_IMPORT_FAILURE.json')
 assert sha(W/'FROZEN_INITIAL_IMPORT_FAILURE.json')==amendment['initial_frozen_sha256'] and sha(W/'FROZEN.json')==amendment['amended_frozen_sha256']
 assert {k:v for k,v in initial.items() if k!='pins'}=={k:v for k,v in cfg.items() if k!='pins'}
 assert sha(W/'quality_model.py')==sha(W/'initial_import_failure_code/model.py')
 for c in amendment['changes']:
  assert sha(c['snapshot'])==c['before_sha256']
  assert sha(c.get('path',c.get('after_path')))==c['after_sha256']
 traincode=(W/'train.py').read_text('utf-8');assert "['INPUTS_COMPLETE.json','FROZEN.json']" in traincode and "admission['pins'][name]" in traincode
 result=dict(status='PASS_GENERATED_INPUTS_ELIGIBLE_FOR_MATCHED_CONTINUATION_NOT_PERFORMANCE_GAIN',timestamp=datetime.now().astimezone().isoformat(),
 reviewer='Separate Codex pure source/hash/numpy audit, not independent human annotations',
 counts=dict(source_clips=30,source_contexts=60,accepted_samples=853,train_samples=sum(v for k,v in counts.items() if k.startswith('train/')),head_probe_samples=sum(v for k,v in counts.items() if k.startswith('head_probe/')),unknown=1746,unique_candidate_denominator=2599,extent_counts=dict(counts),unknown_reasons=dict(unknown_reasons),missing_exclusions_by_intended_label=dict(missing_target),missing_exclusions_by_view=dict(missing_by_view),aux_samples=0,source_sha_train_groups=len(splitsha['train']),source_sha_probe_groups=len(splitsha['head_probe']),axes=[dict(axis=k[0],side=k[1],samples=v) for k,v in sorted(seen_axes.items())]),
 verification=dict(all_hashes_match=True,source_frame_ids_exact=True,qnorm_max_abs_error=maxq,summary_max_abs_error=maxsummary,sequence_max_abs_error=maxsequence,all_853_target_and_mask_fields_match_manifest=True,all_1746_dispositions_reproduced=True,all_60_candidate_denominators_balance=True,source_sha_split_disjoint=True,auxiliary_all_masks_zero=True,import_only_amendment_labels_settings_unchanged=True,training_admission_hash_checks_present=True),
 training_admission_recommendation='Root may approve the fixed B/C×42/43/44 continuation after pinning these exact INPUTS_COMPLETE/FROZEN digests. No claim of improved feedback or fresh-person accuracy.',
 admission_pins={n:sha(W/n) for n in ['FROZEN.json','INPUTS_COMPLETE.json']},
 limitations=['REAL07 excludes this continuation only; original encoder, localizer and copied quality MLP already saw these records.','All admitted candidates have complete body-relation observation; this does not test recovery of missing joints.','All movement annotation subjects are R. Code supports L but no L supervision is supplied.','Class3 excludes movement+terminal hold, unlike old expanded adapter semantics.','Endpoint-range extent supervision is interval-supported; no new direct RGB progress witnesses were created.','No boundary or quality neural outputs were rerun; proposed intervals originate the root saved preparation.'],
 source_checks=source_checks,sample_arithmetic=samples,
 scope=dict(model_forward=0,encoder_inference=0,training=0,raw_video=0,private=0,development_arrays=0,checkpoint_state_read_for_qnorm_only=True),
 pins={str(p):sha(p) for p in [W/'FROZEN.json',W/'INPUTS_COMPLETE.json',W/'UNKNOWN_CANDIDATES.json',W/'SOURCE_INPUT_AUDIT.json',W/'FROZEN_IMPORT_AMENDMENT.json',normalization,Path(__file__)]})
 (U/'reviews/local_quality_input_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps({k:result[k] for k in ['status','counts','verification','admission_pins']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
