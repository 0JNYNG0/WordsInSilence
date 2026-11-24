from core import *
import itertools,collections,time,traceback

def main():
 assert not (W/'INPUTS_COMPLETE.json').exists();torch.set_num_threads(4);torch.set_num_interop_threads(1)
 sources=read(U/'episode_membership/coverage_review/SOURCE_EPISODES_DENSE.json')['rows'];contrasts={r['clip_id']:r for r in read(U/'episode_stage_contrast/RGB_CONTRAST_LABELS.json')['rows']};assert len(sources)==24
 supplement=read(W/'short_action_review/REVIEW.json');assert supplement['status']=='REVIEWED_SOURCE_PROGRESS_ONLY'
 supplement['rows']+=read(W/'short_action_review/FOREARM_REVIEW.json')['rows']
 for s in sources:
  for r in supplement['rows']:
   if r['clip_id']==s['clip_id']:
    assert r['source_sha256']==s['source_sha256'];e=next(e for e in s['events'] if e['event_id']==r['original_event']['event_id']);e['progression_witness_frames']=sorted(set(e['progression_witness_frames']+r['additional_progression_frames']))
 need={r['clip_id'] for r in sources};raw_sources={r['clip_id']:r for r in read(O/'word100_shared_motion_learning_20261109/READY.json')['rows'] if r['role']=='fit' and r['clip_id'] in need}
 cache={}
 for r in read(U/'context_role_training/INPUTS_COMPLETE.json')['rows']:
  if r['clip_id'] in need:
   raw=raw_sources[r['clip_id']];assert raw['source_sha256']==r['source_sha256'];cache[r['clip_id'],r['view']]=dict(r,raw=raw['path'],raw_sha256=raw['sha256'])
 assert len(cache)==48
 role=load('_cand_desc_role',O/'word100_candidate_role_learning_20261111/role_core.py');net=role.frozen_model();out=[];xx=[];yy=[];mm=[];proposals=[]
 for s in sources:
  cid=s['clip_id'];co=contrasts[cid];assert co['source_sha256']==s['source_sha256'];groups,background=source_witnesses(s,co);split='probe' if s['person'] in [8,9] else 'train'
  teacher=set()
  for fam,g in groups.items():
   pts=g['motion'];assert len(pts)>=2
   # Cover whole observed occurrence, partial occurrence, and reviewed contrasts.
   teacher.add((min(pts),max(pts)))
   if g['held']:teacher.add((min(pts),max(g['held'])))
   for a,b in itertools.combinations(pts,2):teacher.add((a,b))
   for v in groups.values():
    if v is not g:teacher.add((min(pts+v['motion']),max(pts+v['motion'])))
   for neg in [co['entry'],co['other_held_presentation'],co['returning']]:
    if len(neg)>=2:
     teacher.add((min(neg),max(neg)));teacher.add((min(pts+neg),max(pts+neg)))
  for view in ['full','upper']:
   c=cache[cid,view];assert sha(c['path'])==c['sha256'] and sha(c['raw'])==c['raw_sha256'];z=np.load(c['path'])['z'];raw=np.load(c['raw']);fi=raw['frame_id'];assert np.array_equal(fi,np.arange(len(fi)));T=len(z);qn=z[:,292:392].reshape(T,10,10);h=torch.tensor(z[:,228:292][None]);fh=torch.tensor(z[:,392:456][None]);qt=torch.tensor(qn[None]);ids=torch.arange(10);query=torch.cat([torch.nn.functional.one_hot(ids//2,5),torch.nn.functional.one_hot(ids%2,2)],-1).float()[None,None].expand(1,T,-1,-1)
   # candidate_spans uses observed flags and boundary outputs; normalized q retains flags.
   with torch.inference_mode():bp=net.boundary(torch.cat([h[:,:,None].expand(-1,-1,10,-1),qt,fh[:,:,None].expand(-1,-1,10,-1),query],-1)).sigmoid()[0].numpy()
   candidate=role.jc.candidate_spans(bp,qn,[]);proposals.append(dict(clip_id=cid,view=view,candidates=candidate));source_spans=set(teacher)|{(c0['a'],c0['b']) for c0 in candidate if c0['moving']=='R' and c0['axis'] in [AX[f] for f in groups] and c0['a']<c0['b']}
   for a,b in sorted(source_spans):
    y=np.zeros((5,3),np.float32);mask=np.zeros_like(y);basis={}
    for family in groups:
     y0,m0,why=labels(a,b,family,groups,background,qn[:,AX[family]*2,8]>.5);j=F.index(family);y[j]=y0;mask[j]=m0;basis[str(family)]=why
    if not mask.any():continue
    xx.append(features(z,a,b));yy.append(y);mm.append(mask);out.append(dict(clip_id=cid,person=s['person'],word_id=s['word_id'],source_sha256=s['source_sha256'],view=view,split=split,span=[a,b],origin='teacher_and_model' if (a,b) in teacher and any(c0['a']==a and c0['b']==b for c0 in candidate) else 'teacher' if (a,b) in teacher else 'model',basis=basis,cache=c['path'],cache_sha256=c['sha256'],raw=c['raw'],raw_sha256=c['raw_sha256']))
  print('A source',cid,'windows',sum(r['clip_id']==cid for r in out),flush=True)
 path=W/'inputs.npz';np.savez_compressed(path,x=np.array(xx),y=np.array(yy),mask=np.array(mm),train=np.array([r['split']=='train' for r in out]))
 write(W/'CANDIDATE_PROPOSALS.json',proposals)
 counts={}
 for split in ['train','probe']:
  inds=np.array([r['split']==split for r in out]);y=np.array(yy)[inds];m=np.array(mm)[inds];counts[split]=dict(positive=(y*m).sum(0).astype(int).tolist(),negative=((1-y)*m).sum(0).astype(int).tolist(),windows=int(inds.sum()))
 pins=[Path(__file__),W/'core.py',U/'episode_membership/coverage_review/SOURCE_EPISODES_DENSE.json',U/'episode_stage_contrast/RGB_CONTRAST_LABELS.json',U/'context_role_training/INPUTS_COMPLETE.json',W/'short_action_review/REVIEW.json',W/'short_action_review/FOREARM_REVIEW.json']
 write(W/'INPUTS_COMPLETE.json',dict(at=stamp(),rows=out,arrays=str(path),arrays_sha256=sha(path),counts=counts,source_clips=24,source_person_split=dict(train=[2,4,5,6],probe=[8,9]),upstream_probe_exposed=True,source_truth='Directly reviewed progression/other-action/preparation witnesses; no dense per-frame truth invented. Coverage is coverage of observed source progression, not semantic completion.',heads=['contains_reviewed_target_motion','no_reviewed_conflicting_action_in_window','covers_all_reviewed_source_progression'],unknown_masked=True,pins={str(p):sha(p) for p in pins},training_started=False,private_used=False,web_changed=False,novelty='Old membership task scored presence only on12sources. New24-source multitask directly supervises other-action contamination and observed progression coverage, retaining partial attempts. Existing upstream and stage baseline stay frozen.'))
 print('Prepared',len(out),'windows',counts,flush=True)

if __name__=='__main__':
 try:main()
 except BaseException:(W/f'PREP_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
