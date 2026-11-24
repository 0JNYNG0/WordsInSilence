"""Public cached-feature mechanism diagnostic; no private recordings or new inference."""
from core import *
spec=importlib.util.spec_from_file_location('_tail_alignment',O/'word100_learned_guide_feedback_20261102/alignment.py');align=importlib.util.module_from_spec(spec);spec.loader.exec_module(align)

def mapping(q,p):
 d=align.dtw(q,p);path=d['path'];m=np.array([round(float(path[path[:,0]==i,1].mean())) for i in range(len(q))])
 assert np.all(np.diff(m)>=0)
 return m,d['mean_cost']

def main():
 assert not (U/'TAIL_ALIGNMENT_CHECK.json').exists()
 prepared=read(D/'PREPARED.json');assert Path(prepared['baseline_checkpoint'])==BASE
 weights=torch.load(BASE,map_location='cpu',weights_only=True);prototype=weights['prototype'].numpy();rows=[]
 for r in prepared['rows']:
  if r['role']!='development':continue
  assert sha(r['path'])==r['sha256'];z=np.load(r['path'],allow_pickle=False);q=z['h_full'];ts=z['timestamp_us']/1e6;intervals=r['intervals'];main_ix=np.flatnonzero(np.any(np.array([(ts>=a)&(ts<=b) for a,b in intervals]),axis=0));assert len(main_ix)>8
  p=prototype[r['word'],:STATS['prototype_lengths'][r['word']]];base,cost=mapping(q,p)
  # These feature-tail controls model a late visible action, not an actual mouse gesture.
  rest=np.repeat(q[-1:],15,axis=0);late=q[main_ix[-8:]]
  for name,tail in [('rest_only',rest),('late_action',np.concatenate([rest,late,rest]))]:
   m,c=mapping(np.concatenate([q,tail]),p);d=np.abs(m[:len(q)]-base);rows.append(dict(clip_id=r['clip_id'],variant=name,original_frames=len(q),tail_frames=len(tail),main_points=len(main_ix),main_mapping_changed=int((d[main_ix]>0).sum()),main_shift_max_frames=int(d[main_ix].max()),all_original_shift_max_frames=int(d.max()),tail_repeated_reference_steps=int((np.diff(m[len(q):])==0).sum()),tail_total_steps=len(tail)-1,mean_cost=c,baseline_cost=cost,backward_steps=int((np.diff(m)<0).sum()),source=str(r['path']),source_sha256=r['sha256']))
 write(U/'TAIL_ALIGNMENT_CHECK.json',dict(scope='12 public historicalDEV cached latents and frozen prototype; synthetic extra-tail controls only. Not a user-recording reproduction or a semantic error detector.',rows=rows,normal_data_untouched=True,model_inference_calls=0,algorithm_changed=False,conclusion='Global endpoint DTW covers appended samples too. Zero/large forward reference steps are possible; backward reference indices are not produced by this mapping. Action-aware trimming needs independently supported boundaries and must preserve missing-main-action warnings.'))
 print(json.dumps({n:dict(clips_with_main_change=sum(r['main_mapping_changed']>0 for r in rows if r['variant']==n),max_main_shift=max(r['main_shift_max_frames'] for r in rows if r['variant']==n)) for n in ['rest_only','late_action']}))
if __name__=='__main__':main()
