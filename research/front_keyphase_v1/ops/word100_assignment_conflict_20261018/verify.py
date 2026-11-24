"""Separate scalar checks and regression contracts, no learning/inference."""
import json,sys,math,copy
from pathlib import Path
from collections import Counter
import numpy as np
from run import read,write,load,sha,module,call,pinscheck
U=Path(__file__).resolve().parent
def independent_guard(d):
 out=[]
 if len(d['reasons'])!=2:return out
 for i in (0,1):
  if d['reasons'][i]!='label_without_close_pose':continue
  side=d['assigned_sides'][i];opp=1-side;anchor=1-i
  if d['reasons'][anchor]!='temporal_and_pose' or d['assigned_sides'][anchor]!=opp:continue
  p=d['pose_distance'][i];t=d['track_distance'][i]
  if any(v is None or not math.isfinite(v) for v in p):continue
  if not p[opp]<0.7*p[side]:continue
  if t[opp] is None or t[opp]>0.4:continue
  if t[side] is not None and not t[opp]<0.7*t[side]:continue
  out.append((i,side))
 return out
def contracts(old,new):
 results=[]
 def palm(x,y):
  a=np.zeros((21,2));a[:,0]=x+np.tile([0,8,16],7);a[:,1]=y+np.arange(21)*1.2;a[0]=[x,y];return a
 def fixture(m):
  a=m.HandAssigner();a.tracks[1]=m.Track(palm(420,430)[m.PALM],0.0,np.zeros(2),False,0);a.frame=0;a.last_time=0
  p=np.zeros((33,2));p[11]=[700,150];p[12]=[300,150];p[15]=[800,550];p[16]=[400,400];u=np.zeros(33,bool);u[[11,12,15,16]]=True
  return a,[np.stack([palm(450,550),palm(400,400)]),np.array([0,1]),np.array([.95,.99]),p,u,33333,1200,900]
 def check(name,alter,expectedold,expectednew):
  a,x=fixture(old);b,y=fixture(new);alter(a,x,old);alter(b,y,new)
  aa,d=a.assign(*x);bb,e=b.assign(*y);aa=list(aa.values());bb=list(bb.values());assert aa==expectedold and bb==expectednew,(name,aa,bb)
  assert not e['coordinates_generated'];results.append(name)
 check('contradictory_duplicate_weak_edge_withheld',lambda a,x,m:None,[0,1],[None,1])
 def perm(a,x,m):
  for k in (0,1,2):x[k]=x[k][::-1].copy()
 check('raw_order_permutation',perm,[1,0],[1,None])
 def reflect(a,x,m):
  x[0][:,:,0]=1200-x[0][:,:,0];x[3][:,0]=1200-x[3][:,0]
  for tr in a.tracks.values():tr.palm[:,0]=1200-tr.palm[:,0]
 check('screen_horizontal_reflection_preserves_anatomical_identity',reflect,[0,1],[None,1])
 def own(a,x,m):a.tracks[0]=m.Track(x[0][0,m.PALM].copy(),0.0,np.zeros(2),True,0)
 check('independent_own_track_protected',own,[0,1],[0,1])
 def no_pose(a,x,m):x[4][15]=False
 check('missing_pose_is_not_duplicate_evidence',no_pose,[0,1],[0,1])
 check('missing_anchor_track_does_not_reject_label',lambda a,x,m:a.tracks.clear(),[0,1],[0,1])
 def lone(a,x,m):
  for k in (0,1,2):x[k]=x[k][:1]
 check('single_candidate_label_fallback_retained',lone,[0],[0])
 def ownpose(a,x,m):x[3][15]=x[0][0,0]
 check('current_own_pose_protected',ownpose,[0,1],[0,1])
 def crossing(a,x,m):
  x[0]=np.stack([palm(350,400),palm(650,400)]);x[3][15]=[350,400];x[3][16]=[650,400];x[1]=np.array([1,0])
  a.tracks={s:m.Track(x[0][s,m.PALM].copy(),0,np.zeros(2),True,0) for s in (0,1)}
 check('real_crossing_with_wrong_raw_labels_protected',crossing,[0,1],[0,1])
 def coincident_pose(a,x,m):
  crossing(a,x,m);x[3][15]=[500,400];x[3][16]=[500,400]
 check('crossing_coincident_pose_distinct_tracks_protected',coincident_pose,[0,1],[0,1])
 def expired(a,x,m):x[5]=1000000
 check('expired_track_not_used_as_negative_evidence',expired,[0,1],[0,1])
 def reset(a,x,m):x[5]=0
 check('nonmonotonic_time_resets_identity',reset,[0,1],[0,1])
 def empty(a,x,m):
  for k in (0,1,2):x[k]=x[k][:0]
 check('empty_frame_does_not_create_hands',empty,[],[])
 return results
def main():
 c=read(U/'CONTEXT.json');old=module('verify_old_assignment',c['baseline']);new=module('verify_new_assignment',U/'assignment.py')
 tests=contracts(old,new)
 if '--contracts' in sys.argv:
  write(U/'CONTRACT_TESTS.json',{'status':'PASS','tests':tests});print('Contracts PASS',len(tests));return
 pinscheck(c['pins']);pinscheck(read(U/'OUTPUT_PINS.json'));summary=read(U/'SUMMARY.json');changes=read(U/'CHANGES.json');guards=read(U/'GUARDS.json');count=Counter();elig=[];actualchanges=[]
 for r in c['rows']:
  for view in ('masked','sensor'):
   olda=load(r['views'][view]['assignment']);newa=load(U/'results'/f"{r['clip_id']}_{view}.npz");tr=read(Path(r['views'][view]['assignment']).with_name('trace.json'))
   aa=olda['sides'];bb=newa['sides'];assert aa.shape==bb.shape
   for f,d in enumerate(tr):
    count['frames']+=1
    for i,s in independent_guard(d):elig.append((r['clip_id'],view,f,i,s))
    for i,s in enumerate(d['assigned_sides']):
     count['raw_candidates']+=1
     if aa[f,i]!=bb[f,i]:actualchanges.append((r['clip_id'],view,f,i,int(aa[f,i]),int(bb[f,i])))
     if s is not None and d['reasons'][i]!='label_without_close_pose':
      count['non_label_supported']+=1
      if aa[f,i]==bb[f,i]:count['non_label_preserved']+=1
   count['withheld']+=int(((aa>=0)&(bb<0)).sum());count['newly_assigned']+=int(((aa<0)&(bb>=0)).sum());count['side_flip']+=int(((aa>=0)&(bb>=0)&(aa!=bb)).sum())
 assert sorted(actualchanges)==sorted((x['clip_id'],x['view'],x['index'],x['raw_index'],x['old_side'],x['new_side']) for x in changes)
 guardkeys=[]
 for g in guards:
  d=g['trace'];orig=read(Path(next(r for r in c['rows'] if r['clip_id']==g['clip_id'])['views'][g['view']]['assignment']).with_name('trace.json'))[g['index']]
  for q in d['label_conflicts']:
   assert (q['raw_index'],q['withheld_side']) in independent_guard(orig)
   assert d['assigned_sides'][q['raw_index']] is None
   assert d['assigned_sides'][q['anchor_raw_index']]==q['supported_side']
   guardkeys.append((g['clip_id'],g['view'],g['index'],q['raw_index'],q['withheld_side']))
 assert sorted(guardkeys)==sorted(elig),'unexpected guard or skipped eligible baseline event; requires history audit'
 # Full replay on changed clips + geometry-selected crossing examples under raw
 # candidate permutation; no anatomical slot should depend on raw-array order.
 selected=set(x['clip_id'] for x in changes)|set(x['clip_id'] for x in read(U/'CROSSING_SELECTION.json'))
 for r in c['rows']:
  if r['clip_id'] not in selected:continue
  for view in ('masked','sensor'):
   raw=load(r['views'][view]['raw']);saved=load(U/'results'/f"{r['clip_id']}_{view}.npz");a=new.HandAssigner();b=new.HandAssigner();cut=len(raw['raw_count'])//2
   for f in range(len(raw['raw_count'])):
    n=int(raw['raw_count'][f]);x=raw['hand_xy_px'][f,:n];lab=raw['raw_label'][f,:n];sc=raw['raw_score'][f,:n]
    out,_=a.assign(x[::-1],lab[::-1],sc[::-1],raw['pose_xy_px'][f],raw['pose_usable'][f],int(raw['timestamp_us'][f]),int(raw['width'][f]),int(raw['height'][f]))
    reordered=[-1 if out[i] is None else out[i] for i in range(n)][::-1]
    assert reordered==saved['sides'][f,:n].tolist(),('order dependence',r['clip_id'],view,f)
    count['permutation_frames']+=1
    if f<cut:
     out2,_=call(b,raw,f);assert [-1 if out2[i] is None else out2[i] for i in range(n)]==saved['sides'][f,:n].tolist();count['prefix_frames']+=1
 for k in ('frames','raw_candidates','withheld','newly_assigned','side_flip'):assert count[k]==summary['counts'].get(k,0),(k,count[k])
 assert count['non_label_supported']==count['non_label_preserved'],'a formerly supported assignment changed; requires review'
 write(U/'INDEPENDENT.json',{'status':'PASS','counts':dict(count),'contracts':tests,'baseline_guard_eligible_events':len(elig),'all_source_pins_preserved':len(c['pins']), 'operator':'Codex same operator, separate scalar implementation and saved-array recomputation; not independent anatomical ground truth.', 'limits':'Permutation/prefix tested on all changed clips and 8 selected crossing clips; full899 numeric replay does not certify correctness of every landmark. Prefix check checks causal consistency, not recovery accuracy.', 'training':0,'production_changed':False})
 print('Independent PASS',dict(count),flush=True)
if __name__=='__main__':main()
