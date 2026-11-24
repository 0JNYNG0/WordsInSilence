"""Explain saved links only. No new path, model score, or parameter fit."""
from pathlib import Path
import json,hashlib,sys,numpy as np
U=Path(__file__).resolve().parent;OPS=U.parent
sys.path.insert(0,str(OPS/'word100_relational_sequence_compare_20261031'))
from relational import features,GROUPS
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in ['x','mask','frame_id','timestamp_us']}
def contribution(q,r,i,j):
 out={}
 for g in GROUPS:
  x,qm=q[g];y,rm=r[g];den=int(rm[j].sum());shared=qm[i]&rm[j]
  observed=float(np.minimum(np.abs(x[i]-y[j]),1)[shared].sum()/den) if den else 0.
  missing=float((rm[j]&~qm[i]).sum()/den) if den else 0.
  empty=.5 if den==0 else 0.
  out[g]={'expected':den,'shared':int(shared.sum()),'observed':observed,'missing':missing,'empty':empty,'total':observed+missing+empty}
 return out
def motion(a,i):
 out={}
 for h,o,v in [('L',0,61),('R',28,65)]:
  ok=bool(a['mask'][i,v:v+2].all());speed=float(np.linalg.norm(a['x'][i,v:v+2])) if ok else None
  direct=bool(i>0 and a['frame_id'][i]-a['frame_id'][i-1]==1 and a['mask'][[i-1,i],o:o+2].all())
  disp=float(np.linalg.norm(a['x'][i,o:o+2]-a['x'][i-1,o:o+2])) if direct else None
  out[h]={'speed_shoulder_per_second':speed,'consecutive_displacement_shoulder':disp,'direct_hand_position_valid':bool(a['mask'][i,o:o+2].all()),'pose_wrist_valid':bool(a['mask'][i,o+2:o+4].all())}
 return out
def main():
 old=OPS/'word100_phase_correspondence_20261101';inp=read(old/'INPUTS.json');arrays={a:load(v['original']) for a,v in inp['inputs'].items()};ff={a:features(v) for a,v in arrays.items()}
 pins={str(old/'INPUTS.json'):sha(old/'INPUTS.json'),str(U/'audit_costs.py'):sha(U/'audit_costs.py')}
 for a,v in inp['inputs'].items():pins[v['original']]=sha(v['original'])
 for p in inp['natural']:
  for k in ['global_path','same_cost_path']:pins[p[k]]=sha(p[k])
 write(U/'COST_AUDIT_SCOPE.json',{'status':'FROZEN_BEFORE_COMPONENT_SUMMARIES','pairs':12,'parents':39,'signals':'original69; all stored B/C links into fixed main reference cores; all alternatives retained','selection':'all39 parent scopes, not only motivating W06','purpose':'local cost decomposition, not new global causal proof','pins':pins,'new_dtw':0,'new_model_scoring':0})
 rows=[];allcells={};summary=[]
 for p in inp['natural']:
  a,b=p['query'],p['reference'];paths={k:read(p[n])['path'] for k,n in [('B','global_path'),('C','same_cost_path')]}
  parents=[x for x in inp['parents'] if x['query']==a and x['reference']==b]
  for parent in parents:
   lo,hi=parent['reference_range'];chosen={k:[(int(i),int(j)) for i,j in path if lo<=j<=hi] for k,path in paths.items()}
   qids=sorted({i for ps in chosen.values() for i,j in ps});pairs=[]
   for i in qids:
    by={}
    for k,ps in chosen.items():
     ids=[j for q,j in ps if q==i];vals=[]
     for j in ids:
      cellkey=f'{a}_{i}_{b}_{j}'
      if cellkey not in allcells:
       c=contribution(ff[a],ff[b],i,j)
       allcells[cellkey]={'query':a,'reference':b,'q':i,'r':j,'components':c,'query_motion':motion(arrays[a],i),'reference_motion':motion(arrays[b],j),'geometry11':float(sum(v['total'] for g,v in c.items() if not g.endswith('_direction'))/13),'direction2':float(sum(v['total'] for g,v in c.items() if g.endswith('_direction'))/13),'full13':float(sum(v['total'] for v in c.values())/13)}
      vals.append(cellkey)
     by[k]=vals
    if by['B'] and by['C']:
     mean={k:{m:float(np.mean([allcells[c][m] for c in keys])) for m in ['geometry11','direction2','full13']} for k,keys in by.items()}
     # Descriptive local ordering only. No path/score recomputation.
     sign_full=np.sign(mean['B']['full13']-mean['C']['full13']);sign_geom=np.sign(mean['B']['geometry11']-mean['C']['geometry11'])
     flip=bool(sign_full*sign_geom<0)
    else:mean=None;flip=None
    pairs.append({'q':i,'saved_links':by,'local_component_mean':mean,'direction_changes_local_ordering':flip})
   rows.append({'parent_id':parent['id'],'query':a,'reference':b,'reference_range':[lo,hi],'saved_query_range':parent['query_range'],'links':pairs})
   summary.append({'parent_id':parent['id'],'q_rows':len(pairs),'both_conditions_rows':sum(x['local_component_mean'] is not None for x in pairs),'local_ordering_flip_rows':sum(x['direction_changes_local_ordering'] is True for x in pairs)})
 write(U/'COST_COMPONENTS.json',{'scope_sha256':sha(U/'COST_AUDIT_SCOPE.json'),'parents':rows,'cells':allcells,'summary':summary,'new_dtw':0,'global_causation_proven':False,'old_published_scores_changed':False})
 print({'parents':len(rows),'unique_saved_link_cells':len(allcells),'local_ordering_flips':sum(s['local_ordering_flip_rows'] for s in summary)})
 for key in ['P066_68_P063_58','P066_68_P063_55']:
  if key in allcells:
   c=allcells[key];print(key,{k:c[k] for k in ['geometry11','direction2','full13','query_motion','reference_motion']})
if __name__=='__main__':main()
