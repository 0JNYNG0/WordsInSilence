"""Retain out-of-core counterparts for the same audited query frames."""
from pathlib import Path
import numpy as np
from audit_costs import U,OPS,read,sha,write,load,features,contribution,motion
def main():
 inp=read(OPS/'word100_phase_correspondence_20261101/INPUTS.json');prior=read(U/'COST_COMPONENTS.json')
 write(U/'CONTEXT_AUDIT_SCOPE.json',{'primary_scope_sha256':sha(U/'COST_AUDIT_SCOPE.json'),'component_sha256':sha(U/'COST_COMPONENTS.json'),'reason':'For every query frame already audited, retain ALL saved B and C reference links, including reference unlabelled gaps/outside this parent. No new search.','code_sha256':sha(U/'audit_context.py')})
 arr={a:load(v['original']) for a,v in inp['inputs'].items()};feat={a:features(x) for a,x in arr.items()}
 lookup={(p['query'],p['reference']):p for p in inp['natural']};cells=dict(prior['cells']);rows=[]
 for par in prior['parents']:
  a,b=par['query'],par['reference'];p=lookup[a,b];paths={k:read(p[n])['path'] for k,n in [('B','global_path'),('C','same_cost_path')]};records=[]
  for oldrow in par['links']:
   i=oldrow['q'];by={}
   for k,path in paths.items():
    ids=[j for qi,j in path if qi==i];by[k]=[]
    for j in ids:
     key=f'{a}_{i}_{b}_{j}';by[k].append(key)
     if key not in cells:
      c=contribution(feat[a],feat[b],i,j)
      cells[key]={'query':a,'reference':b,'q':i,'r':j,'components':c,'query_motion':motion(arr[a],i),'reference_motion':motion(arr[b],j),'geometry11':float(sum(v['total'] for g,v in c.items() if not g.endswith('_direction'))/13),'direction2':float(sum(v['total'] for g,v in c.items() if g.endswith('_direction'))/13),'full13':float(sum(v['total'] for v in c.values())/13)}
   mean={k:{m:float(np.mean([cells[c][m] for c in keys])) for m in ['geometry11','direction2','full13']} for k,keys in by.items()}
   records.append({'q':i,'saved_links':by,'local_component_mean':mean,'direction_changes_local_ordering':bool(np.sign(mean['B']['full13']-mean['C']['full13'])*np.sign(mean['B']['geometry11']-mean['C']['geometry11'])<0)})
  rows.append({'parent_id':par['parent_id'],'query':a,'reference':b,'reference_range':par['reference_range'],'links':records})
 write(U/'COST_CONTEXT_SUPPLEMENT.json',{'parents':rows,'cells':cells,'scope_sha256':sha(U/'CONTEXT_AUDIT_SCOPE.json'),'new_paths':0,'adds_saved_context_only':True})
 print('cells',len(cells),'added',len(cells)-len(prior['cells']))
 for key in ['P066_68_P063_58','P066_68_P063_55']:
  c=cells[key];print(key,{k:c[k] for k in ['geometry11','direction2','full13']})
if __name__=='__main__':main()
