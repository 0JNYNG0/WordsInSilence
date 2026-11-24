"""New fixed-reference change evidence, replacing old shared-only exploratory fields.
Does not alter stored paths, original scores, ranges, decisions or models.
"""
from pathlib import Path
import json,hashlib,numpy as np
from relational import features,GROUPS
U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def main():
 rows={r['alias']:r for r in read(U/'AUTHORIZATION.json')['rows']};f={a:features(dict(np.load(r['input'],allow_pickle=False))) for a,r in rows.items()};out=[]
 for p in sorted((U/'pairs').glob('*.json')):
  d=read(p);q,r=f[d['query']],f[d['reference']];path=np.asarray(d['path']);stages=[]
  for s in d['stages']:
   thirds=np.array_split(np.arange(s['reference_core'][0],s['reference_core'][1]+1),3);groups={}
   for g in GROUPS:
    x,qm=q[g];y,rm=r[g];records=[];ref_means=[];query_means=[];valid=[]
    for ids in thirds:
     if not len(ids):records.append(dict(expected=0,compared=0,sufficient=False));valid.append(np.zeros(y.shape[1],bool));ref_means.append(np.zeros(y.shape[1]));query_means.append(np.zeros(y.shape[1]));continue
     pp=path[np.isin(path[:,1],ids)];a,b=pp.T;expected=int(rm[ids].sum());den=rm[ids].sum(0);rv=np.divide(np.where(rm[ids],y[ids],0).sum(0),den,out=np.zeros(y.shape[1]),where=den>0)
     q_by_ref=np.zeros((len(ids),y.shape[1]));supported=np.zeros_like(q_by_ref,bool);missing=0;unique_query=set()
     for k,j in enumerate(ids):
      qi=pp[pp[:,1]==j,0];shared=qm[qi]&rm[j];count=shared.sum(0);supported[k]=count>0;q_by_ref[k]=np.divide(np.where(shared,x[qi],0).sum(0),count,out=np.zeros(y.shape[1]),where=count>0);missing+=int((rm[j]&~qm[qi]).sum());unique_query.update(qi[shared.any(1)].tolist())
     count=supported.sum(0);qv=np.divide(np.where(supported,q_by_ref,0).sum(0),count,out=np.zeros(y.shape[1]),where=count>0)
     good=(den>=2)&(count==den)
     if missing or len(unique_query)<2:good[:]=False
     records.append(dict(reference_frames=ids.tolist(),expected=expected,compared=int(supported.sum()),path_missing_cells=missing,distinct_observed_query_frames=len(unique_query),sufficient=bool(good.any()),reference_mean=rv.tolist(),query_observed_mean=qv.tolist(),valid_dimensions=good.tolist()))
     ref_means.append(rv);query_means.append(qv);valid.append(good)
    common=np.logical_and.reduce(valid);delta=np.abs(np.diff(query_means,axis=0)-np.diff(ref_means,axis=0));score=float(np.minimum(delta[:,common],1).mean()) if common.any() else None
    groups[g]=dict(thirds=records,valid_dimensions=int(common.sum()),change_difference=score,status='observed_change_candidate' if score is not None else 'insufficient',semantic_order_proven=False)
   stages.append(dict(segment_id=s['segment_id'],reference_core=s['reference_core'],groups=groups))
  out.append(dict(pair_file=p.name,pair_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),query=d['query'],reference=d['reference'],condition=d['condition'],stages=stages))
 with (U/'CHANGE_EVIDENCE.json').open('x',encoding='utf8') as fp:json.dump(dict(status='COMPLETE_FIXED_REFERENCE_CHANGE_DIAGNOSTIC',replaces='pair.changes (query-conditioned shared-only exploratory field); originals preserved',thirds_are_not_semantic_stages=True,pairs=out),fp,ensure_ascii=False,indent=2)
 print('fixed-reference change evidence',len(out))
if __name__=='__main__':main()
