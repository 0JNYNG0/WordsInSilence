"""Correct hand-only masking's arm QUALITY sidecar; frozen paths/scores untouched."""
import study as S
import numpy as np
from collections import Counter
def qseq(quality,index,first,last,domain):
 a=np.floor(index).astype(int);b=np.ceil(index).astype(int);point=quality[a]&quality[b];out=np.zeros(domain.shape,bool)
 for j in range(2,len(index)-2):out[j]=domain[j-2:j+3].all(0)&quality[first[j-2]:last[j+2]+1].all(0)&point[[j-2,j,j+2]].all(0)
 return out
def main():
 f=S.read(S.U/'FROZEN.json');r=S.read(S.U/'RESULT.json');S.check(r['pins']);S.check(f['input_pins']);rows={x['clip_id']:x for x in f['rows']};dest=S.U/'quality_correction';dest.mkdir();count=Counter();pins={};entries=[]
 for row in r['rows']:
  if row['kind'] not in S.MISSING:continue
  kind=row['kind'];z=S.load(rows[row['clip_id']]['output']);o=S.load(S.U/'cases'/(row['name']+'.npz'));n=len(z['core']);q=z['quality_eligible'].copy();h=0 if kind.endswith('L') else 1;ids=np.arange(n)
  if kind.startswith('gap'):
   active=np.flatnonzero(z['core']);mid=active[len(active)//2];ids=np.arange(mid-1,mid+2)
  q[ids,h,:17]=False;q[ids,h,21:]=False
  # Pose elbow/wrist17:21 remain available and retain quality; no hand transfer.
  assert np.array_equal(q[:,:,17:21],z['quality_eligible'][:,:,17:21])
  ref=qseq(z['quality_eligible'],np.arange(n,dtype=float),np.arange(n),np.arange(n),o['domain'])
  a=qseq(q,o['phase_old'],o['span_first'],o['span_last'],o['domain'])&ref&o['sequence_old_ok']
  b=qseq(q,o['phase_new'],np.minimum(o['span_first'],np.floor(o['phase_new']).astype(int)),np.maximum(o['span_last'],np.ceil(o['phase_new']).astype(int)),o['domain'])&ref
  b=np.where(o['phase_source']==1,b,a)&o['sequence_new_ok']
  old=o['sequence_quality'];new=o['sequence_new_quality'];allow=np.zeros(old.shape,bool);allow[:,h,17:21]=True
  assert np.array_equal(a[~allow],old[~allow]) and np.array_equal(b[~allow],new[~allow]);assert not(old&~a).any() and not(new&~b).any()
  restored_old=int((a&~old).sum());restored_new=int((b&~new).sum());count['conditions']+=1;count['old_arm_quality_restored']+=restored_old;count['new_arm_quality_restored']+=restored_new
  p=dest/(row['name']+'.npz');np.savez_compressed(p,sequence_quality=a,sequence_new_quality=b);pins[str(p)]=S.sha(p);entries.append(dict(name=row['name'],old_arm_quality_restored=restored_old,new_arm_quality_restored=restored_new))
 S.check(r['pins']);S.check(f['pins']);S.check(f['protected'])
 S.write(S.U/'QUALITY_CORRECTION.json',dict(at=S.now(),status='PASS_SUPPLEMENT_NO_PATH_OR_SCORE_RECOMPUTATION',cause='In synthetic hand-only missing controls, study.py also set same-side pose elbow/wrist QUALITY false. Coordinates, comparability, path and discrepancy were unaffected because this flag is sidecar-only. Correct masks only channels0:17 and21:23; preserve body-arm17:21. Frozen first outputs retained.',
  counts=dict(count),rows=entries,pins=pins,original_results_sha256=S.sha(S.U/'RESULT.json'),original_code_sha256=S.sha(S.U/'study.py'),correction_code_sha256=S.sha(__file__),scope='1200hand-only missing controls quality sidecars only; no operational change or detector/model inference.',independent_check='Locality and no quality-to-position substitution checked on all1200. Original4500 result byte hashes unchanged. Same Codex separate arithmetic, not human truth.'))
 print('QUALITY_CORRECTION',dict(count),flush=True)
if __name__=='__main__':main()
