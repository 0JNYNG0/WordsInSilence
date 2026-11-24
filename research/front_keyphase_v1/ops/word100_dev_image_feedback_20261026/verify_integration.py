import prepare as R
from pathlib import Path
import numpy as np,sys,traceback
from collections import Counter
N=R.O/'word100_source_separate_multiref_20261026';V=R.mod('separate_window_verifier',N/'verify_integration.py')
def evidence(z,q,r,key):
 n=len(z['domain']);old=z['phase_old'];new=z['phase_new'];a=z['span_first'];b=z['span_last'];_,_,oq=V.descriptor(q,old,a,b,z['domain'],key);_,_,nq=V.descriptor(q,new,np.minimum(a,np.floor(new).astype(int)),np.maximum(b,np.ceil(new).astype(int)),z['domain'],key);_,_,rq=V.descriptor(r,np.arange(n,dtype=float),np.arange(n),np.arange(n),z['domain'],key);_,_,qp=V.mapped(q,old,key)
 return z['point_ok']&qp&r[key],z['sequence_ok']&np.where(z['source']==1,nq,oq)&rq
def inputs_pairs():
 f=R.read(R.U/'FROZEN.json');p=R.read(R.U/'PREPARED.json');res=R.read(R.U/'PAIRS_DONE.json');R.check(f['pins']);R.check(p['pins']);R.check(res['pins']);samples={};total=Counter();controls=Counter()
 for row in p['rows']:
  z=R.load(row['output']);b=R.load(row['baseline_input']);original=R.load(row['original_input']);o=R.load(R.U/'outputs'/(row['clip_id']+'.npz'));req=o['requested'][...,None];av=original['available'][:,:,:15]
  for k in original:np.testing.assert_array_equal(z[k],original[k])
  ex=np.zeros(av.shape,np.uint8);ex[av]=1;active=req&av;ex[active]=2;ex[active&~o['mp_identity'][...,None]]=3;ex[active&(o['mp_unresolved']|o['pixel_unresolved'])]=4;ex[active&o['mp_identity_conflict'][...,None]]=3;ex[av&o['pixel_eligible']]=5;ex=np.where(req,ex,b['state_after']);np.testing.assert_array_equal(z['state_after'],ex);np.testing.assert_array_equal(z['quality_after'][:,:,:15],ex==5);np.testing.assert_array_equal(z['quality_before'],b['quality_before']);assert not(b['quality_before']&~z['quality_after']).any();assert not(z['core'][:,None]&av.any(-1)&~z['requested_current']).any();assert row['source']['signer_id'] in [11,12,13];samples[row['clip_id']]=z
 for i,row in enumerate(res['rows']):
  z=R.load(row['pair']);b=R.load(row['baseline_pair']);q=samples[row['query']];r=samples[row['reference']]
  for k in b:np.testing.assert_array_equal(z[k],b[k])
  for mode in ['before','after']:
   pp,ss=evidence(z,q,r,'quality_'+mode);np.testing.assert_array_equal(z['point_corroborated_'+mode],pp);np.testing.assert_array_equal(z['sequence_corroborated_'+mode],ss)
  pb=z['point_corroborated_before'];pa=z['point_corroborated_after'];sb=z['sequence_corroborated_before'];sa=z['sequence_corroborated_after'];assert not(pb&~pa).any() and not(sb&~sa).any();c=Counter(pairs=1,domain=int(z['domain'].sum()),angle_domain=int(z['domain'][:,:,:15].sum()),raw_point_angles=int(z['point_ok'][:,:,:15].sum()),raw_sequence_angles=int(z['sequence_ok'][:,:,:15].sum()),point_before=int(pb.sum()),point_after=int(pa.sum()),sequence_before=int(sb.sum()),sequence_after=int(sa.sum()),point_withdrawn=int((pb&~pa).sum()),sequence_withdrawn=int((sb&~sa).sum()),legacy_sequence_quality=int(z['sequence_quality'].sum()));assert dict(c)==row['counts'];total.update(c)
  cells=np.argwhere(pa[:,:,:15])
  if len(cells):
   j,h,a=cells[len(cells)//2];lo=int(np.floor(z['phase_old'][j]));hi=int(np.ceil(z['phase_old'][j]));mut=dict(q,quality_after=q['quality_after'].copy());mut['quality_after'][lo:hi+1,h,a]=False;pp,ss=evidence(z,mut,r,'quality_after');assert not pp[j,h,a] and not(pp&~pa).any() and not(ss&~sa).any();other=np.ones((2,23),bool);other[h,a]=False;np.testing.assert_array_equal(pp[:,other],pa[:,other]);np.testing.assert_array_equal(ss[:,other],sa[:,other]);controls['single_angle']+=1
   mut=dict(q,quality_after=q['quality_after'].copy());mut['quality_after'][lo:hi+1,h]=False;pp,ss=evidence(z,mut,r,'quality_after');assert not pp[j,h].any();np.testing.assert_array_equal(pp[:,1-h],pa[:,1-h]);np.testing.assert_array_equal(ss[:,1-h],sa[:,1-h]);assert not(pp&~pa).any() and not(ss&~sa).any();controls['single_hand']+=1
  if (i+1)%100==0:print('VERIFIED_DEV_PAIR',i+1,600,flush=True)
 assert len(samples)==300 and dict(total)==res['counts'] and total['pairs']==600;R.check(f['protected']);R.write(R.U/'INTEGRATION_VERIFIED.json',dict(at=R.now(),status='PASS300_ORIGINAL_INPUTS600_FIXED_PAIRS',inputs=300,pairs=600,counts=dict(total),negative_controls=dict(controls),pins={str(R.U/'PREPARED.json'):R.sha(R.U/'PREPARED.json'),str(R.U/'PAIRS_DONE.json'):R.sha(R.U/'PAIRS_DONE.json')},method='SameCodex separate source-window equations; original arrays/paths/quality preserved; not human anatomical or semantic groundtruth'));print('DEV_PAIR_PASS',dict(total))
def receipts():
 f=R.read(R.U/'FROZEN.json');p=R.read(R.U/'PREPARED.json');res=R.read(R.U/'PAIRS_DONE.json');out=R.read(R.U/'FEEDBACK_DONE.json');R.check(out['pins']);samples={r['clip_id']:R.load(r['output']) for r in p['rows']};counts=Counter();query_refs={}
 for row,rr in zip(res['rows'],out['rows']):
  b=R.read(row['baseline_receipt']);a=R.read(rr['receipt']);z=R.load(row['pair']);q=samples[row['query']];r=samples[row['reference']];domain=z['domain'][:,:,:15];n=len(domain);t=z['reference_times_us'];src={'query_before':q['state_after'][np.floor(z['phase_old']).astype(int)],'query_after':q['state_after'][np.ceil(z['phase_old']).astype(int)],'reference':r['state_after']}
  assert a['hands']==b['hands'] and a['recognition']==b['recognition'] and a['time_correspondence']==b['time_correspondence'];assert a['whole_gesture_correct'] is a['validated_correction'] is None
  for g in a['individual_source_states']:
   actual=src[g['source']];rebuilt=np.full(actual.shape,255,np.uint8)
   for e in g['states']:
    h=['L','R'].index(e['side']);k=e['angle_index'];got=0
    for sp in e['intervals']:
     lo,hi=sp['first'],sp['last'];assert 0<=lo<=hi<n and sp['samples']==hi-lo+1;assert sp['reference_start_s']==float(t[lo]/1e6) and sp['reference_end_s']==float(t[hi]/1e6);assert np.all(rebuilt[lo:hi+1,h,k]==255);rebuilt[lo:hi+1,h,k]=e['state_code'];got+=hi-lo+1
    assert got==e['cells']
   np.testing.assert_array_equal(rebuilt[domain],actual[domain]);assert np.all(rebuilt[~domain]==255);counts['source_cells']+=int(domain.sum())
  query_refs.setdefault(row['query'],[]).append(row['reference']);counts['pairs']+=1
 for qid,refs in query_refs.items():
  assert len(refs)==len(set(refs))==2;qq=R.read(R.U/'queries'/(qid+'.json'));assert qq['query_id']==qid and {r['reference'] for r in qq['references']}==set(refs);assert qq['recognition']=={k:v['predictions'][qid] for k,v in f['scores'].items()};assert qq['semantic_correctness'] is qq['validated_correction'] is None
 assert len(query_refs)==300 and counts['pairs']==600;R.check(f['protected']);R.write(R.U/'FEEDBACK_VERIFIED.json',dict(at=R.now(),status='PASS600_RECEIPTS300_SAVED_RECOGNITION_QUERIES',counts=dict(counts),queries=300,raw_and_legacy_quality_preserved=True,recognition_reinference=0,method='Saved-score identity and exact unchanged raw/legacy metrics; independent source state interval reconstruction. SameCodex, not human truth.'));print('DEV_FEEDBACK_PASS',dict(counts))
if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException:
  p=R.U/'INDEPENDENT_FIRST_ERROR.txt'
  if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
  raise
