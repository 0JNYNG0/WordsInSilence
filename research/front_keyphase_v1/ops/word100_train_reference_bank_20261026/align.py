"""All TRAIN person-to-person paths, preserving named references and original coordinates."""
import prepare as R
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np,sys,time,os,traceback
N=R.O/'word100_source_separate_multiref_20261026';Z=R.O/'word100_core_image_evidence_20261026'
M=R.mod('train_bank_frozen_pair',R.O/'word100_multipeople_feedback_integration_20261026/run.py')
KEYS=['domain','path','phase_old','phase_new','span_first','span_last','point_ok','point_delta','sequence_old_ok','sequence_ok','sequence_old_delta','sequence_delta','sequence_old_quality','sequence_quality','source','quality_fallback','reference_missing','reference_times_us','query_times_us','mean_path_evidence']

def freeze():
 f=R.read(R.U/'FROZEN.json');assert R.read(R.U/'BASE_VERIFIED.json')['inputs']==899;R.check(f['pins']);old=R.read(N/'PAIRS_DONE.json');R.check(old['pins']);prev={(r['query'],r['reference']):r for r in old['rows']};by=defaultdict(list);pairs=[];pins={};oldinputs={r['clip_id']:r for r in R.read(N/'PREPARED.json')['rows']}
 for r in f['catalog']:by[r['word_code']].append(r)
 # Reuse only when every alignment-relevant original array is exactly identical.
 for r in f['catalog']:
  if r['clip_id'] not in oldinputs:continue
  a=R.load(r['input']);b=R.load(oldinputs[r['clip_id']]['output'])
  for k in ['features','present','core','values','available','quality_eligible','timestamp_us']:np.testing.assert_array_equal(a[k],b[k])
 for word,rows in sorted(by.items()):
  for q in sorted(rows,key=lambda r:r['signer_id']):
   for r in sorted(rows,key=lambda r:r['signer_id']):
    if q['signer_id']==r['signer_id']:continue
    key=(q['clip_id'],r['clip_id']);oldpath=prev[key]['pair'] if key in prev else None
    if oldpath:pins[oldpath]=R.sha(oldpath)
    pairs.append(dict(query=key[0],reference=key[1],query_person=q['signer_id'],reference_person=r['signer_id'],word_code=word,reuse=oldpath))
 assert len(pairs)==7184 and sum(p['reuse'] is not None for p in pairs)==200
 for p in [R.U/'FROZEN.json',R.U/'BASE_VERIFIED.json',Path(__file__),Path(M.__file__),Path(M.P.__file__),Path(M.D.__file__),Path(M.F.__file__),N/'PAIRS_DONE.json',N/'INTEGRATION_VERIFIED.json']:pins[str(p)]=R.sha(p)
 for row in f['catalog']:pins[row['input']]=R.sha(row['input'])
 R.write(R.U/'ALIGNMENT_FROZEN.json',dict(at=R.now(),rows=pairs,catalog=f['catalog'],pins=pins,protected=f['protected'],reused_pairs=200,new_pairs=6984,
  method='Existing unchanged original-C features,flat missing1.5,band.2,original point and bounded1frame fractional sequence with rawavailability fallback. No imageconfidence in alignment,allquality_eligible0. Saved200pairs reused exactly;7184different-recorded-person directed sameword paths. Not averaged pose or bestrefselection.',
  planned_calibration='For each heldoutTRAIN person and each other namedreference,exclude heldout from every calibration contribution. Require>=5other people with currentimage evidence for any strong angle bound. Do not let peer preparation/noncore samples set activity bounds. Retain raw distances and empirical ranges separately,missing staysunknown. Never claim semantic correctness from normalrange.',
  gates='All rawinputs unchanged,monotonic paths,endpoints,sourcewindow arithmetic and reference domain verified. No scores/labels used in path;wordlabel selects comparable public reference pool only. Extra199/200 and personal data forbidden.',
  new_training=0,new_classifier_inference=0))
 for p in ['alignments','alignment_records','alignment_starts']:(R.U/p).mkdir()
 print('ALIGNMENT_FROZEN7184_REUSE200_NEW6984',flush=True)

def run():
 f=R.read(R.U/'ALIGNMENT_FROZEN.json');R.check(f['pins']);R.check(f['protected']);assert not(R.U/'ALIGNMENT_RUNNING.json').exists();R.write(R.U/'ALIGNMENT_RUNNING.json',dict(at=R.now(),pid=os.getpid()));samples={r['clip_id']:R.load(r['input']) for r in f['catalog']};beg=time.perf_counter();rows=[];pins={};counts=Counter()
 for i,row in enumerate(f['rows']):
  name=row['query']+'__'+row['reference'];rec=R.U/'alignment_records'/(name+'.json')
  if rec.exists():
   rr=R.read(rec);assert rr['context_sha256']==R.sha(R.U/'ALIGNMENT_FROZEN.json');R.check(rr['outputs']);rows.append(rr['row']);pins.update(rr['outputs']);counts.update(rr['counts']);continue
  sp=R.U/'alignment_starts'/(name+'.json');assert not sp.exists(),'Inspect preserved partial before recovery';R.write(sp,dict(at=R.now()));q=samples[row['query']];r=samples[row['reference']]
  if row['reuse']:path=Path(row['reuse']);z=R.load(path)
  else:
   z=M.pair(q,r);path=R.U/'alignments'/(name+'.npz');np.savez_compressed(path,**z)
  assert not z['sequence_quality'].any() and not z['sequence_old_quality'].any();assert np.array_equal(z['domain'],np.broadcast_to(r['core'][:,None,None],z['domain'].shape));pp=z['path'];assert (pp[0]==[0,0]).all() and (pp[-1]==[len(q['core'])-1,len(r['core'])-1]).all()
  c=Counter(pairs=1,reused=int(row['reuse'] is not None),new=int(row['reuse'] is None),domain=int(z['domain'].sum()),point_comparable=int(z['point_ok'].sum()),sequence_comparable=int(z['sequence_ok'].sum()));item=dict(row,pair=str(path));out={str(path):R.sha(path)};R.write(rec,dict(at=R.now(),context_sha256=R.sha(R.U/'ALIGNMENT_FROZEN.json'),row=item,outputs=out,counts=dict(c)));rows.append(item);pins.update(out);counts.update(c)
  if (i+1)%100==0:print('TRAIN_PATHS',i+1,7184,'seconds',round(time.perf_counter()-beg,1),flush=True)
 assert len(rows)==7184;R.check(f['pins']);R.check(f['protected']);R.write(R.U/'ALIGNMENT_DONE.json',dict(at=R.now(),rows=rows,pins=pins,counts=dict(counts),seconds=time.perf_counter()-beg,verified=False));(R.U/'ALIGNMENT_RUNNING.json').unlink();print('TRAIN_PATHS_DONE',dict(counts),flush=True)

if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException:
  p=R.U/'ALIGNMENT_FIRST_ERROR.txt'
  if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
  raise
