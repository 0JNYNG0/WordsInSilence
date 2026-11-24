import prepare as R
from pathlib import Path
import sys,numpy as np,traceback,time
from collections import Counter,defaultdict
N=R.O/'word100_source_separate_multiref_20261026';Z=R.O/'word100_core_image_evidence_20261026';M=R.O/'word100_multipeople_feedback_integration_20261026'
sys.path.insert(0,str(N));S=R.mod('dev_image_attach',N/'integrate.py');A=R.mod('dev_source_receipts',Z/'source_feedback_v2.py');C=R.mod('dev_legacy_consumer',R.O/'word100_feedback_evidence_consumer_20261026/consumer.py')
def inputs():
 f=R.read(R.U/'FROZEN.json');assert R.read(R.U/'VERIFIED_IMAGES.json')['clips']==300;R.check(f['pins']);R.check(f['protected']);(R.U/'inputs').mkdir();rows=[];pins={};tot=Counter()
 for row in f['rows']:
  cid=row['clip_id'];b=R.load(row['baseline_input']);original=R.load(row['original_input']);o=R.load(R.U/'outputs'/(cid+'.npz'));R.check(R.read(R.U/'records'/(cid+'.json'))['outputs']);z={k:v.copy() for k,v in b.items()};new=S.state(o,b['available'][:,:,:15]);z['state_after']=np.where(o['requested'][...,None],new,b['state_after']);z['quality_after'][:,:,:15]=z['state_after']==5;z['requested_current']|=o['requested']
  for k in original:np.testing.assert_array_equal(z[k],original[k])
  assert not(b['requested_current']&o['requested']).any() and not(b['quality_before']&~z['quality_after']).any();assert not(z['core'][:,None]&z['available'][:,:,:15].any(-1)&~z['requested_current']).any()
  out=R.U/'inputs'/(cid+'.npz');np.savez_compressed(out,**z);pins[str(out)]=R.sha(out);core=z['core'];cnt=Counter(clips=1,frames=len(core),raw_core_angles=int(z['available'][core,:,:15].sum()),core_before=int(b['quality_before'][core].sum()),core_after=int(z['quality_after'][core].sum()),withdrawn=int((b['quality_before']&~z['quality_after'])[core].sum()));tot.update(cnt);rows.append(dict(**row,output=str(out),counts=dict(cnt)))
 R.write(R.U/'PREPARED.json',dict(at=R.now(),rows=rows,counts=dict(tot),pins=pins));print('DEV_INPUTS300',dict(tot))
def pairs():
 f=R.read(R.U/'FROZEN.json');p=R.read(R.U/'PREPARED.json');R.check(f['pins']);R.check(p['pins']);R.check(f['protected']);(R.U/'pairs').mkdir();samples={r['clip_id']:R.load(r['output']) for r in p['rows']};rows=[];pins={};tot=Counter()
 for i,row in enumerate(f['pairs']):
  z=R.load(row['pair']);q=samples[row['query']];r=samples[row['reference']];pb,sb=S.attach(z,q,r,'quality_before');pa,sa=S.attach(z,q,r,'quality_after');assert not(pb&~pa).any() and not(sb&~sa).any();z.update(point_corroborated_before=pb,point_corroborated_after=pa,sequence_corroborated_before=sb,sequence_corroborated_after=sa)
  op=R.U/'pairs'/Path(row['pair']).name;np.savez_compressed(op,**z);pins[str(op)]=R.sha(op)
  c=Counter(pairs=1,domain=int(z['domain'].sum()),angle_domain=int(z['domain'][:,:,:15].sum()),raw_point_angles=int(z['point_ok'][:,:,:15].sum()),raw_sequence_angles=int(z['sequence_ok'][:,:,:15].sum()),point_before=int(pb.sum()),point_after=int(pa.sum()),sequence_before=int(sb.sum()),sequence_after=int(sa.sum()),point_withdrawn=int((pb&~pa).sum()),sequence_withdrawn=int((sb&~sa).sum()),legacy_sequence_quality=int(z['sequence_quality'].sum()));tot.update(c);rows.append(dict(query=row['query'],reference=row['reference'],word_code=row['word_code'],label=row['label'],baseline_pair=row['pair'],baseline_receipt=row['receipt'],pair=str(op),counts=dict(c)))
  if (i+1)%100==0:print('DEV_PAIRS',i+1,600,flush=True)
 R.write(R.U/'PAIRS_DONE.json',dict(at=R.now(),rows=rows,counts=dict(tot),pins=pins));print('DEV_PAIRS600',dict(tot))
def receipts():
 f=R.read(R.U/'FROZEN.json');p=R.read(R.U/'PREPARED.json');res=R.read(R.U/'PAIRS_DONE.json');assert R.read(R.U/'INTEGRATION_VERIFIED.json')['pairs']==600;R.check(f['protected']);R.check(p['pins']);R.check(res['pins']);(R.U/'receipts').mkdir();(R.U/'queries').mkdir();samples={r['clip_id']:R.load(r['output']) for r in p['rows']};pins={};rows=[];by=defaultdict(list);counts=Counter()
 for i,row in enumerate(res['rows']):
  q=samples[row['query']];r=samples[row['reference']];z=R.load(row['pair']);legacy=R.read(row['baseline_receipt']);image=A.build(z,q,r,row['query'],row['reference'],C);saved={k:v['predictions'][row['query']] for k,v in f['scores'].items()};assert saved==legacy['recognition'];receipt=dict(legacy)
  # Keep legacy body/angle corroboration with its original definition. Add stronger image checks in separate fields.
  for key in ['source_separated_point_evidence','individual_source_states','advice_policy','image_methods']:receipt[key]=image[key]
  receipt['current_image_hands']=image['hands'];receipt['current_image_scope']='New full/crop/original image corroboration applies to15fingerangles only. Legacy hand/body quality and all raw metrics remain in hands unchanged.';receipt['yolo_audit_status']='Old public detector proposals preserved in source records; no new per-angleYOLO mapping in this DEV trial.';receipt['whole_gesture_correct']=None;receipt['validated_correction']=None;receipt['new_recognition_inference']=False
  for lh,ih in zip(legacy['hands'],image['hands']):
   for a,b in zip(lh['groups'],ih['groups']):
    for k in ['audit_cells','point_compared_cells','sequence_compared_cells','point_mean_difference','sequence_level_mean_difference','sequence_change_mean_difference','source_cells','missing_point_intervals','sequence_context_unavailable_intervals','reference_missing_intervals']:assert a[k]==b[k],k
    counts['raw_group_metrics_preserved']+=1
  op=R.U/'receipts'/Path(row['baseline_receipt']).name;R.write(op,receipt);pins[str(op)]=R.sha(op);rows.append(dict(query=row['query'],reference=row['reference'],receipt=str(op)));by[row['query']].append(dict(reference=row['reference'],receipt=str(op),counts=row['counts']));counts['pairs']+=1
  if (i+1)%100==0:print('DEV_RECEIPTS',i+1,600,flush=True)
 for qid,refs in by.items():
  assert len(refs)==2;op=R.U/'queries'/(qid+'.json');R.write(op,dict(query_id=qid,recognition={k:v['predictions'][qid] for k,v in f['scores'].items()},references=refs,semantic_correctness=None,validated_correction=None,reference_aggregation='none',condition='publicDEV_existing_original_predictions_plus_image_evidence'));pins[str(op)]=R.sha(op)
 R.check(f['protected']);R.write(R.U/'FEEDBACK_DONE.json',dict(at=R.now(),rows=rows,pins=pins,counts=dict(counts),query_count=len(by),training=0,inference=0,operating_changed=False));print('DEV_FEEDBACK600',dict(counts))
if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException:
  p=R.U/'INTEGRATION_FIRST_ERROR.txt'
  if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
  raise
