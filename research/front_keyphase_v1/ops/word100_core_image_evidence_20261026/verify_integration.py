import prepare as R
from pathlib import Path
import numpy as np,traceback
from collections import Counter
N=R.O/'word100_source_separate_multiref_20261026'
V=R.mod('fixed_pair_separate_verifier',N/'verify_integration.py')
def main():
 f=R.read(R.U/'FROZEN.json');p=R.read(R.U/'PREPARED.json');res=R.read(R.U/'PAIRS_DONE.json');R.check(f['pins']);R.check(f['protected']);R.check(p['pins']);R.check(p['outputs']);R.check(res['pins']);samples={};inputtot=Counter();total=Counter();tests=Counter();immutable=['xy_px','usable','timestamp_us','frame_id','features','present','core','values','available','quality_eligible','scale','yolo_conflict']
 for row in p['rows']:
  z=R.load(row['output']);b=R.load(row['baseline_input']);o=R.load(R.U/'outputs'/(row['clip_id']+'.npz'));req=o['requested'][...,None];av=b['available'][:,:,:15]
  for k in immutable:np.testing.assert_array_equal(z[k],b[k])
  np.testing.assert_array_equal(z['state_before'],b['state_after']);np.testing.assert_array_equal(z['quality_before'],b['quality_after']);np.testing.assert_array_equal(z['requested_current'],b['requested_current']|o['requested']);assert not (b['requested_current']&o['requested']).any();assert not (b['core'][:,None]&av.any(-1)&~z['requested_current']).any()
  expected=np.zeros(av.shape,np.uint8);expected[av]=1;re=req&av;expected[re]=2;expected[re&~o['mp_identity'][...,None]]=3;expected[re&(o['mp_unresolved']|o['pixel_unresolved'])]=4;expected[re&o['mp_identity_conflict'][...,None]]=3;expected[o['pixel_eligible']&av]=5;expected=np.where(req,expected,b['state_after']);np.testing.assert_array_equal(expected,z['state_after']);np.testing.assert_array_equal(z['quality_after'][:,:,:15],expected==5);assert not z['quality_after'][:,:,15:].any();assert not (z['quality_before']&~z['quality_after']).any()
  core=z['core'];c=Counter(clips=1,frames=len(core),core_angle_domain=int(core.sum())*30,raw_core_angles=int(z['available'][core,:,:15].sum()),before_core_corroborated=int(z['quality_before'][core].sum()),after_core_corroborated=int(z['quality_after'][core].sum()),added_core=int((z['quality_after']&~z['quality_before'])[core].sum()),withdrawn_core=int((z['quality_before']&~z['quality_after'])[core].sum()),yolo_audit_core=int(z['yolo_conflict'][core].sum()));assert dict(c)==row['counts'];inputtot.update(c);samples[row['clip_id']]=z
 assert dict(inputtot)==p['counts'] and len(samples)==300
 def evidence(z,q,r,key):
  domain=z['domain'];n=len(domain);base=z['phase_old'];new=z['phase_new'];first=z['span_first'];last=z['span_last'];_,_,oq=V.descriptor(q,base,first,last,domain,key);_,_,nq=V.descriptor(q,new,np.minimum(first,np.floor(new).astype(int)),np.maximum(last,np.ceil(new).astype(int)),domain,key);_,_,rq=V.descriptor(r,np.arange(n,dtype=float),np.arange(n),np.arange(n),domain,key);_,_,qp=V.mapped(q,base,key);pp=z['point_ok']&qp&r[key];ss=z['sequence_ok']&np.where(z['source']==1,nq,oq)&rq;return pp,ss
 for k,row in enumerate(res['rows']):
  z=R.load(row['pair']);b=R.load(row['baseline_pair']);q=samples[row['query']];r=samples[row['reference']]
  for key in b:
   if not key.startswith(('point_corroborated_','sequence_corroborated_')):np.testing.assert_array_equal(z[key],b[key])
  for mode in ['before','after']:
   pp,ss=evidence(z,q,r,'quality_'+mode);np.testing.assert_array_equal(pp,z['point_corroborated_'+mode]);np.testing.assert_array_equal(ss,z['sequence_corroborated_'+mode])
  pb=z['point_corroborated_before'];pa=z['point_corroborated_after'];sb=z['sequence_corroborated_before'];sa=z['sequence_corroborated_after'];np.testing.assert_array_equal(pb,b['point_corroborated_after']);np.testing.assert_array_equal(sb,b['sequence_corroborated_after']);assert not (pb&~pa).any() and not(sb&~sa).any();assert not(pa&~z['point_ok']).any() and not(sa&~z['sequence_ok']).any()
  c=Counter(pairs=1,domain=int(z['domain'].sum()),angle_domain=int(z['domain'][:,:,:15].sum()),raw_point_angles=int(z['point_ok'][:,:,:15].sum()),raw_sequence_angles=int(z['sequence_ok'][:,:,:15].sum()),point_before=int(pb.sum()),point_after=int(pa.sum()),point_added=int((pa&~pb).sum()),point_withdrawn=int((pb&~pa).sum()),sequence_before=int(sb.sum()),sequence_after=int(sa.sum()),sequence_added=int((sa&~sb).sum()),sequence_withdrawn=int((sb&~sa).sum()));assert dict(c)==row['counts'];total.update(c)
  # Remove a corroborated source angle/hand on the same observed path; never create positive evidence.
  cells=np.argwhere(pa[:,:,:15])
  if len(cells):
   j,h,a=cells[len(cells)//2];lo=int(np.floor(z['phase_old'][j]));hi=int(np.ceil(z['phase_old'][j]));mut=dict(q,quality_after=q['quality_after'].copy());mut['quality_after'][lo:hi+1,h,a]=False;pp,ss=evidence(z,mut,r,'quality_after');assert not pp[j,h,a] and not(pp&~pa).any() and not(ss&~sa).any();other=np.ones((2,23),bool);other[h,a]=False;np.testing.assert_array_equal(pp[:,other],pa[:,other]);np.testing.assert_array_equal(ss[:,other],sa[:,other]);tests['single_angle_no_crosschannel_loss_or_false_evidence']+=1
   mut=dict(q,quality_after=q['quality_after'].copy());mut['quality_after'][lo:hi+1,h]=False;pp,ss=evidence(z,mut,r,'quality_after');assert not pp[j,h].any();np.testing.assert_array_equal(pp[:,1-h],pa[:,1-h]);np.testing.assert_array_equal(ss[:,1-h],sa[:,1-h]);assert not(pp&~pa).any() and not(ss&~sa).any();tests['onehand_evidence_absence_keeps_otherhand_domain_and_raw_values']+=1
  if (k+1)%50==0:print('CORE_INTEGRATION_VERIFIED',k+1,200,flush=True)
 assert dict(total)==res['counts'];R.check(f['protected']);R.write(R.U/'INTEGRATION_VERIFIED.json',dict(at=R.now(),status='PASS300_ORIGINAL_INPUTS_200_IDENTICAL_PATHS_SOURCE_ADDITION',inputs=300,pairs=200,input_counts=dict(inputtot),pair_counts=dict(total),negative_controls=dict(tests),result_sha256=R.sha(R.U/'PAIRS_DONE.json'),prepare_sha256=R.sha(R.U/'PREPARED.json'),verifier_sha256=R.sha(__file__),method='SameCodex separate source-window arithmetic and exact prior array equality; no recognition or alignment rerun. Original negative controls preserved plus additional evidence-withdrawal controls. Not human anatomical groundtruth or semantic normality certification.'));print('CORE_INTEGRATION_PASS',dict(total),dict(tests),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:
  p=R.U/'VERIFY_INTEGRATION_FIRST_ERROR.txt'
  if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
  raise
