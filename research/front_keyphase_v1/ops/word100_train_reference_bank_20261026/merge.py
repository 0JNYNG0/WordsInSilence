"""Complete the original TRAIN bank with separate image evidence, never replacing coordinates."""
import prepare as R
import numpy as np,sys,traceback
from collections import Counter
Z=R.O/'word100_core_image_evidence_20261026'
IMMUTABLE=['xy_px','usable','timestamp_us','frame_id','features','present','core','values','available','quality_eligible','scale','yolo_conflict']

def run():
 f=R.read(R.U/'FROZEN.json');im=R.read(R.U/'VERIFIED_IMAGES.json');assert im['clips']==599;R.check(f['pins']);R.check(f['protected']);R.check(im['pins'])
 (R.U/'inputs').mkdir();rows=[];pins={};total=Counter()
 for row in f['catalog']:
  b=R.load(row['input']);z={k:v.copy() for k,v in b.items()};reused=row['evidence_complete']
  if reused:p=row['input']
  else:
   o=R.load(R.U/'outputs'/(row['clip_id']+'.npz'));av=b['available'][:,:,:15];s=np.zeros_like(av,dtype=np.uint8);s[av]=1;req=o['requested'][...,None]&av;s[req]=2;s[req&~o['mp_identity'][...,None]]=3;s[req&(o['mp_unresolved']|o['pixel_unresolved'])]=4;s[req&o['mp_identity_conflict'][...,None]]=3;s[av&o['pixel_eligible']]=5
   z['state_before']=b['state_after'];z['state_after']=np.where(o['requested'][...,None],s,b['state_after']).astype(np.uint8);z['quality_before']=b['quality_after'];z['quality_after']=b['quality_after'].copy();z['quality_after'][:,:,:15]=z['state_after']==5;z['requested_before']=b['requested_current'];z['requested_current']=b['requested_current']|o['requested'];p=str(R.U/'inputs'/(row['clip_id']+'.npz'));np.savez_compressed(p,**z)
  for key in IMMUTABLE:np.testing.assert_array_equal(z[key],b[key])
  assert not (b['quality_after']&~z['quality_after']).any();assert not(z['core'][:,None]&z['available'][:,:,:15].any(-1)&~z['requested_current']).any()
  c=Counter(clips=1,reused=int(reused),new=int(not reused),frames=len(z['core']),domain_angles=int(z['core'].sum())*30,raw_angles=int(z['available'][z['core'],:,:15].sum()),corroborated_angles=int(z['quality_after'][z['core'],:,:15].sum()),added_angles=int((z['quality_after']&~b['quality_after'])[z['core'],:,:15].sum()),withdrawn_angles=int((b['quality_after']&~z['quality_after'])[z['core'],:,:15].sum()));total.update(c);rows.append(dict(row,output=p,counts=dict(c)));pins[p]=R.sha(p)
 R.check(f['protected']);R.write(R.U/'PREPARED.json',dict(at=R.now(),rows=rows,outputs=pins,counts=dict(total),frozen_sha256=R.sha(R.U/'FROZEN.json'),images_sha256=R.sha(R.U/'VERIFIED_IMAGES.json'),producer_sha256=R.sha(__file__),training=0,operating_changed=False));print('BANK899_CURRENT_IMAGE_PREPARED',dict(total),flush=True)

def verify():
 f=R.read(R.U/'FROZEN.json');p=R.read(R.U/'PREPARED.json');assert p['frozen_sha256']==R.sha(R.U/'FROZEN.json') and p['images_sha256']==R.sha(R.U/'VERIFIED_IMAGES.json');R.check(p['outputs']);R.check(f['pins']);R.check(f['protected']);total=Counter()
 for row in p['rows']:
  b=R.load(row['input']);z=R.load(row['output']);reused=row['evidence_complete']
  for k in IMMUTABLE:np.testing.assert_array_equal(z[k],b[k])
  if reused:assert row['output']==row['input']
  else:
   o=R.load(R.U/'outputs'/(row['clip_id']+'.npz'));a=b['available'][:,:,:15];expected=b['state_after'].copy()
   # Verify each precedence partition separately from the producer's successive assignments.
   good=a&o['pixel_eligible'];identity=a&o['requested'][...,None]&(~o['mp_identity'][...,None]|o['mp_identity_conflict'][...,None]);uncertain=a&o['requested'][...,None]&(o['mp_unresolved']|o['pixel_unresolved'])&~o['mp_identity_conflict'][...,None]
   for h in (0,1):
    use=o['requested'][:,h];e=np.where(~a[:,h],0,np.where(good[:,h],5,np.where(o['mp_identity_conflict'][:,h,None],3,np.where(uncertain[:,h],4,np.where(identity[:,h],3,2))))).astype(np.uint8);expected[use,h]=e[use]
   np.testing.assert_array_equal(z['state_after'],expected);np.testing.assert_array_equal(z['state_before'],b['state_after']);np.testing.assert_array_equal(z['quality_before'],b['quality_after']);np.testing.assert_array_equal(z['quality_after'][:,:,:15],expected==5);np.testing.assert_array_equal(z['requested_current'],b['requested_current']|o['requested']);assert not(b['requested_current']&o['requested']).any()
  assert not z['quality_after'][:,:,15:].any() and not(z['quality_after']&~z['available']).any() and not(b['quality_after']&~z['quality_after']).any()
  assert not(z['core'][:,None]&z['available'][:,:,:15].any(-1)&~z['requested_current']).any()
  core=z['core'];c=Counter(clips=1,reused=int(reused),new=int(not reused),frames=len(core),domain_angles=int(core.sum())*30,raw_angles=int(z['available'][core,:,:15].sum()),corroborated_angles=int(z['quality_after'][core,:,:15].sum()),added_angles=int((z['quality_after']&~b['quality_after'])[core,:,:15].sum()),withdrawn_angles=int((b['quality_after']&~z['quality_after'])[core,:,:15].sum()));assert dict(c)==row['counts'];total.update(c)
 assert dict(total)==p['counts'] and total['clips']==899 and total['new']==599 and total['reused']==300;R.check(f['protected']);R.write(R.U/'INPUTS_VERIFIED.json',dict(at=R.now(),status='PASS899_UNCHANGED_ORIGINALS_SEPARATE_IMAGE_EVIDENCE',counts=dict(total),prepared_sha256=R.sha(R.U/'PREPARED.json'),verifier_sha256=R.sha(__file__),independence='SameCodex separate state-partition arithmetic and bytearray comparisons, prior raw-channel/base checks reused; not human anatomical groundtruth.'));print('BANK899_INPUTS_VERIFIED',dict(total),flush=True)
if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException:
  p=R.U/('MERGE_'+sys.argv[1].upper()+'_FIRST_ERROR.txt')
  if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
  raise
