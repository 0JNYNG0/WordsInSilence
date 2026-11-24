from probe_front import *
import re

def main():
 files=[O/'word100_shared_motion_learning_20261109/READY.json',O/'word100_motion_components_20261109/READY.json',O/'word100_common_action_learning_20261110/READY.json',O/'word100_broad_factorized_learning_20261110/ADDITIONAL_LABELS.json',O/'basic_demo_feedback_20261115/INPUTS.json',O/'basic_demo_feedback_20261115/SOURCE_TRUTH.json',O/'basic_demo_feedback_20261115/acquisition_motion_followup/MOTION_TRUTH.json',O/'word100_finger_repair_20261115/INPUTS.json',O/'word100_clause_finger_diagnostics_20261114/FINGER_INPUTS.json',V/'REVIEW_INPUTS.json',V/'shape_continuation/SOURCE_TRUTH.json',V/'shape_source_repair/SOURCE_TRUTH.json',V/'plane/SOURCE_TRUTH.json',V/'target_region/SOURCE_TRUTH.json',G/'apple_v1/INPUTS.json',G/'apple_v1/SOURCE_TRUTH.json',O/'demo_handshape_learning_20261113/INPUTS.json']
 files.append(N/'geometry/FROZEN.json');rows=[]
 for p in files:
  assert p.is_file(),p
  content=p.read_text(encoding='utf-8-sig');ids=sorted(set(re.findall(r'NIA_SL_(?:WORD|SEN)\d+_REAL\d+_[FLRUD]',content)))
  for cid in ids:require_front(dict(id=cid))
  rows.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),front_clip_ids=len(ids),all_front=True,labels_without_ids=not bool(ids)))
 sources=read(G/'apple_v1/INPUTS.json')['rows']+read(O/'demo_handshape_learning_20261113/INPUTS.json')['rows'];hashes={r['source_sha256']:r for r in sources}
 surface=read(G/'apple_v1/SURFACE_FROZEN.json')['rows']
 for r in surface:
  source=hashes[r['source']];require_front(source);assert source['role']=='fit'
 old=read(M/'web/ADMISSION.json');preserved=[dict(path=p,ok=hashlib.sha256(Path(p).read_bytes()).hexdigest()==h)for p,h in old['pins'].items()];assert all(r['ok']for r in preserved)
 result=read(N/'candidate/RESULT.json');assert all(result[k]==0 for k in ['normal_failed','shape_new_wrong','shape_lost_correct','distinction_failed','temporal_failed'])
 dist=read(N/'candidate/distinctions/RESULT.json');required=[]
 for r in dist['rows']:
  expected=(('left_'if r['kind'].startswith('flex_L')else'right_')+'shape')if r['kind'].startswith('flex_')else None
  if expected:required.append(dict(person=r['person'],kind=r['kind'],expected=expected,passed=expected in r['differences']))
  elif r['kind']=='swap_planes':required.append(dict(person=r['person'],kind=r['kind'],passed={'left_orientation','right_orientation'}.issubset(r['differences'])))
 assert all(r['passed']for r in required)
 # A transplanted fist changes shape, but does not supply an independent palm
 # orientation label. Do not require old incidental orientation warnings.
 write(N/'AUDIT.json',dict(passed=True,source_manifests=rows,surface_fit_labels=len(surface),surface_fit_hashes_linked=True,old_pins=preserved,required_error_localization=required,incidental_old_orientation_changes=result['error_loss'],interpretation='All shape/plane errors with explicit synthetic truth retained. Seven old extra orientation warnings on shape/translation edits are not labelled orientation truth.',limits='Known public development sources, not fresh learner accuracy. Third-party RTMW pretraining is unchanged and not claimed frontal-only.'))
 print('AUDIT',len(rows),'SURFACE_FIT',len(surface),'PRESERVED',len(preserved))
if __name__=='__main__':main()
