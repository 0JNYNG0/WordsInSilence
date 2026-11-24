"""Create an isolated minimal candidate; never edit the saved C extractor."""
import json,hashlib,shutil
from pathlib import Path
from datetime import datetime,timezone,timedelta
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2]
OLD=O/'word100_hand_assignment_joint_20261016'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def write(p,v):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
 for name in ['SCOPE.json','POINT_CONTRACT.json']:
  assert not (U/name).exists();shutil.copy2(OLD/name,U/name)
 src=(OLD/'assignment.py').read_text('utf-8')
 marker='        # Enumerate whole-frame assignments, including absence. Never force 2 hands.'
 assert src.count(marker)==1
 insert='''        # A weak handedness-only edge cannot manufacture a second identity
        # when BOTH current pose preference and a bounded historical link point
        # to the already independently supported hand. This withholds the weak
        # hypothesis; it does not relabel it or recover missing landmarks.
        # Existing gates/ratio are retained, with no word/frame exceptions.
        label_conflicts = []
        if n == 2:
            for i in range(n):
                for side in (0, 1):
                    edge = evidence.get((i, side))
                    if edge is None or edge[1] != 'label_without_close_pose':
                        continue
                    other = 1-side
                    anchor = 1-i
                    anchor_edge = evidence.get((anchor, other))
                    if anchor_edge is None or anchor_edge[1] != 'temporal_and_pose':
                        continue
                    # Require present pose evidence for both sides. Missing
                    # pose alone is not evidence of a duplicate.
                    pose_opposes_label = np.isfinite(pc[i]).all() and pc[i,other] < PARAMS['ambiguity_ratio']*pc[i,side]
                    track_opposes_label = clear_side(tc,i,other,PARAMS['track_gate'])
                    if not (pose_opposes_label and track_opposes_label):
                        continue
                    del evidence[i,side]
                    blocked.add(i)  # no label-based bootstrap in this frame
                    label_conflicts.append({'raw_index':i,'withheld_side':side,
                        'supported_side':other,'anchor_raw_index':anchor,
                        'action':'withhold_weak_label_only_edge',
                        'duplicate_identity_is_ground_truth':False})

'''
 src=src.replace(marker,insert+marker)
 marker2='        joint_evidence = []'
 src=src.replace(marker2,"        for conflict in label_conflicts:\n            i = conflict['raw_index']\n            if result[i] is None:\n                reasons[i] = 'label_conflict_with_supported_hand'\n"+marker2)
 src=src.replace("'coordinates_generated':False}","'label_conflicts':label_conflicts,'coordinates_generated':False}")
 assert not (U/'assignment.py').exists();(U/'assignment.py').write_text(src,'utf-8')
 ctx=read(O/'word100_position_state_audit_20261018/CONTEXT.json')
 pins=dict(ctx['pins'])
 for p in [OLD/'assignment.py',OLD/'SCOPE.json',OLD/'POINT_CONTRACT.json',U/'assignment.py',U/'SCOPE.json',U/'POINT_CONTRACT.json',U/'prepare.py',U/'run.py',U/'verify.py']:
  pins[str(p)]=sha(p)
 for r in ctx['rows']:
  assert r['role']=='train_candidate' and 1<=r['signer_id']<=9
  for view in ('masked','sensor'):
   p=Path(r['views'][view]['assignment']).with_name('trace.json');pins[str(p)]=sha(p)
 for d in ['before','results','evidence']:(U/d).mkdir(exist_ok=False)
 for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:shutil.copy2(p,U/'before'/p.name)
 write(U/'CONTEXT.json',{'at':datetime.now(timezone(timedelta(hours=9))).isoformat(),
 'authorization':'중복 후보/라벨 단독 배정 공통 보완 및 공개 검증. 정상 교차와 기존 정상 배정 보존. C 유지. 사용자/멘토자료 금지. 학습은 결과 확인 후.',
 'scope':'Public TRAIN899, full native detector and cropped sensor streams; no DEV/heldout/user data. masked means native detector stream here, not masked output landmarks.',
 'rows':ctx['rows'],'pins':pins,'baseline':str(OLD/'assignment.py'),
 'frozen_rule':'Withhold only a label_without_close_pose edge when other raw candidate has temporal_and_pose for opposite identity, present pose distances for both sides prefer that opposite identity by existing 0.7 ratio, and own candidate has clear_side opposite temporal distance within existing 0.4 gate. Block its bootstrap this frame. Never assign remaining side by elimination or screen-half.',
 'thresholds':'No new/widened thresholds. Original SCOPE/POINT_CONTRACT byte-identical.',
 'preimplementation_survey':{'frames_per_view':113181,'native_label_only':13,'sensor_label_only':444,'native_two_candidate':9,'sensor_two_candidate':149,'native_strong_other_anchor':6,'sensor_strong_other_anchor':44,'rule_matching_existing_frames':1},
 'not_blind_evaluation':'Known public TRAIN defect motivates this correction; all899 replay is a regression/contract audit, not unseen identity accuracy or classifier performance.',
 'success':['Exact baseline assignment reproduction on every saved frame.', 'Only the explicit weak edge is directly withheld; track propagation is audited, all changed intervals visually reviewed.', 'No new correct-hand suppression in reviewed changes; unresolved evidence retained and disclosed.', 'Genuine-crossing sequence/contract cases and all unaffected saved assignments retained.', 'Independent scalar guard recomputation, source hash checks, raw-order and causal tests pass.'],
 'failure_policy':'Preserve first evidence; no threshold sweep or silent excluded clips. Any new material error => candidate withheld, report to user.',
 'training':0,'classifier_inference':0,'production_changed':False,'user_mentor_dev_heldout_reads':0,
 'implementation_and_verification':'Codex local; independent formula/script verification by same operator, no external reviewer.'})
 print('Frozen isolated assignment candidate; training not permitted.')
if __name__=='__main__':main()
