from pathlib import Path
import json,hashlib,shutil
U=Path(__file__).resolve().parent; O=U.parent; D=O/'word100_action_feedback_demo_20261108'; ROOT=O.parents[2]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
if __name__=='__main__':
 assert not (U/'FROZEN.json').exists()
 (U/'before').mkdir(exist_ok=True)
 for p in [D/'consumer.py',D/'evidence.py',D/'app.js',D/'style.css',D/'CARDS.json',ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:
  shutil.copyfile(p,U/'before'/p.name)
 f=read(D/'FROZEN.json'); pins=dict(f['protected'])
 for p in [D/'runs/42/model.pt',O/'word100_rtmw_joint_learning_20261108/feedback_viewfix/feedback_runs/pose_transfer_42/model.pt',O/'word100_rtmw_joint_learning_20261108/tcn_runs/finetune_C_20261013/model.pt']:
  pins[str(p)]=sha(p)
 write(U/'FROZEN.json',dict(goal='Positive handshape support and signed body-relative motion; absence of a detected error is not evidence of agreement',
  authorization='2026-11-09 user: proceed with all proposed improvements. Public-only; existing C/T3-F preserved.',
  training=False,private_used=False,new_download=False,protected=pins,
  positive=dict(threshold=.65,min_evidence_ms=60,min_available_fraction=.8,mean_support_threshold=.65,contradiction_requires_expected_low=True),
  direction=dict(phase_support=.4,min_frames=6,min_available_fraction=.8,end_fraction=.2,noise_multiple=3.,min_displacement_forearm_units=.1,max_gap_ms=120),
  fit='Existing public ID01-09 only for implementation diagnostics; no threshold search',
  evaluation='Fixed existing DEV ID11-13, all four demo words; existing 12 audit predictions reused without rerunning inference; all conditions reported',
  controls=['normal','speed075','speed125','global_translation_scale','short_hand_gap','long_hand_gap','right_closed_hand_main','main_reversed','closed_hand_and_reversed'],
  deployment_gate='No new direction/shape contradiction on normal public controls. Changed error controls must not be green. Unknown and semantic scope must be explicit. Never claim educational-error accuracy.',
  limitation='Same Codex engineering verification, previously exposed DEV, synthetic perturbations, no independent learner-error gold. No whole-sign pass or similarity percentage.'))
 # The shared primitive is configured by the existing guide meaning, never by a class-score exception.
 write(U/'GUIDE_CONTRACTS.json',dict(provenance='Existing CARDS.json brush_arm description and canonical guide: open right hand follows left forearm toward wrist. Optional endpoint hold does not waive direction.',
  source_sha256=sha(D/'CARDS.json'),contracts={'brush_arm':dict(kind='along_forearm',moving_hand='R',anchor_arm='L',expected_sign=1,phase_groups=[3,4]),'reach_wrist':dict(kind='along_forearm',moving_hand='R',anchor_arm='L',expected_sign=1,phase_groups=[3,4])}))
 print(json.dumps({'phase':f['phase'],'protected':len(pins)}))
