"""Serialize root's actual image review; labels were not calculated from coordinates."""
from pathlib import Path
from datetime import datetime, timezone
import json
import hashlib

U=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    out=U/'REVIEW_FIRST_1133.json';assert not out.exists()
    p=U/'FIRST_RUNTIME_INTERVAL_REVIEW_MANIFEST.json'
    source=json.loads(p.read_text(encoding='utf8'))
    notes={
      1:(0,'Within native44-56 the right hand approaches the left proximal arm and rests/slides near the elbow/proximal forearm. The end image still shows the palm beside the proximal forearm, distinct from the left wrist/fist. A connected proximal-to-distal forearm/wrist traversal is not contained in these blue frames; later gray context cannot supply it.'),
      2:(0,'Within43-53 the right hand is raised to the left upper/proximal arm. By53 it remains above the elbow/proximal forearm while the left wrist is visibly forward/lower. The later distal traversal is not present inside this interval. This is partial observed scope, not a signer error.'),
      3:(0,'Within49-57 the right hand stays near the proximal left arm with a small initial movement. The blue endpoint still lies above the proximal forearm, separated from the left wrist. No connected transition to the distal forearm/wrist is visibly included.'),
      4:(0,'Within71-82 the hand approaches and remains near the proximal left arm/elbow. At82 the palm is still beside that upper/proximal location while the forward left fist/wrist is lower. The nominal interval ends before the queried proximal-to-distal traversal is shown.'),
      5:(0,'Within42-52 the right hand lifts and comes over the left proximal arm. The last blue frame and enlarged original endpoint retain that proximal relationship; the left wrist is forward/lower. The interval does not contain the connected travel to the distal forearm/wrist.'),
      6:(None,'Within39-48 an approach and early sweep are visible. At the final blue image the foreshortened left forearm and overlapping right palm make the precise proximal-versus-distal relationship ambiguous in projection. I cannot reliably affirm or deny the connected queried extent inside this short interval. Gray49-52 is only context and does not turn this nominal target positive.'),
      7:(0,'Within41-50 the right hand moves upward onto the left upper/proximal arm. The endpoint remains beside the elbow/proximal forearm, with the left wrist/fist separately visible forward/lower. No proximal-to-distal traversal is contained in these nominal frames.'),
      8:(0,'Within66-78 the right hand rises and comes to the proximal left arm. At78 it remains near the upper arm/elbow while the wrist is visibly farther down the projected forearm. The queried travel to the distal forearm/wrist is outside the nominal evidence shown.'),
      9:(0,'Within41-53 the right hand rises toward the upper/proximal left arm and begins to move. The final blue frame still places the palm near the proximal arm, above the left forearm and wrist. Later gray frames show further motion but cannot establish a connected distal traversal inside41-53.')
    }
    rows=[]
    for r in source['rows']:
        if r['word_id']!='WORD1133':continue
        label,reason=notes[r['person']]
        pages=[dict(path=x['path'],sha256=x['sha256']) for x in r['review_pages']]
        assert len(pages)==1
        for page in pages:assert sha(page['path'])==page['sha256']
        end=r['nominal_source_frames'][-1]
        endpoint=Path('W:/WordsInSilence_derived/word100_action_completion_supervision_20261102/frames')/f"{r['clip_id']}_f{end:04d}.jpg"
        rows.append(dict(review_id=r['review_id'],case_id=r['case_id'],
            source_frames=r['nominal_source_frames'],source_timestamps_us=r['nominal_source_timestamps_us'],
            observed_process_evidence=label,reason=reason,reviewed_pages=pages,
            additional_reviewed_endpoint=dict(path=str(endpoint),sha256=sha(endpoint)),
            source_registered_id=r['person'],all_nominal_frames_visually_reviewed=True,
            outside_context_is_not_missing_process_evidence=True,
            exact_skin_contact_or_required_sweep_extent_not_asserted=True,
            whole_word_correctness=None,learner_error=None,required_repetitions=None))
    assert len(rows)==9
    result=dict(status='DIRECT_NOMINAL_PHYSICAL_REVIEW_COMPLETE',
        at=datetime.now(timezone.utc).isoformat(),reviewer='root Codex actual contact sheets plus enlarged stored original endpoints',
        manifest_sha256=sha(p),rows=rows,labels={'0':8,'1':0,'unknown':1},
        independent_human_expert=False,new_model_inference=0,new_training=0,
        labels_not_derived_from_old_model_probability=True,whole_goal_complete=False)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print('Root nominal1133 review recorded:8partial1unknown; whole-video positives retained separately')

if __name__=='__main__':main()
