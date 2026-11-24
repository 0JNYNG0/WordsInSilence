"""Contract regression checks, not evidence of visual model accuracy."""
from pathlib import Path
import hashlib
import json
import numpy as np
from process_runtime_contract import qualify, known_seam_diagnostic

U = Path(__file__).resolve().parent

def main():
    dest = U / 'RUNTIME_CONTRACT_CHECKED.json'
    assert not dest.exists()
    point = [[.95, .1], [.9, .1], [.1, .95]]
    good = qualify(.9, point, [True]*3, [.9])
    assert good['qualified_display'] == 1
    assert good['witnessed_chain']['output_indices'] == [0, 1, 2]
    assert good['complete_action_verified'] is False
    assert good['selected_span_truth'] is None
    assert good['rgb_context_may_extend_beyond_nominal_span'] is True
    assert good['learner_error'] is None and good['whole_word_correctness'] is None
    assert qualify(.9, point, [True]*3, [.1])['qualified_display'] is None
    assert qualify(.1, point, [True]*3, [.9])['qualified_display'] is None
    assert qualify(.9, point, [True, False, True], [])['qualified_display'] is None
    assert qualify(.99, point, [False]*3, [])['status'] == 'INSUFFICIENT_OBSERVATION'
    short = qualify(.9, [[.9,.1],[.1,.9]], [True,True], [.9])
    assert short['qualified_display'] == 1  # no minimum dwell removing short events
    assert short['witnessed_chain']['output_indices'] == [0,1]
    privileged = known_seam_diagnostic(good, [1])
    assert privileged['qualified_display'] is None
    assert good['qualified_display'] == 1  # no oracle silently changing runtime
    assert privileged['same_runtime_candidates_and_scores_reused']
    for p in [np.nan, -1, 2]:
        try:
            qualify(p, point, [True]*3, [.9])
        except AssertionError:
            pass
        else:
            raise AssertionError('Invalid probability accepted')
    out = dict(status='PASS_PURE_CONTRACT_NOT_MODEL_ACCURACY',
               scenarios=10, actual_model_forwards=0,
               observations=['short event retained', 'missing image breaks chain',
                             'local and global process required', 'net motion not substituted',
                             'oracle diagnostic never replaces runtime',
                             'no exact boundary/whole-word/error certification'],
               code_sha256=hashlib.sha256((U/'process_runtime_contract.py').read_bytes()).hexdigest())
    dest.write_text(json.dumps(out, indent=2)+'\n', encoding='utf8')
    print(out['status'])

if __name__ == '__main__':
    main()
