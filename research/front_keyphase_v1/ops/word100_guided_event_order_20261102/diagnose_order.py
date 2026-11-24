"""Explain the frozen comparison from saved outputs only; no inference."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone, timedelta
import json

U = Path(__file__).resolve().parent
OUT = Path('W:/WordsInSilence_derived/word100_guided_event_order_20261102')
FLOW = Path('W:/WordsInSilence_derived/word100_learned_flow_role_20261102/evaluation.json')
NORMAL = {'native', 'reencoded_normal', 'speed_0.8', 'speed_1.25', 'trim_initial_rest'}

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def main():
    verified = read(U / 'VERIFIED.json')
    old = {s['key']: s for s in read(FLOW)['sequences']}
    summaries = {}
    for tag in read(U / 'FROZEN.json')['conditions']:
        report = read(OUT / 'evaluations' / (tag + '.json'))
        seq = report['sequences']
        assert len(seq) == len(old) == 88
        normal = [s for s in seq if s['kind'] in NORMAL]
        summaries[tag] = dict(
            normal_status_counts=dict(Counter(s['status'] for s in normal)),
            normal_physical_status_counts=dict(Counter(s['physical_order_status'] for s in normal)),
            changes_from_saved_flow=[dict(key=s['key'], kind=s['kind'], before=old[s['key']]['status'], after=s['status'])
                                     for s in seq if s['status'] != old[s['key']]['status']],
            learned_order_vetoes=[dict(key=s['key'], physical=s['physical_order_status'], final=s['status'], edges=s['learned_order'])
                                  for s in seq if s['physical_order_status'] != s['status']],
            reverse_main=[dict(key=s['key'], status=s['status'], physical=s['physical_order_status'], edges=s['learned_order'],
                               physical_edges=s['precedences'], unexpected=s['unexpected_action_frames'])
                          for s in seq if s['kind'] == 'actual_reverse_main'],
            native=[dict(key=s['key'], status=s['status'], physical=s['physical_order_status'], edges=s['learned_order'],
                         unexpected=s['unexpected_action_frames'], raw_errors=s['evaluation_content_errors'],
                         qualified_errors=s['evaluation_qualified_errors'], core_unconfirmed=s['previous_reviewed_core_unconfirmed'])
                    for s in seq if s['kind'] == 'native'],
            unique_excluded_main_source_points=sorted({(p['clip_id'], p['source_frame']) for p in report['main_eligibility_losses']}),
            rest_diagnostic=dict(Counter(str(p['pred']) for p in report['role_points'] if p['kind'] in NORMAL and p['y'] == 2)),
        )
    result = dict(at=datetime.now(timezone(timedelta(hours=9))).isoformat(), conditions=summaries,
                  frozen_gate_status=verified['status'], no_gate_revision=True, old_model_reexecution=0,
                  existing_scores_recomputed=False, whole_goal_complete=False,
                  interpretation='Counts and explanations of stored decisions; weighted order scores are not calibrated whole-sign correctness. Same Codex arithmetic, not expert KSL ground truth.')
    with (U / 'CAUSE_AUDIT.json').open('x', encoding='utf8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: {q: v[q] for q in ['normal_status_counts', 'normal_physical_status_counts', 'changes_from_saved_flow']}
                      for k, v in summaries.items()}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
