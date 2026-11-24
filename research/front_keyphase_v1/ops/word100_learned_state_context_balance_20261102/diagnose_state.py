"""Saved predictions only: expose regression and coverage, never retune."""
from collections import Counter
from state_model import *

def main():
    assert not (T/'DIAGNOSIS.json').exists()
    oldreport=read(T/'evaluations/old_phase.json');cur=read(T/'evaluations/current_42.json');ctx=read(T/'evaluations/anchored_context_42.json')
    key=lambda p:(p['key'],p['frame']);oldpts={key(p):p for p in oldreport['points']};cp={key(p):p for p in cur['points']}
    changes=[]
    for p in ctx['points']:
        q=cp[key(p)];before=oldpts[key(p)]
        if p['pred']!=q['pred'] or p['pred']!=p['y']:
            changes.append(dict(case=p['key'],frame=p['frame'],source_frame=p['source_frame'],truth=p['y'],old=before['pred'],current=q['pred'],context=p['pred'],current_conf=max(q['probability']),context_conf=max(p['probability'])))
    examples=[]
    for s in ctx['sequences']:
        if s['kind'] not in ['native','actual_reverse_main','action_cut']:continue
        examples.append(dict(key=s['key'],status=s['status'],precedences=s['precedences'],missing=s['missing_groups'],
          states=[dict(group=z['group'],spans=[[run[0],run[-1]] for run in z['runs']],isolated=z['isolated_candidate_frames']) for z in s['states']],
          rest_frames=len(s['rest_observed_frames']),uncertain_frames=len(s['uncertain_frames']),total_frames=s['comparison_frames'],
          reviewed_points=[dict(frame=p['frame'],source=p['source_frame'],y=p['y'],pred=p['pred'],confidence=max(p['probability'])) for p in ctx['points'] if p['key']==s['key']]))
    exclusive(T/'DIAGNOSIS.json',dict(at=clock(),changed_or_wrong_points=changes,examples=examples,old_order_coarse_source=str(S/'evaluations/old_phase.json'),
      no_new_predictions=True,no_new_training=True,no_threshold_or_candidate_selection=True,
      scope='A sampled-state accuracy increase is not proof that every confident interval identifies the right guide phase. Unlabelled intervals remain unvalidated.'))
    print('Saved diagnosis',len(changes),'changed/wrong labelledpoints;',len(examples),'native/local/cut examples')

if __name__=='__main__':main()
