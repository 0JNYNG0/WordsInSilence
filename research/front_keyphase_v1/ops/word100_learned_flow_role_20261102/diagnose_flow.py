"""Saved-output comparison; diagnoses do not change acceptance gates."""
from verify_flow import *
from collections import Counter

def main():
    verified=read(U/'VERIFIED.json');r=read(OUT/'evaluation.json')
    old=read(Path('W:/WordsInSilence_derived/word100_content_role_bridge_20261102/evaluations/role_only.json'))
    before={s['key']:s for s in old['sequences']};changes=[]
    for s in r['sequences']:
        b=before[s['key']]
        if s['status']!=b['status']:
            changes.append(dict(key=s['key'],kind=s['kind'],before=b['status'],after=s['status'],
                content_errors=s['evaluation_content_errors'],qualified_errors=s['evaluation_qualified_errors'],
                previous_core_unconfirmed=s['previous_reviewed_core_unconfirmed'],
                unexpected_frames=s['unexpected_action_frames']))
    role_rest={}
    for tag,report in [('saved_control',old),('candidate',r)]:
        rest=[p for p in report['role_points'] if p['kind'] in NORMAL and p['y']==2]
        role_rest[tag]=dict(expected_rest=len(rest),predicted_role_counts=dict(Counter(p['pred'] for p in rest)),
            note='0=nonmain context,1=main,2=rest. Confusing rest with nonmain context differs from treating it as main. Prespecified exact-rest gate is unchanged.')
    unique=sorted({(p['clip_id'],p['source_frame']) for p in r['main_eligibility_losses']})
    normal=[s for s in r['sequences'] if s['kind'] in NORMAL]
    save(U/'CAUSE_AUDIT.json',dict(at=datetime.now(timezone(timedelta(hours=9))).isoformat(),changes=changes,
        normal_status_counts=dict(Counter(s['status'] for s in normal)),rest_diagnostic=role_rest,
        unique_excluded_main_source_points=unique,prespecified_gate_status=verified['status'],
        no_gate_revision=True,no_old_inference_or_rescoring=True,whole_goal_complete=False))
    print('Changed statuses',[(x['key'],x['before'],x['after']) for x in changes]);print(role_rest)

if __name__=='__main__':main()
