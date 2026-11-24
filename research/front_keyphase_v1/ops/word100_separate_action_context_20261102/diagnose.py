from model import *
from collections import Counter
NORMAL=['native','reencoded_normal','speed_0.8','speed_1.25','trim_initial_rest']

def main():
    assert not (F/'DIAGNOSIS.json').exists();verified=read(F/'VERIFIED.json');reports={}
    old=read(D/'evaluations/episode_42.json');oldpoints={(p['key'],p['frame']):p for p in old['points']}
    for name in verified['conditions']:
        r=read(F/'evaluations'/f'{name}.json');raw={(p['key'],p['frame']):p for p in r['raw_points']};cmp=[]
        for p in r['points']:
            if p['kind'] not in NORMAL or p['y']>=8:continue
            k=(p['key'],p['frame']);a=raw[k];b=oldpoints[k];assert a['y']==b['y']==p['y']
            cmp.append(dict(key=p['key'],frame=p['frame'],truth=p['y'],before=b['pred'],raw=a['pred'],effective=p['pred'],
              old_correct_lost=b['pred']==b['y'] and p['pred']!=p['y'],old_error_fixed=b['pred']!=b['y'] and p['pred']==p['y'],
              correct_content_lost_by_role=a['pred']==a['y'] and p['pred']!=p['y']))
        confusion=Counter((p['y'],p['pred']) for p in r['points'] if p['kind'] in NORMAL and p['y']<8 and p['y']!=p['pred'])
        sequences=[dict(key=q['key'],kind=q['kind'],word=q['word'],status=q['status'],edges=q['precedences'],unexpected=q['unexpected_action_frames'],core_unconfirmed=q['previous_reviewed_core_unconfirmed'],action_errors=q['evaluation_action_errors'],
          confirmed_context=q['context_frames'],role_uncertain=q['role_uncertain_frames'],states=q['states'],raw_states=q['raw_action_states']) for q in r['sequences'] if q['kind'] in NORMAL]
        reports[name]=dict(old_correct_lost=sum(p['old_correct_lost'] for p in cmp),old_error_fixed=sum(p['old_error_fixed'] for p in cmp),correct_content_lost_by_role=sum(p['correct_content_lost_by_role'] for p in cmp),
          changed=[p for p in cmp if p['before']!=p['effective'] or p['raw']!=p['effective']],confusions=[dict(truth=a,prediction=b,count=n) for (a,b),n in sorted(confusion.items())],normal_sequences=sequences)
    exclusive(F/'DIAGNOSIS.json',dict(at=clock(),scope='Saved outputs only; no inference, relabeling, revised old score or candidate selection. Separate content errors from role-induced losses.',reports=reports,whole_goal_complete=False))
    print({k:{n:v[n] for n in ['old_correct_lost','old_error_fixed','correct_content_lost_by_role']} for k,v in reports.items()})
if __name__=='__main__':main()
