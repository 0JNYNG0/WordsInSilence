from pathlib import Path
import json,collections,numpy as np
U=Path(__file__).resolve().parent;O=U.parent
OUT=Path('W:/WordsInSilence_derived/word100_role_retention_20261102')
NORMAL={'native','reencoded_normal','speed_0.8','speed_1.25','trim_initial_rest'}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def main():
    assert not (U/'CAUSE_AUDIT.json').exists();v=read(U/'VERIFIED.json');details={}
    for tag in v['conditions']:
        r=read(OUT/'evaluations'/f'{tag}.json');pts=[p for p in r['points'] if p['kind'] in NORMAL and p['y']<8]
        hidden=[p for p in pts if p['y']==p['raw_pred'] and not p['eligible']];bad=[p for p in pts if p['y']!=p['raw_pred']]
        for p in hidden:
            a=np.load(OUT/'predictions'/tag/(p['key']+'.npz'));f=p['frame']
            p.update(activity=a['activity_probability'][f].tolist(),roles=a['role_probability'][f].tolist())
        ns=[s for s in r['sequences'] if s['kind'] in NORMAL]
        details[tag]=dict(correct_main_excluded=hidden,raw_errors=bad,
            confusions=[dict(expected=a,predicted=b,count=n) for (a,b),n in collections.Counter((p['y'],p['raw_pred']) for p in bad).most_common()],
            normal_status_counts=dict(collections.Counter(s['status'] for s in ns)),
            cases=[dict(key=s['key'],status=s['status'],content_errors=s['evaluation_content_errors'],qualification_errors=s['evaluation_action_errors'],
                previously_unconfirmed=s['previous_reviewed_core_unconfirmed'],unexpected=s['unexpected_action_frames'],
                states=[dict(group=z['group'],frames=z['frames'],isolated=z['isolated_candidate_frames']) for z in s['states']]) for s in ns])
    p=v['conditions']['new_labels_with_retention'];c=v['conditions']['new_labels_only']
    out=dict(status=v['status'],matched_effect={k:p[k]-c[k] for k in ['used_correct','normal_order','clean_order','short_reverse','false_order','reverse_false_pass']},
        conditions=details,scope='Reused development and sparse physical annotations. No claim of entire KSL meaning, no new labels or tuning from diagnosis.',whole_goal_complete=False)
    with (U/'CAUSE_AUDIT.json').open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
    print(json.dumps(dict(status=v['status'],effect=out['matched_effect'],excluded={k:len(z['correct_main_excluded']) for k,z in details.items()}),ensure_ascii=False))
if __name__=='__main__':main()
