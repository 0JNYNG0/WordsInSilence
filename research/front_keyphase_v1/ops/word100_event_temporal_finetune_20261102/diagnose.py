"""Factor attribution from completed saved predictions, never re-inference."""
from pathlib import Path
import json,collections
U=Path(__file__).resolve().parent
OUT=Path('W:/WordsInSilence_derived/word100_event_temporal_finetune_20261102')
NORMAL={'native','reencoded_normal','speed_0.8','speed_1.25','trim_initial_rest'}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def pointkey(p):return p['key'],p['frame']
def main():
    v=read(U/'VERIFIED.json');reports={t:read(OUT/'evaluations'/f'{t}.json') for t in v['conditions']}
    values=v['conditions'];effects={}
    for metric in ['raw_correct','used_correct','normal_order','short_reverse','clean_order']:
        a,b,c,d=[values[t][metric] for t in ['old_frozen','new_frozen','old_adapt','new_adapt']]
        effects[metric]=dict(labels_with_frozen=b-a,labels_with_adapt=d-c,temporal_with_old=c-a,temporal_with_new=d-b,interaction=d-c-b+a)
    detail={}
    for tag,rep in reports.items():
        raw={pointkey(p):p for p in rep['raw_points'] if p['kind'] in NORMAL and p['y']<8}
        used={pointkey(p):p for p in rep['points'] if p['kind'] in NORMAL and p['y']<8}
        roles={pointkey(p):p for p in rep['role_points'] if p['kind'] in NORMAL and p['y']==1}
        bad=[p for p in raw.values() if p['y']!=p['pred']]
        hidden=[p for k,p in raw.items() if p['y']==p['pred'] and used[k]['pred']!=p['y']]
        conf=collections.Counter((p['y'],p['pred']) for p in bad)
        normal=[s for s in rep['sequences'] if s['kind'] in NORMAL]
        bycase=[]
        for s in normal:
            points=[p for p in raw.values() if p['key']==s['key']]
            dropped=[p['frame'] for p in hidden if p['key']==s['key']]
            bycase.append(dict(key=s['key'],word=s['word'],kind=s['kind'],status=s['status'],
                known_raw_errors=sum(p['y']!=p['pred'] for p in points),correct_main_excluded=dropped,
                unexpected=s['unexpected_action_frames'],missing_role_families=[w['group'] for w in s['role_main_witnesses'] if not w['observed']],
                states=[dict(group=z['group'],frames=z['frames'],isolated=z['isolated_candidate_frames']) for z in s['states']]))
        detail[tag]=dict(raw_errors=bad,correct_main_excluded=hidden,confusions=[dict(expected=a,predicted=b,count=n) for (a,b),n in conf.most_common()],
            normal_status_counts=dict(collections.Counter(s['status'] for s in normal)),cases=bycase)
    out=dict(factor_effects=effects,conditions=detail,
        interpretation='Fixed four-way development diagnosis. Main effects and interaction are descriptive, single seed and reused DEV, not independent generalization evidence.',
        no_threshold_or_model_selection=True,whole_goal_complete=False)
    with (U/'CAUSE_AUDIT.json').open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
    print(json.dumps(dict(effects=effects,conditions={k:dict(raw_errors=len(z['raw_errors']),correct_main_excluded=len(z['correct_main_excluded']),normal_status=z['normal_status_counts']) for k,z in detail.items()}),ensure_ascii=False))
if __name__=='__main__':main()
