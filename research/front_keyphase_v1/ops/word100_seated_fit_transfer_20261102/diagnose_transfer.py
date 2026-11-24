"""Explain the fixed result without new inference or tuning."""
from prepare_seated import *
from collections import Counter
DEST=Path('W:/WordsInSilence_derived/word100_seated_fit_transfer_20261102')
NORMAL={'native','reencoded_normal','speed_0.8','speed_1.25','trim_initial_rest'}
def main():
    assert not (L/'CAUSE_AUDIT.json').exists();out={}
    for tag in ['original_replay','seated_variation']:
        r=read(DEST/'evaluations'/f'{tag}.json');raw={(p['key'],p['frame']):p for p in r['raw_points']};points=[p for p in r['points'] if p['kind'] in NORMAL and p['y']<8]
        causes=Counter();losses=[]
        for p in points:
            q=raw[p['key'],p['frame']]
            if p['pred']==p['y']:continue
            if q['pred']!=q['y']:causes['raw_content_wrong']+=1;continue
            a=np.load(DEST/'predictions'/tag/f"{p['key']}.npz");f=p['frame'];rp=a['role_probability'][f];role=int(rp.argmax())
            causes['role_discarded_correct_content']+=1
            earliest=min(x['source_frame'] for x in points if x['clip_id']==p['clip_id'] and x['y']==p['y'])
            losses.append(dict(p,role='context' if role==0 else 'rest' if role==2 else 'main',role_probability=rp.tolist(),
                current_rest=float(a['current_probability'][f,9]),activity_probability=a['activity_probability'][f].tolist(),raw_probability=a['raw_action_probability'][f].tolist(),
                source_distance_from_first_labelled_same_family=int(p['source_frame']-earliest)))
        assert sum(causes.values())==sum(p['pred']!=p['y'] for p in points)
        out[tag]=dict(error_causes=dict(causes),role_losses=losses,unique_source_points=sorted({(p['clip_id'],p['source_frame']) for p in losses}),
            role_counts=dict(Counter(x['role'] for x in losses)),at_first_labelled_family_point=sum(p['source_distance_from_first_labelled_same_family']==0 for p in losses))
    exclusive(L/'CAUSE_AUDIT.json',dict(at=clock(),conditions=out,
        conclusion='Camera-view variation reduces raw-content errors but increases correct-content suppression by learned process role. The known regression is primarily the interpretation gate; not evidence that joints disappeared or all RGB features failed.',
        next='Audit sparse context/main boundary supervision in FIT and contract of simultaneous action content/process role. Keep correct raw evidence visible; evaluate omission/return confusion separately. Do not lower thresholds or force target order from exposed DEV.',
        no_new_inference=True,no_score_change=True,whole_goal_complete=False))
    print({k:dict(causes=v['error_causes'],roles=v['role_counts'],first=v['at_first_labelled_family_point'],unique=len(v['unique_source_points'])) for k,v in out.items()})
if __name__=='__main__':main()
