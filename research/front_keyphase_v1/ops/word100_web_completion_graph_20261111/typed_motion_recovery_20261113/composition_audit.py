from common import *
from collections import Counter
import copy,itertools

def main():
    D=HERE/'composition_audit';D.mkdir(exist_ok=True);assert not(D/'FROZEN.json').exists()
    pp=HERE/'paired_static_progression_learning/evaluation/COMPARISON.json';dp=HERE/'evaluation/RESULTS.json'
    cfg=dict(at=stamp(),conditions=['local_old','direction_repair_only','progression_only','both'],
        inputs='Only previously saved same-seed/same-scenario/same-span outputs. No new predictions or score changes.',
        composition='direction_repair_only uses reencoded_projected_geometry; progression uses saved paired-static presence support on unchanged oldextent. Unsupported correspondence retains raw feature evidence but cannot assert a linked feature difference. No-support never becomes omission.',
        limitation='Combined accounting cannot prove final educational feedback accuracy. Existing failures retained; no changed threshold or choice per scenario.',pins={str(p):sha(p)for p in[Path(__file__),pp,dp]},web_changed=False,private_access=False,final8=False)
    write(D/'FROZEN.json',cfg);loc=read(pp);lk={(r['seed'],r['clip_id'],r['kind']):r for r in loc['rows']};out=[]
    for r in read(dp)['rows']:
        lr=lk[r['seed'],r['clip_id'],r['kind']];cards={c['card']:c for c in lr['conditions']['progression_separate']['cards']};conditions={}
        for name in cfg['conditions']:
            source='reencoded_projected_geometry'if name in ['direction_repair_only','both']else'local_old';cc=copy.deepcopy(r['conditions'][source]);use_gate=name in ['progression_only','both']
            for c in cc:
                gate=cards[c['card']];raw=gate.get('original_extent',gate['chosen']);assert c['span']==(raw['span']if raw else None)
                c['raw_status']=c['status'];c['progression_state']=gate.get('progression_state','unsupported_family_original_preserved')
                c['linked_feature_evidence']=(not use_gate or gate['chosen']is not None)and c['span']is not None
                if not c['linked_feature_evidence']:c['status']='association_unknown'
            order=copy.deepcopy(lr['conditions']['progression_separate'if use_gate else'local_full']['order'])
            conditions[name]=dict(cards=cc,order=order)
        out.append(dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],conditions=conditions))
    summary=[]
    for seed,name,kind in itertools.product([42,43,44],cfg['conditions'],sorted({r['kind']for r in out})):
        rr=[r for r in out if r['seed']==seed and r['kind']==kind];cc=[c for r in rr for c in r['conditions'][name]['cards']];axis2=[e for c in cc if c['linked_feature_evidence']for e in c['evidence']if e['task']=='direction'and e.get('axis')==2]
        summary.append(dict(seed=seed,condition=name,kind=kind,cases=len(rr),cards=len(cc),statuses=dict(Counter(c['status']for c in cc)),linked_axis2_states=dict(Counter(e['state']for e in axis2)),order=dict(Counter(o['state']for r in rr for o in r['conditions'][name]['order']))))
    write(D/'RESULTS.json',dict(at=stamp(),rows=out,summary=summary,reinference=False,adopted=False))
    for r in summary:
        if r['condition']=='both'and r['kind']in ['full_normal','right_closed_hand_main','main_reversed','closed_hand_and_reversed','dynamic_complete_blocks','omit_final_event']:print(r,flush=True)

if __name__=='__main__':main()
