"""One fixed diagnostic on saved rank scores; no training or model inference."""
from pathlib import Path
import sys,itertools,copy
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import read,write,sha,stamp,np
from relation_mass import candidate_distribution,relation_mass
U=Path(__file__).resolve().parent;H=U.parent;J=H/'splice_rank_learning'

def main():
    assert not (U/'FROZEN.json').exists()
    cp=J/'evaluation/COMPARISON.json';comp=read(cp)
    pins={str(p):sha(p)for p in[Path(__file__),U/'relation_mass.py',U/'test_relation_mass.py',cp]}
    for p in comp['predictions']:
        for path,h in [(p['path'],p['sha256']),(p['original_prediction'],p['original_prediction_sha256']),(p['source'],p['source_sha256'])]:
            assert sha(path)==h;pins[path]=h
    cfg=dict(at=stamp(),purpose='Diagnostic of learned alternative interval mass, NOT G2 acceptance or deployed rule.',
      conditions=['local_full','splice_rank'],temperature=1.,threshold=.65,
      threshold_warning='Existing .65 evidence convention reused as one frozen diagnostic value; rank mass is not calibrated correctness probability.',
      interpretation='Conditional on both candidate pools; action presence, occurrence identity, completeness and KSL correctness NOT established.',
      hypotheses='Listwise positive-mass supervision versus two hard argmax boundaries. Preserve raw spans and full positive-duration overlap mass.',
      risks=['independent marginal candidate-product assumption','repeated occurrence pairing ambiguous','rank score scale and candidate density sensitivity','missing action may still have a ranked candidate'],
      prohibited=['expected order or word input','new labels','threshold tuning','new inference','G2 success declaration','web replacement'],
      tests='Eight pure interval contracts passed before reading any outcome.',pins=pins)
    write(U/'FROZEN.json',cfg)
    meta={(p['seed'],p['clip_id'],p['kind']):p for p in comp['predictions']};out=[]
    for row in comp['rows']:
        p=meta[row['seed'],row['clip_id'],row['kind']];ts=np.load(p['source'])['ts'];cards=row['conditions']['local_full']['cards']
        pair=[(i,j)for i,j in itertools.combinations(range(len(cards)),2)if sorted([cards[i]['family'],cards[j]['family']])==[2,3]]
        if not pair:continue
        for mode,path,prefix in [('local_full',p['original_prediction'],'local_full'),('splice_rank',p['path'],'splice_rank')]:
            z=np.load(path)
            for i,j in pair:
                dist=[candidate_distribution(z[f'spans_{k}'],z[f'{prefix}_logits_{k}'],z[f'presence_{k}'],ts)for k in[i,j]]
                m=relation_mass(*dist)
                out.append(dict(seed=row['seed'],clip_id=row['clip_id'],kind=row['kind'],condition=mode,cards=[cards[i]['card'],cards[j]['card']],
                  raw_chosen=[copy.deepcopy(row['conditions'][mode]['cards'][k]['chosen'])for k in[i,j]],relation=m))
    summary=[]
    for seed,mode,kind in itertools.product([42,43,44],cfg['conditions'],sorted({r['kind']for r in out})):
        rr=[r for r in out if r['seed']==seed and r['condition']==mode and r['kind']==kind]
        summary.append(dict(seed=seed,condition=mode,kind=kind,count=len(rr),states=dict(Counter(r['relation']['state']for r in rr))))
    for p,h in pins.items():assert sha(p)==h
    write(U/'RESULTS.json',dict(at=stamp(),rows=out,summary=summary,inference=False,training=False,g2_pass_claim=False,web_changed=False,pins_preserved=True))
    print('conditional relation rows',len(out))
    for r in summary:
        if r['condition']=='splice_rank'and r['kind']in['full_normal','speed075','speed125','dynamic_complete_blocks','omit_final_event']:print(r)

if __name__=='__main__':main()
