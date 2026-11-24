"""Sparse, native RGB interval truth. It labels visible trends, not word correctness."""
from pathlib import Path
import json,hashlib
D=Path(__file__).resolve().parent;U=D.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')

# Each interval was inspected in native RGB contact sheets before new predictions.
# Classes: 0 little net separation change, 1 moving closer, 2 moving apart.
PAIRS={
 1:{1:[(47,50),(56,62)],2:[(50,56),(62,68)],0:[(68,74),(74,80)]},
 2:{1:[(49,52),(58,61)],2:[(52,58),(61,67)],0:[(67,73),(73,79)]},
 3:{1:[(56,62),(65,68)],2:[(62,65),(71,77)],0:[(77,83)]},
 4:{1:[(84,90),(96,99)],2:[(90,96),(99,105)],0:[(105,111),(111,117)]},
 5:{1:[(41,44),(50,53)],2:[(44,50),(53,59)],0:[(59,65)]},
 6:{1:[(45,48),(51,54)],2:[(48,51),(54,60)],0:[(60,66)]},
 7:{1:[(48,54),(60,63)],2:[(54,60),(63,69)],0:[(69,75)]},
 8:{1:[(48,54),(60,66)],2:[(54,60),(66,72)],0:[(75,81)]},
 9:{1:[(54,57),(63,66)],2:[(57,63),(66,72)],0:[(72,78)]},
 11:{1:[(49,52),(58,61)],2:[(52,58),(64,70)],0:[(70,76)]},
 12:{1:[(37,40),(46,52)],2:[(40,46),(52,58)],0:[(58,64),(64,70)]},
 13:{1:[(55,58),(64,67)],2:[(58,64)],0:[(70,76),(76,82)]},
}
def main():
    assert not(D/'MOTION_FROZEN.json').exists()
    rows=[]
    for r in read(U/'INPUTS.json')['rows']:
        assert sha(r['video']) == r['source_sha256']
        for y,intervals in PAIRS[r['person']].items():
            for a,b in intervals:
                rows.append(dict(clip_id=r['clip_id'],person=r['person'],split=r['role'],start=a,end=b,label=y,
                    source_sha256=r['source_sha256'],sheet=str(U/'review'/f'{r["person"]:02}_0.jpg')))
    write(D/'MOTION_TRUTH.json',dict(rows=rows,reviewer='Codex native RGB visual review, not independent KSL specialist',
        unit='Net projected inter-hand motion in the stated native interval; no dense boundary interpolation.',
        classes=['little_net_change','approaching','separating'],physical_contact=None,whole_word_correct=None))
    write(D/'MOTION_FROZEN.json',dict(truth_sha256=sha(D/'MOTION_TRUTH.json'),fit_people=list(range(1,10)),
        development=[11,12,13],diagnostic_only=[14,15,16,17],reserve_unopened=[10,18],
        input='Native interval hand-palm relative displacement/distance changes, shoulder-scale normalization; no word/person/expected repetition/time-position input.',
        models=['linear','mlp32'],seeds=[42,43,44],steps=400,lr=.003,weight_decay=.001,confidence=.7,
        inference='Three-seed unanimous confidence >=.7; missing coordinates -> unknown; independent of static near/far confidence.',
        acceptance='No extra wrong development trend versus same-label linear comparator; do not infer final word success from this component.',
        protections={str(p):sha(p)for p in [U/'basic_runtime.py',U/'core.py',U/'shape_consensus.py',U.parent/'word100_live_g6_20261114/engine.py']},
        scope='Isolated common movement observation experiment. Existing100 Conv+GRU/TCN/C·T3-F and web unchanged. No private data.'))
    print('REVIEWED_INTERVALS',len(rows),{s:sum(r['split']==s for r in rows) for s in ['fit','dev']})
if __name__=='__main__':main()
