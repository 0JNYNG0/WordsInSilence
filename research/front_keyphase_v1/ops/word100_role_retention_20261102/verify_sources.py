"""Audit unchanged DEV labels against an already verified evaluation record."""
from pathlib import Path
import json,hashlib
U=Path(__file__).resolve().parent;O=U.parent
OUT=Path('W:/WordsInSilence_derived/word100_role_retention_20261102')
BASE=Path('W:/WordsInSilence_derived/word100_event_temporal_finetune_20261102/evaluations/new_adapt.json')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not (U/'SOURCES_VERIFIED.json').exists()
    fields=['key','kind','word','person','frame','source_frame','y']
    def normalized(points):return {tuple(p[k] for k in ['key','frame']):{k:p[k] for k in fields} for p in points}
    base=normalized(read(BASE)['raw_points']);pins={str(BASE):sha(BASE)}
    for tag in read(U/'TRAINING_FREEZE.json')['conditions']:
        p=OUT/'evaluations'/f'{tag}.json';report=read(p);points=report['points']
        assert len(points)==len(base) and normalized(points)==base
        assert {p['person'] for p in points}=={11,12,13};pins[str(p)]=sha(p)
    out=dict(status='UNCHANGED_DEVELOPMENT_LABELS_VERIFIED',known_points_per_condition=len(base),recorded_ids=[11,12,13],
        comparison='Exact clip/kind/word/recorded-ID/native mapping/label match to prior independently checked saved evaluation; no relabeling or private/final-holdout access.',pins=pins,
        old_model_inference=0,whole_goal_complete=False)
    with (U/'SOURCES_VERIFIED.json').open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
    print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
