"""Mechanical union of disjoint raw A reviews; never adjudicates or modifies labels."""
from pathlib import Path
import json,hashlib,datetime
N=Path(__file__).resolve().parent;F=N/'full24_native'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    plan=read(F/'A_ASSIGNMENT.json')
    assert plan['merge_code_sha256']==sha(Path(__file__))
    source=read(F/'BLIND_NATIVE.json');rows=[];parts=[]
    for name,expected in [('REVIEW_A_PART1.json',[f'P{i:02}' for i in range(1,10)]),('REVIEW_A_PART2.json',[f'P{i:02}' for i in range(10,16)])]:
        p=F/name;d=read(p)
        assert d['status']=='COMPLETE_PART' and len(d['rows'])==len(expected)
        assert sorted(r['alias'] for r in d['rows'])==expected
        assert d['native_manifest_sha256']==source['native_manifest_sha256'] and d['rubric_sha256']==source['rubric_sha256']
        rows.extend(d['rows']);parts.append(dict(file=str(p),sha256=sha(p),reviewer=d.get('reviewer'),aliases=expected))
    assert len(rows)==len({r['sample_id'] for r in rows})==15
    out=dict(status='COMPLETE',reviewer='A: two disjoint Codex readers, each blind to B and old answers',
        native_manifest_sha256=source['native_manifest_sha256'],rubric_sha256=source['rubric_sha256'],rows=rows,
        raw_parts=parts,merge='Exact unchanged raw rows; no averaging, adjudication, filtering, or disagreement resolution',
        assignment_sha256=sha(F/'A_ASSIGNMENT.json'))
    with (F/'REVIEW_A.json').open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False)
    print('COMPLETE15_RAW_ROWS_MERGED_WITHOUT_CHANGE')
if __name__=='__main__':main()
