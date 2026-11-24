from pathlib import Path
import json,hashlib,gzip,sys
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent; P=U.parent/'word100_hold_invariant_trajectory_20261101'
def read(p):
    p=Path(p)
    return json.loads(gzip.decompress(p.read_bytes()).decode('utf8')) if p.suffix=='.gz' else json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def new(p,x):
    with Path(p).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def main():
    assert not (P/'RUNNING.json').exists()
    prev=read(P/'COMPLETION.json'); pins={}
    for path,h in prev['pins'].items():assert sha(path)==h,path
    oldf=read(P/'FROZEN.json')
    for group in ['pins','protected']:
        for path,h in oldf[group].items():assert sha(path)==h,path
    for f in ['COMPLETION.json','FROZEN.json','TASKS.json','RESULT_MANIFEST.json','EVALUATION.json','NEXT_PLAN_KO.txt','review/RESULT_VERIFICATION.json','review/POSTEXECUTION_REVIEW.json']:
        pins[str(P/f)]=sha(P/f)
    manifest=read(P/'RESULT_MANIFEST.json'); rows=[]; complete=0
    for r in manifest['results']:
        assert sha(r['path'])==r['sha256'];pins[r['path']]=r['sha256']
        d=read(r['path']);c=d['comparison']['candidate'];complete+=int(c['complete'])
        rows.append(dict(**r,old_complete=c['complete'],old_distance=c['distance']))
    assert len(rows)==2825 and complete==1417 and len(manifest['missing_component_opportunities'])==10
    new(U/'INPUTS.json',dict(source_unit=str(P),pins=pins,protected=oldf['protected'],records=rows,
        missing=manifest['missing_component_opportunities'],possible_opportunities=2835,
        public_source_count=16,public_word_count=4,natural_count=585,complete_count=1417,unknown_count=1408,
        atol=1e-7,max_iterations=60,source_metadata_evaluation_only=True))
    new(U/'AUTHORIZATION.json',dict(user_request='설명 후 바로 다음 작업 진행; 주먹 회전·손목 꺾임과 작은 흔들림을 구분',
        scope='fixed continuous polyline comparator research only',old_models_preserved=True,private_access=False,
        training=0,model_inference=0,production_change=False,word_specific_exceptions=False))
    new(U/'CURRENT_EXECUTION.json',dict(status='PREPARED_PENDING_METHOD_REVIEW_AND_FREEZE',active=None))
    print('prepared',len(rows),complete,'unknown',len(rows)-complete)
if __name__=='__main__':main()
