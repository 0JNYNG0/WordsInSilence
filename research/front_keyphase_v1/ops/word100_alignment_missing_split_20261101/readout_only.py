"""Add evidence to a pre-existing path without searching/changing alignment."""
from pathlib import Path
import sys,json,copy,gzip,hashlib
sys.dont_write_bytecode=True
import run as driver
import split_alignment as candidate
U=Path(__file__).resolve().parent

def compare_existing_path(query,reference,path):
    saved=copy.deepcopy(path)
    if not saved:raise ValueError('Existing full path required')
    if saved[0]!=[0,0] or saved[-1]!=[len(query['x'])-1,len(reference['x'])-1]:
        raise ValueError('Existing path must keep full endpoints')
    if any((b[0]-a[0],b[1]-a[1]) not in ((0,1),(1,0),(1,1)) for a,b in zip(saved,saved[1:])):
        raise ValueError('Existing path is not a monotonic DTW path')
    evidence=candidate.readout(candidate.features(query),candidate.features(reference),saved)
    assert path==saved
    return dict(path=saved,evidence=evidence,new_alignment=False,new_model_inference=False,whole_sign_verdict=None,
                change_scope='Additional evidence only; original alignment stays exact')

def main():
    driver.checkpins();t=driver.read(U/'TASKS.json')
    files=[U/'readout_only.py',U/'split_alignment.py',U/'FROZEN.json',U/'EVALUATION.json',U/'TASKS.json']
    pins={str(p):driver.sha(p) for p in files}
    for p in t['pairs']:pins[p['baselines']['global_dtw']]=driver.sha(p['baselines']['global_dtw'])
    driver.new(U/'READOUT_ONLY_CONTRACT.json',dict(status='FROZEN_BEFORE_EVIDENCE_ONLY_REPLAY',paths=36,seeds=[42,43,44],records=108,
        pins=pins,old_alignment_unchanged=True,new_path_search=False,new_scores=False,
        failed_candidate_preserved=True,not_a_second_alignment_candidate=True))
    # Enforce the absence of path search in this compatibility replay.
    def forbidden(*args,**kwargs):raise RuntimeError('Path search forbidden in evidence-only replay')
    candidate.align=forbidden;candidate.solve=forbidden;candidate.cost=forbidden
    store=driver.previous.original.RuntimeStore(driver.read(driver.previous.PREV/'REFERENCE_BANK.json'))
    rows=[]
    for task in t['pairs']:
        source=Path(task['baselines']['global_dtw']);path=driver.read(source)['path']
        for seed in driver.SEEDS:
            q=driver.previous.select(store.clip(task['query']),seed);r=driver.previous.select(store.clip(task['reference']),seed)
            out=compare_existing_path(q,r,path)
            assert out['path']==path
            out.update(query=task['query'],reference=task['reference'],seed=seed,source=str(source),source_sha256=driver.sha(source),
                contract_sha256=driver.sha(U/'READOUT_ONLY_CONTRACT.json'))
            dest=U/'readout_replay'/f"{task['query']}_{task['reference']}_{seed}.json.gz";dest.parent.mkdir(exist_ok=True)
            payload=json.dumps(out,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode('utf8')
            with dest.open('xb') as f:f.write(gzip.compress(payload,mtime=0))
            summary={g:{side:data[side]['counts'] for side in ('query','reference')} for g,data in out['evidence']['groups'].items()}
            rows.append(dict(path=str(dest),sha256=driver.sha(dest),source=str(source),source_sha256=driver.sha(source),query=task['query'],reference=task['reference'],seed=seed,
                path_exact=True,path_cells=len(path),group_counts=summary))
    for p,h in pins.items():assert driver.sha(p)==h,p
    driver.new(U/'READOUT_ONLY_RESULT.json',dict(status='COMPLETE_ADDITIVE_EVIDENCE_COMPATIBILITY_REPLAY',records=rows,record_count=len(rows),
        original_paths_unchanged=True,new_path_search=0,new_model_inference=0,new_training=0,new_score_calculation=0,
        existing_development_stage_score_reused=82.16435185185185,failed_new_candidate_still_rejected=True,production_change=False))
    print('evidence-only replay complete',len(rows),'existing path unchanged')
if __name__=='__main__':main()
