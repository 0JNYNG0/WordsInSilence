"""Build a training-only descriptive bank, then score36 development clips once."""
from pathlib import Path
import sys,json,time,hashlib
import numpy as np
U=Path(__file__).resolve().parent;B=U.parents[1]
sys.path.insert(0,str(B/'src'))
from wis_lab.service import word100_analysis as A,word100_multiref as M,word100 as W
from wis_lab.core.observation import Observation
from wis_lab.core import features,dtw,observation
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
def write(p,v):
    p=Path(p);p.parent.mkdir(exist_ok=True,parents=True)
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),'utf-8');temp.replace(p)

def main():
    assert not (U/'BASELINE.json').exists(),'Use saved baseline, do not rebuild'
    start=time.monotonic();plan=read(U/'FROZEN_PLAN.json');cfg=W.config()
    assert plan['frozen_model_sha256']==W.PIN and sha(cfg['checkpoint'])==W.PIN
    codes=plan['words'];manifestpath=B/'ops/word100_training_20261013/MANIFEST.json'
    manifest=read(manifestpath);records=Path(read(manifestpath.parent/'FROZEN_EXPERIMENT.json')['output'])/'records'
    old=read(B/'ops/word100_robustness_20261013/dtw96/PAIRS.json')
    original_pin=read(B/'ops/word100_robustness_20261013/dtw96/FROZEN_PLAN.json')['source_sha256']
    assert sha(U/'before/word100_analysis.py')==original_pin[str(B/'src/wis_lab/service/word100_analysis.py')]
    for p in (features.__file__,dtw.__file__,observation.__file__):assert sha(p)==original_pin[str(Path(p))]
    reused={(p['query_clip'],p['reference_clip']):p['result'] for p in old}
    bank=dict(schema=M.SCHEMA,model_sha256=W.PIN,manifest_sha256=sha(manifestpath),words={},
              code_sha256={str(Path(p)):sha(p) for p in [A.__file__,M.__file__,features.__file__,dtw.__file__,observation.__file__]},
              quality_gate=plan['quality_gate'],aggregation=plan['baseline'],user_data_used=False)
    train_pairs=[];dev_results=[];samples={};allrecords=[];new=0;reuse=0
    for wi,code in enumerate(codes):
        rows=sorted([r for r in manifest if r['word_code']==code],key=lambda r:r['signer_id'])
        train=[r for r in rows if r['role']=='train_candidate'];dev=[r for r in rows if r['role']=='development_only']
        assert [r['signer_id'] for r in train]==list(range(1,10))
        assert [r['signer_id'] for r in dev]==[11,12,13]
        refs=[]
        for row in rows:
            record=read(records/(row['clip_id']+'.json'))
            assert record['role']==row['role'] and record['manifest_sha256']==bank['manifest_sha256']
            assert sha(record['observation'])==record['observation_sha256']
            seq=A._sample(Observation.load(record['observation']));samples[row['clip_id']]=seq
            samplepath=U/'samples'/(row['clip_id']+'.npz');samplepath.parent.mkdir(exist_ok=True)
            np.savez_compressed(samplepath,**seq)
            info=dict(clip_id=row['clip_id'],signer_id=row['signer_id'],role=row['role'],
                observation=record['observation'],observation_sha256=record['observation_sha256'],
                sample_path=str(samplepath),sample_sha256=sha(samplepath))
            allrecords.append(info)
            if row['role']=='train_candidate':refs.append(info)
        loo=[]
        for row in train:
            pairs=[]
            for ref in refs:
                if ref['clip_id']==row['clip_id']:continue
                key=(row['clip_id'],ref['clip_id'])
                if key in reused:result=reused[key];reuse+=1
                else:result=A.compare_samples(samples[key[0]],samples[key[1]]);new+=1
                pair=dict(word_id=code,query_clip=key[0],reference_clip=key[1],
                          query_signer=row['signer_id'],reference_signer=ref['signer_id'],result=result)
                pairs.append(pair);train_pairs.append(pair)
            loo.append(dict(clip_id=row['clip_id'],signer_id=row['signer_id'],metrics=M.aggregate(pairs)))
        baseline={}
        for axis in M.AXES:
            metrics=[next(m for m in p['metrics'] if m['key']==axis) for p in loo]
            baseline[axis]=M.distribution([m['value'] for m in metrics])
            baseline[axis]['sides']={s:M.distribution([next(d for d in m['sides'] if d['side']==s)['value'] for m in metrics]) for s in ('L','R')}
        bank['words'][code]=dict(label=train[0]['canonical_label'],references=refs,leave_one_out=loo,baseline=baseline)
        for row in dev:
            pairs=[dict(reference_clip=r['clip_id'],reference_signer=r['signer_id'],
                        result=A.compare_samples(samples[row['clip_id']],samples[r['clip_id']])) for r in refs]
            dev_results.append(dict(word_id=code,query_clip=row['clip_id'],signer_id=row['signer_id'],
                                pairs=pairs,metrics=M.relative(M.aggregate(pairs),baseline)))
        progress=dict(status='RUNNING',words_complete=wi+1,total_words=len(codes),new_pairs=new,reused_pairs=reuse,seconds=time.monotonic()-start)
        write(U/'PROGRESS.json',progress);print(json.dumps(progress),flush=True)
    write(U/'INPUTS.json',allrecords);write(U/'TRAIN_PAIRS.json',train_pairs)
    write(U/'DEVELOPMENT.json',dev_results);write(U/'BASELINE.json',bank)
    write(U/'PROGRESS.json',dict(status='COMPLETE_PENDING_CHECK',words_complete=12,new_pairs=new,reused_pairs=reuse,seconds=time.monotonic()-start))

if __name__=='__main__':main()
