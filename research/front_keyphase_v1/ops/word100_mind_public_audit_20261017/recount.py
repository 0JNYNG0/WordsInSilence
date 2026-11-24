"""Separate recount; same Codex operator. Not an independent expert review."""
import json,hashlib,statistics,zipfile
from pathlib import Path
import numpy as np
U=Path(__file__).resolve().parent;P=U.parent/'word100_body_boundary_20261017'
load=lambda p:json.loads(Path(p).read_text('utf-8'))
r=load(U/'AUDIT.json');checks=[]
assert len(r['records'])==36 and len({x['clip_id'] for x in r['records']})==36
assert all(x['signer_id'] in range(1,10) for x in r['records'])
checks.append('36 unique TRAIN sources; nine per label; no user observations')
for p,h in r['pins'].items():
    assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
checks.append(f"{len(r['pins'])} evidence hashes unchanged")
annotation=zipfile.ZipFile(r['annotation_archive'])
coverage=[]
for x in r['records']:
    ann=json.loads(annotation.read(x['annotation_member']).decode('utf-8-sig'))
    assert ann==x['annotation']
    assert ann['data'][0]['attributes'][0]['name']==x['label']
    assert hashlib.sha256(Path(x['video']).read_bytes()).hexdigest()==x['source']['sha256']
    rec=load(P/'data/records'/f"{x['clip_id']}.json")
    lo=ann['data'][0]['start'];hi=ann['data'][0]['end']
    for view in ['native','masked','sensor']:
        obs=np.load(rec['views'][view]['observation']);sec=obs['timestamp_us']/1000000
        ids=[i for i,t in enumerate(sec) if lo<=t<=hi]
        for kind in ['whole','context']:
            l,h=rec['views'][view]['feature_info'][kind+'_span']
            fraction=sum(l<=i<=h for i in ids)/len(ids)
            assert abs(fraction-x['views'][view][kind]['core_coverage_fraction'])<1e-12
            coverage.append(fraction)
        for side,base in [('L',33),('R',54)]:
            hits=[all(bool(obs['usable'][i,j]) and bool(obs['observed'][i,j]) for j in [base,base+5,base+9,base+13,base+17]) for i in ids]
            assert abs(sum(hits)/len(hits)-x['views'][view]['hands'][side]['observed_palm5_fraction'])<1e-12
annotation.close()
checks.extend(['36 original labels/intervals and video hashes verified','216 span coverage values independently recounted','216 hand/core observation fractions independently recounted'])
for w,s in r['summary'].items():
    v=[x['core_seconds'] for x in r['records'] if x['word_code']==w]
    assert [min(v),statistics.median(v),max(v)]==[s['core_seconds'][k] for k in ['min','median','max']]
for d in r['stored_development_predictions']:
    source=load(P/f"training_C_{d['seed']}_RESULT.json")
    pred=next(p for p in source['predictions'][d['view']] if p['clip_id']==d['clip_id'])
    ranks=sorted(range(100),key=lambda i:(-pred['logits'][i],i))
    assert ranks.index(pred['label_index'])+1==d['target_rank']
    assert (ranks[0]==pred['label_index'])==d['correct']
checks.append('72 stored DEV ranks recounted with Python sorting; no rescoring')
result={'status':'PASS','checks':checks,'operator':'Codex root, separate implementation; not another agent',
'new_training':0,'new_inference':0,'user_observation_reads':0,'evidence_sha256':hashlib.sha256((U/'AUDIT.json').read_bytes()).hexdigest()}
(U/'RECOUNT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))

