"""Separate stdlib-only recomputation; does not import the analysis or model code."""
import hashlib
import json
import zipfile
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def read(p): return json.loads(p.read_text('utf-8'))
report=read(HERE/'RESULT.json')
session=ROOT/'runs'/report['scope']['session']
meta=read(session/'session.json')
config=read(ROOT/'config/word100_practice.json')
counts={'M0':[0,0,0],'C':[0,0,0]}
verified=[]
for j,tid in enumerate(meta['trial_ids']):
    d=read(session/'trials'/f'{tid}.result.json')
    i=config['word_codes'].index(d['requested_word_id'])
    for key,v in [('C',d['practice100']),('M0',d['comparison']['baseline'])]:
        logits=v['logits']; target=logits[i]
        rank=1+sum(x>target or (x==target and k<i) for k,x in enumerate(logits))
        assert rank==report['rows'][j][key]['rank']
        if j<12:
            for n,top in enumerate([1,3,5]): counts[key][n]+=int(rank<=top)
    with zipfile.ZipFile(session/'trials'/f'{tid}.capture.zip') as z:
        m=json.loads(z.read('manifest.json'))
        assert m['trial_id']==tid and m['observation_sha256']==d['observation_sha256']
        assert not m['training_use']
        for f in m['frames']:
            assert hashlib.sha256(z.read(f['image'])).hexdigest()==f['sha256']
    verified.append({'trial':tid,'count_greater_rank_check':True,'stored_capture_images_hash_verified':len(m['frames'])})
assert counts=={'M0':[8,10,10],'C':[11,11,11]}
for k,values in counts.items():
    for top,n in zip([1,3,5],values): assert report['statistics']['first12'][k][f'top{top}_count']==n
for p,pin in report['source_hashes'].items(): assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==pin,p
out={'status':'PASS','method':'independent stdlib count-greater rank calculation and full diagnostic JPEG hashes; same Codex operator, no second human/agent review',
     'first12_counts_top1_top3_top5':counts,'checks':verified,'all_source_hashes_unchanged':True,
     'no_training_tuning_or_inference':True}
(HERE/'INDEPENDENT_RECOMPUTE.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n','utf-8')
print(json.dumps({'status':'PASS','counts':counts,'trials':len(verified),'capture_images':sum(x['stored_capture_images_hash_verified'] for x in verified)},ensure_ascii=False))
