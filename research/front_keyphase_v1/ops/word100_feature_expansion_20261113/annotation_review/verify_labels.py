from pathlib import Path
import json,hashlib,collections
import numpy as np
U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n','utf-8')
def main():
    files=[U/'ORDINARY_LABELS.json',U/'EXTERNAL_LABELS.json'];rows=sum([read(p)['rows'] for p in files],[]);keys=set();sources={};raws={};imgs={};counts={}
    for r in rows:
        k=(r['clip_id'],r['frame'],r['side']);assert k not in keys,k;keys.add(k)
        assert len(r['y'])==5 and all(v in [-1,0,1,2] for v in r['y'])
        assert r['person'] in ([1,2,3,4,7] if r['role']=='fit' else [11,12,13])
        assert ('REAL%02d_'%r['person']) in r['clip_id'];assert not r['derived_from_word_identity'] and not r['derived_from_model']
        sources[r['source_video']]=r['source_sha256'];raws[r['raw_path']]=r['raw_sha256'];imgs[r['image']]=r['image_sha256']
    for p,h in {**sources,**raws,**imgs}.items():assert sha(p)==h,p
    for p in raws:
        raw=np.load(p)
        for r in [r for r in rows if r['raw_path']==p]:
            ix=np.flatnonzero(raw['frame_id']==r['frame']);assert len(ix)==1
            assert int(raw['timestamp_us'][int(ix[0])])==r['pts_us']
    hashroles=collections.defaultdict(set)
    for r in rows:hashroles[r['source_sha256']].add(r['role'])
    overlaps=[h for h,roles in hashroles.items() if len(roles)>1];assert not overlaps,overlaps
    for role in ['fit','development']:
        rr=[r for r in rows if r['role']==role]
        counts[role]=dict(frames=len({(r['clip_id'],r['frame']) for r in rr}),clips=len({r['clip_id'] for r in rr}),hand_rows=len(rr),known_hand_rows=sum(r['known']>0 for r in rr),finger_cells=dict(collections.Counter(s for r in rr for s in r['states'])),by_finger={f:dict(collections.Counter(r['states'][i] for r in rr)) for i,f in enumerate(['thumb','index','middle','ring','little'])})
    audit=dict(status='structural_and_provenance_pass',label_sha256={p.name:sha(p) for p in files},frames=len(imgs),clips=len(sources),raws=len(raws),hand_rows=len(rows),counts=counts,source_hash_fit_dev_overlap=overlaps,role_integrity=True,frame_timestamp_integrity=True,source_and_raw_and_image_sha_verified=True,review_pending='root independent RGB spot review; not independent human or KSL expert ground truth')
    write(U/'VERIFICATION.json',audit);print(json.dumps(audit,ensure_ascii=False))
if __name__=='__main__':main()
