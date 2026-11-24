"""Overlay directly reviewed FIT labels only. Reuse frozen public observations."""
from model_core import *
from collections import Counter

def main():
    assert not (U/'DATA_READY.json').exists()
    approval=read(U/'ROOT_LABEL_REVIEW.json');assert approval['status']=='PASS_RESEARCH_LABELS'
    lp=U/'shape_recovery/EXTRA_LABELS.json';labels=read(lp)['rows']
    assert sha(lp)==approval['labels_sha256']
    assert labels,'No speculative labels admitted'
    before=read(PREV/'DATA_READY.json');indexed={}
    for l in labels:
        assert l['role']=='fit' and 1<=l['person']<=9
        assert len(l['y'])==5 and set(l['y'])<=set([-1,0,1,2])
        key=(l['clip_id'],int(l['frame']),l['side']);assert key not in indexed;indexed[key]=l
        for pathkey,hashkey in [('source_video','source_sha256'),('image','image_sha256'),('raw_path','raw_sha256')]:
            assert sha(l[pathkey])==l[hashkey],pathkey
    out=[];applied=set();added=Counter();new_by_clip={}
    for r in before['rows']:
        relevant=[(k,l) for k,l in indexed.items() if k[0]==r['clip_id']]
        if not relevant:out.append(r);continue
        assert r['role']=='fit' and sha(r['path'])==r['sha256']
        with np.load(r['path'],allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
        for k,l in relevant:
            assert l['source_sha256']==r['source_sha256'];ix=np.flatnonzero(a['frame_id']==l['frame']);assert len(ix)==1
            i=int(ix[0]);s=0 if l['side']=='L' else 1
            assert int(a['timestamp_us'][i])==int(l['pts_us'])
            for f,y in enumerate(l['y']):
                prior=int(a['fy'][i,s,f]);assert prior<0 or y<0 or prior==y,(k,f,prior,y)
                if y>=0 and prior<0:a['fy'][i,s,f]=y;added[str(y)]+=1
            applied.add(k)
        dst=U/'cache'/f"{r['clip_id']}.npz";dst.parent.mkdir(exist_ok=True,parents=True);np.savez_compressed(dst,**a)
        out.append(dict(r,path=str(dst),sha256=sha(dst),finger_known=int((a['fy']>=0).sum()),parent_cache=r['path'],parent_sha256=r['sha256']))
        new_by_clip[r['clip_id']]=len(relevant)
    assert applied==set(indexed),'Label source lacks prepared common input'
    for r in out:
        if r['role']=='development':assert r==next(x for x in before['rows'] if x['clip_id']==r['clip_id'])
    write(U/'DATA_READY.json',dict(rows=out,extra_words=before['extra_words'],new_labels=str(lp),labels_sha256=sha(lp),
        added_known_cells=dict(added),reused_caches=len(out)-len(new_by_clip),changed_label_caches=len(new_by_clip),
        no_new_detector_inference=True,development_labels_unchanged=True,common100_preserved=True))
    event('inputs_prepared',added=dict(added),clips=len(new_by_clip),total=len(out))
if __name__=='__main__':main()
