"""Separate decode/index/label audit. Does not certify human KSL semantics."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import Counter
import json, hashlib
import av, numpy as np
from PIL import Image
U=Path(__file__).resolve().parent;O=U.parent

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not (U/'VERIFIED.json').exists()
    manifest=read(U/'REVIEW_INPUTS.json');review=read(U/'REVIEW_DECISIONS.json');frozen=read(U/'FROZEN.json')
    old={r['clip_id']:r for r in read(O/'word100_guide_conditioned_episode_20261102/CONTEXT_TARGETS.json')['rows']}
    by={r['clip_id']:r for r in review['rows']};checked=0;errors=[];counts=Counter();role_counts=Counter()
    for r in manifest['rows']:
        assert r['person'] in [2,4,8] and sha(r['source'])==r['source_sha256']
        wanted={i for p in r['panels'] for i in p['frames']};frames={};times={}
        with av.open(r['source']) as container:
            origin=None
            for i,f in enumerate(container.decode(video=0)):
                if origin is None:origin=float(f.time)
                if i in wanted:frames[i]=Image.fromarray(f.to_ndarray(format='rgb24'));times[i]=float(f.time)-origin
        assert set(frames)==wanted
        for p in r['panels']:
            assert sha(p['image'])==p['sha256']
            assert p['frames']==list(range(p['frames'][0],p['frames'][-1]+1))
            panel=np.asarray(Image.open(p['image']).convert('RGB'),dtype=float)
            for j,f in enumerate(p['frames']):
                assert abs(times[f]-p['times'][j])<1e-7
                im=frames[f];w,h=im.size;im=im.crop((int(w*.23),0,int(w*.77),h));im.thumbnail((356,330))
                x=j%4*360;y=j//4*360+27
                a=np.asarray(im,dtype=float);b=panel[y:y+im.height,x:x+im.width]
                error=float(abs(a-b).mean());assert error<4,(p['image'],f,error);errors.append(error);checked+=1
        labels={p['frame']:p['action_family'] for p in old[r['clip_id']]['points'] if p['mask']}
        seen=set()
        for p in by[r['clip_id']]['points']:
            f=p['frame'];assert f in wanted and f not in seen and f not in labels;seen.add(f)
            assert abs(p['timestamp']-times[f])<1e-7 and p['role'] in [0,1,2]
            assert p['action_mask']==(p['action_family']>=0)
            if 0<=p['action_family']<8:assert p['role']==1
            if p['action_family']==8:assert p['role']==2
            if p['role']==0:assert p['action_family']==-1
            counts[r['word_id']]+=1;role_counts[p['role']]+=1
    # Verify that the disputed old full-size review image depicts native f62,
    # rather than accidentally comparing an off-by-one or a sampled frame.
    correction=review['correction_candidates'][0];r=next(r for r in manifest['rows'] if r['clip_id']==correction['clip_id'])
    with av.open(r['source']) as c:
        frame=next(f for i,f in enumerate(c.decode(video=0)) if i==62)
        raw=Image.fromarray(frame.to_ndarray(format='rgb24'))
    stored=Image.open(correction['full_image']).convert('RGB');assert sha(correction['full_image'])==correction['full_image_sha256']
    if raw.size!=stored.size:raw=raw.resize(stored.size,Image.Resampling.LANCZOS)
    difference=float(np.abs(np.asarray(raw,dtype=float)-np.asarray(stored,dtype=float)).mean())
    assert difference<6, difference
    for p,h in frozen['pins'].items():assert sha(p)==h
    assert sum(counts.values())==review['new_points']==172 and checked==sum(len(p['frames']) for r in manifest['rows'] for p in r['panels'])
    out=dict(at=datetime.now(timezone(timedelta(hours=9))).isoformat(),status='NATIVE_FRAME_AND_ANNOTATION_STRUCTURE_VERIFIED',
        clips=12,panels=40,panel_cells_checked=checked,max_panel_jpeg_mae=max(errors),new_points=172,
        new_points_by_word=dict(counts),new_roles=dict(role_counts),disputed_frame_source_index_verified=True,disputed_image_jpeg_mae=difference,
        correction_visual_verdict='Separate visual evidence record required; numerical checks establish frame identity, not handshape.',
        no_old_labels_or_scores_changed=True,new_training=0,whole_goal_complete=False,
        pins={str(p):sha(p) for p in [U/'FROZEN.json',U/'REVIEW_INPUTS.json',U/'REVIEW_DECISIONS.json',U/'record_review.py',U/'verify_review.py']},
        reviewer='Same Codex separate structural arithmetic and source decoding; not independent KSL expert labels.')
    with (U/'VERIFIED.json').open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in out.items() if k!='pins'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
