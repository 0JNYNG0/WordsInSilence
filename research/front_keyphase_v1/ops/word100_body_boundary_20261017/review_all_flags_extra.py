"""Explicit extra TRAIN source frames for an ambiguous frozen review window."""
import argparse
from datetime import datetime,timezone
import sys
sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import review_all_flags_build as b

def main(wid,frame_text,all_window=False,zoom_side=None):
    assert wid.startswith('W') and len(wid)==5 and wid[1:].isdigit()
    material_path=b.OUT/f'{wid}.json';material=b.read(material_path);job=material['job']
    assert material['pins']['generator']==b.sha(b.__file__)
    context=b.read(b.U/'PREP_CONTEXT.json')
    assert b.sha(b.U/'PREP_CONTEXT.json')==material['pins']['context']
    row=next(r for r in context['ordered_rows'] if r['clip_id']==job['clip_id'])
    assert row['role']=='train_candidate' and 1<=row['signer_id']<=9
    rp=Path(material['record_path']);assert b.sha(rp)==material['record_sha256'];rec=b.read(rp)
    obs={v:b.loadobs(rec['views'][v]['observation']) for v in ('native','masked','sensor')}
    assert all(b.sha(p)==h for p,h in material['observation_pins'].items())
    requested=set(job['window']['frame_ids']) if all_window else set(int(x) for x in frame_text.split(','))
    assert requested and requested<=set(job['window']['frame_ids']),'Extra frames must stay in this exact frozen review window'
    selection=sorted(requested);tag=b.hashlib.sha256((','.join(map(str,selection))+'|'+str(zoom_side)).encode()).hexdigest()[:12]
    out=b.OUT/f'{wid}_extra_{tag}.json';assert not out.exists(),'Preserve earlier extra-frame evidence'
    data,method=b.source_bytes(job['clip_id'],row,rec);frames,pts,origin=b.decode(data,requested,obs['native']['timestamp_us'])
    images=[];shown=[];pages=(len(selection)+3)//4
    renderjob=dict(job)
    if zoom_side:
        renderjob['conditions']=[c for c in job['conditions'] if c['side']==zoom_side]
        assert renderjob['conditions'],'Requested side is not flagged in this window'
    for page,at in enumerate(range(0,len(selection),4),1):
        image,meta=b.render_page(renderjob,selection[at:at+4],obs,frames,pts,page,pages)
        p=b.OUT/f'{wid}_extra_{tag}_p{page:02d}.jpg';assert not p.exists();image.save(p,quality=90,subsampling=0)
        images.append(dict(path=str(p),sha256=b.sha(p),displayed_frame_ids=[r['frame_id'] for r in meta]));shown+=meta
    result=dict(status='EXTRA_FRAMES_CREATED_PENDING_HUMAN_REVIEW',window_id=wid,clip_id=job['clip_id'],
        source_sha256=rec['source']['sha256'],record_sha256=b.sha(rp),material_path=str(material_path),material_sha256=b.sha(material_path),
        source_read=method,source_absolute_pts_origin_us=origin,frames=shown,images=images,
        requested_frame_ids=selection,zoom_side=zoom_side,all_window_frames_included=all_window,human_viewing_completed=False,
        semantic_pass=False,training_allowed=False,detector_runs=0,classifier_runs=0,training_runs=0,
        created_utc=datetime.now(timezone.utc).isoformat(),generator_sha256=b.sha(__file__))
    b.write_new(out,result);print(str(out))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--window',required=True);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--frames');g.add_argument('--all-window',action='store_true');p.add_argument('--zoom-side',choices=['left','right'])
    a=p.parse_args();main(a.window,a.frames,a.all_window,a.zoom_side)
