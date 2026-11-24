"""Extract one frozen 54 MB local sample and verify actual MP4 timestamps using PyAV."""
from pathlib import Path
import hashlib
import json
import math
import os
import shutil
import traceback
import zipfile
import av

U=Path(__file__).resolve().parent
plan=json.loads((U/'LOCAL_PILOT_PLAN.json').read_text(encoding='utf-8'))
assert not (U/'PILOT_RESULT.json').exists(), 'Pilot is already complete'
dest=Path(plan['destination']).resolve()
assert dest==Path('W:/WordsInSilence/data/AIHub_103/pilot12_20261011').resolve()
assert len(plan['rows'])==12 and plan['selected_bytes']<100_000_000
assert shutil.disk_usage('W:/').free>1_000_000_000
dest.mkdir(parents=True,exist_ok=True)
results=[]
try:
    with zipfile.ZipFile(plan['archive']) as z:
        for r in plan['rows']:
            info=z.getinfo(r['video']['member'])
            assert info.file_size==r['video']['bytes'] and info.CRC==r['video']['crc32']
            out=dest/(r['clip_id']+'.mp4')
            assert out.parent==dest and not out.exists()
            source_hash=hashlib.sha256()
            with z.open(info) as src,out.open('xb') as f:
                while block:=src.read(1024*1024):
                    source_hash.update(block);f.write(block)
                f.flush();os.fsync(f.fileno())
            with out.open('rb') as f:
                assert hashlib.file_digest(f,'sha256').hexdigest()==source_hash.hexdigest()
            with av.open(str(out)) as container:
                st=container.streams.video[0]
                fps=float(st.average_rate)
                pts=[];last_duration=None
                for frame in container.decode(st):
                    assert frame.pts is not None
                    pts.append(float(frame.pts*frame.time_base))
                    last_duration=float(frame.duration*frame.time_base) if frame.duration else 1/fps
                assert pts and all(math.isfinite(x) for x in pts)
                monotonic=all(a<b for a,b in zip(pts,pts[1:]))
                media_duration=pts[-1]-pts[0]+last_duration
                interval_ok=all(0<=a<b<=media_duration for a,b in r['intervals'])
                declared_match=abs(media_duration-r['duration'])<=1/fps+.001
                rec={'clip_id':r['clip_id'],'label':r['labels'][0],'word_code':r['word_code'],
                     'path':str(out),'bytes':out.stat().st_size,'sha256':source_hash.hexdigest(),
                     'source_member_crc_pass':True,'decoded_frames':len(pts),'width':st.width,'height':st.height,
                     'average_rate':str(st.average_rate),'first_pts_s':pts[0],'last_pts_s':pts[-1],
                     'actual_duration_s':media_duration,'annotation_duration_s':r['duration'],
                     'intervals':r['intervals'],'monotonic_pts':monotonic,'annotation_within_actual_duration':interval_ok,
                     'declared_duration_within_one_frame':declared_match,
                     'pass':monotonic and interval_ok and declared_match}
            results.append(rec)
    result={'status':'PASS' if all(x['pass'] for x in results) else 'COMPLETE_WITH_MEDIA_REVIEW_REQUIRED',
            'scope':'12 local public source files and timestamps only; no coordinate extraction/model inference/learning',
            'plan_sha256':hashlib.sha256((U/'LOCAL_PILOT_PLAN.json').read_bytes()).hexdigest(),
            'media_tool':'PyAV '+av.__version__,'tool_change_before_execution':'ffprobe unavailable; used existing PyAV for full decode and PTS validation',
            'bytes_extracted':sum(x['bytes'] for x in results),'clips':results,'signers':[1],
            'new_network_downloads':0,'training':0,'inference':0,'new_person_accuracy_measured':False}
    (U/'PILOT_RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (dest/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':result['status'],'bytes':result['bytes_extracted'],'pass':sum(x['pass'] for x in results),'frames':sum(x['decoded_frames'] for x in results),'dimensions':sorted({(x['width'],x['height']) for x in results})}))
except Exception:
    (U/'FIRST_PILOT_FAILURE.txt').write_text(traceback.format_exc(),encoding='utf-8')
    raise
