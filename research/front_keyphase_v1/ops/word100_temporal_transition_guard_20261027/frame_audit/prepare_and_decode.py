import hashlib,json,traceback
from pathlib import Path
from datetime import datetime,timezone
import av
import numpy as np
from PIL import Image

OUT=Path(__file__).resolve().parent
OLD=OUT.parent.parent/'word100_temporal_shape_context_20261027'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'PIXEL_DECODE_AUDIT.json').exists():raise RuntimeError('Completed audit already exists; do not overwrite')
    index=json.loads((OLD/'visual_inputs/INDEX.json').read_text(encoding='utf-8'))
    pins={str(p):sha(p) for p in [OLD/'FROZEN.json',OLD/'RESULT_KO.html',OLD/'visual_inputs/INDEX.json',OLD/'visual_inputs/build.py']}
    expect=[];tilechecks=[];step=[];ordered_k=[]
    for r in index['rows']:
        assert sha(r['source'])==r['source_sha256'];pins[r['source']]=r['source_sha256']
        fs=r['frames'];sheet=Image.open(r['ordered_contact']).convert('RGB');pins[r['ordered_contact']]=sha(r['ordered_contact'])
        tile_rows=[];frame_rows=[]
        for j,f in enumerate(fs):
            focus=Image.open(f['focus_file']).convert('RGB');full=Image.open(f['full_file']).convert('RGB')
            for q in ['focus_file','full_file']:pins[f[q]]=sha(f[q])
            assert hashlib.sha256(np.asarray(full).tobytes()).hexdigest()==f['rgb_sha256']
            tile=focus.copy();tile.thumbnail((472,360),Image.Resampling.LANCZOS)
            x=(j%4)*480+(480-tile.width)//2;y=(j//4)*400+36
            assert np.array_equal(np.asarray(tile),np.asarray(sheet.crop((x,y,x+tile.width,y+tile.height))))
            tile_rows.append({'index':j,'frame_id':f['frame_id'],'image_key':f['image_key'],'ordered_tile_pixel_exact':True})
            frame_rows.append({'index':j,'frame_id':f['frame_id'],'pts_us':f['actual_pts_us'],'offset_frames':f['offset_frames'],'image_key':f['image_key'],'focus_file':f['focus_file'],'full_file':f['full_file'],'rgba_sha256':hashlib.sha256(np.asarray(focus.convert('RGBA')).tobytes()).hexdigest(),'width':focus.width,'height':focus.height})
            if j:
                assert f['frame_id']-fs[j-1]['frame_id']==1
                step.append(f['actual_pts_us']-fs[j-1]['actual_pts_us'])
        ids=[f['image_key'] for f in fs];ordered_k.append({'case_id':r['case_id'],'keys_in_time_order':ids,'numerically_sorted':ids==sorted(ids)})
        tilechecks.append({'case_id':r['case_id'],'tile_count':13,'tiles':tile_rows})
        expect.append({'case_id':r['case_id'],'frames':frame_rows})
    decoded=[]
    for cid in ['T15','T17']:
        r=next(x for x in index['rows'] if x['case_id']==cid);selected={f['frame_id']:f for f in r['frames']};pts=[];checks=[]
        with av.open(r['source']) as c:
            s=c.streams.video[0];meta={'average_rate':str(s.average_rate),'base_rate':str(s.base_rate),'time_base':str(s.time_base),'reported_frames':s.frames,'codec':s.codec_context.name}
            first=None
            for n,fr in enumerate(c.decode(s)):
                if first is None:first=fr.pts
                t=round(float((fr.pts-first)*fr.time_base)*1e6);pts.append(t)
                if n in selected:
                    f=selected[n];arr=fr.to_ndarray(format='rgb24');saved=np.asarray(Image.open(f['full_file']).convert('RGB'))
                    assert np.array_equal(arr,saved);assert t==f['actual_pts_us']
                    checks.append({'frame_id':n,'pts_us':t,'rgb_exact':True,'image_key':f['image_key'],'source_frame_rgb_sha256':hashlib.sha256(arr.tobytes()).hexdigest()})
        assert len(checks)==13
        if cid=='T15':assert {58,59}<={x['frame_id'] for x in checks}
        decoded.append({'case_id':cid,'source':r['source'],'source_sha256':sha(r['source']),'stream':meta,'whole_source_decoded_frames':len(pts),'whole_source_adjacent_pts_delta_us_values':sorted(set(np.diff(pts).tolist())),'selected':checks})
    after={p:sha(p) for p in pins};assert pins==after
    dump('SOURCE_PINS_BEFORE.json',pins);dump('SOURCE_PINS_AFTER_PIXEL.json',after)
    dump('EXPECTED_BROWSER.json',{'html':str(OLD/'RESULT_KO.html'),'rows':expect})
    result={'status':'PASS','at':datetime.now(timezone.utc).isoformat(),'all_ordered_tiles_exact':234,'windows':18,'adjacent_forward_transitions':len(step),'adjacent_pts_delta_us_values':sorted(set(step)),'frame_id_step':1,'neutral_key_mapping':ordered_k,'tile_checks':tilechecks,'source_decode':decoded,'source_pin_count':len(pins),'original_files_modified':False,'source_capture_limit':'Stored stream has sequential frames/PTS. This cannot establish camera capture frame rate before encoding, exposure duration, dropped upstream frames, or unsampled faster motion.'}
    dump('PIXEL_DECODE_AUDIT.json',result)
    dump('EARLY_COMMAND_NOTES.json',{'nonpipeline_failures_preserved':['Read-only discovery command used default cp949 and raised UnicodeDecodeError while reading UTF-8 INDEX; rerun explicitly UTF-8 succeeded.','Initial new-unit directory listing failed before parent created unit; no source/output replaced.'],'python_playwright_missing':'Using bundled Node Playwright, no install/download.'})
    print(json.dumps({'status':'PASS','tiles':234,'source_decode_frames':26,'adjacent_deltas_us':sorted(set(step)),'pins':len(pins)}))
if __name__=='__main__':
    try:main()
    except Exception:
        if not (OUT/'FIRST_ERROR.txt').exists():(OUT/'FIRST_ERROR.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
