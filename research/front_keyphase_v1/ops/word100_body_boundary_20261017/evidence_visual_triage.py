"""Priority TRAIN geometry source evidence, without new detector or learning."""
from pathlib import Path
import json
import sys
import numpy as np
import av
from PIL import Image, ImageDraw

sys.dont_write_bytecode = True
U = Path(__file__).resolve().parent
sys.path.insert(0, str(U))
import prepare
from evidence_fallback import draw_frame, font, sha, read

IDS = ['NIA_SL_WORD1000_REAL02_F', 'NIA_SL_WORD0759_REAL07_F',
       'NIA_SL_WORD1133_REAL06_F', 'NIA_SL_WORD0760_REAL07_F',
       'NIA_SL_WORD1085_REAL06_F']


def sheet(cid, label, bottom, ids, images, xy, old, ts, core, gap, suffix):
    cols, tw, th = 3, 640, 360
    h = ((len(ids) + cols - 1) // cols) * (th + 35) + 80
    canvas = Image.new('RGB', (cols * tw, h), (16, 22, 32))
    d = ImageDraw.Draw(canvas)
    d.text((14, 7), f'{label} / {cid} / {suffix} / 하단 {bottom}px', font=font(25), fill='white')
    d.text((14, 45), '원본 전체 화면·기존 관측 / 노랑 하단·붉은 제외 영역 / 초록 L·하늘 R', font=font(21), fill=(205,214,226))
    for n, i in enumerate(ids):
        x, y = (n % cols) * tw, 80 + (n // cols) * (th + 35)
        canvas.paste(draw_frame(images[i], xy[i], old[i], bottom, f'f{i} / {ts[i]/1e6:.3f}s'), (x,y))
        label2 = ('핵심 주석: ' if core[i] else '구간 밖: ') + ('ROI 적용 후 유효 손 관측 0' if gap[i] else '유효 손 관측 남음')
        d.text((x+8,y+th+3), label2, font=font(18), fill=(245,150,130) if gap[i] else (220,225,235))
    out = U / f'evidence_visual_triage_{cid}_{suffix}_v2.png'
    assert not out.exists()
    canvas.save(out)
    return out


def main():
    prepare.configure()
    context = read(U / 'PREP_CONTEXT.json')
    rows = {r['clip_id']:r for r in context['ordered_rows'] if r['role']=='train_candidate'}
    triage = read(U / 'training_geometry_triage_RESULT.json')
    chosen = {r['clip_id']:r for r in triage['rows'] if r['clip_id'] in IDS}
    assert len(chosen) == len(IDS)
    for cid in IDS:
        row, item = rows[cid], chosen[cid]
        rec = read(row['legacy_record_path'])
        assert sha(rec['observation']) == rec['observation_sha256']
        local = U / 'review_sources' / (cid + '.mp4')
        if local.exists():
            assert sha(local) == rec['source_sha256']
            source = local
        else:
            source, _ = prepare.legacy.source_file(row, rec, U / 'review_sources')
        with np.load(rec['observation'], allow_pickle=False) as z:
            xy, us, ob, ts, fid = (z[k] for k in ('xy_px','usable','observed','timestamp_us','frame_id'))
        assert np.array_equal(fid, np.arange(len(fid)))
        old = us & ob & np.isfinite(xy).all(-1)
        bottom, width = item['bottom_pixels'], item['source_width']
        new = old & (xy[:,:,0]>=0) & (xy[:,:,0]<width) & (xy[:,:,1]>=0) & (xy[:,:,1]<bottom)
        core = (ts >= round(item['core_seconds'][0]*1e6)) & (ts <= round(item['core_seconds'][1]*1e6))
        hand_before = old[:,33:75].any(1)
        gap = core & hand_before & ~new[:,33:75].any(1)
        ci = np.flatnonzero(core)
        overview = sorted(set([0,len(ts)-1,*np.linspace(ci[0],ci[-1],8).round().astype(int).tolist()]))
        focus = set()
        for loss in item['all_hand_loss_runs']:
            focus.update(range(max(0,loss['start_index']-3),min(len(ts),loss['end_index']+4)))
        for hand in item['hands'].values():
            for e in hand['evidence_frames']:
                k = e['frame_id']
                focus.update(range(max(0,k-1),min(len(ts),k+2)))
        focus = sorted(focus)
        selected = set(overview) | set(focus)
        images, pts = {}, []
        with av.open(str(source)) as stream:
            for i, frame in enumerate(stream.decode(video=0)):
                pts.append(round(float(frame.pts * frame.time_base) * 1e6))
                if i in selected:
                    images[i] = frame.to_ndarray(format='rgb24')
        assert len(pts)==len(ts) and set(images)==selected
        pts = np.asarray(pts,dtype=np.int64)
        assert np.max(np.abs((pts-pts[0])-(ts-ts[0]))) <= 1
        post_roi_no_hand = ~new[:,33:75].any(1)
        files = [sheet(cid,item['label'],bottom,overview,images,xy,old,ts,core,post_roi_no_hand,'overview')]
        # Split into twelve tiles for readable loss-before/after evidence.
        for offset in range(0,len(focus),12):
            files.append(sheet(cid,item['label'],bottom,focus[offset:offset+12],images,xy,old,ts,core,post_roi_no_hand,f'gap_{offset//12+1}'))
        meta = {'clip_id':cid,'label':item['label'],'role':'train_candidate','source':str(source),
                'source_sha256':sha(source),'observation':rec['observation'],'observation_sha256':rec['observation_sha256'],
                'source_entry':row['video_entries'][0], 'method':item['method'],'bottom':bottom,
                'frame_count':len(ts),'core_seconds':item['core_seconds'],'core_frames':[int(ci[0]),int(ci[-1])],
                'overview_frames':overview,'focus_frames':focus,
                'selected_frame_timestamps':[{'frame':int(i),'timestamp_us':int(ts[i]),'source_pts_us':int(pts[i]),'core':bool(core[i]),'no_valid_hand_after_roi':bool(post_roi_no_hand[i])} for i in sorted(selected)],
                'PTS_alignment_verified_max_error_us':int(np.max(np.abs((pts-pts[0])-(ts-ts[0])))),
                'triage_row':item,'triage_sha256':sha(U/'training_geometry_triage_RESULT.json'),
                'images':{str(p):sha(p) for p in files},'new_detector_runs':0,'training_runs':0,
                'limits':'Original RGB with existing legacy joints. Sparse overall samples plus all-hand-loss windows and selected near-boundary motion. No linguistic expert judgement; unsampled partial finger losses remain possible.',
                'display_correction':'v1 caption said both hands outside ROI when no valid hand remained. This incorrectly included a source-visible hand already missed by the legacy detector. v2 uses no-valid-hand wording, also evaluated outside core. First v1 evidence preserved; boundaries and observations unchanged.'}
        out=U/f'evidence_visual_triage_{cid}_v2.json'
        with out.open('x',encoding='utf-8') as f:json.dump(meta,f,ensure_ascii=False,indent=2)
        print(json.dumps({'clip_id':cid,'images':list(meta['images']),'focus_frames':focus},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
