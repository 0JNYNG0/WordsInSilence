"""Public time-local evidence preparation; no labels, training, scores or network.

Existing 48 feature arrays are referenced by immutable hash. Only eight fixed
new-expression clips invoke the existing frozen SignRep GPU encoder. All new
files are contained in this unit; annotation manifests contain opaque IDs only.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
import random
import shutil
import sys
import time
import traceback
import zipfile

sys.dont_write_bytecode = True
U = Path(__file__).resolve().parent
OPS = U.parent
PRIOR = OPS / 'word100_stage_feedback_cycle_20261031'
BODY = OPS / 'word100_body_boundary_20261017'
RUNTIME = OPS / 'word100_signrep_feasibility_20261029'


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value, replace=False):
    path = Path(path).resolve()
    assert path.is_relative_to(U)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(path)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('x', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(tmp, path)


def event(**data):
    row = dict(at=now(), **data)
    with (U / 'PREP_EVENTS.jsonl').open('a', encoding='utf8') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(json.dumps(row, ensure_ascii=False), flush=True)


def adapters():
    spec = importlib.util.spec_from_file_location('unchanged_public_preparer', PRIOR / 'prepare_bundle.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.U = U
    return mod


def known_feature_history(new_ids):
    # Bounded metadata inventories, never label arrays or score/results files.
    names = [
        'word100_stage_feedback_cycle_20261031/SELECTION.json',
        'word100_feature_support_streaming_20261031/fresh/SELECTION.json',
        'word100_signrep_temporal_head_20261029/INPUTS.json',
        'word100_shared_feature_bundle_20261027/FROZEN.json',
        'word100_semantic_episode_pilot_20261031/AUTHORIZATION.json',
        'word100_description_evidence_feedback_20261101/AUTHORIZATION.json',
    ]
    checks = []
    for name in names:
        p = OPS / name
        d = read(p)
        ids = {r['clip_id'] for r in d['rows'] if 'clip_id' in r}
        matches = sorted(set(new_ids) & ids)
        assert not matches, ('Previously selected feature-head clip', name, matches)
        checks.append(dict(path=str(p), sha256=sha(p), clips=len(ids), matches=matches))
    return dict(checked_manifests=checks,
        result='No overlap with these existing feature-head cohorts; not an exhaustive project-history absence proof.',
        prior_history='All new clips are public TRAIN / native body-boundary and reference-bank lineage. TCN TRAIN use and prior numeric bank work remain disclosed; not universally unseen.')


def select():
    dest = U / 'SELECTION.json'
    if dest.exists():
        result = read(dest)
        for p, h in result['pins'].items():
            assert sha(p) == h, p
        event(stage='selection_reused', clips=len(result['rows']))
        return result
    old = read(PRIOR / 'SELECTION.json')
    old_inputs = {r['clip_id']: r for r in read(PRIOR / 'INPUTS.json')['rows']}
    old_visuals = {r['alias']: r for r in read(PRIOR / 'VISUAL_INDEX.json')['rows']}
    known = set(old['words'])
    excluded_words = known | set(old['novel_words'])
    assert len(known) == 12 and len(excluded_words) == 16
    pool = read(BODY / 'PREP_CONTEXT.json')['ordered_rows']
    pool = [r for r in pool if r['role'] == 'train_candidate']
    new_words = sorted({r['word_code'] for r in pool} - excluded_words)[:4]
    previous = [r for r in old['rows'] if r['word_id'] in known and r['person'] in (1, 2, 3, 4)]
    fresh = sorted((r for r in pool if r['word_code'] in new_words and r['signer_id'] in (7, 8)),
                   key=lambda r: (r['word_code'], r['signer_id']))
    assert len(previous) == 48 and len(fresh) == 8
    assert {(r['word_code'], r['signer_id']) for r in fresh} == {(w, i) for w in new_words for i in (7, 8)}
    history = known_feature_history([r['clip_id'] for r in fresh])
    rows = []
    for r in previous:
        inp = old_inputs[r['clip_id']]
        assert sha(inp['input']) == inp['input_sha256']
        rec = read(r['record'])
        vis = old_visuals[r['alias']]
        centers = [next(f['frame_id'] for f in vis['frames'] if f['ordinal'] == o) for o in (3, 6, 9)]
        rows.append(dict(clip_id=r['clip_id'], word_id=r['word_id'], label=r['label'], person=r['person'],
            role=r['role'], intervals=r['intervals'], record=r['record'], record_sha256=r['record_sha256'],
            observation=r['observation'], observation_sha256=sha(r['observation']), archive_source=r['archive_source'],
            box=r['box'], frames=rec['frames'], input=inp['input'], input_sha256=inp['input_sha256'],
            feature_names=inp['feature_names'], windows=inp['windows'], reused_input=True,
            existing_source=inp['source'], centers=centers, old_alias=r['alias']))
    for r in fresh:
        p = BODY / 'data/records' / (r['clip_id'] + '.json')
        rec = read(p)
        assert rec['role'] == 'train_candidate' and rec['word_code'] == r['word_code']
        assert rec['signer_id'] == r['signer_id'] and rec['native_capture']['box'] is None
        s = rec['source']; e = r['video_entries']
        assert len(e) == 1 and all(s[k] == e[0][k] for k in ('archive', 'member', 'bytes', 'crc32'))
        with zipfile.ZipFile(s['archive']) as z:
            zi = z.getinfo(s['member'])
            assert zi.CRC == s['crc32'] and zi.file_size == s['bytes']
        observation = rec['views']['native']['observation']
        assert Path(observation).is_relative_to(BODY)
        rows.append(dict(clip_id=r['clip_id'], word_id=r['word_code'], label=r['canonical_label'], person=r['signer_id'],
            role='new_feature_expression_evaluation', intervals=r['intervals'], record=str(p), record_sha256=sha(p),
            observation=observation, observation_sha256=sha(observation), archive_source=s,
            box=rec['calibration']['box_source_pixels'], frames=rec['frames'], reused_input=False,
            existing_source=r.get('retained_video'), centers=None))
    assert len({r['clip_id'] for r in rows}) == len({r['archive_source']['sha256'] for r in rows}) == 56
    rows.sort(key=lambda r: (r['word_id'], r['person']))
    random.Random(2026110101).shuffle(rows)
    for i, row in enumerate(rows, 1):
        row['alias'] = f'S{i:03}'
    protected = read(OPS / 'word100_semantic_episode_pilot_20261031/FROZEN.json')['protected']
    assert len(protected) == 32
    for p, h in protected.items():
        assert sha(p) == h, p
    free = shutil.disk_usage(U).free
    assert free >= 2 * 1024 ** 3, ('Free space below bounded preparation estimate', free)
    pins = [Path(__file__), U/'AUTHORIZATION.json', PRIOR/'SELECTION.json', PRIOR/'INPUTS.json',
        PRIOR/'VISUAL_INDEX.json', PRIOR/'prepare_bundle.py', BODY/'PREP_CONTEXT.json',
        OPS/'word100_signrep_temporal_head_20261029/prepare_inputs.py',
        OPS/'word100_guide_video_feedback_20261028/motion_observer.py', RUNTIME/'common.py',
        RUNTIME/'converted/signrep_features_fp32.xml', RUNTIME/'converted/signrep_features_fp32.bin']
    result = dict(status='SELECTION_FIXED_BEFORE_NEW_RGB_LABELS_AND_HEAD_SCORES', at=now(), rows=rows,
        roles={'fit':36, 'development':12, 'new_feature_expression_evaluation':8},
        new_words=new_words, excluded_prior16_words=sorted(excluded_words), opaque_alias_seed=2026110101,
        new_selection='First four lexicographic words outside prior16 x recordedIDs7/8, no outcome-based replacement.',
        history_audit=history, protected=protected, pins={str(p):sha(p) for p in pins},
        reuse_contract='48 existing full arrays by path+SHA; no recomputation. Only8 new clips encode frozen SignRep features.',
        window_contract='Three centers at previous uniform ordinals3/6/9, each center-6..center+6 original frames; no boundary/label-based center selection.',
        annotation_separation='VISUAL_INDEX has only opaque aliases, original frame/PTS and opaque image paths. This SELECTION is not an annotator input.',
        no_labels=True, no_training=True, no_scores_read=True, free_bytes=free)
    write(dest, result)
    event(stage='selection_frozen', clips=56, reused_inputs=48, new_gpu_clips=8)
    return result


def source_file(row):
    s = row['archive_source']
    if row.get('existing_source') and Path(row['existing_source']).is_file():
        p = Path(row['existing_source'])
        assert sha(p) == s['sha256']
        return p, 'existing_exact_public_cache'
    p = U/'videos'/(row['alias']+'.mp4')
    if p.exists():
        assert sha(p) == s['sha256']
        return p, 'new_exact_cache_reused'
    p.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(s['archive']) as z:
        zi = z.getinfo(s['member'])
        assert zi.CRC == s['crc32'] and zi.file_size == s['bytes']
        data = z.read(zi)
    assert hashlib.sha256(data).hexdigest() == s['sha256']
    tmp = p.with_suffix('.mp4.part')
    with tmp.open('xb') as f:
        f.write(data)
    os.replace(tmp, p)
    return p, 'existing_public_archive_exact_sha'


def centers_for(row, obs):
    import numpy as np
    if row['centers'] is not None:
        centers = row['centers']
    else:
        ts = obs['timestamp_us']; a = min(v[0] for v in row['intervals']); b = max(v[1] for v in row['intervals'])
        wanted = np.linspace(max(0, a-.12), min(ts[-1]/1e6, b+.12), 13)*1e6
        indices = [int(np.abs(ts-v).argmin()) for v in wanted]
        assert len(set(indices)) == 13
        centers = [indices[i] for i in (3, 6, 9)]
    assert len(set(centers)) == 3 and all(6 <= c < row['frames']-6 for c in centers)
    return centers


def make_visuals(row, source, obs, centers):
    import av
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    folder = U/'visuals'/row['alias']; folder.mkdir(parents=True, exist_ok=True)
    wanted = sorted({j for c in centers for j in range(c-6, c+7)})
    images = {}; timestamps = {}
    with av.open(str(source)) as video:
        for j, frame in enumerate(video.decode(video=0)):
            if j in wanted:
                images[j] = frame.to_image()
                timestamps[j] = round(float(frame.pts*frame.time_base)*1e6)
            if j >= wanted[-1]:
                break
    assert set(images) == set(wanted)
    assert all(abs(timestamps[j]-int(obs['timestamp_us'][j])) <= 1 for j in wanted)
    xy = obs['xy_px']; observed = obs['observed'] & np.isfinite(xy).all(-1)
    shoulders = xy[wanted][:, [11, 12]].reshape(-1, 2)
    center = np.nanmedian(shoulders, axis=0)
    width, height = images[wanted[0]].size
    if not np.isfinite(center).all():
        center = np.array([width/2, height/3])
    body_box = (max(0, int(center[0]-620)), 0, min(width, int(center[0]+620)), height)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 19)
    bigfont = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 23)
    source_files = {}
    for fid, im in images.items():
        p = folder/f'f{fid:04}.jpg'
        if not p.exists():
            im.save(p, quality=94)
        source_files[fid] = dict(frame_id=fid, timestamp_us=timestamps[fid], seconds=timestamps[fid]/1e6, file=str(p))

    def body(im):
        annotated = im.copy(); draw = ImageDraw.Draw(annotated)
        x0,y0,x1,y1 = row['box']
        draw.line((x0,y1,x1,y1), fill=(240,65,55), width=5)
        return annotated.crop(body_box)

    def hand_box(fid, side):
        off = 54 if side == 'R' else 33
        pts = xy[fid,off:off+21][observed[fid,off:off+21]]
        proposal = 'observed_hand_points'
        if len(pts):
            lo, hi = pts.min(0), pts.max(0); cx,cy = (lo+hi)/2
            radius = max(float(np.max(hi-lo))*.8, 75)
        else:
            cx,cy = xy[fid,16 if side == 'R' else 15]; radius = 140; proposal = 'body_wrist_fallback_visual_only'
        if not np.isfinite([cx,cy]).all():
            cx,cy = center; radius = 280; proposal = 'body_center_fallback_visual_only'
        bb = (max(0,int(cx-radius)),max(0,int(cy-radius)),min(width,int(cx+radius)),min(height,int(cy+radius)))
        if bb[2] <= bb[0] or bb[3] <= bb[1]:
            bb = body_box; proposal = 'body_view_fallback_visual_only'
        return bb, proposal

    def paste_fit(canvas, im, box):
        cut = im.copy(); cut.thumbnail((box[2]-box[0],box[3]-box[1]))
        canvas.paste(cut, (box[0]+(box[2]-box[0]-cut.width)//2,box[1]+(box[3]-box[1]-cut.height)//2))

    windows=[]
    for wi,c in enumerate(centers):
        sid=f"{row['alias']}_W{wi}"; ids=list(range(c-6,c+7))
        sheet=Image.new('RGB',(1600,2400),'#20252b'); draw=ImageDraw.Draw(sheet)
        draw.text((10,6),sid+' | 13 original frames | red line: fixed ROI bottom | R/L crops are proposals',font=bigfont,fill='white')
        frame_rows=[]
        for n,fid in enumerate(ids):
            xx=(n%2)*800; yy=40+(n//2)*335
            draw.text((xx+8,yy+3),f"{'CENTER ' if fid==c else ''}f{fid} {timestamps[fid]/1e6:.3f}s",font=font,fill='#ffdd78' if fid==c else 'white')
            paste_fit(sheet,body(images[fid]),(xx+3,yy+30,xx+280,yy+330))
            proposals={}
            for si,side in enumerate(('R','L')):
                bb,method=hand_box(fid,side); proposals[side]=dict(box=list(bb),method=method)
                paste_fit(sheet,images[fid].crop(bb),(xx+285+si*255,yy+52,xx+535+si*255,yy+325))
                draw.text((xx+290+si*255,yy+30),side+' proposal',font=font,fill='white')
            frame_rows.append(dict(**source_files[fid],crop_proposals=proposals))
        sheetpath=folder/f'window_{wi}.jpg'; sheet.save(sheetpath,quality=94)
        windows.append(dict(sample_id=sid,alias=row['alias'],window_index=wi,center_frame=c,
            center_timestamp_us=timestamps[c],frame_ids=ids,timestamps_us=[timestamps[j] for j in ids],
            center_file=source_files[c]['file'],sheet=str(sheetpath),frames=frame_rows,
            roi_box_source_pixels=row['box'],source_image_size=[width,height],
            source_frame_convention='zero-based native original frame; inclusive13-frame window',
            visual_note='Original RGB; detector hand crops only proposals. Check anatomical arm chain and whole-body view. Red line is fixed ROI bottom; MP missing is not an RGB label.'))
    detail=Image.new('RGB',(1800,1500),'#20252b'); draw=ImageDraw.Draw(detail)
    for wi,c in enumerate(centers):
        yy=wi*500
        draw.text((10,yy+7),f"{row['alias']}_W{wi} CENTER f{c} {timestamps[c]/1e6:.3f}s | red=ROI bottom",font=bigfont,fill='white')
        paste_fit(detail,body(images[c]),(0,yy+42,600,yy+495))
        for si,side in enumerate(('R','L')):
            bb,_=hand_box(c,side);paste_fit(detail,images[c].crop(bb),(600+si*600,yy+65,1200+si*600,yy+495))
            draw.text((608+si*600,yy+38),side+' proposal - verify arm connection',font=font,fill='white')
    statepath=folder/'states.jpg'; detail.save(statepath,quality=94)
    for w in windows:w['states']=str(statepath)
    paths=[statepath]+[Path(w['sheet']) for w in windows]+[Path(v['file']) for v in source_files.values()]
    return dict(alias=row['alias'],states=str(statepath),windows=windows),paths


def publish_visuals(visuals):
    write(U/'VISUAL_INDEX.json',dict(status='PARTIAL' if len(visuals)<56 else 'COMPLETE',
        rows=visuals,samples=[w for r in visuals for w in r['windows']],
        clips_complete=len(visuals),windows_complete=3*len(visuals),expected_clips=56,expected_windows=168,
        metadata_blind=True,no_labels=True,source_words_roles_people_absent=True),replace=True)


def prepare(selected):
    import numpy as np
    mod=adapters();comp=None;inputs=[];visuals=[]
    for row in selected['rows']:
        alias=row['alias'];checkpoint=U/'prepared'/(alias+'.json')
        if checkpoint.exists():
            saved=read(checkpoint)
            for p,h in saved['pins'].items():assert sha(p)==h,p
            inputs.append(saved['input']);visuals.append(saved['visual']);publish_visuals(visuals)
            event(stage='verified_checkpoint_reused',alias=alias)
            continue
        assert sha(row['record'])==row['record_sha256'] and sha(row['observation'])==row['observation_sha256']
        source,method=source_file(row);obs=mod.read_observation(row['observation'])
        assert len(obs['frame_id'])==row['frames'] and np.array_equal(obs['frame_id'],np.arange(row['frames']))
        centers=centers_for(row,obs)
        visual,image_paths=make_visuals(row,source,obs,centers)
        # Publish images before encoding so independent annotators can begin.
        visuals.append(visual);publish_visuals(visuals)
        event(stage='images_ready',alias=alias,samples=[w['sample_id'] for w in visual['windows']],clips_ready=len(visuals))
        if row['reused_input']:
            assert sha(row['input'])==row['input_sha256']
            with np.load(row['input'],allow_pickle=False) as z:
                assert z['x'].shape==(row['frames'],69) and z['rgb'].shape==(row['frames'],768)
                assert np.array_equal(z['frame_id'],obs['frame_id']) and np.max(abs(z['timestamp_us']-obs['timestamp_us']))<=1
            feature_names=row['feature_names'];dest=Path(row['input']);seconds=0;windows=row['windows']
        else:
            b=row['box'];xy=obs['xy_px']
            obs['usable'] &= (xy[:,:,0]>=b[0])&(xy[:,:,0]<b[2])&(xy[:,:,1]>=b[1])&(xy[:,:,1]<b[3])
            features=mod.extract_features(obs);x=np.nan_to_num(features['x']).astype(np.float32)
            assert x.shape==(row['frames'],69)
            if comp is None:event(stage='gpu_compile_start');comp=mod.compile_gpu()
            wf,ts,starts,seconds=mod.encode(source,b,len(x),comp)
            assert wf.shape==(len(starts),768) and np.isfinite(wf).all()
            assert np.max(abs(ts-obs['timestamp_us']))<=1
            mid=(ts[np.array(starts)]+ts[np.array(starts)+15])/2
            rgb=np.stack([np.interp(ts,mid,wf[:,j]) for j in range(768)],1).astype(np.float32)
            dest=U/'inputs'/(alias+'.npz');dest.parent.mkdir(exist_ok=True)
            assert not dest.exists(),'Uncheckpointed input exists; preserve and inspect before recovery'
            temp=dest.with_suffix('.npz.part')
            with temp.open('xb') as f:
                np.savez_compressed(f,x=x,mask=features['mask'],rgb=rgb,window_features=wf,window_starts=starts,
                    window_center_us=mid,frame_id=obs['frame_id'],timestamp_us=ts)
            os.replace(temp,dest);feature_names=features['feature_names'];windows=len(starts)
        out={k:row[k] for k in ['alias','clip_id','word_id','label','person','role','record','record_sha256','observation','observation_sha256','box','frames','reused_input']}
        out.update(source=str(source),source_sha256=sha(source),source_method=method,input=str(dest),input_sha256=sha(dest),
            centers=centers,windows=windows,rgb_seconds=seconds,feature_names=feature_names)
        paths=[dest,source,Path(row['record']),Path(row['observation'])]+image_paths
        write(checkpoint,dict(input=out,visual=visual,pins={str(p):sha(p) for p in paths}))
        inputs.append(out);event(stage='clip_complete',alias=alias,frames=row['frames'],reused_input=row['reused_input'],gpu_seconds=seconds)
    assert len(inputs)==56 and sum(r['reused_input'] for r in inputs)==48
    assert len({r['source_sha256'] for r in inputs})==56
    assert sum(len(v['windows']) for v in visuals)==168
    for p,h in selected['protected'].items():assert sha(p)==h,p
    for r in inputs:assert sha(r['input'])==r['input_sha256']
    write(U/'INPUTS.json',dict(status='PREPARED_NO_LABELS',rows=inputs,roles=selected['roles'],
        feature_names=inputs[0]['feature_names'],selection_sha256=sha(U/'SELECTION.json')))
    write(U/'PREP_COMPLETED.json',dict(at=now(),clips=56,windows=168,unique_center_samples=168,
        old_inputs_reused_unchanged=48,new_fixed_signrep_inputs=8,protected_sha_verified=32,
        original_frames_preserved=True,visuals_full_source=True,roi_line_display_only=True,
        labels_created=False,training_started=False,scores_read=False,private_data_used=False,
        freshness='Eight new clips absent from bounded known feature-cohort manifests; previous public TCN TRAIN/reference-bank history remains.'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    U.mkdir(parents=True,exist_ok=True);lock=U/'PREP_EXECUTION_LOCK.json'
    with lock.open('x',encoding='utf8') as f:json.dump(dict(pid=os.getpid(),at=now()),f)
    active=dict(pid=os.getpid(),started=now(),argv=sys.argv,stage='PREPARING' if args.prepare else 'SELECT_ONLY')
    write(U/'PREP_CURRENT_EXECUTION.json',dict(active=active),replace=True)
    status='FAILED'
    try:
        if args.prepare:assert not (U/'PREP_COMPLETED.json').exists(),'Completed preparation must not be rerun'
        selected=select()
        if args.prepare:prepare(selected)
        status='PREPARED_NO_LABELS' if args.prepare else 'SELECTION_READY'
    except BaseException:
        error=traceback.format_exc();first=U/'PREP_FIRST_FAILURE.txt'
        if not first.exists():first.write_text(error,encoding='utf8')
        failure=U/'prep_failures'/f'{time.time_ns()}.txt';failure.parent.mkdir(exist_ok=True);failure.write_text(error,encoding='utf8')
        event(stage='failure',file=str(failure));raise
    finally:
        write(U/'PREP_CURRENT_EXECUTION.json',dict(active=None,status=status,last=active,finished=now()),replace=True)
        if read(lock)['pid']==os.getpid():lock.unlink()


if __name__=='__main__':
    main()
