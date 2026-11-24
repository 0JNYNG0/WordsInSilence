"""Fixed public ID7/8 quality-evaluation inputs; --select never runs inference.

Reuses the prior full-source native observations and the exact 69-feature /
768-SignRep adapter. No labels, training, quality scores, private files or network.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
import traceback
import zipfile

sys.dont_write_bytecode = True
U = Path(__file__).resolve().parent
UNIT = U.parent
OPS = UNIT.parent
PRIOR = OPS / 'word100_stage_feedback_cycle_20261031'
BODY = OPS / 'word100_body_boundary_20261017'
RUNTIME = OPS / 'word100_signrep_feasibility_20261029'


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value, replace=False):
    path = Path(path)
    assert path.resolve().is_relative_to(U)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not replace and path.exists():
        raise FileExistsError(path)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temp.open('x', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
    os.replace(temp, path)


def event(**data):
    data = dict(at=now(), **data)
    with (U / 'EVENTS.jsonl').open('a', encoding='utf8') as stream:
        stream.write(json.dumps(data, ensure_ascii=False) + '\n')
    print(json.dumps(data, ensure_ascii=False), flush=True)


def adapters():
    path = PRIOR / 'prepare_bundle.py'
    spec = importlib.util.spec_from_file_location('frozen_prior_prepare_bundle', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.U = U  # Only the image renderer uses its output root; old files stay read-only.
    return mod


def select():
    existing = U / 'SELECTION.json'
    if existing.exists():
        selected = read(existing)
        for p, digest in selected['pins'].items():
            assert sha(p) == digest, p
        for row in selected['rows']:
            assert sha(row['record']) == row['record_sha256']
            assert sha(row['observation']) == row['observation_sha256']
        event(stage='selection_reused', clips=len(selected['rows']))
        return selected
    prior = read(PRIOR / 'SELECTION.json')
    known, novel = prior['words'], prior['novel_words']
    words = sorted(known + novel)
    assert len(words) == len(set(words)) == 16
    candidates = read(BODY / 'PREP_CONTEXT.json')['ordered_rows']
    candidates = sorted((r for r in candidates if r['role'] == 'train_candidate'
                         and r['signer_id'] in (7, 8) and r['word_code'] in words),
                        key=lambda r: (r['word_code'], r['signer_id']))
    assert len(candidates) == 32
    assert {(r['word_code'], r['signer_id']) for r in candidates} == {(w, i) for w in words for i in (7, 8)}
    old_ids = {r['clip_id'] for r in prior['rows']}
    old_hashes = {r['archive_source']['sha256'] for r in prior['rows']}
    old_members = {(r['archive_source']['archive'], r['archive_source']['member']) for r in prior['rows']}
    rows, seen, archives = [], set(), {}
    try:
        for index, r in enumerate(candidates, 1):
            record_path = BODY / 'data/records' / (r['clip_id'] + '.json')
            rec = read(record_path)
            assert rec['clip_id'] == r['clip_id'] and rec['role'] == 'train_candidate'
            assert rec['word_code'] == r['word_code'] and rec['signer_id'] == r['signer_id']
            assert r['split'] == '1.Training' and rec['source']['sha256'] not in old_hashes
            assert r['clip_id'] not in old_ids and rec['source']['sha256'] not in seen
            source = rec['source']
            assert (source['archive'], source['member']) not in old_members
            assert len(r['video_entries']) == 1
            entry = r['video_entries'][0]
            assert all(source[k] == entry[k] for k in ('archive', 'member', 'crc32', 'bytes'))
            if source['archive'] not in archives:
                archives[source['archive']] = zipfile.ZipFile(source['archive'])
            zi = archives[source['archive']].getinfo(source['member'])
            assert zi.CRC == source['crc32'] and zi.file_size == source['bytes']
            seen.add(source['sha256'])
            observation = Path(rec['views']['native']['observation'])
            assert observation.is_relative_to(BODY) and observation.is_file()
            assert rec['native_capture']['box'] is None
            rows.append(dict(alias=f'F{index:03}', clip_id=r['clip_id'], word_id=r['word_code'],
                label=r['canonical_label'], person=r['signer_id'],
                role='fresh_known' if r['word_code'] in known else 'fresh_expression',
                intervals=r['intervals'], record=str(record_path), record_sha256=sha(record_path),
                observation=str(observation), observation_sha256=sha(observation),
                frames=rec['frames'], box=rec['calibration']['box_source_pixels'],
                archive_source=source, retained_video=r.get('retained_video'),
                history='Public AIHub TRAIN / train_candidate; previously native body-boundary extraction and project TRAIN lineage. Fresh only relative to prior80 physical-head cohort, not universal unseen/C/T3F-independent data.'))
    finally:
        for archive in archives.values():
            archive.close()
    required = sum(r['archive_source']['bytes'] for r in rows) + 1024 ** 3
    free = shutil.disk_usage(U).free
    assert free >= required, f'Insufficient free space: {free} < conservative {required}'
    pins = [Path(__file__), UNIT/'AUTHORIZATION.json', PRIOR/'SELECTION.json', PRIOR/'prepare_bundle.py',
            BODY/'PREP_CONTEXT.json', OPS/'word100_signrep_temporal_head_20261029/prepare_inputs.py',
            OPS/'word100_guide_video_feedback_20261028/motion_observer.py',
            RUNTIME/'common.py', RUNTIME/'converted/signrep_features_fp32.xml',
            RUNTIME/'converted/signrep_features_fp32.bin', RUNTIME/'assets/ckpt.pt']
    selected = dict(status='FROZEN_BEFORE_FRESH_RGB_LABELS_OR_QUALITY_SCORES', at=now(),
        words=known, novel_words=novel, aliases='F001..F032 in (word_id, recorded_id) order',
        roles={'fresh_known': 24, 'fresh_expression': 8}, rows=rows,
        prior80_disjoint={'clip_ids': True, 'archive_members': True, 'expected_source_sha256': True,
                          'actual_extracted_source_sha256_verified_later': True},
        source_integrity='Record and PREP_CONTEXT archive/member/CRC/bytes agree; actual MP4 SHA verified before any decode/inference.',
        disk_preflight={'free_bytes': free, 'required_bytes_conservative': required},
        pins={str(p):sha(p) for p in pins},
        input_contract={'geometry_features':69, 'native_points':75, 'rgb_features':768,
                        'native_detector':'already extracted full source frame, then geometric ROI mask',
                        'rgb':'full source RGB frame, neutralize outside fixed ROI, resize224,16-frame SignRep/stride2, interpolate window centers',
                        'post_capture_future_context':True, 'labels_absent':True},
        restrictions=['No old held32 quality tuning or selection','No fresh quality labels/scores for design',
                      'No download/private/mentor/camera/Git','No old source or result writes'],
        review='Code/selection self-check only; independent parent review pending')
    write(existing, selected)
    event(stage='selection_frozen', clips=32, expected_unique_source_hashes=len(seen), source_bytes=sum(r['archive_source']['bytes'] for r in rows))
    return selected


def source_file(row):
    source = row['archive_source']
    if row['retained_video']:
        cache = Path(row['retained_video'])
        if cache.is_file():
            assert sha(cache) == source['sha256'], f'Public cache mismatch: {cache}'
            return cache, 'existing_exact_public_cache'
    dest = U / 'videos' / (row['alias'] + '.mp4')
    if dest.exists():
        assert sha(dest) == source['sha256']
        return dest, 'verified_fresh_cache'
    dest.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(source['archive']) as archive:
        zi = archive.getinfo(source['member'])
        assert zi.CRC == source['crc32'] and zi.file_size == source['bytes']
        data = archive.read(zi)
    assert hashlib.sha256(data).hexdigest() == source['sha256']
    temp = dest.with_suffix('.mp4.part')
    with temp.open('xb') as stream:
        stream.write(data)
    os.replace(temp, dest)
    return dest, 'extracted_existing_public_archive_exact_sha'


def prepare(selected):
    import numpy as np
    mod = adapters()
    comp, rows, visuals = None, [], []
    for row in selected['rows']:
        a = row['alias']
        checkpoint = U/'prepared'/(a+'.json')
        if checkpoint.exists():
            saved = read(checkpoint)
            for path, digest in saved['pins'].items():
                assert sha(path) == digest, path
            rows.append(saved['input']); visuals.append(saved['visual'])
            event(stage='completed_clip_reused', alias=a)
            continue
        source, method = source_file(row)
        assert sha(row['record']) == row['record_sha256'] and sha(row['observation']) == row['observation_sha256']
        obs = mod.read_observation(row['observation'])
        assert len(obs['frame_id']) == row['frames'] and np.array_equal(obs['frame_id'], np.arange(row['frames']))
        frames, sequence, states = mod.images(row, source, obs)
        b = row['box']; xy = obs['xy_px']
        obs['usable'] &= (xy[:,:,0]>=b[0]) & (xy[:,:,0]<b[2]) & (xy[:,:,1]>=b[1]) & (xy[:,:,1]<b[3])
        features = mod.extract_features(obs)
        x = np.nan_to_num(features['x']).astype(np.float32)
        assert x.shape == (row['frames'],69)
        if comp is None:
            event(stage='gpu_compile_start')
            comp = mod.compile_gpu()
        wf, ts, starts, seconds = mod.encode(source,b,len(x),comp)
        assert wf.shape == (len(starts),768) and np.isfinite(wf).all()
        assert np.max(np.abs(ts-obs['timestamp_us'])) <= 1
        mid = (ts[np.array(starts)]+ts[np.array(starts)+15])/2
        rgb = np.stack([np.interp(ts,mid,wf[:,j]) for j in range(768)],1).astype(np.float32)
        dest=U/'inputs'/(a+'.npz'); dest.parent.mkdir(exist_ok=True)
        if dest.exists():
            raise FileExistsError(f'Uncheckpointed input requires explicit recovery: {dest}')
        temp=dest.with_suffix('.npz.part')
        with temp.open('xb') as stream:
            np.savez_compressed(stream,x=x,mask=features['mask'],rgb=rgb,window_features=wf,
                                window_starts=starts,window_center_us=mid,frame_id=obs['frame_id'],timestamp_us=ts)
        os.replace(temp,dest)
        out={k:row[k] for k in ('alias','clip_id','word_id','label','person','role','intervals','record','record_sha256','observation','observation_sha256','box','history')}
        out.update(source=str(source),source_sha256=sha(source),source_method=method,input=str(dest),input_sha256=sha(dest),frames=len(x),windows=len(starts),rgb_seconds=seconds,reused_rgb=False,feature_names=features['feature_names'])
        visual=dict(alias=a,sequence=sequence,states=states,frames=frames,static_ordinals=[3,6,9],
                    left_right='Anatomical; non-mirrored. R usually screen left, follow arm chain when crossed.',
                    note='Original full-height visual evidence; ROI affects later inputs. Unlabeled public fresh head-evaluation cohort.')
        paths=[dest,Path(sequence),Path(states),source,Path(row['record']),Path(row['observation'])]+[Path(z['file']) for z in frames]
        write(checkpoint,dict(input=out,visual=visual,pins={str(p):sha(p) for p in paths}))
        rows.append(out);visuals.append(visual)
        event(stage='clip_complete',alias=a,frames=len(x),windows=len(starts),rgb_seconds=seconds)
    assert len(rows)==32 and len({r['source_sha256'] for r in rows})==32
    old_hashes={r['archive_source']['sha256'] for r in read(PRIOR/'SELECTION.json')['rows']}
    assert not old_hashes & {r['source_sha256'] for r in rows}
    write(U/'INPUTS.json',dict(status='UNLABELED_FRESH_PREPARED',rows=rows,feature_names=rows[0]['feature_names'],roles=selected['roles'],selection_sha256=sha(U/'SELECTION.json')))
    write(U/'VISUAL_INDEX.json',dict(rows=visuals,labels_absent=True,scope='Fresh head-evaluation inputs only; same-Codex future visual labels are not independent human gold'))
    write(U/'PREP_COMPLETED.json',dict(at=now(),clips=len(rows),frames=sum(r['frames'] for r in rows),windows=sum(r['windows'] for r in rows),source_hashes_unique_and_disjoint_old80=True,training_started=False,labels_created=False,old_sources_unchanged=True))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--gpu-slot-confirmed',action='store_true');args=parser.parse_args()
    if args.prepare and not args.gpu_slot_confirmed:
        raise SystemExit('Coordinate GPU slot with parent before --prepare --gpu-slot-confirmed')
    U.mkdir(parents=True,exist_ok=True)
    lock=U/'EXECUTION_LOCK.json'
    with lock.open('x',encoding='utf8') as stream:
        json.dump(dict(pid=os.getpid(),at=now(),prepare=args.prepare),stream)
    active=dict(pid=os.getpid(),started=now(),argv=sys.argv,stage='PREPARE' if args.prepare else 'SELECT_ONLY')
    write(U/'CURRENT_EXECUTION.json',dict(active=active,training_started=False),replace=True)
    status='FAILED'
    try:
        if args.prepare and (U/'PREP_COMPLETED.json').exists():
            raise RuntimeError('Completed extraction is frozen; do not rerun.')
        selected=select()
        if args.prepare:prepare(selected)
        status='UNLABELED_PREP_COMPLETE' if args.prepare else 'CODE_AND_SELECTION_READY_GPU_NOT_STARTED'
    except BaseException:
        error=traceback.format_exc();first=U/'FIRST_FAILURE.txt'
        if not first.exists():first.write_text(error,encoding='utf8')
        failure=U/'failures'/f'{time.time_ns()}.txt';failure.parent.mkdir(exist_ok=True);failure.write_text(error,encoding='utf8')
        event(stage='failure',file=str(failure))
        raise
    finally:
        write(U/'CURRENT_EXECUTION.json',dict(active=None,status=status,last=active,finished=now(),training_started=False),replace=True)
        if read(lock)['pid']==os.getpid():lock.unlink()


if __name__=='__main__':
    main()
