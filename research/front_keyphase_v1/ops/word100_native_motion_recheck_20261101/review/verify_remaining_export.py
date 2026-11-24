"""Independent remaining15 native export verification; reuse prior successful checker logic.

No producer script is imported or rerun. Only the final verification JSON is
written. Prior raw9 files are read only to verify exact preservation.
"""
from pathlib import Path
from fractions import Fraction
import collections
import datetime
import hashlib
import json
import sys
sys.dont_write_bytecode = True
import av
import numpy as np
from PIL import Image, ImageDraw

N = Path(__file__).resolve().parent.parent
S = N / 'full24_native'
U = N.parent / 'word100_core_feature_split_20261101'
W = N.parent / 'word100_interval_motion_20261101'
OUT = N / 'review/REMAINING_EXPORT_VERIFICATION.json'
checks, read_pins = [], {}


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def ck(group, name, okay):
    checks.append(dict(group=group, name=name, passed=bool(okay)))


def pin(group, path, expected):
    p = str(path)
    actual = sha(p)
    ck(group, p, actual == expected)
    if p in read_pins:
        ck('pin_consistency', p, read_pins[p] == actual)
    read_pins[p] = actual


def png(p):
    with Image.open(p) as im:
        ck('png_encoding', str(p), im.format == 'PNG' and im.mode == 'RGB')
        return np.array(im)


def main():
    if OUT.exists():
        raise FileExistsError('Reuse the immutable completed verification.')
    manifest, blind = read(S / 'MANIFEST.json'), read(S / 'BLIND_NATIVE.json')
    plan = read(S / 'PLAN_FROZEN.json')
    previous = read(N / 'review/SUPPORT_OUTCOME_VERIFICATION.json')
    protected = dict(previous['protected_pins'])
    for origin in [previous['source_pins'], plan['pins'], manifest['pins']]:
        for p,h in origin.items():
            ck('pin_consistency',p,p not in protected or protected[p]==h)
            protected[p]=h
    for p,h in protected.items():
        pin('protected_sha',p,h)
    for p in [Path(__file__),S/'PLAN_FROZEN.json',S/'MANIFEST.json',S/'BLIND_NATIVE.json']:
        read_pins[str(p)] = sha(p)
    prior=read(W/'PILOT_MANIFEST.json')['rows']
    done=read(N/'MANIFEST.json')['rows']
    oldids={r['sample_id'] for r in prior}
    doneids={r['sample_id'] for r in done}
    remaining=[r for r in prior if r['sample_id'] not in doneids]
    supportids={r['sample_id'] for r in read(N/'support_inventory/SELECTION.json')['selected']}
    inputs={r['alias']:r for r in read(U/'INPUTS.json')['rows']}
    ck('scope','original24 exact9 plus remaining15',len(prior)==len(oldids)==24 and len(done)==len(doneids)==9 and len(remaining)==15)
    ck('scope','frozen exact reuse9/list15',plan['reuse9']==sorted(doneids) and plan['review15']==[r['sample_id'] for r in remaining])
    ck('scope','support12 disjoint original24',not supportids.intersection(oldids))
    ck('manifest_scope','complete15windows195frames',manifest['status']=='COMPLETE' and manifest['windows']==15 and manifest['native_frames']==195 and len(manifest['rows'])==15)
    ck('manifest_scope','exact original remaining order', [r['sample_id'] for r in manifest['rows']]==[r['sample_id'] for r in remaining])
    chosen=[]
    for i,row in enumerate(remaining,1):
        inp=inputs[row['sample_id'].split('_')[0]]
        ck('fit_lineage',row['sample_id'],row['role']==inp['role']=='fit' and row['clip_id']==inp['clip_id'] and row['person']==inp['person'] and row['word_id']==inp['word_id'] and inp['clip_id'].startswith('NIA_SL_WORD') and inp['source_method'] in ('existing_exact_public_cache','existing_public_archive_exact_sha'))
        chosen.append(dict(support_alias=f'P{i:02}',sample_id=row['sample_id'],source=inp['source'],source_sha256=inp['source_sha256'],person=row['person'],word_id=row['word_id'],roi_bottom=inp['box'][3],frame_ids=row['frame_ids'],timestamps_us=row['timestamps_us']))
    for who in 'AB':
        raw=read(N/'review'/f'NATIVE_{who}.json')
        ck('reused9_raw','exact9 file/IDs '+who,len(raw['rows'])==9 and {r['sample_id'] for r in raw['rows']}==doneids)
        n008=next(r for r in raw['rows'] if r['sample_id']=='N008_W1')
        ck('reused9_raw','N008 left down unknown '+who,n008['presence']['L']['down'] is None)
    ck('blind_scope', 'exact pinned output/rubric', blind['status'] == 'COMPLETE' and blind['native_manifest_sha256'] == sha(S / 'MANIFEST.json') and
       blind['rubric'] == str(W / 'RUBRIC.json') and blind['rubric_sha256'] == sha(W / 'RUBRIC.json') and len(blind['rows']) == 15)
    exported = []
    for i, (row, chosen_row, blind_row) in enumerate(zip(manifest['rows'], chosen, blind['rows']), 1):
        alias, sid = chosen_row['support_alias'], chosen_row['sample_id']
        ck('export_identity', alias, row['alias'] == alias and row['sample_id'] == sid and
           row['source'] == chosen_row['source'] and row['source_sha256'] == chosen_row['source_sha256'] and row['person'] == chosen_row['person'] and
           row['word_id'] == chosen_row['word_id'] and row['role'] == 'fit' and row['roi_bottom'] == chosen_row['roi_bottom'])
        ck('blind_scope', alias, blind_row == {k: v for k, v in row.items() if k not in ['source', 'source_sha256', 'person', 'word_id', 'role']})
        pin('source_video_sha', row['source'], chosen_row['source_sha256'])
        expected_ids, expected_times = chosen_row['frame_ids'], chosen_row['timestamps_us']
        ck('export_frame_ids', alias, len(row['frames']) == 13 and [f['frame_id'] for f in row['frames']] == expected_ids and [f['timestamp_us'] for f in row['frames']] == expected_times)
        saved_frames = {f['frame_id']: f for f in row['frames']}
        decoded, frames_seen, source_pts = {}, [], []
        with av.open(row['source'], mode='r') as video:
            for fid, frame in enumerate(video.decode(video=0)):
                if fid in saved_frames:
                    metadata = saved_frames[fid]
                    rgb = frame.to_ndarray(format='rgb24')
                    height, width = rgb.shape[:2]
                    ts = round(Fraction(frame.pts) * Fraction(frame.time_base) * 1000000)
                    ck('absolute_source_pts', alias + '/' + str(fid), ts == metadata['timestamp_us'] and abs(ts - expected_times[expected_ids.index(fid)]) <= 1)
                    box = (round(Fraction(width, 5)), round(Fraction(2 * height, 25)), round(Fraction(4 * width, 5)), round(Fraction(19 * height, 20)))
                    ck('native_crop_geometry', alias + '/' + str(fid), tuple(metadata['crop_box']) == box and metadata['source_size'] == [width, height])
                    x0, y0, x1, y1 = box
                    crop = rgb[y0:y1, x0:x1]
                    for kind, expected_pixels in [('full', rgb), ('fixed', crop)]:
                        path = Path(metadata[kind])
                        ck('export_paths', alias + '/' + kind + str(fid), path == S / 'visuals' / alias / f'{kind}_{fid:04}.png')
                        pin('export_image_sha', path, metadata[kind + '_sha256'])
                        ck(kind + '_pixels_vs_source', alias + '/' + str(fid), np.array_equal(png(path), expected_pixels))
                    decoded[fid] = crop.copy()
                    frames_seen.append(fid)
                    source_pts.append(dict(frame_id=fid, pts=frame.pts, time_base=str(frame.time_base), absolute_timestamp_us=ts))
                if fid >= max(expected_ids):
                    break
        ck('decoded_full13', alias, frames_seen == expected_ids and len(decoded) == 13)
        ck('fixed_all13_geometry', alias, len({tuple(f['crop_box']) for f in row['frames']}) == 1 and len({tuple(f['source_size']) for f in row['frames']}) == 1)
        ck('pair_count', alias, len(row['pairs']) == 7)
        for j, pair in enumerate(row['pairs']):
            ids = expected_ids[j * 2:j * 2 + 2]
            ck('pair_ids_paths', alias + '/' + str(j), pair['frame_ids'] == ids and Path(pair['file']) == S / 'visuals' / alias / f'pair_{j:02}.png')
            pin('export_image_sha', pair['file'], pair['sha256'])
            pixels = png(pair['file'])
            ch, cw = decoded[ids[0]].shape[:2]
            ck('pair_dimensions', alias + '/' + str(j), pixels.shape == (ch + 36, cw * 2, 3))
            header = Image.new('RGB', (cw * 2, 36), (32, 37, 43))
            draw = ImageDraw.Draw(header)
            for panel, fid in enumerate(ids):
                ck('pair_pixels_vs_source', alias + '/' + str(fid), np.array_equal(pixels[36:, panel * cw:(panel + 1) * cw], decoded[fid]))
                text = f'{alias} f{fid} {expected_times[expected_ids.index(fid)]/1e6:.6f}s native1:1'
                draw.text((panel * cw + 12, 8), text, fill='white')
            ck('pair_headers', alias + '/' + str(j), np.array_equal(pixels[:36], np.array(header)))
            if len(ids) == 1:
                ck('last_blank_panel', alias, np.all(pixels[:, cw:] == np.array([32, 37, 43], dtype=np.uint8)))
        expected_files = {f'{kind}_{fid:04}.png' for fid in expected_ids for kind in ('full', 'fixed')} | {f'pair_{j:02}.png' for j in range(7)}
        ck('export_file_set', alias, {p.name for p in (S / 'visuals' / alias).iterdir()} == expected_files)
        exported.append(dict(alias=alias, sample_id=sid, source_sha256=row['source_sha256'], absolute_source_pts=source_pts))
        print(json.dumps(dict(verified_remaining_windows=i, total=15, alias=alias)), flush=True)
    for path,digest in read_pins.items():
        ck('read_set_unchanged',path,sha(path)==digest)
    ck('earlier_results','old24 FAIL unchanged',read(W/'PRESENCE_PILOT_AUDIT.json')['status']=='REPAIR_PRESENCE_PILOT_BEFORE_BROAD_LABELS_OR_TRAINING')
    ck('earlier_results','native9 not training readiness',read(N/'NATIVE_DIAGNOSIS.json')['status']=='SOURCE_DIAGNOSIS_COMPLETE_TRAINING_NOT_READY')
    groups={}
    for group in sorted({r['group'] for r in checks}):
        subset=[r for r in checks if r['group']==group]
        groups[group]=dict(checks=len(subset),passed=sum(r['passed'] for r in subset),failed=sum(not r['passed'] for r in subset))
    failures=[r for r in checks if not r['passed']]
    report=dict(status='PASS_INDEPENDENT_REMAINING15_NATIVE_EXPORT' if not failures else 'FAIL_REMAINING15_NATIVE_EXPORT',
      at_kst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),
      reviewer='Separate Codex subagent; own previous pixel-check logic reused in a new verifier, no producer imported or rerun.',
      counts=dict(original_windows=24,reused_windows=9,new_windows=15,new_frames=195,full_pngs=195,fixed_pngs=195,pair_pngs=105,blank_panels=15,protected_files=len(protected),protected_production_model_baseline_files=32,rehashed_read_files=len(read_pins),checks=len(checks),failed=len(failures)),
      check_groups=groups,checks=checks,failures=failures,exported_windows=exported,protected_pins=protected,
      source_pins={str(p):sha(p) for p in [Path(__file__),S/'PLAN_FROZEN.json',S/'MANIFEST.json',S/'BLIND_NATIVE.json',N/'review/NATIVE_A.json',N/'review/NATIVE_B.json',W/'PILOT_GATES.json']},
      protocol=dict(exact_original24=True,old9_labels_exact_reuse=True,N008_left_down_unknown_preserved=True,support12_excluded=True,source_frame_time_rule_unchanged=True,new_model_or_label_claim=False),
      limitations=['Export integrity is not label reliability or readiness.','Same PyAV/Pillow dependencies are used with independently checked source selection and pixels.','New15 plus exact9 reuse is retrospective same24 assessment, not unseen validation.'],
      execution=dict(exporter_rerun=False,previous_verifiers_rerun=False,model_inference_runs=0,training_runs=0,original_input_reextraction=0,private_recording_access=0,production_changes=False,root_documents_changed=False))
    with OUT.open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:report[k] for k in ('status','counts')},ensure_ascii=False))
    return bool(failures)

if __name__=='__main__':
    raise SystemExit(main())
