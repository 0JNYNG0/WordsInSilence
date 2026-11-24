"""Local acquisition audit: one approved ZIP, directory and exact train/dev members only."""
from collections import defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import zipfile
import zlib
from zoneinfo import ZoneInfo

import av


EVIDENCE = Path(__file__).resolve().parent
WEB = EVIDENCE.parents[1]
ARCHIVE = Path('C:/Users/local-user/Downloads/수어 영상/1.Training/[원천]03_real_sen_video.zip')
OUT = WEB / 'runs/apology_source_review_v1/raw'
REQUIRED = WEB / 'ops/native_data_pair_20260919/web_domain_required_sources.json'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    access = json.loads((EVIDENCE / 'access_result.json').read_text(encoding='utf-8'))
    assert access['single_archive_proposal']['download_approval'] == 'USER_APPROVED'
    assert access['single_archive_proposal']['file'] == ARCHIVE.name
    assert ARCHIVE.is_file() and ARCHIVE.stat().st_size > 0
    assert not list(ARCHIVE.parent.glob(ARCHIVE.name.replace('[', '[[]') + '.irx*')), 'partial transfer remains'
    report_path = EVIDENCE / 'archive03_inspection.json'
    assert not report_path.exists(), 'first result must be preserved'
    required = json.loads(REQUIRED.read_text(encoding='utf-8'))['required_missing']
    expected = {row['filename']: row for row in required}
    assert len(expected) == 11
    assert {row['signer_id'] for row in required} == set(range(2, 10)) | {11, 12, 13}
    for row in required:
        assert row['filename'] == f"NIA_SL_SEN0356_REAL{row['signer_id']:02d}_F.mp4"
        assert row['split_role'] == ('train_candidate' if row['signer_id'] < 10 else 'development_only')
    result = {
        'started_kst': datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
        'status': 'RUNNING', 'archive': ARCHIVE.as_posix(),
        'archive_bytes': ARCHIVE.stat().st_size,
        'required_source_manifest_sha256': sha(REQUIRED),
        'whole_archive_crc_test': 'NOT_RUN; only selected video members are decompressed and CRC checked',
        'selected': [], 'missing': [], 'new_training_runs': 0,
        'held_out_or_unassigned_video_contents_opened': 0,
        'registration_or_model_promotion': False,
    }
    try:
        before = (ARCHIVE.stat().st_size, ARCHIVE.stat().st_mtime_ns)
        with zipfile.ZipFile(ARCHIVE, 'r') as archive:
            infos = archive.infolist()
            assert len({i.filename for i in infos}) == len(infos), 'duplicate ZIP paths'
            groups = defaultdict(list)
            matches = defaultdict(list)
            unmatched = []
            pattern = re.compile(r'NIA_SL_SEN(\d+)_REAL(\d+)_([FDLRU])\.mp4', re.IGNORECASE)
            for info in infos:
                if info.is_dir():
                    continue
                name = info.filename.replace('\\', '/').rsplit('/', 1)[-1]
                match = pattern.fullmatch(name)
                if match:
                    sentence, person, view = match.groups()
                    groups[(int(person), view.upper())].append(int(sentence))
                else:
                    unmatched.append(info.filename)
                if name in expected:
                    matches[name].append(info)
            result['directory_entries'] = len(infos)
            result['video_groups'] = [
                {'signer_id': key[0], 'view': key[1], 'count': len(ids),
                 'min_sentence': min(ids), 'max_sentence': max(ids),
                 'complete_integer_range': sorted(ids) == list(range(min(ids), max(ids) + 1))}
                for key, ids in sorted(groups.items())
            ]
            result['unmatched_member_names'] = unmatched
            result['missing'] = sorted(set(expected) - set(matches))
            for name, items in sorted(matches.items()):
                assert len(items) == 1, f'duplicate target basename: {name}'
                info = items[0]
                assert not info.flag_bits & 1, 'encrypted ZIP member'
                assert 0 < info.file_size < 64 * 1024 * 1024, 'unexpected clip size; review before extraction'
                source = expected[name]
                data = archive.read(info)  # zipfile verifies member CRC while decompressing.
                assert len(data) == info.file_size
                assert zlib.crc32(data) & 0xffffffff == info.CRC
                checksum = hashlib.sha256(data).hexdigest()
                destination = OUT / name  # Never extract an archive-controlled directory path.
                OUT.mkdir(parents=True, exist_ok=True)
                existed = destination.exists()
                if existed:
                    assert sha(destination) == checksum, 'existing original differs; preserve and stop'
                else:
                    with destination.open('xb') as handle:
                        handle.write(data)
                    assert sha(destination) == checksum
                entry = {'source': source, 'archive_member': info.filename,
                         'bytes': len(data), 'crc32': f'{info.CRC:08x}', 'sha256': checksum,
                         'path': destination.as_posix(), 'already_existed': existed,
                         'semantic_review': 'Family mapping user-confirmed from REAL01; this signer video not visually reviewed',
                         'training_registered': False}
                result['selected'].append(entry)
                timestamps = []
                missing_pts = 0
                with av.open(str(destination)) as container:
                    stream = container.streams.video[0]
                    entry['width_height'] = [stream.codec_context.width, stream.codec_context.height]
                    entry['time_base'] = str(stream.time_base)
                    for frame in container.decode(video=0):
                        if frame.pts is None:
                            missing_pts += 1
                        else:
                            timestamps.append(float(frame.pts * frame.time_base))
                assert timestamps and missing_pts == 0, 'missing PTS'
                assert all(b > a for a, b in zip(timestamps, timestamps[1:])), 'non-monotonic PTS'
                entry.update(decoded_frames=len(timestamps), missing_pts=missing_pts,
                             first_last_pts_s=[timestamps[0], timestamps[-1]], monotonic_pts=True,
                             annotation_time_within_video=(timestamps[0] <= source['annotation_start_s'] < source['annotation_end_s'] <= timestamps[-1]),
                             keypoint_to_video_alignment='UNVERIFIED', full_action_boundaries='UNVERIFIED')
                print(f'EXTRACTED_AND_DECODED signer={source["signer_id"]} frames={len(timestamps)} bytes={len(data)}', flush=True)
        assert before == (ARCHIVE.stat().st_size, ARCHIVE.stat().st_mtime_ns), 'archive changed during inspection'
        result['status'] = 'COMPLETE_TARGETS_FOUND' if result['selected'] else 'COMPLETE_NO_REQUIRED_TARGETS'
        result['new_raw_videos'] = sum(not item['already_existed'] for item in result['selected'])
        result['archive_stat_unchanged'] = True
    except Exception as exc:
        result['status'] = 'FAIL_FIRST_RESULT_PRESERVED'
        result['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        result['finished_kst'] = datetime.now(ZoneInfo('Asia/Seoul')).isoformat()
        report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'new_raw_videos': result['new_raw_videos'],
                      'remaining_missing': len(result['missing']), 'video_groups': result['video_groups']}, ensure_ascii=True))


if __name__ == '__main__':
    main()
