"""Copy existing authorized WORD labels; download bounded public documentation only."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import urllib.request
import zipfile

UNIT = Path(__file__).resolve().parent
PROJECT = UNIT.parents[3]
PRIOR = UNIT.parent / 'aihub_modu_source_review_20261011'
ROOT = Path('W:/WordsInSilence/data')
DEADLINE = dt.datetime(2026, 10, 11, 20, 0, tzinfo=dt.timezone(dt.timedelta(hours=9)))
LIMIT = 25 * 1024 * 1024

def now():
    return dt.datetime.now(DEADLINE.tzinfo)

def check_time():
    if now() >= DEADLINE:
        raise RuntimeError('20:00 KST cutoff reached; no new transfer')

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def copy_checked(src, dst, expected=None):
    check_time()
    src = Path(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if expected is not None and src.stat().st_size != expected:
        raise RuntimeError('Source size differs from official catalog')
    source_hash = sha(src)
    if dst.exists():
        if sha(dst) != source_hash:
            raise RuntimeError('Existing destination differs; not overwritten')
        action = 'existing_identical_reused'
    else:
        part = dst.with_name(dst.name + '.partial')
        with src.open('rb') as a, part.open('xb') as b:
            while block := a.read(1024 * 1024):
                check_time()
                b.write(block)
            b.flush()
            os.fsync(b.fileno())
        if sha(part) != source_hash:
            raise RuntimeError('Copied bytes differ')
        part.rename(dst)
        action = 'copied_from_existing_local_file'
    return {'source': str(src), 'destination': str(dst), 'bytes': dst.stat().st_size,
            'sha256': source_hash, 'source_destination_match': True, 'action': action}

def fetch_bounded(url):
    check_time()
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=30) as r:
        info = {'url': url, 'final_url': r.url, 'content_type': r.headers.get('Content-Type')}
        length = r.headers.get('Content-Length')
        if length and int(length) > LIMIT:
            raise RuntimeError('Public document exceeds 25 MiB cap; not downloaded')
        blocks, n = [], 0
        while block := r.read(256 * 1024):
            check_time()
            n += len(block)
            if n > LIMIT:
                raise RuntimeError('Public document exceeded 25 MiB cap')
            blocks.append(block)
        return b''.join(blocks), info

def save_received(dst, body):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if dst.read_bytes() != body:
            raise RuntimeError('Different destination exists; not overwritten')
    else:
        with dst.open('xb') as f:
            f.write(body)
            f.flush()
            os.fsync(f.fileno())
    return {'destination': str(dst), 'bytes': len(body), 'sha256': sha(dst)}

def main():
    if (UNIT / 'ACQUISITION_RESULT.json').exists():
        raise RuntimeError('Completed result already exists; read it instead of repeating')
    before = shutil.disk_usage('W:/')
    if before.free < 1024**3:
        raise RuntimeError('Less than 1 GiB free')
    catalog = json.loads((PRIOR / 'AIHUB103_PUBLIC_CATALOG.json').read_text(encoding='utf-8-sig'))
    local = json.loads((PRIOR / 'LOCAL_LABEL_AVAILABILITY.json').read_text(encoding='utf-8-sig'))
    result = {'started_at_kst': now().isoformat(), 'actor': 'Codex local',
              'scope': 'Existing REAL/WORD annotation copies and bounded public documentation/catalog; no video/keypoint acquisition',
              'deadline_kst': DEADLINE.isoformat(), 'disk_before': before._asdict(),
              'labels': [], 'documents': [], 'catalog': None}
    try:
        for item in local['archives']:
            split = item['split']
            matches = [f for f in catalog['files'] if f['streFileNm'] == '[라벨]01_real_word_morpheme.zip'
                       and '/' + split + '/' in f['fileStreCours']]
            if len(matches) != 1:
                raise RuntimeError('Official annotation mapping is not unique')
            record = copy_checked(item['path'], ROOT / 'AIHub_103' / 'annotations' / split / Path(item['path']).name,
                                  matches[0]['fileSize'])
            record['official_catalog_entry'] = matches[0]
            with zipfile.ZipFile(record['destination']) as z:
                files = [i for i in z.infolist() if not i.is_dir()]
                bad = z.testzip()
                if bad is not None:
                    raise RuntimeError('CRC failure: ' + bad)
                record.update(crc_all_entries_pass=True, entries=len(files),
                              uncompressed_bytes=sum(i.file_size for i in files),
                              all_entries_json=all(i.filename.endswith('.json') for i in files))
            result['labels'].append(record)
            print(json.dumps({'label_complete': split, 'bytes': record['bytes'], 'crc': True}, ensure_ascii=False), flush=True)
        cat_url = catalog['source_url']
        try:
            body, info = fetch_bounded(cat_url)
            parsed = json.loads(body.decode('utf-8-sig'))
            info.update(save_received(ROOT / 'AIHub_103' / 'manifests' / 'official_file_list_20261011.json', body))
            info['json_valid'] = True
            result['catalog'] = info
        except Exception as e:
            result['catalog'] = {'url': cat_url, 'error': str(e)}
        specs = [
            ('AIHub_103/docs/data_description.pdf', 'https://www.aihub.or.kr/file/down.do?dataMnlFileSn=7969&dataSetSn=103&fileSn=7969'),
            ('AIHub_103/docs/construction_usage_guide.pdf', 'https://www.aihub.or.kr/file/down.do?cnstcPrcuseFileSn=18941&dataSetSn=103&fileSn=18941'),
            ('Modu_2025/docs/NIKL_Sign_Language_Parallel_Corpus_2025_v1.0.pdf', 'https://kli.korean.go.kr/images/des/293/NIKL_Sign%20Language%20Parallel%20Corpus_2025_v1.0.pdf'),
        ]
        for name, url in specs:
            record = {'url': url}
            try:
                body, info = fetch_bounded(url)
                record.update(info)
                if not body.startswith(b'%PDF-'):
                    record.update(status='not_saved_as_pdf', received_bytes=len(body),
                                  reason='Response is not PDF; login/error HTML may require user action')
                else:
                    record.update(save_received(ROOT / name, body))
                    record['status'] = 'downloaded'
            except Exception as e:
                record.update(status='failed', error=str(e))
            result['documents'].append(record)
            print(json.dumps({'document': name, 'status': record['status']}, ensure_ascii=False), flush=True)
        for name in ('AIHUB103_PUBLIC_CATALOG.json', 'ACQUISITION_SCOPE.json'):
            copy_checked(PRIOR / name, ROOT / 'AIHub_103' / 'manifests' / name)
        result['completed_at_kst'] = now().isoformat()
        result['disk_after'] = shutil.disk_usage('W:/')._asdict()
        result['status'] = 'COMPLETE_LABELS_COPIED_CRC_VERIFIED_PUBLIC_DOC_ATTEMPTS_RECORDED'
        write_json(UNIT / 'ACQUISITION_RESULT.json', result)
        write_json(ROOT / 'acquisition_20261011.json', result)
        print(json.dumps({'status': result['status'], 'result': str(UNIT / 'ACQUISITION_RESULT.json')}, ensure_ascii=False))
    except Exception as e:
        result['failure'] = {'at': now().isoformat(), 'error': repr(e)}
        write_json(UNIT / ('FIRST_FAILURE_' + now().strftime('%H%M%S') + '.json'), result)
        raise

if __name__ == '__main__':
    main()
