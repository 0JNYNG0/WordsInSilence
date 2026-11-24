"""Prepare the explicitly approved WORD-only acquisition; preserve old evidence."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil

UNIT = Path(__file__).resolve().parent
PROJECT = UNIT.parents[3]
DEST = Path('W:/WordsInSilence/data/AIHub_103')
SOURCE = Path('C:/Users/local-user/Downloads/수어 영상/1.Training/[원천]01_real_word_video.zip')

def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def main():
    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).isoformat()
    catalog_path = DEST / 'manifests/AIHUB103_PUBLIC_CATALOG.json'
    catalog = json.loads(catalog_path.read_text(encoding='utf-8-sig'))['files']
    rows = [r for r in catalog if 'real_word_video.zip' in r['streFileNm']]
    assert len(rows) == 33 and len({r['fileSn'] for r in rows}) == 33
    assert sum(r['fileSize'] for r in rows) == 1181218593950
    prior = json.loads((UNIT.parent / 'word_acquisition_inventory_20261011/VIDEO_ACQUISITION_QUEUE.json').read_text(encoding='utf-8-sig'))
    assert {(r['fileSn'],r['fileStreCours'],r['fileSize']) for r in rows} == {(r['fileSn'],r['fileStreCours'],r['fileSize']) for r in prior}
    assert SOURCE.stat().st_size == 32547266828
    free = shutil.disk_usage('W:/').free
    assert free > sum(r['fileSize'] for r in rows) + 3*max(r['fileSize'] for r in rows)
    rows.sort(key=lambda r: (r['fileSn'] != 39546, r['fileSn'] == 39483, r['fileSn']))
    for r in rows:
        assert '/REAL/' in r['fileStreCours'] and '/SEN/' not in r['fileStreCours']
        split = '2.Validation' if '/2.Validation/' in r['fileStreCours'] else '1.Training'
        r['destination'] = str(DEST / 'videos' / split / r['streFileNm'])
        r['action'] = 'copy_existing_no_download' if r['fileSn'] == 39546 else 'download'
        if r['fileSn'] == 39546:
            r['existing_source'] = str(SOURCE)
    result = {'schema':'word_video_acquisition_v1','created_at_kst':now,'actor':'Codex',
      'authorization':'User explicitly removed the time restriction and requested all required isolated-word data. REAL/WORD only; not entire AIHub103.',
      'time_cutoff':None,'datasetkey':103,'destination_root':str(DEST),
      'source_catalog_sha256':hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
      'all_archive_count':33,'all_archive_bytes':1181218593950,
      'new_download_count':32,'new_download_bytes':1148671327122,
      'available_disk_bytes':free,'archives':rows,
      'excluded':['SEN sentence videos','SYN synthetic videos','CROWD','OpenPose keypoint archives'],
      'reuse_labels':['annotations/1.Training/[라벨]01_real_word_morpheme.zip','annotations/2.Validation/[라벨]01_real_word_morpheme.zip'],
      'guards':['One archive at a time; preserve interrupted files; no silent overwrite.',
        'Verify expected size, ZIP structure and CRC before marking an archive complete.',
        'Preserve original videos and signer roles; no automatic training or model replacement.',
        'No credential values in logs, manifests or command arguments.',
        'Source archive download includes supplied views; later model input preparation prioritizes front view.'],
      'status':'QUEUE_READY_LOGIN_REQUIRED_NETWORK_NOT_STARTED'}
    write(UNIT/'ACQUISITION_PLAN.json', result)
    write(DEST/'downloads_20261011/ACQUISITION_PLAN.json', result)
    write(UNIT/'PREFLIGHT.json', {'status':'PASS','catalog_queue_equal':True,'archive_count':33,'new_download_count':32,'space_with_3x_largest_temporary_reserve':True,'existing_source_size_match':True,'time_cutoff_removed':True,'new_video_network_bytes':0})
    print(json.dumps({k:result[k] for k in ['status','all_archive_count','new_download_count','new_download_bytes','available_disk_bytes']}))

if __name__ == '__main__':
    main()
