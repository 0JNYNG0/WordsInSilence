"""Index one training demonstration per class, preserving every original artifact."""
import json
import hashlib
import zipfile
from pathlib import Path

U=Path(__file__).resolve().parent
B=U.parents[1]
def read(p): return json.loads(Path(p).read_text('utf-8-sig'))
def write(p,d): Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n','utf-8')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

if __name__=='__main__':
    config=read(B/'config/word100_practice.json')
    plan=read(B/'ops/word100_training_20261013/FROZEN_EXPERIMENT.json')
    manifest=read(B/'ops/word100_training_20261013/MANIFEST.json')
    output=Path(config['items'][0]['video_path']).parent
    selections=[min((r for r in manifest if r['word_code']==code and r['role']=='train_candidate'),
                    key=lambda r:(r['signer_id'],r['clip_id'])) for code in config['word_codes']]
    scope=dict(user_request='Top5 ranking, score bars, requested-reference DTW measurements, top3 videos, preserve2x2',
        model_frozen=True, no_training_or_threshold_tuning=True, reference_policy='Lowest training signer ID for every100 class; not selected by user results',
        sample_budget=96, band_ratio=0.2, dtw_alignment='Existing grouped XY cost; no old18 pass line, no hand-role exclusion',
        measurements='2D finger interior angle MAE degrees; wrist location and relative trajectory deviation in shoulder widths; observation coverage',
        original_results_readonly=True, no_real_camera_for_verification=True,
        reference_total_bytes=sum(r['video_entries'][0]['bytes'] for r in selections))
    write(U/'SCOPE.json',scope)
    guard={str(p):sha(p) for p in (B/'runs/ses_79977f57893d').rglob('*') if p.is_file()}
    guard[config['checkpoint']]=sha(config['checkpoint'])
    write(U/'ORIGINAL_GUARD.json',guard)
    items=[];zips={};new_bytes=0
    try:
        for r in selections:
            assert 1<=r['signer_id']<=9
            record=read(Path(plan['output'])/'records'/(r['clip_id']+'.json'))
            assert record['role']=='train_candidate'
            e=r['video_entries'][0];dest=output/(r['clip_id']+'.mp4')
            if not dest.exists():
                if e['archive'] not in zips:zips[e['archive']]=zipfile.ZipFile(e['archive'])
                z=zips[e['archive']];info=z.getinfo(e['member'])
                assert info.file_size==e['bytes'] and info.CRC==e['crc32']
                data=z.read(e['member']);assert hashlib.sha256(data).hexdigest()==record['source_sha256']
                with dest.open('xb') as out:out.write(data)
                new_bytes+=len(data)
            assert sha(dest)==record['source_sha256']
            assert sha(record['observation'])==record['observation_sha256']
            items.append(dict(word_id=r['word_code'],label=r['canonical_label'],clip_id=r['clip_id'],
                signer_id=r['signer_id'],source_role=r['role'],video_path=str(dest),
                source_sha256=record['source_sha256'],observation=record['observation'],
                observation_sha256=record['observation_sha256'],duration_ms=round(r['duration_metadata']*1000)))
    finally:
        for z in zips.values():z.close()
    write(B/'config/word100_analysis_references.json',dict(schema='wis_word100_analysis_refs_v1',items=items))
    report=dict(status='PASS',reference_count=len(items),new_bytes=new_bytes,total_bytes=sum(Path(x['video_path']).stat().st_size for x in items),
                signer_ids=sorted({x['signer_id'] for x in items}),all_video_and_observation_hashes_match=True)
    write(U/'REFERENCE_CHECK.json',report);print(json.dumps(report))
