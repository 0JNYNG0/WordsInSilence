"""Read existing public ZIP metadata only; no extraction, inference or training."""
from pathlib import Path
import datetime
import json
import zipfile

U = Path(__file__).resolve().parent
O = U.parent.parent
data = json.loads((O / 'demo_handshape_learning_20261113/INPUTS.json').read_text('utf-8-sig'))
selected = [r for r in data['rows'] if r['clip_id'] in
            ['NIA_SL_WORD0099_REAL01_F', 'NIA_SL_WORD1290_REAL01_F']]
assert len(selected) == 2
rows = []
for row in selected:
    archive = Path(row['archive'])
    clip = row['clip_id'][:-2]
    with zipfile.ZipFile(archive) as z:
        matches = {v: [i for i in z.infolist() if Path(i.filename).name == f'{clip}_{v}.mp4']
                   for v in 'FLRUD'}
        rows.append(dict(clip=clip, archive=str(archive),
                         front_reference_exists=Path(row['video']).is_file(),
                         all_five_views_present=all(len(x) == 1 for x in matches.values()),
                         views={v: [dict(member=i.filename, bytes=i.file_size, crc32=i.CRC)
                                    for i in infos] for v, infos in matches.items()}))
report = dict(
    at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),
    scope='Public ZIP directory metadata only; source payloads not extracted or decoded.',
    rows=rows,
    reused_pipeline='avatar_motion_pilot_20261113 + avatar_motion_smoothing_20261113',
    available_rig='Existing Ch29 Mixamo rig and skin; web GLB export already exists.',
    ready_teaching_animations=[],
    existing_ancestor_animation='Technical prototype only, not fully validated for teaching.',
    limitations=['No verified calibration supplied to existing reconstruction pipeline.',
                 'New clips need observation/geometry review; camera reuse is not assumed.',
                 'Face/hand relative placement, fingers, contact and clipping need checking.'],
    training_runs=0, private_recordings_used=False, web_changed=False)
(U / 'SOURCE_AUDIT.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', 'utf-8')
print(json.dumps({r['clip']: {'five_views_present': r['all_five_views_present'],
                             'bytes': sum(i['bytes'] for infos in r['views'].values() for i in infos)}
                  for r in rows}, ensure_ascii=False))
