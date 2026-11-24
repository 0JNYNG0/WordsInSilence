"""Append only explicitly supplied human-reviewed condition notes, no inference."""
from pathlib import Path
import json
import hashlib
import datetime

U=Path(__file__).resolve().parent
OUT=U/'review_all_flags_materials'
LEDGER=U/'review_all_flags_protocol_ledger.jsonl'

def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def append(pages,notes,details=None):
    details=details or {}
    existing={r['window_id'] for r in (json.loads(line) for line in LEDGER.read_text('utf-8').splitlines())}
    rows={}
    for pn in pages:
        p=OUT/f'batch_2_overview_v3_{pn:03d}.json';overview=read(p)
        assert sha(overview['image'])==overview['image_sha256']
        for r in overview['rows']:
            wid=r['window_id']
            if wid in existing:continue
            key=r['condition_review_id'];assert key in notes, f'Explicit viewed note missing: {key}'
            m=read(r['detail_material']);assert sha(r['detail_material'])==overview['material_pins'][r['detail_material']]
            entry=rows.setdefault(wid,dict(window_id=wid,batch=2,reviewer='roi_training_protocol',at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),clip_id=r['clip_id'],material_manifest=r['detail_material'],material_manifest_sha256=sha(r['detail_material']),record_sha256=m['record_sha256'],source_sha256=m['source_sha256'],viewed_images=[],actual_viewed_frames=[],actual_overlay_compared_frame_ids=[],source_only_context_frames=[],condition_reviews=[],conditions=m['job']['conditions']))
            entry['viewed_images'].append(dict(path=overview['image'],sha256=overview['image_sha256']))
            entry['actual_viewed_frames']+=r['overview_displayed_frame_ids']
            entry['actual_overlay_compared_frame_ids']+=r['actual_overlay_compared_frame_ids']
            entry['source_only_context_frames']+=r['source_only_context_frames']
            note=notes[key]
            entry['condition_reviews'].append(dict(condition_review_id=key,condition=r['condition'],actual_overlay_compared_frame_ids=r['actual_overlay_compared_frame_ids'],observation_ko=note[0],unresolved_essential_geometric_cut=note[1],additional_frames_needed=note[1]))
    for wid,e in rows.items():
        assert len(e['condition_reviews'])==len(e['conditions']),f'Incomplete condition pages for {wid}'
        if wid in details:
            m=read(e['material_manifest'])
            e['viewed_images']+=m['images']
            e['actual_viewed_frames'] += [x['frame_id'] for x in m['selected_frames']]
            e['actual_overlay_compared_frame_ids'] += [x['frame_id'] for x in m['selected_frames']]
            e['detail_observation_ko']=details[wid]
        e['viewed_images']=list({r['path']:r for r in e['viewed_images']}.values())
        for k in ['actual_viewed_frames','actual_overlay_compared_frame_ids','source_only_context_frames']:e[k]=sorted(set(e[k]))
        e['interval_covered_by_sampled_sequence']=True
        e['every_source_frame_viewed']=False
        e['unresolved_essential_geometric_cut']=any(r['unresolved_essential_geometric_cut'] for r in e['condition_reviews'])
        e['additional_frames_needed']=e['unresolved_essential_geometric_cut']
        e['disposition']='NEEDS_EXTRA_VISUAL_REVIEW' if e['unresolved_essential_geometric_cut'] else 'NO_NEW_ESSENTIAL_GEOMETRIC_CUT_OBSERVED_SENSOR_RISK_RETAINED'
        e['expert_semantic_guarantee']=False;e['training_allowed_by_this_row']=False
        e['review_limit']='직접열람한조건별peak와전후원본표본만기록. 모호한경우상세추가. 연속모든프레임/언어학적의미보존증명아님.'
        with LEDGER.open('a',encoding='utf-8') as f:f.write(json.dumps(e,ensure_ascii=False)+'\n')
    print(json.dumps(dict(appended_windows=list(rows),total=len(existing)+len(rows)),ensure_ascii=False))
