"""Attach metadata to manually written, actually viewed condition notes."""
import json,hashlib,sys
from pathlib import Path
from datetime import datetime,timezone
U=Path(__file__).resolve().parent;D=U/'review_all_flags_materials'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def main():
    notes={}
    for path in sorted(U.glob('review_all_flags_batch1_tail_NOTES*.json')):
        part=read(path)
        assert not(set(notes)&set(part))
        notes.update(part)
    ledger=U/'review_all_flags_batch1_tail_ledger.jsonl'
    existing={json.loads(l)['condition_review_id'] for l in ledger.read_text(encoding='utf-8').splitlines()} if ledger.exists() else set()
    for page_key,manual in notes.items():
        page=int(page_key);mp=D/f'batch_1_overview_v3_{page:03d}.json';meta=read(mp)
        assert 30<=page<=57
        for row in meta['rows']:
            cid=row['condition_review_id']
            if cid not in manual:continue
            note=manual[cid]
            assert note['actually_viewed'] is True and note['observation_ko']
            if cid in existing:continue
            material=read(row['detail_material']);j=material['job']
            images=[{'path':meta['image'],'sha256':meta['image_sha256']}]
            frames=set(row['overview_displayed_frame_ids']);overlays=set(row['actual_overlay_compared_frame_ids'])
            for extra in note.get('detail_images',[]):
                found=next(i for i in material['images'] if Path(i['path']).name==extra)
                images.append(found);frames.update(found['displayed_frame_ids']);overlays.update(found['displayed_frame_ids'])
            assert all(sha(i['path'])==i['sha256'] for i in images)
            item=dict(condition_review_id=cid,window_id=row['window_id'],batch=1,reviewer='roi_missing_protocol',at_utc=datetime.now(timezone.utc).isoformat(),
                clip_id=j['clip_id'],comparison=row['condition']['comparison'],side=row['condition']['side'],condition=row['condition'],
                material_manifest=row['detail_material'],material_manifest_sha256=sha(row['detail_material']),record_sha256=material['record_sha256'],source_sha256=material['source_sha256'],
                overview_manifest=str(mp),overview_manifest_sha256=sha(mp),viewed_images=images,
                actual_viewed_frames=sorted(frames),actual_overlay_compared_frame_ids=sorted(overlays),
                observation_ko=note['observation_ko'],finding=note['finding'],
                disposition='REVIEWED_NO_NEW_ESSENTIAL_GEOMETRIC_CUT_OBSERVED' if note['unresolved_essential_geometric_cut'] is False else 'UNRESOLVED_REQUIRES_ROOT_REVIEW',
                unresolved_essential_geometric_cut=note['unresolved_essential_geometric_cut'],additional_frames_needed=note['additional_frames_needed'],
                all_window_frames_viewed=False,expert_semantic_guarantee=False,training_allowed_by_this_row=False,
                review_limit='Explicit actually viewed overview samples and individually named extra detailed pages only; no all-frame or linguistic semantic guarantee.')
            with ledger.open('a',encoding='utf-8') as f:f.write(json.dumps(item,ensure_ascii=False)+'\n')
            existing.add(cid)
    print(json.dumps({'recorded_conditions':len(existing)}))
if __name__=='__main__':main()
