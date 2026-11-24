"""Append only explicitly supplied observations after actual image viewing."""
import json
from datetime import datetime,timezone
from pathlib import Path
from review_all_flags_build import U,read,sha
D=U/'review_all_flags_materials'
P=U/'review_all_flags_budget_ledger.jsonl'

def record(page,notes,extra_by_condition=None):
    extra_by_condition=extra_by_condition or {}
    metadata=read(D/f'batch_1_overview_v3_{page:03d}.json')
    assert len(notes)==len(metadata['rows'])
    assert sha(metadata['image'])==metadata['image_sha256']
    previous=[json.loads(x) for x in P.read_text('utf-8').splitlines()] if P.exists() else []
    done={x['condition_review_id'] for x in previous}
    out=[]
    for r,observation in zip(metadata['rows'],notes):
        cid=r['condition_review_id'];assert cid not in done
        mp=Path(r['detail_material']);m=read(mp)
        assert sha(mp)==metadata['material_pins'][str(mp)]
        images=[{'path':metadata['image'],'sha256':metadata['image_sha256']}]
        frames=set(r['overview_displayed_frame_ids']);compared=set(r['actual_overlay_compared_frame_ids'])
        for ep in extra_by_condition.get(cid,[]):
            match=[x for x in m['images'] if Path(x['path']).name==ep]
            assert len(match)==1 and sha(match[0]['path'])==match[0]['sha256']
            images.append({k:match[0][k] for k in ('path','sha256')})
            frames.update(match[0]['displayed_frame_ids']);compared.update(match[0]['displayed_frame_ids'])
        out.append({'created_utc':datetime.now(timezone.utc).isoformat(),'reviewer':'roi_data_budget',
          'condition_review_id':cid,'window_id':r['window_id'],**{k:r['condition'][k] for k in ('comparison','side')},
          'material_manifest':str(mp),'material_manifest_sha256':sha(mp),'record_sha256':m['record_sha256'],'source_sha256':m['source_sha256'],
          'viewed_images':images,'actual_viewed_frames':sorted(frames),'actual_overlay_compared_frame_ids':sorted(compared),
          'observation_ko':observation,'disposition':'VISUAL_SAMPLE_REVIEWED_WITH_STATED_LIMITS',
          'additional_frames_needed':False,'unresolved_essential_geometric_cut':False,'every_source_frame_viewed':False,
          'training_allowed_by_this_row':False,'semantic_pass':False,
          'limits_ko':'표시된 시작·끝 원본과 조건 peak 관측을 직접 확인한 표본 검토. 모든 프레임/수어 의미 보존 또는 학습 허가를 뜻하지 않음.'})
    with P.open('a',encoding='utf-8') as f:
        for r in out:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    print(f'page {page}: appended {len(out)}; total {len(previous)+len(out)}',flush=True)
