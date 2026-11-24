"""Validate and index complete visual materials without interpreting them."""
from collections import Counter
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import review_all_flags_audit as a


def main():
    final=a.read(a.U/'review_all_flags_RESULT.json')
    assert final['windows']==685 and final['detector_runs']==final['training_runs']==final['classifier_runs']==0
    expected=a.expected_windows(a.read(a.U/'review_flag_coverage_RESULT.json'))
    target_ids={c['condition_review_id'] for w in expected.values() for c in w['conditions']}
    inventory=[];seen=[]
    for batch in (1,2,3):
        files=sorted(a.D.glob(f'batch_{batch}_overview_v3_*.json'))
        conditions=[];pages=[]
        content=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>조건별 전체 구간 검토</title>',
            '<style>body{font:16px sans-serif;background:#101722;color:#e7edf5;margin:24px}img{display:block;width:100%;max-width:1920px}article{margin:28px 0}a{color:#96c8ff}small{color:#d8e0e9}</style>',
            f'<h1>묶음 {batch}: 조건별 시각 검토 자료</h1>',
            '<p>각 행은 한 구간의 한 손/비교 조건입니다. 같은 W번호의 모든 조건을 확인하세요. 생성과 수치 검산은 사람의 관찰 또는 학습 승인이 아닙니다.</p>',
            '<p>처음/끝 원본과 조건별 최대손실의 관측 비교를 표시합니다. 움직임·받침손이 애매하면 상세 시퀀스나 추가 프레임을 확인하고 기록하세요. 모든 프레임을 보았다는 뜻은 아닙니다.</p>']
        for path in files:
            page=a.read(path)
            assert page['batch']==batch and a.sha(page['image'])==page['image_sha256']
            assert all(a.sha(p)==h for p,h in page['material_pins'].items())
            ids=[r['condition_review_id'] for r in page['rows']]
            assert len(ids)==len(set(ids))
            for row in page['rows']:
                wid=row['window_id'];m=a.read(row['detail_material'])
                candidates=[c for c in expected[wid]['conditions'] if c['condition_review_id']==row['condition_review_id']]
                assert len(candidates)==1 and expected[wid]['batch']==batch
                assert row['selected_peak_frame_id']==a.peak(m,candidates[0])
            conditions+=ids;seen+=ids
            pages.append(dict(path=str(path),sha256=a.sha(path),image=page['image'],image_sha256=page['image_sha256'],condition_ids=ids))
            content.append(f'<article id="page{page["page"]}"><h2>페이지 {page["page"]}</h2><a href="{Path(page["image"]).name}"><img loading="lazy" src="{Path(page["image"]).name}"></a><small>')
            for wid in dict.fromkeys(r['window_id'] for r in page['rows']):
                content.append(f'<a href="batch_{batch}.html#{wid}">{wid} 상세</a> · ')
            content.append(f'<a href="{path.name}">조건/프레임/핀 기록</a></small></article>')
        required={c['condition_review_id'] for w in expected.values() if w['batch']==batch for c in w['conditions']}
        assert set(conditions)==required and len(conditions)==len(required),'Missing/duplicate condition overview'
        index=a.D/f'batch_{batch}_overview_v3.html'
        with index.open('x',encoding='utf-8') as f:f.write('\n'.join(content))
        inventory.append(dict(batch=batch,windows=sum(w['batch']==batch for w in expected.values()),conditions=len(conditions),
            overview_pages=len(pages),overview_html=str(index),overview_html_sha256=a.sha(index),pages=pages,
            detail_html=str(a.D/f'batch_{batch}.html'),ledger_template=str(a.D/f'batch_{batch}_ledger_TEMPLATE.json')))
    assert set(seen)==target_ids and len(seen)==len(target_ids)==838
    result=dict(status='ALL_WINDOW_CONDITION_MATERIALS_INDEXED_PENDING_REVIEW',created_utc=datetime.now(timezone.utc).isoformat(),
        windows=685,window_conditions=838,clips=631,batches=inventory,total_overview_pages=sum(i['overview_pages'] for i in inventory),
        detailed_material_result=str(a.U/'review_all_flags_RESULT.json'),detailed_material_result_sha256=a.sha(a.U/'review_all_flags_RESULT.json'),
        source_pins={str(a.U/'review_flag_coverage_RESULT.json'):a.sha(a.U/'review_flag_coverage_RESULT.json')},
        all_conditions_explicitly_displayed=True,all_frames_displayed=False,actual_viewing_complete=False,semantic_pass=False,training_allowed=False,
        generator_sha256=a.sha(__file__))
    with (a.U/'review_all_flags_MATERIAL_INVENTORY.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:result[k] for k in ('windows','window_conditions','total_overview_pages')},ensure_ascii=False))
    for batch in inventory:print(json.dumps({k:batch[k] for k in ('batch','windows','conditions','overview_pages','overview_html','detail_html','ledger_template')},ensure_ascii=False))


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');main()
