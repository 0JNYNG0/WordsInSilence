from __future__ import annotations
from pathlib import Path
import json, html, hashlib
from datetime import datetime, timezone, timedelta
from feature_profiles import read, dump, ROOT

def h(x): return html.escape(str(x))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    profiles=read(ROOT/'FEATURE_PROFILES.json'); public=read(ROOT/'public_comparison/RESULT.json')
    index=read(ROOT/'local_audit/PUBLIC_TRAIN_DEV_INPUT_INDEX.json')['rows']
    byword={p['word_id']:p for p in profiles['profiles']}
    public_ids=list(dict.fromkeys(x['query'] for x in public['pairs']))
    bridged=[]
    for cid in public_ids:
        pairs=[x for x in public['pairs'] if x['query']==cid];p=byword[pairs[0]['word_id']]
        bridged.append({'query_clip':cid,'word_id':p['word_id'],'display_label':p['display_label'],
            'dictionary_candidates':[{'source_id':x['source_id'],'entry_id':x['entry_id'],'url':x['url']} for x in p['candidate_entries']],
            'reference_comparisons':pairs,'source_to_video_variant_verified':False,
            'feature_importance_weights':None,'source_semantic_comparison':'NOT_ENABLED_UNVERIFIED_VARIANT_AND_PHASE',
            'educational_judgment':None,'message_ko':'공개 참고와 측정값은 비교했지만, 사전 표현과 같은 형태인지 및 허용 기준은 미확인입니다.'})
    dump(ROOT/'INTEGRATED_EVIDENCE_INDEX.json',{'scope':'Source lookup alongside measurements, not semantic fusion or model learning','queries':bridged})
    # Preparing eligible public records for annotation is not assigning weak targets.
    manifest=[]
    for p in profiles['profiles']:
        if not p['candidate_entries']: continue
        rows=[x for x in index if x['word_id']==p['word_id']]
        manifest.append({'word_id':p['word_id'],'label':p['display_label'],
            'sources':[{'source_id':e['source_id'],'entry_id':e['entry_id'],'url':e['url']} for e in p['candidate_entries']],
            'public_records':[{'clip_id':r['clip_id'],'recorded_person_id':r['recorded_person_id'],'role':r['role'],'word_interval_seconds':r['word_interval_seconds'],'record':r['record']} for r in rows],
            'verified_feature_targets':None,'eligible_for_training':False,
            'annotation_fields_needed':['matched_source_variant','feature_type','hand','start_time','end_time','visible_evidence','annotation_provenance'],
            'correctness_fields_separate':['accepted_variation_or_confirmed_error','linguistic_review_provenance']})
    dump(ROOT/'FEATURE_LABEL_PREPARATION.json',{'rows':manifest,'count':len(manifest),'labeled_tokens':0,'warning':'An annotation index only. No target assignments, learner accuracy, or dataset expansion performed.'})
    coverage=[]
    for word in dict.fromkeys(p['word_id'] for p in public['pairs']):
        pairs=[x for x in public['pairs'] if x['word_id']==word];stats={}
        for key in public['totals']:
            n=sum(x['stats'][key]['jointly_observed'] for x in pairs);d=sum(x['stats'][key]['planned'] for x in pairs)
            stats[key]={'numerator':n,'denominator':d,'percent':100*n/d}
        coverage.append({'word_id':word,'label':pairs[0]['label'],'pairs':len(pairs),'metrics':stats})
    final_review=ROOT/'review/FINAL_REVIEW.json'
    report={
        'task':'word100_linguistic_feedback_pipeline_20261027','at':datetime.now(timezone(timedelta(hours=9))).isoformat(),
        'status':'SOURCE_PROFILE_AND_PUBLIC_MEASUREMENT_COMPLETE_SEMANTIC_VALIDATION_AND_LEARNING_UNMET',
        'stages':[
            {'stage':1,'status':'COMPLETE_BOUNDED_AUDIT','result':'100 names,1199 original annotations inspected;32 lexical-linked names; no verified variant or acceptability targets'},
            {'stage':2,'status':'IMPLEMENTED_RESEARCH_SCHEMA_AND_COMPARISON_COMPONENT','result':'100 profiles,49 candidate source links,168 source-feature mentions/metadata records; not a learned semantic importance model'},
            {'stage':3,'status':'TECHNICAL_REPLAY_COMPLETE_LINGUISTIC_VALIDATION_NOT_ESTABLISHED','result':'18 public clips/6 expressions/3 recorded IDs;36 directed reference pairs; coverage and missing-data contracts verified; actual acceptable/error labels unavailable'},
            {'stage':4,'status':'NOT_RUN_LABEL_PREREQUISITES_UNMET','result':'Authorized conditionally;0 gradient updates. Neither video-feature learning nor pedagogical-correctness improvement established.'}],
        'source_summary':profiles['summary'],'acquisition':read(ROOT/'source_audit/ACQUISITION_SUMMARY.json'),
        'measurement_totals':public['totals'],'public_word_coverage':coverage,
        'program_checks':{'root':19,'public':9,'independent_review_available':final_review.exists()},
        'verified_real_error_examples':0,'new_classification_accuracy':None,'new_feedback_accuracy':None,
        'preserved':{'C':True,'T3F':True,'operating_web':True,'operating_DTW':True,'protected_and_input_hashes':90},
        'user_mentor_access':0,'training_runs':0,'source_variant_mappings_verified':0,
        'next_priority':'Match a small source variant to actual public clips, then obtain visible token-feature labels and separately validated acceptable/error examples. Keep recorded people separate. Do not start another detector search or tune thresholds to current user.'}
    dump(ROOT/'RESULT.json',report)
    rows=[]
    for label,key in [('손가락별 굽힘각','angles'),('몸 기준 손목·팔꿈치 위치','body_position'),('얼굴 부위와의 투영 거리','face_distance'),('직접 비교 근거까지 갖춘 얼굴 거리','face_direct')]:
        t=public['totals'][key]
        rows.append(f'<tr><th>{label}</th><td>{t["planned"]:,}</td><td>{t["jointly_observed"]:,}</td><td><b>{100*t["joint_coverage"]:.2f}%</b></td></tr>')
    wordbars=''.join(f'<div class="wordbar"><b>{h(x["label"])}</b><div><span style="width:{x["metrics"]["angles"]["percent"]:.2f}%"></span></div><strong>{x["metrics"]["angles"]["percent"]:.1f}%</strong></div>' for x in coverage)
    cards=[]
    for p in profiles['profiles']:
        entries=[]
        for e in p['candidate_entries']:
            short=' '.join((e['description'] or '').split()[:15])
            families=list(dict.fromkeys(c['family'] for c in e['claims']))
            names={'handshape':'손 모양','body_relation':'몸 부위','orientation':'손 방향','contact_description':'접촉 기술','movement':'움직임','order':'순서 언급','repetition_description':'반복 언급','hand_count':'손 수','two_hand_relation':'양손 관계'}
            tags=' '.join(f'<span class="tag">{names.get(f,f)}</span>' for f in families)
            entries.append(f'<article class="entry"><a href="{h(e["url"])}" target="_blank" rel="noreferrer">{h(e["entry_label"])} ↗</a><small>{h(e["source_id"])} · {h(e["entry_id"])}</small><p>{h(short)+"…" if short else "구조화된 수형·손 수 등의 메타데이터"}</p>{tags}<p class="muted">이름 연결 후보 · 영상 형태 일치 미확인 · 필수 여부/허용 오차 미확정</p></article>')
        cards.append(f'<details class="profile" data-label="{h(p["display_label"])} {h(p["word_id"])}" data-mapped="{int(bool(entries))}"><summary><b>{h(p["display_label"])}</b><span>{len(entries)}개 사전 후보</span></summary>{"".join(entries) if entries else "<p>이번 제한 검색에서는 이름 연결 후보를 확보하지 못했습니다. 사전에 없다는 뜻은 아닙니다.</p>"}</details>')
    reviewtext='별도 에이전트 검산 완료' if final_review.exists() else '별도 에이전트 최종 검산 진행 중'
    doc='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>수어 특징 기반 피드백 · 자료와 구현 검증</title><style>
    *{box-sizing:border-box}body{margin:0;background:#f3f5f7;color:#142236;font:16px/1.75 "Malgun Gothic",sans-serif}main{max-width:1120px;margin:auto;padding:38px 28px 70px}h1{font-size:34px;line-height:1.35;letter-spacing:-1.5px}h2{font-size:25px;letter-spacing:-.7px;margin:0 0 20px}h3{font-size:18px;margin:0 0 8px}p{margin:10px 0}a{color:#1458a4;overflow-wrap:anywhere}section{background:white;border:1px solid #dee4ea;border-radius:18px;padding:30px;margin:25px 0}.eyebrow{color:#22766b;letter-spacing:2px;font-size:13px;font-weight:bold}.lead{font-size:18px;max-width:900px}.status{border-left:5px solid #c78a24;background:#fff6e7;padding:20px 24px;border-radius:8px}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:15px}.card{padding:20px;border:1px solid #dce5eb;border-radius:13px;background:#f8fafb}.card b{font-size:27px;display:block}.muted,small{color:#586779;font-size:13px}small{display:block}.good{color:#167363}.hold{color:#94630e}.flow{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px}.flow>div{border:1px solid #dce5eb;border-radius:12px;padding:18px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:13px 9px;border-bottom:1px solid #e1e6eb;text-align:left}th{font-weight:600}thead{background:#edf4f6}.tablewrap{overflow-x:auto}.wordbar{display:grid;grid-template-columns:90px 1fr 65px;align-items:center;gap:12px;margin:13px 0}.wordbar>div{height:13px;background:#e6ecef;border-radius:10px;overflow:hidden}.wordbar span{height:100%;display:block;background:#2b8175;border-radius:10px}.wordbar strong{text-align:right;font-size:14px}img{max-width:100%;width:100%;height:auto;border:1px solid #e2e6ea;border-radius:10px}.tag{font-size:12px;display:inline-block;padding:2px 9px;background:#edf3f9;border-radius:20px;margin:3px}.entry{border-top:1px solid #dee4e9;margin-top:15px;padding-top:15px}.profile{border:1px solid #dae2e8;border-radius:9px;margin:10px 0;padding:12px 16px}.profile summary{cursor:pointer;display:flex;justify-content:space-between;gap:12px}.profile summary span{color:#647489;font-size:13px}.tools{display:flex;gap:10px;margin:18px 0;flex-wrap:wrap}input,select{font:inherit;padding:9px 12px;max-width:100%;border:1px solid #bccbd6;border-radius:8px}input{flex:1;min-width:160px}details[hidden]{display:none}ol,ul{padding-left:23px}li{margin:9px 0}.warn{background:#fff7e9;border:1px solid #ecdab5;border-radius:12px;padding:20px}.pill{background:#e7f2ee;color:#1c6657;padding:5px 10px;border-radius:5px;font-size:13px}.foot{font-size:13px;color:#5a6879}@media(max-width:700px){main{padding:22px 14px 45px}section{padding:21px 16px}h1{font-size:27px}h2{font-size:22px}.grid{grid-template-columns:1fr 1fr}.flow{grid-template-columns:1fr}.wordbar{grid-template-columns:65px 1fr 58px;gap:8px}.lead{font-size:16px}.card{padding:14px}table{font-size:12px}td,th{padding:9px 5px}}
    </style><main><div class="eyebrow">WORDS IN SILENCE · 2026.10.27</div><h1>좌표 차이에서<br>이유를 설명하는 피드백으로</h1><p class="lead">사전의 설명을 찾고, 공통 특징으로 정리하고, 실제 공개 영상의 비교 근거와 연결했습니다. 다만 <b>“이 차이가 틀린 수어다”를 학습시킬 정답은 아직 확보되지 않았습니다.</b></p>
    <div class="status"><b>현재 결론</b><br>자료 조사·공통 구조·공개 관측 비교의 구현과 검산은 완료했습니다. 교육적 정오 검증은 미완료이며 새 학습은 실행하지 않았습니다. 기존 C·T3-F와 웹 테스트는 그대로입니다.</div>
    <section><h2>승인한 네 단계는 어디까지 왔나</h2><div class="grid"><div class="card"><b>01</b><h3>공식 자료 확인</h3><span class="good">제한 범위 조사 완료</span><p>100개 단어와 1,199개 원래 주석 확인. 32개 이름에 사전 후보 확보.</p></div><div class="card"><b>02</b><h3>공통 특징 구조</h3><span class="good">연구용 구현 완료</span><p>출처·표현 후보·적용 시점·미확정을 분리. 의미를 영상에 확정 연결한 모델은 아님.</p></div><div class="card"><b>03</b><h3>공개 자료 검증</h3><span class="hold">기술 검증 완료</span><p>18영상·36쌍 비교. 관측 누락 보존 확인. 정상/오류의 언어적 판정은 미검증.</p></div><div class="card"><b>04</b><h3>특징 모델 학습</h3><span class="hold">정답 자료 부족으로 미실행</span><p>학습 승인 문제는 없음. 영상별 특징 정답과 허용/오류 정답이 필요.</p></div></div></section>
    <section><h2>왜 사전 설명을 바로 학습시키지 않았나</h2><div class="flow"><div><h3>이름이 같아도 뜻이 다름</h3><p>‘감사’ 검색에는 고마움과 감찰·검사의 뜻이 함께 나옵니다. 이름만 보고 자동 연결하면 엉뚱한 동작을 가르칠 수 있습니다.</p></div><div><h3>처음 모양 ≠ 전체 모양</h3><p>누리사전 수형은 가장 먼저 나타나는 모양입니다. 이 값을 모든 프레임의 정답으로 붙이면 도중에 바뀌는 정상 손 모양을 오답으로 만들 수 있습니다.</p><a href="https://ksldict.korean.go.kr/kslserv/other/information.do">공식 일러두기 ↗</a></div><div><h3>설명 ≠ 허용 경계</h3><p>사전의 코 접촉 설명만으로 “몇 cm 벗어나면 틀림”을 정할 수 없습니다. 여러 정상 영상의 최솟값·최댓값도 전체 허용 범위는 아닙니다.</p></div></div><p class="muted">구사전의 1지=검지, 5지=엄지입니다. MediaPipe 관절 번호와 구분했습니다. 첫 페이지 중심의 제한 검색이므로 미연결 68개가 사전에 없다는 뜻은 아닙니다.</p></section>
    <section><h2>실제로 만든 연결 구조</h2><div class="flow"><div><h3>사전 근거</h3><p>손 모양 · 몸 부위 · 손 방향 · 접촉 기술 · 움직임 · 순서 · 반복</p><p class="muted">현재100 프로필 / 49개 항목 연결 / 168개 특징 언급·메타데이터 후보. 아직 영상 정답이 아님.</p></div><div><h3>영상 측정</h3><p>손가락별 2D 굽힘각 · 몸 기준 손목/팔꿈치 위치 · 얼굴 부위와 거리</p><p class="muted">사람별 참고를 따로 유지. 좌우 손이나 원래 관측을 새로 배정하지 않음.</p></div><div><h3>분리해서 표시</h3><p>얼마나 다른지 / 어디를 못 봤는지 / 시간 정렬이 미확인인지 / 사전 형태 연결이 미확인인지</p><p class="muted">아직 단어별 중요도를 학습해 교정하는 기능이 아닙니다.</p></div></div></section>
    <section><h2>얼마나 실제로 비교할 수 있었나</h2><p>6표현 × 기록 ID 3개 = 18영상. 각 영상을 다른 두 사람의 참고와 비교한 <b>36개 방향쌍</b>입니다. 이미 연구에서 살펴본 개발 자료이므로 새로운 실사용 성능시험은 아닙니다.</p><div class="tablewrap"><table><thead><tr><th>비교 항목</th><th>예정 항목 수</th><th>양쪽에서 관측</th><th>비교 가능</th></tr></thead><tbody>__ROWS__</tbody></table></div><p class="muted">단위: 고정된 주석 시간의 프레임 × 손 × 측정 항목 × 개별 참고. 손이 사라져도 분모는 유지합니다. 정상 한 손 동작의 쉬는 손도 포함된 기술적 분모이므로, 미관측을 곧바로 실패나 수어 오류로 해석하지 않습니다. 관측된 좌표 자체도 틀릴 수 있습니다.</p><h3>표현별 손가락 각도 비교 가능 비율</h3>__BARS__<p><b>77.78%는 수어 정답률이 아닙니다.</b> 손가락 차이를 계산할 근거가 양쪽에 함께 있던 비율입니다. 나머지는 차이가 0인 것처럼 채우지 않았습니다.</p></section>
    <section><h2>실제 비교 예시</h2><p>같은 공개 표현을 한 세 사람의 오른손 검지 중간 굽힘각입니다. 진한 선이 비교 대상, 나머지 두 선이 시간축을 맞춘 각각의 참고입니다. 선이 끊긴 구간은 측정 근거가 부족한 곳입니다.</p><img src="public_comparison/example_1.svg" alt="남아 공개 세 사람의 검지 굽힘각 비교"><img src="public_comparison/example_2.svg" alt="냉커피 공개 세 사람의 검지 굽힘각 비교"><details><summary>세 번째 고정 예시 보기</summary><img src="public_comparison/example_3.svg" alt="멀다 공개 세 사람의 검지 굽힘각 비교"></details><p class="muted">그래프는 화면에 투영된 2D 각도이며 실제 3D 관절 각도가 아닙니다. 기존 시간 정렬을 재사용했으므로 모든 시점이 의미상 같은 단계라는 보장도 없습니다. 두 참고 사이에 들었다는 이유로 정답을 선언하지 않습니다.</p></section>
    <section><h2>100개 단어의 사전 연결 확인</h2><p>자료를 바로 살펴볼 수 있게 검색 가능한 표로 정리했습니다. 후보가 여러 개면 하나를 임의로 고르거나 평균 내지 않았습니다.</p><div class="tools"><input id="search" placeholder="단어 이름 또는 WORD 번호 검색" aria-label="단어 검색"><select id="filter" aria-label="표시 범위"><option value="mapped">사전 후보 있는 32개</option><option value="all">전체 100개</option></select></div><p id="count" class="muted"></p>__CARDS__</section>
    <section><h2>검증 결과와 한계</h2><ul><li>출처 처리 19개 + 공개 측정 처리 9개 검사를 통과했습니다. 수어 정오 정확도 시험과는 다릅니다.</li><li>손을 제거한 시험에서 비교 분모는 그대로이고 관측률만 낮아졌습니다. 이웃 각도와 손 위치는 독립적으로 유지됐습니다.</li><li>좌우가 화면에서 교차해도 손 ID를 바꾸지 않았고, 전체 화면의 평행이동·동일비율 크기 변화에도 정규화 측정값이 유지됐습니다.</li><li>원본·입력·운영 보호 90개 해시가 유지됐습니다. 사용자·멘토 촬영 접근 0, 새 학습 0입니다.</li><li>__REVIEW__. 별도 Codex의 계산·코드 검토이며 한국수어 전문가의 정답 검토를 대신하지 않습니다.</li></ul><div class="warn"><b>아직 입증하지 않은 것</b><p>정상 동작에 잘못 경고하는 비율, 실제 잘못된 동작을 찾아내는 비율, 새로운 TCN 인식률 개선은 이번 자료로 계산할 수 없습니다. 0%나 100%로 보고하지 않았습니다.</p></div></section>
    <section><h2>이제 무엇이 있어야 학습을 시작할 수 있나</h2><ol><li><b>작은 공개 표현 묶음에서 사전 형태와 실제 영상을 연결합니다.</b> 같은 이름인지 확인하는 것으로 끝내지 않고, 어느 뜻·변이의 동작인지 확인해야 합니다.</li><li><b>눈에 보이는 특징 정답을 붙입니다.</b> 어떤 손의 어느 시간에 어떤 모양·부위 관계가 보이는지 확인합니다. 이 자료로 특징을 읽는 모델을 학습할 수 있습니다.</li><li><b>교정을 위한 허용/오류 자료를 따로 갖춥니다.</b> 확실한 기술적 관측과 언어적 정오를 구분하고, 불확실한 허용 경계만 한국수어 검토에 집중합니다.</li><li><b>다른 사람을 남겨 비교합니다.</b> 단어 인식률과 별개로 잘못된 교정·오류 놓침·판정 가능한 비율을 함께 평가합니다.</li></ol><p>32개 후보와 관련 공개 기록을 <a href="FEATURE_LABEL_PREPARATION.json">주석 준비 목록</a>으로 저장했습니다. 모든 단어의 모든 프레임을 수동 검토하라는 목록이 아닙니다. 새 검출기를 더 붙이거나 근거 없는 임계값을 반복 조정하는 단계로 돌아가지 않습니다.</p></section>
    <p class="foot">근거: <a href="RESULT.json">종합 결과</a> · <a href="FEATURE_PROFILES.json">100개 특징 프로필</a> · <a href="INTEGRATED_EVIDENCE_INDEX.json">사전·공개 비교 연결 색인</a> · <a href="public_comparison/RESULT.json">36쌍 비교</a> · <a href="TRAINING_READINESS.json">학습 조건 확인</a> · <a href="source_audit/SOURCE_PROVENANCE.json">공식 출처</a> · <a href="review/EVAL_PROTOCOL.json">검증 절차</a></p>
    </main><script>const box=document.getElementById('search'),filter=document.getElementById('filter');function update(){let n=0;const q=box.value.trim().replaceAll(' ','').toLowerCase();document.querySelectorAll('.profile').forEach(el=>{const ok=(filter.value==='all'||el.dataset.mapped==='1')&&el.dataset.label.replaceAll(' ','').toLowerCase().includes(q);el.hidden=!ok;if(ok)n++;});document.getElementById('count').textContent=n+'개 표시 · 이름 일치 후보이며 형태 확인은 별도';}box.addEventListener('input',update);filter.addEventListener('change',update);update();</script></html>'''
    doc=doc.replace('__ROWS__',''.join(rows)).replace('__BARS__',wordbars).replace('__CARDS__',''.join(cards)).replace('__REVIEW__',reviewtext)
    (ROOT/'RESULT_KO.html').write_text(doc,encoding='utf-8')
    text='''수어 특징 기반 피드백: 자료 조사·공통 구조·공개 비교 검증

1. 현재100개 단어 및 AIHub 주석1199개를 확인했다. 원래 주석은 단어이름/시간/영상정보이며 손모양·허용오류 정답은 없다.
2. 두 공식사전의 제한검색에서32개 이름에49항목 연결후보를 확보했다. 전체 획득상세는 보조 커피 포함50항목이다. 100개프로필과168개 출처 특징언급/메타데이터를 만들었으나 실제영상의 의미·변이·단계 연결은 미확인이다.
3. 연구용 공통 수치비교·출처표시 구조를 구현했다. 공개18영상/6표현/3기록ID를 다른2사람과 각각 비교한36방향쌍에서 손각도77.78%, 몸위치91.32%, 얼굴거리83.29%(직접근거81.17%)의 양쪽관측을 확인했다. 정확도가 아니라 비교근거비율이다. 필수손/휴식손 의미분모가 아니며 누락을 틀린수어로 판단하지 않았다.
4. 19개 출처처리 계약+9개 공개측정 계약 통과. 원본/모델/운영90핀 보존. C/T3F/웹/DTW 운영변경0, 사용자·멘토접근0, 새학습0.
5. 교육적 정오 검증은 미완료이며 특징학습도 실행하지 않았다. 실제영상-사전변이 연결0, 독립 영상특징라벨0, 검증된오류정답0이므로 학습조건이 충족되지 않았다. 승인이 없어서 멈춘 것이 아니다. 새 TCN성능이나 피드백정확도가 개선됐다고 보고하지 않는다.
6. 다음은 작은공개표현에서 실제변이와 특징시간을 확인한 자료를 확보하고, 허용/오류판정은 별도검증한다. 같은사람으로 학습과평가를 섞지 않는다. 32개후보 주석준비색인은 마련했다. 모든프레임 수동검토나 새검출기 추가를 요구하는 계획이 아니다.

실제 완료: 제한자료조사/공통출처구조/기술적공개비교/검산/학습전제확인.
미완료: 언어적 중요도·허용범위의 실제영상 연결검증, 특징모델학습, 피드백성능입증, 운영적용.
'''
    (ROOT/'RESULT_KO.txt').write_text(text,encoding='utf-8')
    print(report['status'])

if __name__=='__main__':main()
