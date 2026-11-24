from pathlib import Path
from datetime import datetime, timezone, timedelta
import json, hashlib, shutil
U=Path(__file__).resolve().parent
O=U.parent
R=U.parents[3]
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
now=datetime.now(timezone(timedelta(hours=9))).isoformat()
paths=[R/'WordsInSilence_PROJECT_CONTEXT.md',R/'NEW_CHAT_HANDOFF.md',O/'STATE.json']
(U/'before').mkdir(exist_ok=True)
for p in paths:
    if not (U/'before'/p.name).exists(): shutil.copy2(p,U/'before'/p.name)
if not (U/'FROZEN.json').exists():
    original=read(O/'word100_shared_feature_bundle_20261027/FROZEN.json')
    files=[O/'word100_shared_feature_bundle_20261027'/f for f in ['FROZEN.json','SOURCE_PROFILES.json','DRAFT_LABELS.json','LEARNING_MANIFEST.json','visual_inputs/INDEX.json']]
    files += [O/'word100_source_token_alignment_20261027'/f for f in ['SOURCE_FORM_LINKS.json','TOKEN_OBSERVATIONS.json']]
    pins={str(p):sha(p) for p in files}
    for row in original['rows']:
        p=Path(row['record']); assert sha(p)==row['record_sha256']
        pins[str(p)]=sha(p)
        rec=read(p); obs=Path(rec['views']['masked']['observation']); pins[str(obs)]=sha(obs)
    protection=read(O/'word100_face_relation_alignment_20261027/FROZEN.json')['protected']
    for p,h in protection.items(): assert sha(p)==h,p
    protocol=dict(created_at=now,unit=U.name,public_clips=36,expressions=12,reference_people=[1,2],query_people=[3],
        observer_policy='guide/word-blind; observed and inferred distinguish; no fill missing',
        existing_draft_labels='same-AI consistency only, not independent educational truth',
        evaluation='development/replay of existing public TRAIN; not new C/T3-F test',
        numerical_alignment=dict(method='DTW',groups_equal_weight=True,missing_pair_cost=1.0,point_cost_clip=2.0),
        controls=['timestamp_speed_0.75','timestamp_speed_1.5','global_translate_scale','hand_observation_removed','middle_interval_removed','reverse_time'],
        mismatch_policy='numeric differences and missing evidence, never verified pedagogical correction',
        acceptance=['same-observations time-change invariance','translation/scale invariance','no expected answer in observer','missing evidence cannot pass as zero difference','whole reference scope retained','each reference preserved'],
        guide_requiredness='unknown unless independently substantiated',
        input_pins=pins,protected=protection)
    save(U/'FROZEN.json',protocol)
goal=read(U/'AUTHORIZATION.json')['goal_ko']
ctx=paths[0].read_text(encoding='utf-8-sig')
marker='## 2. 프로젝트 정체성과 최종 목표'
add='\n\n### 2026-10-28 사용자 확정: 현재 AI 피드백의 기준 목표\n\n> '+goal+'\n\n기술 부품의 점수 향상보다 이 전체 경험의 검증을 우선한다. V4는 보류하고 기존 C·T3-F는 보존한다. 한국수어 설명과 실제 영상의 표현형·시간 근거를 연결하되, 기대한 정답을 관측한 것처럼 채워 넣지 않는다. 허용 차이·중요 차이·관측 부족을 구분하며 미검증 값을 교육적 정오로 단정하지 않는다. 사용자·멘토 촬영은 학습·튜닝·선택에 사용하지 않는다. 새 외부 자료·모델은 출처/용도/규모/조건을 먼저 설명하고 확인받는다. 아래의 오래된 Unity 데모 범위는 이 연구 계보이며 현재 단일 작업은 최신 §8·§9가 우선한다.\n'
if '### 2026-10-28 사용자 확정: 현재 AI 피드백의 기준 목표' not in ctx: ctx=ctx.replace(marker,marker+add,1)
summary='사용자가 가이드·여러 사람 영상 기반 단계·관계 피드백 목표를 확정하고 1~4단계 자율 수행을 승인했다. 기존 공개36영상 자료 연결, 가이드와 분리한 관찰, 학습 정답 계약, 전체 비교 경로를 검증한다. V4보류·C/T3F보존·개인자료제외. 새 영상언어 모델 다운로드는 상세 설명 후 확인 대기이며 나머지 로컬 검증 진행 중.'
nxt='word100_guide_video_feedback_20261028/FROZEN.json에 고정한 공개36영상으로 관측→단계/가이드 대응→다중 참고 비교 경로 구현 및 검증. 새외부모델 확인 전 다운로드 금지. 교육적정답을 꾸미거나 관측누락을 정상으로 처리하지 않는다.'
for marker,title,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인','2026-10-28 사용자 확정 목표의 4단계 전체 경로 실행 중',summary),('## 9. 다음 단일 작업','2026-10-28 한국수어 가이드-영상 피드백 4단계 수행',nxt)]:
    if '### '+title not in ctx:ctx=ctx.replace(marker,marker+'\n\n### '+title+'\n'+now+'\n'+body+'\n근거: research/front_keyphase_v1/ops/'+U.name+'/WORK_PLAN_KO.txt.\n',1)
ctx=ctx.replace('최종 갱신: 2026-10-10 KST','최종 갱신: 2026-10-28 KST',1)
paths[0].write_text(ctx,encoding='utf-8')
paths[1].write_text('# WordsInSilence 최신 인수인계\n\n'+now+'\n\n목표: '+goal+'\n\n'+summary+'\n\n다음: '+nxt+'\n\n현재 단위: '+U.name+'\n기존 모델 운영 변경 없음. 새 다운로드 확인 대기 외에는 로컬 작업을 계속한다. 소스 감사/독립 평가 검토/관측 모듈 구현은 Codex 에이전트가 역할을 나누며 전문가 정답 검증으로 부르지 않는다.\n',encoding='utf-8')
s=read(paths[2]);s.update(current_stage='GUIDE_VIDEO_FEEDBACK_FOUR_STEPS_RUNNING',updated_at=now,next_single_action=nxt)
s[U.name]=dict(status='RUNNING',at=now,goal=goal,summary=summary,next=nxt,v4_on_hold=True,private_data_used=False,operating_changed=False)
save(paths[2],s)
save(U/'CURRENT_EXECUTION.json',dict(status='LOCAL_IMPLEMENTATION',updated_at=now,training_active=False,download_started=False))
print('Goal, execution and frozen public-input protocol recorded.')
