"""Synchronize the completed public diagnostic without modifying prior evidence."""
import hashlib,json
from pathlib import Path
from datetime import datetime,timezone,timedelta
U=Path(__file__).resolve().parent;O=U.parent;R=O.parents[2]
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
now=datetime.now(timezone(timedelta(hours=9))).isoformat()
key=U.name
f=read(U/'FROZEN.json');v=read(U/'VERIFY.json');a=read(U/'FINAL_AUDIT.json')
for p,h in {**f['pins'],**f['protected']}.items():assert sha(p)==h,p
for n,p in [('WordsInSilence_PROJECT_CONTEXT.md',R/'WordsInSilence_PROJECT_CONTEXT.md'),
            ('NEW_CHAT_HANDOFF.md',R/'NEW_CHAT_HANDOFF.md'),('STATE.json',O/'STATE.json')]:
    assert sha(p)==sha(U/'before'/n),f'Concurrent document change: {p}'
next_action='公開TRAIN36영상의 LK 추적 비교 결과를 사용자와 검토한다. 후보는 미채택: 자연누락9개1프레임 위치보조(본동작2),좌우오류12유지,손회전시중심드리프트반례. 다음은 손영역추적과 해부학적손바닥중심을 분리하고 회전·재등장근거를 검증하는 작은 후보의 조건 검토다. 첫등장/완전가림은추적만으로해결불가,현재이미지재검출필요성분리. C·T3-F/운영DTW/웹유지,사용자·멘토자료학습·튜닝·선택금지. 결과확인전추가학습·자동다운로드·운영반영없음.'
next_action=next_action.replace('公開','공개')
title='### 2026-10-20 공개 영상 추적 보조 비교 완료 — 짧은 위치 보조·중심 드리프트 한계, 미채택'
entry=f"""{title}
- 단위 `research/front_keyphase_v1/ops/{key}/RESULT_KO.html` 및 `RESULT_KO.txt`. 사용자 승인 공개 영상 시각화·현C 대 영상추적 비교 완료. 학습/새MediaPipe/분류재평가/사용자자료 접근0.
- 공개 TRAIN36영상·23단어·9촬영자·4641프레임. 같은sensor범위·저장C관측 사용,현재/과거 영상만. 0.20초 상한 LK 추적, 실제 팔·손 근거 시작점, 관측손가락 생성0. 단어별예외없음.
- 자연 손바닥위치 누락에 추정9개 추가(8영상·5단어),전부1프레임약0.033초. Codex원영상 검토상 손영역내9개;정밀관절정답아님. 주석상 본동작2개/밖7개(주로손내림). 긴가림복원성과로 확대금지.
- 기존정상42/42보존,기하학적교차54프레임배정변화0,알려진오류12/12그대로(이전9포함). 전체배정변경0이므로 좌우정정 성공아님. 전체4641프레임수기판정/새사용자/전체100검증아님.
- 검출만숨긴70×6=420손프레임부가시험:위치추정61(14.52%),보류359. 숨긴MediaPipe중심proxy오차중앙1.79%어깨너비/최대19.81%. 10%초과4프레임은상처07한구간;영상점은같은손안이지만회전시해부학적중심과벌어짐. 10%는분석요약선,수어허용기준아님.
- `VERIFY.json`:저장추적2769건산술·인과시간검산. `VISUAL_REVIEW.json`:자연9추정/부가4반례원영상검토. 동일Codex별도코드·시각검토이며 독립사람/에이전트검증아님.
- 최초22영상후NaN진단JSON저장오류보존. 저장형식만복구,완료22해시보존하고나머지14완료. 동결188핀/보호32해시유지. `FIRST_ERROR.json`,`STORAGE_RECOVERY.json`.
- 보고서36영상·36이동그래프·장면확대완료. 로컬링크/영상프레임수/JS구문확인. 브라우저도구file://정책차단으로자동미리보기미확인,우회없음. 영상/그래프이미지는view_image로검토.
- 개인피드백은 관측불가안내와실제동작차이교정을분리. 여러참고분포는허용범위출발자료이지그밖=오답의검증기준아님. 가림은정상수어에서도가능.
- C·T3-F/운영DTW/웹 유지,현재후보미채택. 완료명령종료,실행잠금없음. 다음은결과검토후중심드리프트/재등장후보설계;학습시작금지.

"""
t=(R/'WordsInSilence_PROJECT_CONTEXT.md').read_text('utf-8-sig')
assert title not in t
h8='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인'
h9='## 9. 다음 단일 작업'
t=t.replace(h8,h8+'\n\n'+entry,1).replace(h9,h9+'\n\n'+title+'\n'+next_action+'\n\n',1)
(R/'WordsInSilence_PROJECT_CONTEXT.md').write_text(t,encoding='utf8')
s=read(O/'STATE.json')
s.update(updated_date_kst='2026-10-20',updated_at=now,
    current_stage='WORD100_IMAGE_TRACKING_PAIR_REVIEW',activity_state='AWAITING_USER_REVIEW',
    next_single_action=next_action,resume_command=next_action)
s[key]=dict(status='COMPLETED_LIMITED_POSITION_SUPPORT_NOT_ADOPTED',at=now,unit=key,
    role='Codex implementation and execution; separate saved arithmetic and visual review by same Codex, no independent agent/human',
    scope=dict(videos=36,words=23,train_people=9,frames=4641),
    result=v,final_audit=a,visual_review='VISUAL_REVIEW.json',
    artifacts=['RESULT_KO.html','RESULT_KO.txt','FROZEN.json','VERIFY.json','FINAL_AUDIT.json','VISUAL_REVIEW.json'],
    error_recovery='FIRST_ERROR preserved; serialization-only resume,22 completed hashes preserved and14 completed. Frozen algorithm unchanged.',
    production_adoption=False,existing_C_T3F_unchanged=True,user_recordings_used=False,
    browser_preview='Blocked by browser file:// URL policy; no workaround. Local assets inspected, links/video metadata/JS syntax verified.',
    next_single_action=next_action)
write(O/'STATE.json',s)
handoff=f"""# WordsInSilence 최신 인수인계

갱신 {now}

현재 목표: 본동작의 관측 손실을 줄이되 없는 손가락 정보로 피드백하지 않기.
마지막 완료: 공개 TRAIN36영상의 C 관측 대 LK 영상 추적 보조 및 손 이동 시각화.
다음 한 작업: 이번 결과 검토 후 손영역 추적과 손바닥 중심을 분리하는 후보 조건 검토. 추가 학습은 아직 하지 않는다.

- 현재 단위: `{U}`. 보고서 `RESULT_KO.html`, 상세 `RESULT_KO.txt`.
- 36영상/23단어/9촬영자/4641프레임. sensor범위의 동일픽셀·저장C raw/assignment 사용. LK 현재/과거만,max0.20초. 화면밖/미래이미지나 사용자촬영 사용0.
- 자연누락9개 순간에 위치보조,8영상·5단어,모두1프레임약0.033초. 원영상검토상 손영역내9개이나 정밀관절복원아님. 주석상본동작2/밖7(주로손내림). 긴가림 해결아님.
- 정상42/42보존,교차54프레임배정변화0. 알려진좌우오류12개그대로(이전9포함),전체배정변경0. 따라서 정상보존은있지만 좌우정정성과없음.
- 합성검출누락70구간×6프레임420중추정61/보류359. RGB와팔정보유지,영상복제/시간삭제없음. 숨긴MediaPipe중심은proxy:오차중앙어깨너비1.79%,최대19.81%. 10%초과4프레임은상처07한구간;손회전시특징점이손바닥중심과다른위치. 같은손영역이며잘못된다른손으로추적확정아님.
- 손가락관측생성0. 자연표본의손위치보조와손모양복원을구분. 관측불가=촬영/확인안내,관측된차이=분석근거,수어오답/교정은별도검증.
- `VERIFY.json`:추적2769건산술·시점검산. `VISUAL_REVIEW.json`:추가자연9/부가반례4시각검토. 같은Codex별도검산이며다른사람/에이전트독립검토아님.
- 최초22영상뒤NaN진단JSON저장오류. 최초증거/완료22해시보존,저장형식만별도복구해14완료. 동결코드·계수변경없음. 완료실험 재실행하지않기.
- 기존C/T3-F6가중치포함32보호파일및188핀해시유지. 새학습/분류재평가/MediaPipe실행0. 운영웹·DTW변경없음. 새후보미채택. 기존후속연구후보들이웹C에모두반영됐다고설명하지않기.
- 원본/W:/sources/과거동결결과읽기전용. 사용자·멘토촬영학습/튜닝/후보선택금지. DEV/보류14~18접근없음.
- 36영상/36그래프/검토확대시트완료. 로컬링크·영상프레임수·JS구문검사통과. 브라우저file://정책차단으로자동미리보기미확인,우회하지않음. 이미지는view_image로검토.
- 실행명령종료,새RUNNING잠금없음. 연산중프로세스/추가자동실행없음.

{next_action}
"""
(R/'NEW_CHAT_HANDOFF.md').write_text(handoff,encoding='utf8')
write(U/'DELIVERY_CHECK.json',dict(
    at=now,local_links='PASS',video_files=36,video_frame_counts='PASS',plots=36,
    javascript_syntax='PASS_node_check',browser_visual_preview='NOT_CHECKED_file_protocol_policy_block',
    image_visual_review='contact sheets and scientific plot inspected',
    docs_synced=['WordsInSilence_PROJECT_CONTEXT.md','NEW_CHAT_HANDOFF.md','ops/STATE.json'],
    protected_hashes='PASS_32',source_and_frozen_hashes='PASS_188'))
write(U/'ARTIFACT_HASHES.json',{str(p.relative_to(U)):sha(p) for p in U.rglob('*')
    if p.is_file() and 'before' not in p.parts and p.name!='ARTIFACT_HASHES.json'})
assert read(O/'STATE.json')[key]['result']['counts']['missing_with_image_estimate']==9
assert next_action in (R/'NEW_CHAT_HANDOFF.md').read_text('utf8')
assert not (U/'RUNNING.json').exists()
print('CLOSED: docs synchronized; frozen/protected unchanged; no training; awaiting result review')
