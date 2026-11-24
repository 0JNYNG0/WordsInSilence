"""Close verified preparation unit and synchronize latest project state."""
import json
import re
from pathlib import Path
from datetime import datetime, timedelta, timezone

U=Path(__file__).resolve().parent
ROOT=U.parents[3]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    final=read(U/'review/FINAL_REVIEW.json')
    ui=read(U/'UI_VERIFY.json')
    assert str(final.get('status','')).startswith('PASS'),final
    assert ui['status']=='PASS'
    now=datetime.now(timezone(timedelta(hours=9))).isoformat()
    unit=U.name
    summary=('직전 단계의 이름 연결에서 실제 공개 RGB 표본의 뜻/표현형·시간/손/특징 연결로 진행했다. 기존6표현 중 사전후보가 있는 멀다·믿다·감사·별 4표현, 기록ID01~03 각3개 총12공개TRAIN영상을 결과 확인 전에 선택했다. 각16장 총192RGB를 실제 검토하고 141고유표본에38관찰묶음을 기록했다. 멀다는 한손 얼굴옆→전방 지시가 보여 두손 시작 사전8171의 강제연결을 보류했고 추가6공식조회에서도 맞는 설명을 확보하지 못했다(세영상이 틀렸다는 뜻 아님). 믿다는 검지/반대펼친손 접근을 잠정연결, 감사는 검사·감찰의 동음이의12468을 제외하고 고마움2372의 큰구조만 연결하되 아래손면 방향·접촉·정확횟수를 보류, 별은 오므림/펼침만 부분연결하고 별1/2 및 정확경로/높이허용을 미확정으로 남겼다. 손모양 정적표본과 전후표본의 움직임근거를 분리했고, 최초검토에서 발견한 내림말미 정지프레임의 motion오해를 before_motion_scope_review에 보존 후 수정했다. 별도Codex가 사전표 읽기 전12클립표본을 관찰하고, 원본 별도디코딩192장 RGB일치·PTS차이0us·49원본/216이미지/32운영핀 불변을 검산했다. HTML12슬라이더24끝점·데스크톱/모바일 렌더PASS. 큰특징 연결3개이나 전체변이최종확정0/교육적정오정답0/추가학습0. 같은AI교차검토이며 독립사람/수어전문가 정답이나 정확도향상이 아니다. C·T3F·운영웹/DTW변경0, 사용자/멘토접근0, 표정/새검출기없음.')
    nxt=('다음 한 작업은 TOKEN_OBSERVATIONS.json의 검지펴기/펼친손/모음·펼침/손-손·손-얼굴관계를 공통시각특징으로 정리해, 같은특징을 가진 다른 공개표현과 대조사례를 추가한 주석·평가묶음을 만드는 것이다. 4표현단어암기를 특징이해로 오인하지 않도록 기록사람 및 표현을 나눈 평가를 사전동결하고, 독립검증 가능한 공개특징정답/검토근거를 확보한다. 관측모델에는 목표단어ID/Gloss/정답사전설명을 입력하지 않는다. 표본사이를 자동보간해 dense정답으로 채우지 않고 sequence변화는 단일프레임motion정답으로 쓰지 않는다. 접촉·가려진손가락·정확횟수·필수조건·허용범위는 null을 유지하며, 특징학습과 교육적정오학습을 구분한다. 승인된 일반 준비는 재승인 없이 진행 가능하나 정답자료없이 성능검증완료로 올리지 않는다. 기존C/T3F/운영보존 및 사용자·멘토 학습튜닝선택금지 유지. 활성 실행 없음.')
    evidence=f'research/front_keyphase_v1/ops/{unit}/RESULT_KO.html'
    ctx=ROOT/'WordsInSilence_PROJECT_CONTEXT.md'
    text=ctx.read_text(encoding='utf-8-sig')
    title='2026-10-27 공개 실제표현형·시간별 시각특징 주석 준비'
    assert title not in text,'Already finalized'
    for section,body in [(8,summary),(9,nxt)]:
        pattern=rf'(^## {section}\. [^\n]*\n)'
        insertion=f'\n### {title}\n{now}\n{body}\n근거: {evidence}, TOKEN_OBSERVATIONS.json, SOURCE_FORM_LINKS.json, review/FINAL_REVIEW.json, UI_VERIFY.json.\n\n'
        text,n=re.subn(pattern,lambda m:m.group(1)+insertion,text,count=1,flags=re.M)
        assert n==1
    ctx.write_text(text,encoding='utf-8')
    sp=ROOT/'research/front_keyphase_v1/ops/STATE.json'
    state=read(sp);state['updated_date_kst']=now;state['current_stage']='SOURCE_TOKEN_VISIBLE_FEATURE_DRAFTS_VERIFIED_LEARNING_TARGETS_PENDING'
    state['next_single_action']=nxt
    state[unit]=dict(status='PUBLIC_RGB_SOURCE_FORM_REVIEW_AND_SAMPLE_ANNOTATION_COMPLETE',at=now,summary=summary,next=nxt,
        evidence=evidence,counts=read(U/'RESULT.json')['counts'],implementation='Codex root + local_feature_inventory + ksl_source_audit',
        verification='Separate Codex linguistic_eval_review; not independent human or sign-language ground truth',
        training_runs=0,active_execution=None,authority='User requested next core task; earlier conditional autonomous four-stage authorization retained.')
    save(sp,state)
    handoff=f'''# WordsInSilence 최신 인수인계

{now}

## 현재 목표와 마지막 완료
{summary}

## 다음 한 작업
{nxt}

## 실행·검토
- 구현: root(표본주석·출처형태판정·보고서), local_feature_inventory(공개RGB12개·192표본), ksl_source_audit(멀다 공식6건 추가조회).
- 별도 검토: linguistic_eval_review. 사전표 읽기 전 OBSERVATIONS_STAGE1.json 고정 후 출처비교. 같은AI, 완전블라인드/한국수어전문가 아님.
- 최초문제/복구: 내림그룹에 이미 멈춘 말미표본 포함. 최초파일을 before_motion_scope_review에 보존 후 sequence범위를 명시하고 lowering_and_or_low_rest_samples로 정정. 정적표본과 시퀀스운동 구분.
- 멀다 표현형 미연결, 감사의 세부면방향/접촉/정확횟수, 별의 전체경로/변이 미확정은 결과이지 검산실패가 아니다.
- 12개영상 전체를 연속 재생한 것이 아니라 16장씩 표본 확인. 표본시간사이를 dense정답으로 만들지 않음.
- 학습0, 새추론/재정렬0, 운영교체0, 사용자/멘토접근0. 활성 학습/서버/자동화 없음.

## 근거
- `{evidence}`: 사진 슬라이더와 쉬운 설명.
- `{unit}/TOKEN_OBSERVATIONS.json`: 38묶음/141고유표본의 시점·손·특징 초안.
- `{unit}/SOURCE_FORM_LINKS.json`: 동음이의 제외, 큰형태/미확정 구분.
- `{unit}/ANNOTATION_READINESS.json`, `LEARNING_CONTRACT.json`: 조건부 학습의 남은 정답/분리평가 요건.
- `{unit}/review/FINAL_REVIEW.json`: 별도검산 및 시각판단의 한계.
- `{unit}/UI_VERIFY.json`: 데스크톱/390px·12슬라이더24끝점 검증.

기존 반복 승인 대기 문구는 이번 자율 승인 범위를 취소하지 않는다. 이번 완료는 실제표본을 확인한 준비 단위이며 수어의 필수특징·허용범위를 학습한 피드백 모델 완성이 아니다.
'''
    (ROOT/'NEW_CHAT_HANDOFF.md').write_text(handoff,encoding='utf-8')
    save(U/'CLOSEOUT.json',dict(status='COMPLETE_PREPARATION_UNIT',at=now,root_docs_synced=True,report_ui_pass=True,review_status=final['status'],active_execution=None,training_runs=0))
    print('Synchronized latest context, STATE and handoff')

if __name__=='__main__':main()
