from pathlib import Path
from datetime import datetime, timezone, timedelta
import json, shutil, hashlib

U=Path(__file__).resolve().parent
R=U.parents[3]
KEY=U.name
def load(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    assert (R/'AGENTS.md').exists()
    final=load(U/'RESULT.json'); review=load(U/'review/FINAL_REVIEW.json');render=load(U/'HTML_RENDER_VERIFIED.json')
    assert review['status'].startswith('PASS') and render['status']=='PASS'
    assert final['training_runs']==0 and final['source_summary']['source_claims']==168
    # Verify the complete protected/input set after report production too.
    pins=load(U/'public_comparison/BEFORE.json')
    assert all(sha(p)==s for p,s in pins.items())
    state_path=R/'research/front_keyphase_v1/ops/STATE.json'
    if not (U/'before/STATE.json').exists():shutil.copy2(state_path,U/'before/STATE.json')
    stamp=datetime.now(timezone(timedelta(hours=9))).isoformat()
    title='한국수어 특징 출처 연결·공개 비교 및 학습 조건 검증'
    summary=('사용자가 제안1~4단계의 자율 실행을 승인했다. 공식 두사전 제한검색 및 실제 AIHub 주석1199개를 확인하고, '
             '100개프로필·32개이름연결후보·49상세연결·168출처특징후보를 구현했다. 획득상세전체는 보조커피포함50개이며 '
             '검색되지않은68개를 사전에없다고 판정하지 않았다. 초기수형 메타데이터는 첫손모양만, 구사전1지는검지/5지는엄지로 '
             '범위를 명시했고 감사의 동음이의·별의복수항목을 보존했다. source_id+entry_id 출처·시간범위·변이후보와 '
             '측정/미관측/정렬불확실성을 분리한 연구부품 및 사전-공개측정 병렬색인을 구현했으나, 사전의 중요특징을 '
             '실제영상단계에 연결해 판단하는 모델은 아직 아니다. 공개18영상/6표현/기록ID01~03의36방향쌍을 기존B정렬로 '
             '재사용 비교했다. 고정분모에서 손각도36959/47520=77.78%, 몸위치8679/9504=91.32%, 얼굴거리158322/190080=83.29%, '
             '직접근거154296/190080=81.17%는 양측관측비율이지 정답률·개선폭이 아니다. 이 분모는 양손전체 기술단위이며 '
             '필수손/휴식손 언어분모가 아니다. 손누락시 분모유지·이웃각도/위치보존·좌우재배정없음·평행이동/동일크기변환 '
             '불변 등 root19+공개9계약 통과. 별도Codex검산은 원래관측에서 각도/몸위치 별도스칼라재구성 차이0·36쌍통계·'
             '90원본/운영핀을 확인했다. 이는 한국수어 사람정답 검증이 아니다. 연구HTML100개검색/3그래프·데스크톱/390px '
             '검증완료. 사전-영상동일변이확인0/독립영상특징정답0/검증된실제오류0이므로 단계3언어적정오검증미완, '
             '단계4학습미실행0회. 승인부족이 아닌 정답자료 전제불충족이다. C/T3F·운영웹/DTW변경0, 사용자/멘토접근0, '
             '표정/새검출기/운영교체없음. 학습완료나 피드백정확도개선을 주장하지 않는다.')
    nxt=('현재 준비된32개 이름후보와 FEATURE_LABEL_PREPARATION.json에서 작은 공개 표현묶음의 사전 뜻/변이를 '
         '실제영상에 확인 연결하고, 눈에 보이는 손모양·부위관계의 손/시간/관측근거를 표본주석으로 확보하는 것이 다음 단일 작업이다. '
         '이후 기록사람을 나눠 특징학습을 비교할 수 있다. 교육적 교정용 허용변이/실제오류 정답은 별도로 검증한다. '
         '이름·사전첫수형·MediaPipe값·다중참고최솟값최댓값을 정답으로 자동승격하지 않는다. 전체단어모든프레임 '
         '수동검토나 새검출기/임계값반복탐색으로 확대하지 않는다. 사용자 승인범위 내 자율진행 권한은 유지하되 '
         '실제 학습정답 없이 학습이 통과됐다고 꾸미지 않는다. 기존C/T3F/웹보존·사용자/멘토자료학습튜닝선택금지. 활성실행없음.')
    evidence=f'research/front_keyphase_v1/ops/{KEY}/RESULT_KO.html'
    state=load(state_path);state['updated_date_kst']=stamp
    state['current_stage']='LINGUISTIC_SOURCE_AND_PUBLIC_MEASUREMENT_VERIFIED_LEARNING_PREREQUISITES_UNMET'
    state['next_single_action']=nxt
    state[KEY]={'status':final['status'],'at':stamp,'summary':summary,'next':nxt,'evidence':evidence,
                'implementation':'Codex root + source/local measurement agents',
                'verification':'Separate Codex linguistic_eval_review; arithmetic and code, not human sign truth',
                'stages':final['stages'],'active_execution':None,'authority':'User approved all four stages conditionally; no extra approval sought for routine implementation.'}
    save(state_path,state)
    p=R/'WordsInSilence_PROJECT_CONTEXT.md';s=p.read_text(encoding='utf-8')
    assert f'### 2026-10-27 {title}' not in s
    for marker,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',summary),('## 9. 다음 단일 작업',nxt)]:
        assert s.count(marker)==1
        s=s.replace(marker,marker+f'\n\n### 2026-10-27 {title}\n{stamp}\n{body}\n근거: {evidence}, RESULT.json, review/FINAL_REVIEW.json, HTML_RENDER_VERIFIED.json.\n',1)
    p.write_text(s,encoding='utf-8')
    handoff=f'''# WordsInSilence 최신 인수인계

{stamp}

## 현재 목표와 결과
{summary}

## 다음 한 작업
{nxt}

## 실행·검토
- 구현: Codex root(출처프로필/비교부품/보고서), ksl_source_audit(공식자료), local_feature_inventory(1199주석감사/공개측정).
- 별도 검토: linguistic_eval_review. source 상세 추가 중 최초스냅샷 불일치 보존 후 최종입력으로 검산 PASS. 전문가 수어정답 아님.
- 최초문제/복구: 사전상세295누락1개 추가확보(원래49획득 before_entry295보존); null위치명검사 약점은 원문대조로수정; 참고끼리중복hash거부추가. HTML선택항목 닫는태그 오류는 최초로그보존후수정, 재렌더PASS.
- 본 공개 비교는 저장결과36쌍을 재추론/재정렬하지 않았다. 기존B정렬과 원본관측 유지.
- 활성학습/서버/자동화신규 없음. 훈련0, C/T3F/운영웹/DTW교체0, 사용자/멘토접근0.

## 근거
- `{evidence}`
- `{KEY}/FEATURE_PROFILES.json`:100개출처프로필.
- `{KEY}/FEATURE_LABEL_PREPARATION.json`:32개후보의 주석준비 목록. 정답을 붙인 자료가 아님.
- `{KEY}/TRAINING_READINESS.json`:승인됨, 실제정답전제미충족.
- `{KEY}/public_comparison/RESULT.json`, `review/FINAL_REVIEW.json`:측정과 별도검산.

과거 학습승인대기라는 문구는 이번사용자 자율승인을 취소하지 않는다. 단, 정답부재·운영교체 범위는 별도다. 현재 마무리는 자료/공통구조/기술검산 완료이며, 단어별 중요도를 학습한 피드백 모델 완성이 아니다.
'''
    (R/'NEW_CHAT_HANDOFF.md').write_text(handoff,encoding='utf-8')
    save(U/'SYNC_CHECK.json',{'status':'PASS','at':stamp,'root_state':KEY,'protection_pins':len(pins),'all_pins_unchanged':True,
         'source_summary':final['source_summary'],'active_execution':None,'training_runs':0,
         'documents':{str(p.relative_to(R)):sha(p) for p in [state_path,R/'NEW_CHAT_HANDOFF.md',R/'WordsInSilence_PROJECT_CONTEXT.md']},
         'limits':'Semantic validation and model training are not complete; readiness decision and technical checks are complete.'})
    print('SYNC_PASS',KEY)

if __name__=='__main__':main()
