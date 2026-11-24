"""Publish verified facts and the next learning question, preserving history."""
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import shutil
import hashlib

U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2)


def main():
    assert not (U/'FINAL.json').exists()
    required=['REVERSE_RESULT.json','REVERSE_RUN_FINISHED.json','SAVED_REVERSE_VERIFIED.json',
        'COMPOSITION_RESULT.json','COMPOSITION_FINISHED.json','SAVED_COMPOSITION_VERIFIED.json',
        'COMPOSITION_RECOVERY.json','COMPOSER_REVIEW_v2.json','COVERAGE_REVIEW.json',
        'REQUIREMENT_REVIEW.json','NEXT_PAIR_LEARNING_REVIEW.json']
    for n in required:assert (U/n).is_file(),n
    reverse=read(U/'REVERSE_RESULT.json');combined=read(U/'COMPOSITION_RESULT.json')
    joined=read(U/'REQUIREMENT_REVIEW.json')['joined_pointwise_rows']
    assert len(joined)==67 and len({(r['clip_id'],r['native_frame']) for r in joined})==47
    assert sum(r['context']==1 and r['physical_y']==1 and r['physical_mask'] for r in joined)==52
    assert sum(r['context']==0 and r['physical_y']==0 and r['physical_mask'] for r in joined)==12
    assert sum(r['context']==1 and r['physical_y']==0 and r['physical_mask'] for r in joined)==0
    assert reverse['model_forwards']==4 and reverse['summary']['high_occurrence']==5
    assert len(combined['rows'])==16 and combined['model_forwards']==0
    for name, result_name in [('SAVED_REVERSE_VERIFIED.json','REVERSE_RESULT.json'),('SAVED_COMPOSITION_VERIFIED.json','COMPOSITION_RESULT.json')]:
        verified=read(U/name)
        assert verified['status'].startswith('PASS') and not verified['failed_checks'], name
        assert verified['result_sha256']==sha(U/result_name), name
    assert sha(U/'REVERSE_RESULT.json')==read(U/'REVERSE_RUN_FINISHED.json')['result_sha256']
    assert sha(U/'COMPOSITION_RESULT.json')==read(U/'COMPOSITION_FINISHED.json')['result_sha256']
    before=U/'before_docs';before.mkdir(exist_ok=False)
    paths=[ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']
    for p in paths:shutil.copy2(p,before/p.name)
    at=datetime.now(timezone(timedelta(hours=9))).isoformat()
    summary=('AN 공개 피드백 구별범위 검증 완료. 출처/맥락/물리정답67행·고유47시점 연결: 같은맥락 특징부합52, 다른정상단계 반대특징12, 불확실3, 같은맥락 실제반대특징0. '
        '따라서 준비단계의 열린손을 잘못된주먹 정답으로 학습시키지 않았고, 중요한잘못된손모양 민감도는 미검증이다. '
        '기존공개 ID12의 실제픽셀역순4영상471native행에 아직미실행 동결AH를각1회 적용했다. 기존R고정5범위 high5이나 발생gold0이며 전체정답/완료로 해석하지 않는다. '
        '기존AA 부분순서5관계는역순2/교차근거보류3,조상은distinct pair0이라분모제외. source·order·local이 서로다른질문임을 명확히했다. '
        'AA+AE+AJ/newreverse의16저장조건을 공통결합부로 연결하고 12속도/대기조건의기존요약을별도보존했다. 새점수/DTW/원기준재실행0,기존모든필드/누락/범위보존. '
        '초기low미확인문구과장·order index/nativeID혼동·계보검사부족을 실행전수정했다. 실제결합은7개저장후무관측high_confirmation키생략으로실패; 원코드/첫오류/첫7출력보존후 명시무관측schema호환 v2로나머지9개만완료했다. '
        '구현과별도저장검산은같은Codex팀이며한국수어전문가정답이아니다. 이번은미검증부분을좁힌실제추론/결합완료이지 정확도향상이나교차해결아님. '
        'AM8810데모/기존C·T3F/8804·8808·8809·원본보존. 사용자·멘토자료접근/학습/튜닝/선택0,새훈련0,새HTML0. 전체goal active·미완료.')
    next_action=('AN 완료역순4/16결합/AJ24/AH48 평가를다시실행하지않는다. AN/NEXT_PAIR_LEARNING_REVIEW.json을따라 '
        '기존Y 시간별사건/순서학습에 AH5방향 시간별움직임+가용성 정보를 추가하는 별도학습후보의 입력호환부터 준비한다. '
        'Y는이미BiGRU+pairloss를썼으므로새GRU만반복/물리관측삭제/기준완화로해결했다고하지않는다. '
        '공개FIT01~09의원본36+기존실제RGB역순36=72캐시의sourceSHA/nativeaxis/전처리를확인하고,AH완료48값은정확동일입력일때만재사용한다. '
        '확률/특징을단순역순복사해실제역순방향입력으로쓰지않는다. 호환검사와명확한target정의후동일72자료·동일초기치·동일학습예산의대조와AH추가조건을고정해 '
        '필요한지도학습/부분미세조정1회씩비교한다. DEV11~13은이미노출된개발비교로표시하고평가점수로seed/문턱/epoch/범위를선택하지않는다. '
        '실제로같이존재하는이동/후반자세를억지배타화하지않으며,어떤관측쌍의순서인지와짧은반대근거/공백을보존한다. '
        '잘못된손모양same-context반례0은해결된것이아니므로별도공개정답범위확보전교정정확도주장금지. '
        '현재새학습은아직0이고서버AM8810만기존대로실행중. 중복서버/기존완료추론금지. 사용자·멘토/잠금평가/원본/C·T3F보존,다운로드·외부전송·Git·운영교체없음.')
    final=dict(status='PUBLIC_FACTOR_COMPARISON_COMPLETE_NO_ACCURACY_GAIN_CLAIM',at=at,
        new_reverse_clips=4,new_reverse_native_frames=471,new_local_forwards=4,new_training=0,
        saved_combinations=16,existing_speed_variants=12,known_same_context_wrong_feature_cases=0,
        summary_ko=summary,next=next_action,whole_goal_complete=False,
        pins={str((U/n).resolve()):sha(U/n) for n in required})
    write(U/'FINAL.json',final)
    report=('이번에 확인한 핵심\n\n'
        '이전에는 움직임을 찾는 결과와 순서를 보는 결과가 떨어져 있었습니다. 이번에는 같은 공개 영상에서 두 결과와 설명에 연결된 특징을 함께 확인했습니다.\n\n'
        '- 순서를 뒤집은 4영상에서도 움직임 근거는 5구간 모두 나왔습니다. 움직임이 있다는 것만으로 잘 따라했다는 뜻은 아닙니다.\n'
        '- 기존 순서 판정부는 비교 가능한 앞뒤 관계5쌍 중2쌍에서 역순을 확인했고3쌍은 근거가 서로 섞여 보류했습니다. 위의 움직임5구간과는 다른 집계입니다. 조상은 반복을 판정할 별도 앞뒤 관계가 없어 이5쌍에 포함하지 않았습니다.\n'
        '- 정상·삭제·역순·가림16조건을 함께 연결했습니다. 보이지 않은 시간과 미확인은 계속 남겼고, 이미 끝낸 속도·대기변이12조건은 저장된 결과를 사용했습니다.\n'
        '- 중요한 정답자료 부족도 찾았습니다. 주요 순간의 올바른 특징52개와 정상적인 다른 단계의 반대 특징12개는 있지만, 같은 주요 순간에서 실제로 틀린 손모양을 한 반례는 이 감사범위에0개입니다. 준비 때 손을 편 것을 오류로 가르치면 안 됩니다.\n\n'
        '이번에는 새로운 정확도 향상을 확인한 것이 아니라, 무엇을 이미 구별하고 무엇을 아직 못 구별하는지 실제 공개 입력과 저장 결과로 분리했습니다. 새 손모양오류 모델을 곧바로 학습시키지 않았습니다.\n\n'
        '다음은 기존 순서 신경망에 새로 학습된 시간별 움직임 정보를 추가했을 때 겹친 동작 근거를 더 잘 구분하는지, 같은 학습조건의 대조실험으로 확인하는 것입니다. 먼저 입력의 영상·시간·가용성이 정확히 맞는지 검사합니다.\n\n'
        '처리시간: 이번 작은 판정부만의4회 합계는 약0.018초입니다. 전체 영상 분석시간이 아닙니다. 기존 데모의 전체 약11초 대기 문제는 남아 있습니다.\n'
        '새 HTML과 기존 데모 교체는 하지 않았습니다. C·T3-F와 원본을 보존했고 사용자·멘토 촬영은 사용하지 않았습니다.\n')
    (U/'RESULT_KO.txt').write_text(report,encoding='utf-8')
    state=read(O/'STATE.json');state['updated_date_kst']='2026-11-03'
    state['current_stage']='PUBLIC_FEEDBACK_FACTOR_COMPARISON_COMPLETE_MOTION_AWARE_ORDER_LEARNING_NEXT'
    state['next_single_action']=next_action
    state[U.name]=dict(final,directory=str(U))
    (O/'STATE.json').write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    cp=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';s=cp.read_text(encoding='utf-8-sig')
    for heading,title,text in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인','공개 동작·순서 결합 검증과 실제 반례 부족 확인',summary),('## 9. 다음 단일 작업','공개 동작·순서 결합 검증 이후 다음 단일 작업',next_action)]:
        assert heading in s;s=s.replace(heading,heading+'\n\n### 2026-11-03 '+title+'\n\n'+text,1)
    cp.write_text(s,encoding='utf-8')
    (ROOT/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+at+'\n\n'+summary+'\n\n다음: '+next_action+'\n\n근거: '+str(U)+'\n\n직전 goal turn: progress. 공개미계산4추론과16결합/별도검산완료. 전체goal active.\n',encoding='utf-8')
    assert read(O/'STATE.json')[U.name]['whole_goal_complete'] is False
    assert summary in cp.read_text(encoding='utf-8') and next_action in (ROOT/'NEW_CHAT_HANDOFF.md').read_text(encoding='utf-8')
    write(U/'DOCS_SYNCED.json',dict(at=at,paths={str(p):sha(p) for p in paths},goal='active',progress=True))
    print(final['status'])


if __name__=='__main__':main()
