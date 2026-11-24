"""Record exact sensor preservation; no rescoring or model operations."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json, hashlib, re, shutil

U=Path(__file__).resolve().parent; O=U.parent; ROOT=O.parents[2]
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f: json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def prepend(text,n,entry):
    m=re.search(r'(?m)^## '+str(n)+r'\.[^\n]*\n',text); assert m
    return text[:m.end()]+'\n'+entry+'\n\n'+text[m.end():].lstrip('\r\n')

def main():
    result=read(U/'RESULT.json'); verified=read(U/'SAVED_CAPTURE_VERIFIED.json')
    assert result['all_exact'] and len(result['rows'])==2
    assert verified['status']=='PASS_EXACT_SAVED_CAPTURE_ARRAYS' and verified['failed_count']==0
    assert sha(U/'RESULT.json')==read(U/'RUN_FINISHED.json')['result_sha256']
    assert verified['result_sha256']==sha(U/'RESULT.json')
    for p,h in read(U/'EXECUTION_FROZEN.json')['pins'].items(): assert sha(p)==h,p
    count=verified['check_count']; assert count==5307
    at=datetime.now(timezone(timedelta(hours=9))).isoformat()
    summary=(
        'AL은 같은 native RGB 프레임의 몸·손·얼굴 검출만 3개 전용worker로 병렬화했다. 각 VIDEO상태·조정ms·프레임barrier·원래좌우배정/저장 코드와 모델·옵션·범위를 보존했다. '
        '실행전 marker파일 충돌을 수정했고 fake45/정적95검토·55핀동결 이후 고정 공개2입력을 후보각1회만 처리했다. exec36177 terminal0. '
        'AK WORD1133 ID10 118프레임과 AA WORD0739 ID12 공개앉은시야파생123프레임, 총241프레임에서 raw/기존관측/최종관측/face/할당의 모든키·dtype·shape·값·NaN·원시간과 trace/sidecar 의미가 완전일치했다. '
        f'별도 저장검산{count} PASS/실패0; 검산은 모델/디코드0이며 같은Codex 검산이지 수어전문가 정답검증은 아니다. '
        '관측후보 실제7.5089초/4.2935초, 과거직렬11.3440초/8.1687초. 과거는 RGB 등 동시작업부하가 있어 차이를 병렬화만의 인과효과나 전체앱속도로 주장하지 않는다. '
        '첫입력 pose1.4966초/hand2.8659초/faceprocess0.8629초는 겹쳐 실행되므로 합산금지. 프레임처리wall3.2089초, preflight+probe 재읽기1.3105초가 남아 있다. '
        '얼굴삭제·프레임생략·수치허용오차변경 없이 두조건 관측보존을 확인한 실행최적화 후보이며 정확도향상은 아니다. '
        '기존학습/추론/평가 재실행0, 새학습/SignRep/피드백head/DTW0. C·T3F/8804/8808/8809/원본/사용자·멘토자료 보존, HTML추가없음. 전체goal active·미완료.')
    next_action=(
        'AL 두후보·저장검산 완료이며 완료센서출력/옛학습·평가를 재실행하지 않는다. 다음은 AK 시간별움직임+검증된 AL 검출스케줄을 새격리 연구데모에 연결하는 작업이다. '
        '기존 AA/AK/AL/X 코드는 읽기전용으로 보존하고 새adapter/worker/output경로를 준비한다. analyze전체 직렬lock, 요청별 원영상hash/target/nativeaxis/추론전용계약을 유지한다. '
        'source문구와 원래사용자시간을 중심으로 확인한 손모양·관계·국소움직임, 기존 전체방향/부분순서·관측공백을 각각 표시한다. 새local high로 옛 전체방향미확인·접촉·필수반복·문구전체완료·교육적오류를 덮어쓰지 않는다. '
        '저장공개결과 재생은 재추론 없이 표시검증하고, 원시/관측/추정 공백과 실제보존범위를 검산한다. 새로운 전체실행 비교가 필요하면 입력·목적·예산을 먼저고정하고 별도후보실행으로 기록한다. '
        '실제전체지연은 관측만 시간과 분리하며 과거부하차이로 속도향상을 주장하지 않는다. 정상/짧은변화/순서/누락의 기존저장검증을 표시근거로 연결하고 아직없는정답은 주장하지 않는다. '
        '사용자 촬영은 요청한 추론만, 사용자·멘토자료 접근·학습·튜닝·선택 금지. 원본/모델/기존웹/잠금평가 보존, 새다운로드/외부전송/Git/운영교체없음. 전체목표는 완료아님.')
    final=dict(status='EXACT_PARALLEL_SENSOR_CANDIDATE_VERIFIED',at=at,
        candidate_captures=2,native_frames=241,saved_checks=count,failed_checks=0,
        actual_times_seconds=[r['capture_wall_seconds'] for r in result['rows']],
        historical_times_seconds=[r['baseline_capture_seconds'] for r in result['rows']],
        exact_observation_preserved=True,causal_speedup_proven=False,whole_app_latency_measured=False,
        new_training=0,semantic_model_forwards=0,old_live_changed=False,whole_goal_complete=False,
        summary_ko=summary,next=next_action,
        evidence={str(U/n):sha(U/n) for n in ['CODE_REVIEW.json','SYNTHETIC_CHECKS.json','EXECUTION_FROZEN.json','RESULT.json','RUN_FINISHED.json','verify_saved.py','SAVED_CAPTURE_VERIFIED.json']})
    write(U/'FINAL.json',final)
    report=('검출하는 내용은 유지하면서 계산을 동시에 하도록 보완했습니다.\n\n'
        '이전: 한 프레임에서 몸을 찾고, 다음에 손을 찾고, 다음에 얼굴을 찾았습니다.\n'
        '현재 후보: 같은 프레임에서 세 계산을 동시에 합니다. 모두 끝난 뒤 다음 프레임으로 넘어갑니다. 손의 좌우 배정과 관절 저장 방식은 같습니다.\n\n'
        '공개 영상 2개, 총 241프레임에서 좌표·검출 누락·손 배정·시간값이 기존 저장 결과와 모두 같았습니다. 별도 저장검산 5,307개도 통과했습니다.\n'
        '관측 처리시간은 잘하다 공개 원본 7.51초, 순종 공개 파생영상 4.29초였습니다. 과거 측정은 각각 11.34초와 8.17초였지만, 과거에는 다른 분석도 동시에 실행됐으므로 차이를 전부 이번 보완 효과라고 말할 수 없습니다. 이번 시간은 전체 피드백 대기시간도 아닙니다.\n\n'
        '새롭게 확인한 점: 얼굴 분석을 없애지 않아도 계산을 겹칠 수 있습니다. 첫 영상에서 손 검출 2.87초와 몸 검출 1.50초, 얼굴 처리 0.86초가 겹쳐 실행됐습니다. 영상 확인을 위한 반복 읽기와 다른 준비 비용은 남았습니다.\n'
        '남은 작업: 이 후보와 시간별 움직임 판정부를 별도 데모에 연결하고, 설명 문구와 영상 시간대별로 무엇이 확인됐고 무엇이 불확실한지 쉽게 표시해야 합니다. 이후 전체 대기시간을 확인합니다.\n'
        '기존 C·T3-F와 웹 데모를 바꾸지 않았고 사용자·멘토 촬영을 학습·튜닝·선택에 사용하지 않았습니다. 이번 결과는 관측 보존 검증이며 수어 정확도 향상이나 전체 기능 완성은 아닙니다.\n')
    with (U/'RESULT_KO.txt').open('x',encoding='utf-8') as f: f.write(report)
    docs=[ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']
    before=U/'before'; before.mkdir(exist_ok=False)
    for p in docs: shutil.copy2(p,before/p.name)
    heading='2026-11-03 입력 검출 병렬화의 공개 관측 보존 검증 완료'
    text=docs[0].read_text(encoding='utf-8-sig'); assert heading not in text
    text=prepend(text,8,'### '+heading+'\n\n'+summary)
    text=prepend(text,9,'### '+heading+' 이후 다음 단일 작업\n\n'+next_action)
    docs[0].write_text(text,encoding='utf-8')
    docs[1].write_text('# WordsInSilence 최신 인수인계\n\n'+at+'\n\n'+summary+'\n\n다음: '+next_action+'\n\n근거: '+str(U)+
        '\n\n직전 goal turn: progress. 병렬화 구현·고정후보2실행·별도검산 완료. exec36177 terminal0, detector전용thread종료. 전체goal active.\n',encoding='utf-8')
    state=read(docs[2]); state['updated_date_kst']='2026-11-03'
    state['current_stage']='PARALLEL_SENSOR_EXACT_PRESERVATION_VERIFIED_ISOLATED_DEMO_NEXT'
    state['next_single_action']=next_action; state[U.name]=dict(final,directory=str(U))
    docs[2].write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
    assert summary in docs[1].read_text(encoding='utf-8') and read(docs[2])[U.name]['whole_goal_complete'] is False
    write(U/'DOCS_SYNCED.json',dict(at=at,files={str(p):sha(p) for p in docs},whole_goal_complete=False))
    print('AL exact preservation verified; context/state/handoff synchronized; goal remains active')

if __name__=='__main__': main()
