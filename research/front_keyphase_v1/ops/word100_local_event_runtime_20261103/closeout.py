"""Record verified runtime evidence, without changing scored outputs."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,hashlib,re,shutil
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def prepend(text,n,entry):
    m=re.search(r'(?m)^## '+str(n)+r'\.[^\n]*\n',text);assert m
    return text[:m.end()]+'\n'+entry+'\n\n'+text[m.end():].lstrip('\r\n')

def main():
    result=read(U/'PUBLIC_RUN_RESULT.json');verified=read(U/'SAVED_RUNTIME_VERIFIED.json')
    assert verified['status']=='PASS_SAVED_ARITHMETIC_NOT_ACCURACY' and verified['failed_count']==0
    assert verified['check_count']==874
    assert sha(U/'PUBLIC_RUN_RESULT.json')==read(U/'PUBLIC_RUN_FINISHED.json')['result_sha256']
    feedback=read(result['result']);event=feedback['local_event_feedback']['observations'][0]
    assert len(result['actual_dtw_calls'])==1 and feedback['local_event_feedback']['new_model_forwards']==1
    at=datetime.now(timezone(timedelta(hours=9))).isoformat()
    summary=(
      'AK 별도 실행부 구현과 신규 공개 영상의 실제 처음부터 끝까지 분석을 완료했다. 기존 ZIP에서 WORD1133 등록 ID10 원본1개만 C: AK/public_input으로 복사했고 W:원본은 보존했다. '
      '이 입력은 통합 개발 노출이며 이후 독립 holdout 주장금지. 모델/자료 선택용으로 쓰지 않았고 직접 시점별 정답은 없다. '
      '새 full-frame 118 native frame/PTS를 관측하고 GPU.0 SignRep RGB52창을 실행했다. 실제 window features/prelinear를 추가forward 없이 저장해 native 값과 완전일치 검산했다. '
      '사용하지 않는 baseline engine2개 중복부팅을 새 wrapper에서 생략했다. 학습된 prototype DTW실측1회, 원영상별 참고비교0, 기존 부분순서 head유지, AH 시간별움직임1회. '
      '54 local/24 wrapper 합성검사·62 정적검토 이후 실행 terminal0, 별도874저장검산/466핀 보존 통과. '
      '같은 제안구간48~64에서 기존 전체방향 점수0.0844 미확인, 새 국소발생0.9947 high(최대 원48frame/1.6초). 서로 다른 학습질문이고 ID10 직접정답이 없으므로 정확도 향상으로 주장하지 않는다. '
      '기존 SOURCE→ORDER→LOCAL→LIVE 필드 및 원시간 보존. 기존 요약 IMPORTANT_MOTION_UNCONFIRMED는 남겨 두었으며 새로운 국소발생과 다른 관측임을 다음 사용자 표시에서 분리해야 한다. '
      '실제 분석12.824초, 최초import/준비포함27.197초; MediaPipe포함 관측11.344초, 병렬 RGB5.517초, 새head0.00280초. 11.344초에는 detector생성/기존capture·assignment·저장도 포함되어 손/팔/얼굴별 원인은 아직 분리하지 않았다. '
      '기존 C/T3F/8804/8808/8809와 사용자·멘토자료 보존, 새학습/외부전송/다운로드/기존웹교체0. same-Codex 검산이며 독립 KSL전문가 정답아님. 전체goal active·미완료.')
    next_action=(
      'AK 단회 신규공개통합실행·저장검산 완료, PUBLIC_RUN_FINISHED exit0. 완료 추론/학습/점수 재실행금지. '
      '다음 단일작업은 느린 입력관측의 비용을 나누어 확인하고 관측 내용을 유지하는 계산 병렬화 후보를 격리 검증하는 것이다. '
      '현재 Landmarker는 VIDEO모드 pose→hands 순차, Capture는 뒤에 face를 수행한다. 전체11.344초만으로 얼굴이 주원인이라 단정하거나 프레임/손가락/얼굴근거를 삭제하지 않는다. '
      '먼저 detector생성·decode·pose·hands·face·assignment·저장 시간을 구분할 계측과 동일frame 독립detector 병렬 후보를 준비한다. '
      '각 detector의 이전 VIDEO상태/순증가 timestamp·동일RGB·좌우배정·원시간·고정범위를 보존하며 데이터/프레임수/모델/문턱을 바꾸지 않는다. '
      '기존 AK raw/observations/face를 기준으로 후보만 새추론하고 전후 좌표·검출·마스크·할당·시간축을 비교하는 조건을 결과 전 고정한다. '
      '처리순서 병렬화가 비동등하면 미적용하고 다른 원인을 검토한다. 기존 학습head를 재점수화하거나 ID10 결과로 가중치·규칙·후보선택을 튜닝하지 않는다. '
      '병렬화를 일반 실행 최적화로 채택할 근거는 관측 동등성 및 계측이며 인식률 향상이 아니다. '
      '이후 별도 데모에 시간별 움직임과 기존 전체방향/순서의 상태를 구분해 문구·시간대 중심으로 표시한다. 웹동시요청 적용 전 analyze전체 직렬lock이 필수다. '
      'source schema/출처조건은 유지하고 높은 local score로 문구전체·접촉·반복완료를 확정하지 않는다. '
      '실제 사용자 입력은 추론만, 사용자·멘토/잠금평가자료 접근·학습·튜닝·선택 금지. 기존 C/T3F와웹/원본 보존. 전체goal active·미완료.')
    final=dict(status='NEW_PUBLIC_RUNTIME_VERIFIED_LATENCY_REMAINS',at=at,source_registered_id=10,
       source_sha256=result['source_sha256'],public_input_role='development_integration_not_accuracy',
       native_frames=118,rgb_windows=52,learned_prototype_DTW_calls=1,reference_video_comparisons=0,
       new_local_event_forwards=1,new_training=0,old_outputs_preserved=True,independent_saved_checks=874,
       server_analysis_seconds=result['server_analysis_seconds'],cold_job_seconds=result['cold_job_seconds'],
       old_direction_probability=feedback['motion_observations'][0]['probability'],
       new_local_occurrence_probability=event['probability'],accuracy_improvement_claimed=False,
       existing_live_changed=False,whole_goal_complete=False,next=next_action,
       pins={str(U/n):sha(U/n) for n in ['PUBLIC_INPUT_FROZEN.json','EXECUTION_FROZEN.json','CODE_REVIEW.json',
            'PUBLIC_RUN_RESULT.json','PUBLIC_RUN_FINISHED.json','SAVED_RUNTIME_VERIFIED.json']})
    write(U/'FINAL.json',final)
    report=('새 공개 영상의 실제 분석 연결 결과\n\n'
      '한 영상에서 관절·RGB 관측부터 학습된 기준 흐름과의 시간 정렬, 손모양·관계·순서·새 움직임 근거까지 연결했습니다.\n'
      '여러 참고 영상을 각각 비교하지 않았습니다. 학습된 기준 흐름에 DTW 한 번, 새 움직임 판정부도 한 번 실행했습니다.\n'
      '이번 새 입력에서는 기존 전체방향 판정부가 확신하지 못한 범위에서도, 새 시간별 판정부가 팔을 따라 이동하는 흐름의 근거를 반환했습니다. 다만 직접 정답을 새로 확인한 시험이 아니므로 정답률 향상을 증명한 것은 아닙니다.\n'
      '원래 118프레임의 시간과 기존 결과를 유지했습니다. 실제 RGB 창 52개의 값도 저장하여 연결이 정확한지 별도로 검산했습니다.\n\n'
      '남은 문제는 속도와 안내입니다. 전체 분석은 12.82초로 길었습니다. 새 판정부는 약0.003초였고 관절 관측 등 입력 준비가 11.34초였습니다. RGB 분석5.52초는 이와 동시에 실행되므로 두 시간을 더하면 안 됩니다. 최초 모델 준비까지 포함한27.20초는 매 촬영의 분석시간과 구분합니다.\n'
      '기존의 전체방향 미확인과 새 국소움직임 확인은 질문이 다릅니다. 다음 화면에서는 이 둘을 뒤섞어 틀렸다고 말하지 않고, 확인한 특징과 아직 모르는 부분을 나누어 표시해야 합니다.\n\n'
      'C·T3-F와 기존 웹은 그대로입니다. 사용자·멘토 촬영을 학습이나 튜닝에 쓰지 않았습니다. 새 후보의 실제 분석 연결을 마쳤으며 전체 피드백 기능은 아직 완성되지 않았습니다.\n\n다음: '+next_action+'\n')
    with (U/'RESULT_KO.txt').open('x',encoding='utf-8') as f:f.write(report)
    before=U/'before';before.mkdir(exist_ok=True)
    docs=[ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']
    for p in docs:
        assert not (before/p.name).exists();shutil.copy2(p,before/p.name)
    heading='2026-11-03 새 공개 입력의 학습형 피드백 실행부 통합 검증 완료'
    text=docs[0].read_text(encoding='utf-8-sig');assert heading not in text
    text=prepend(text,8,'### '+heading+'\n\n'+summary)
    text=prepend(text,9,'### '+heading+' 이후 다음 단일 작업\n\n'+next_action)
    docs[0].write_text(text,encoding='utf-8')
    docs[1].write_text('# WordsInSilence 최신 인수인계\n\n'+at+'\n\n'+summary+'\n\n다음: '+next_action+
        '\n\n근거: '+str(U)+'\n\n직전 goal turn progress: 새 실행부 구현·공개 원본1개 신규분석·별도검산을 완료함. 현재 프로세스 종료0, 별도worker도 stdin닫힘 후 wait종료. 재실행하지 않는다. 전체goal active.\n',encoding='utf-8')
    state=read(docs[2]);state['updated_date_kst']='2026-11-03'
    state['current_stage']='LOCAL_EVENT_NEW_INPUT_RUNTIME_VERIFIED_CAPTURE_LATENCY_NEXT'
    state['next_single_action']=next_action;state[U.name]=dict(final,directory=str(U),summary_ko=summary)
    docs[2].write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
    assert read(docs[2])[U.name]['whole_goal_complete'] is False
    assert summary in docs[1].read_text(encoding='utf-8')
    write(U/'DOCS_SYNCED.json',dict(at=at,files={str(p):sha(p) for p in docs},whole_goal_complete=False))
    print('AK runtime verified; current context/state/handoff synchronized; overall goal remains active')
if __name__=='__main__':main()
