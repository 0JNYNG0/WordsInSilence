"""Root-owned evidence closeout and synchronized project handoff."""
import json,re,shutil
from pathlib import Path
from datetime import datetime,timezone,timedelta
from train_local import U,O,read,write,sha
ROOT=O.parents[2]

def latest(text,section,entry):
    pattern=r'(?m)^## '+str(section)+r'\.[^\n]*\n'
    m=re.search(pattern,text);assert m
    return text[:m.end()]+'\n'+entry+'\n\n'+text[m.end():].lstrip('\r\n')

def main():
    r=read(U/'RESULT.json');v=read(U/'SAVED_VERIFIED.json');tr=read(U/'TRAIN_RESULT.json')
    assert r['gate_pass'] and not r['whole_goal_complete']
    # Reviewer schema is read as evidence, not guessed from the filename.
    assert v.get('status','').startswith('PASS')
    at=datetime.now(timezone(timedelta(hours=9))).isoformat()
    complete=('공개 4표현의 등록 ID01~09 36영상으로 시간별 움직임 판정부를 1회 지도학습했다. '
      '39,013개 파라미터, seed42, 80epoch·320step이며 기존 관절/RGB 투영 가중치 복사본을 미세조정했다. SignRep 본체·C·T3-F·기존웹은 보존했다. '
      '기존 양성 구간을 모든 프레임의 정답으로 복사하지 않고, 그 안에 움직임이 있다는 구간 감독으로 사용했다. 새 연속 원RGB 검토는45창, AG3창 재사용이며 FIT 국소43음성/1양성/1불확실, DEV13음성/2양성이다. '
      '원본 시간·입력 준비1689검사, 별도 자료1707검사, 코드77검사·합성18검사 후 실행했다. 학습 진입 첫 항목명 KeyError는 모델 로드·optimizer 이전 실패이며 원오류와 initial코드를 보존하고 REVIEW_CODE_v2로 읽기호환만 수정했다. '
      '48영상의 full-native 평가 1회 및 별도 저장4535검산 완료. 등록 ID11~13의 DEV12영상에서 기존 이동15/15와 새 국소 이동2/2를 .8 이상 확인, 알려진 국소 비발생13개 오확인0. '
      '전체 최고점이 검토 양성 범위 안인 것은9/15. 밖인6개 중5개는1~2프레임 차이이며 정확한 시작끝 주석이 없어 이를 오류6개로 단정하지 않는다. '
      '새 작은 판정부 캐시 계산 중앙값 약2.24ms는 전체 앱 대기시간이 아니다. 고정 연구후보 기준 통과지만 새 사용자·속도변이·동작 전체누락·교육적 정오 및 웹개선은 미검증이다. '
      '같은 Codex의 물리 관측/별도 검산이며 독립 수어 전문가 정답은 아니다. 사용자·멘토·잠금평가 자료 미사용. 전체 학습형 피드백 목표는 active·미완료.')
    nxt=('AH 학습·native 평가·저장검산은 완료되어 재학습/재추론하지 않는다. 새 시간별 점수를 기존 학습 prototype의 1회 DTW와 이미 고정된 motion proposal에 연결하는 격리 adapter를 준비한다. '
         '공개 DEV12의 저장 path·원시간·AH predictions를 재사용해 source guide 문구 및 관측 상태가 어떻게 달라지는지 확인한다. '
         '움직임 점수로 비교 범위를 고르거나 단어별 예외/문턱 조정은 금지. 기존 AE의 검증된 source-component 계약을 따르고, '
         '얼굴→몸통 이동을 옆으로 든 팔을 내림 전체와 같다고 바꾸거나 상승/하강 하나를 번갈아 수행 완료로 승격하지 않는다. '
         '높은 관측 근거와 전체 동작 충분성, 낮은 점수와 관측 부족은 분리한다. 기존 C/T3F/8804/8808/8809·원본 보존. '
         '현재 모델 프로세스는 TRAIN/EVAL 종료코드0으로 완료; 다음은 학습이 아닌 저장 결과 연결·검증이다.')
    final=dict(status='TRAINED_PUBLIC_OCCURRENCE_CANDIDATE_VERIFIED',at=at,
        public_fit_clips=36,public_development_clips=12,registered_fit_ids=list(range(1,10)),registered_dev_ids=[11,12,13],
        training_runs=1,training_steps=320,model_file=tr['model'],model_sha256=tr['model_sha256'],
        native_evaluation_forwards=48,saved_verification_checks=4535,
        dev_native_positive_high=[15,15],dev_additional_local_positive_high=[2,2],dev_local_negative_false_high=[0,13],
        broad_argmax_in_positive=[9,15],candidate_gate_pass=True,live_changed=False,
        next=nxt,whole_goal_complete=False,
        pins={str(U/n):sha(U/n) for n in ['DATASET_FROZEN.json','REVIEW_CODE_v2.json','DATA_REVIEW.json','TRAIN_RESULT.json','EVAL_RESULT.json','RESULT.json','SAVED_VERIFIED.json','TIMING_INTERPRETATION.json']})
    write(U/'FINAL.json',final)
    oldstatus=read(U/'STATUS.json');write(U/'STATUS_initial.json',oldstatus)
    status=dict(final,actual_training_steps=320,checkpoint_loads=2,public_video_model_inferences=48,
        checkpoint_load_scope='one frozen projection initialization plus one new checkpoint evaluation',
        previous_goal_turn='progress: prepared direct controls, trained one candidate, evaluated once, verified saved arithmetic')
    (U/'STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    before=U/'before';before.mkdir(exist_ok=True)
    docs=[ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']
    for p in docs:
        dest=before/p.name
        if not dest.exists():shutil.copy2(p,dest)
    context=docs[0].read_text(encoding='utf-8-sig')
    heading='2026-11-03 시간별 움직임 판정부 지도학습·공개 비교 완료'
    assert heading not in context
    context=latest(context,8,'### '+heading+'\n\n'+complete)
    context=latest(context,9,'### '+heading+' 이후 다음 단일 작업\n\n'+nxt)
    docs[0].write_text(context,encoding='utf-8')
    docs[1].write_text('# WordsInSilence 최신 인수인계\n\n'+at+'\n\n'+complete+'\n\n다음: '+nxt+'\n\n근거: '+str(U)+'\n\n전체 goal active·미완료. 완료 모델 실행을 반복하지 말고 저장 결과와 현재 프로세스를 확인한다.\n',encoding='utf-8')
    state=read(docs[2]);state['updated_date_kst']='2026-11-03'
    state['current_stage']='LOCAL_EVENT_HEAD_TRAINED_VERIFIED_SOURCE_GUIDE_ADAPTER_NEXT'
    state['next_single_action']=nxt
    state['word100_local_event_learning_20261103']=dict(status,directory=str(U),summary_ko=complete)
    docs[2].write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
    assert read(docs[2])['word100_local_event_learning_20261103']['whole_goal_complete'] is False
    assert heading in docs[0].read_text(encoding='utf-8') and complete in docs[1].read_text(encoding='utf-8')
    write(U/'DOCS_SYNCED.json',dict(at=at,files={str(p):sha(p) for p in docs},whole_goal_complete=False))
    print('Verified AH closeout; project context/state/handoff synchronized; goal remains active')
if __name__=='__main__':main()
