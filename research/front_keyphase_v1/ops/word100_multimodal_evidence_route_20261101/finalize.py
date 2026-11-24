from pathlib import Path
from datetime import datetime
import json,hashlib,shutil
U=Path(__file__).resolve().parent; OPS=U.parent; ROOT=OPS.parents[2]
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert read(U/'VERIFY.json')['status']=='PASS'
assert read(U/'MISSINGNESS.json')['unique_frames']==2084
assert not (U/'COMPLETION.json').exists()
at=datetime.now().astimezone().isoformat()
before=U/'before'; before.mkdir(exist_ok=False)
paths=[ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',OPS/'STATE.json',ROOT/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt']
for p in paths: shutil.copyfile(p,before/p.name)
current=('### 2026-11-01 관측 누락 범위·이미지/관절 추정 출력 분리 완료\n'
'`word100_multimodal_evidence_route_20261101`에서 공개16영상2084frame 저장기록감사. 사진은P065r48(1.6초),전체120frame중L미관측13·관측107;사진구간45~49의5frame약0.17초. 기존주석1.28~2.825초내46frame중L7/R5미관측. 원래기록이며재검출과합산X,미관측≠오류율. 현연구판정부는SignRep RGB16frame특징+69pose/69mask/시간정보를이미사용;관절proxy gate와모델추정분리출력추가. 저장30비교×3seed원26지원/4unknown불변,보류4에도모델추정존재를노출하되정오/정렬/회차/재촬영확정X. 새추론/검출/학습/DTW0. root별도저장검산2461PASS(독립사람정답아님),한글타임라인png/txt,HTML0. 기존C/T3F·운영/개인자료보존. 이는정보분리개선이며피드백정확도향상아님.\n\n')
nexttext=('### 2026-11-01 다음 단일 작업 — 중요한 구간의 영상·시간 추정 근거 평가\n'
'완벽한좌표복구가목표아님. 현재분리출력/누락집계재실행X. 기존공개정답과저장예측으로관절부족/정상관측을나누고,지원단어의중요손모양·단계차이에대한RGB+관절시간모델의실제지원범위를확인한다. 기록ID별분리/진단라벨훈련제외/미관측분모유지. RGB존재/높은확률/앞뒤일치만으로정답승격X,좌표가없다는이유만으로모델추정도없다고처리X. 중요구간근거가남는지평가하고필요한부분만재확인하며재촬영은중요근거끝내부족할때만설계. 새검출기/비용식순회/운영교체자동확대X. 학습은지원오류와적합정답이있을때만.\n\n')
p=paths[0]; text=p.read_text(encoding='utf-8-sig')
for heading,addition in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',current),('## 9. 다음 단일 작업',nexttext)]:
    assert text.count(heading)==1
    text=text.replace(heading,heading+'\n\n'+addition,1)
p.write_text(text,encoding='utf8')
entry=dict(status='COMPLETE_EVIDENCE_CHANNEL_SEPARATION_NOT_ACCURACY_GAIN',at=at,unit=str(U),
    active_session=None,public_clips=16,native_frames=2084,example='P065 frame48 at1.60s',
    example_left_missing_full=13,example_total_frames=120,example_gap_frames=5,example_gap_seconds_approx=.17,
    saved_supported_per_seed=26,saved_unknown_per_seed=4,unknown_model_estimates_retained_per_seed=4,
    support_promotions=0,recognition_gain=None,educational_accuracy=None,new_training=0,new_inference=0,
    new_detector_calls=0,new_dtw_paths=0,verify_checks=2461,verifier='root separate checker; not independent human truth',
    production_changed=False,report=str(U/'RESULT_KO.txt'),figure=str(U/'P065_TIMELINE_KO.png'))
s=read(paths[2]); s['current_stage']='MULTIMODAL_EVIDENCE_ROUTE_COMPLETE_NOT_ACCURACY_GAIN'
s['next_single_action']='완벽좌표복구보다중요동작근거우선. 저장공개정답/예측의관절부족·정상관측을분리평가해모델추정의피드백사용범위검증. 완료집계재실행/보류승격/운영교체X.'
s[U.name]=entry
paths[2].write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
paths[1].write_text('# WordsInSilence 최신 인수인계\n\n'+at+'\n\n'+current+nexttext+
    '근거 RESULT_KO.txt/MISSINGNESS.json/EVIDENCE_CHANNELS.json/ARCHITECTURE.json/VERIFY.json. 실제실행종료. 새프로세스없음.\n'
    '이번 root구현/root별도저장검산,독립에이전트·사람판독을했다고주장X. 이전복원결과는 word100_feedback_causal_pilot_20261101에읽기전용보존.\n'
    '관측은원래전체화면/비교는고정상반신영역. 이미지특징은해당영역밖을가린전체화면224리사이즈,손확대crop으로착각X.\n'
    '운영웹과TCN C/T3F 그대로. 사용자·멘토촬영접근/학습/튜닝/선택0. 카메라/다운로드/전송/Git0.\n',encoding='utf8')
with paths[3].open('a',encoding='utf8') as f:
    f.write('\n\n2026-11-01 — 완벽한 관절 복원보다 중요한 동작의 근거를 우선\n'
      '사용자는 소수의 누락 프레임을 영상 전체 문제처럼 다루거나 좌표 완벽성을 추구하고 있는지 질문했다. 원래 공개16영상2084프레임을 확인했고,예시사진은120프레임중연속5프레임약0.17초의왼손누락에속했다. 전체왼손누락은13프레임이다.\n'
      '현재연구모델이이미RGB영상특징과관절·mask·시간흐름을사용한다는코드계보를확인했다. 관절지원부족때모델추정까지없는것처럼설명하지않도록두출력분리. 3seed각기존26지원/4보류유지,보류4의모델추정도표시하되정답승격X.\n'
      '완벽한21점은모든종류의피드백의필수조건이아니다. 위치·모양·움직임·순서에필요한근거를각각평가하고,재관측/이미지·시간추정/중요근거보류를구분한다. 짧은누락을무조건오류또는무조건무해하다고하지않는다.\n'
      'root별도검산2461PASS. 새학습/추론0,인식·교육정확도개선아님. 다음은기존공개라벨로관절부족에서의모델추정신뢰범위를분리평가하는것이며또다른검출기추가가아니다.\n'
      '근거: '+str(U/'RESULT_KO.txt')+'\n')
artifacts=list(U.glob('*.json'))+list(U.glob('*.py'))+[U/'RESULT_KO.txt',U/'P065_TIMELINE_KO.png']
completion=dict(**entry,artifacts={str(p):sha(p) for p in artifacts},document_pins={str(p):sha(p) for p in paths})
with (U/'COMPLETION.json').open('x',encoding='utf8') as f: json.dump(completion,f,ensure_ascii=False,indent=2)
assert read(paths[2])[U.name]['verify_checks']==2461
for p,h in read(U/'FROZEN.json')['pins'].items(): assert sha(p)==h,p
for p in artifacts: assert p.exists(),p
print('Complete: report, verified evidence channels, timeline and four project records synchronized.')
