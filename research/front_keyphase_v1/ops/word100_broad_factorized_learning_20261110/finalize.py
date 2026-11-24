from core import *
import datetime
def main():
 assert read(U/'VERIFICATION.json')['status']=='PASS' and read(U/'CONFLICT_GUARD_CHECK.json')['status']=='PASS'
 e=read(U/'EVALUATION_COMPLETE.json');s=read(U/'SUMMARY.json');b=read(B/'EVALUATION_COMPLETE.json');ready=read(U/'READY.json');cfg=read(U/'FROZEN.json')
 errors=[r for r in read(U/'scores/factorized_42.json') if r['task']=='movement' and r['variant']=='normal' and r['pred']!=r['target']]
 key='NIA_SL_WORD1133_REAL11_F';normal=read(U/'edited_channels/coupled_42'/f'{key}_main_reversed/after.json')['cards'][0]['motion_evidence'];both=read(U/'edited_channels/coupled_42'/f'{key}_closed_hand_and_reversed/after.json')['cards'][0]['motion_evidence']
 cause=dict(observed=['same frozen stage network chooses different spans when observation handshape changes','separating finger/latent inputs makes motion invariant within the SAME supplied interval, but does not make the upstream interval invariant','body-wrist-only motion mislabels3 reviewed hold intervals; all seeds repeat these3 errors','same learned order head accepts3 reversed pairs at primary42, even with supplied reviewed endpoints'],example=dict(clip=key,direction_only_span=normal['span'],combined_span=both['span'],direction_only_state=normal['state'],combined_state=both['state'],combined_geometry_state=both['old_geometry_state'],combined_wrist_trajectory=both['learned_evidence']['trajectory']),inference='Automatic window includes a different boundary trajectory. Training on clean supplied intervals does not by itself establish automatic-window accuracy. These observations do not prove a single exclusive neural cause.',factorized_hold_errors=errors,posthoc_safety='Strong geometry/neural disagreement withheld; not adjudicated as ground truth.',no_extra_development_tuning=True)
 write(U/'CAUSE_AUDIT.json',cause)
 nextplan=dict(objective='Retain100-word common learning and improved shape checkpoint; teach evidence for attempted event windows and motion/order separately without deleting mistakes.',steps=[dict(step=1,task='FIT-only audit automatic proposed intervals versus reviewed core/order anchors; distinguish boundary contamination, actual repeated/returning motion, and missing/wrong wrist observations',evidence='Current combined control span54..76→47..78 while same frozen detector/phase branch used; no word-specific exception'),dict(step=2,task='Build shared FIT contrasts: same movement with different handshape, reversed stroke, multiple strokes with return, small supporting-hand sway, sensor loss; give attempted-event extent and per-stroke direction/order labels separately',evidence='31new motion intervals alone did not cover automatic boundary mixtures; real learners were not evaluated'),dict(step=3,task='Maintain899FIT100-word auxiliary; preserve shape weights and stable output path; compare motion-only vs limited-context/order heads with automatic-window supervision, fixed conditions and all seeds before evaluation',evidence='Shape227/228 and normal9/12 preserved; isolated wrist-only classifier29/32 vs coupled32/32, so full separation is not automatically best'),dict(step=4,task='Evaluate public normal/speed/crossing/short-event/shape-only/direction-only/combined/omission/unknown; keep every case in denominator and only consider an explicit research-web option if component-specific errors and normal reliability improve',evidence='Some-error alert9/9 hid wrong direction in1combined case. No production replacement before review.')],constraints=['No private/user/mentor data','C/T3F/current RTMW TCN/default web unchanged','No selecting best seed or thresholds from historical DEV','No rerunning completed6trainings/414comparisons','New performance must distinguish historic regression from fresh independent test'])
 write(U/'NEXT_PLAN.json',nextplan)
 table=[]
 for name,v in e['models'].items():
  n=v['by_variant']['normal'];rev=v['by_variant']['reversed_main'];table.append(f"{name}: 손모양 {n['reviewed_shape']['correct']}/228, 추가8표현 {n['expanded_shape']['correct']}/133, 방향/유지 {n['movement']['correct']}/32, 정상순서 {n['order']['correct']}/12, 뒤집힌순서 {rev['order']['correct']}/12,100단어보조 {v['aux_word100']['correct']}/300")
 seconds=sum(read(p)['seconds'] for p in (U/'runs').glob('*/COMPLETE.json'))
 report=f'''100단어 공통 학습 유지·세부 정답 보강 결과 — 2026-11-10

요청한 공개 정답 보강→학습→기존 비교→검산을 모두 마쳤다.
손모양 관측은 좋아졌고, 이번에는 기존 동작 구간 찾기를 흔들어 정상 결과가 나빠지던 문제도 피했다.
그러나 새 방향·순서 판정부까지 완성된 것은 아니다. 기본 웹은 교체하지 않았다.

무엇을 얼마나 배웠나
100단어·공개899영상(기록ID01~09)을 모든 학습 회차에 빠짐없이 사용했다. 사람ID11~13의300영상은 학습에 넣지 않았다.
이300영상은 과거에도 비교했던 개발 자료다. 완전히 새로운 사람에 대한 최종 시험으로 부르지 않는다.
현재100단어 공통학습은 단어 구분을 돕는 보조 학습이다. 100단어 모두에 완성된 세부 피드백 정답을 붙였다는 뜻은 아니다.
세부 모양 정답은12표현60FIT영상, 움직임 정답은10표현54FIT영상에 연결했다. 실제 피드백 확인은4표현을 우선 대상으로 했다.
공개6표현18영상의90RGB시점을 확인해 움직임31구간과 모양76개 확인항목을 추가했다. 예: 펴진 손이 움직임/일부 손가락만 편 손이 움직임/다른 손이 움직일 동안 받치는 손목은 유지.
기존에 RGB로 검토했던 사건들을 연결해 순서63쌍을 추가했다. 기존12쌍과 합쳐75쌍이다. 75개 단어나75개 독립 영상이라는 뜻은 아니다.
모양·방향·순서의 새 정답과 과거 DTW로 옮긴 약한 구간 주석은 구분했다. 모르는 엄지/손끝 접촉/3D 회전은 정답으로 만들어 채우지 않았다.
이번 시각 확인과 기존 주석은 같은Codex의 검토다. 독립적인 농인 강사/전문가 정답으로 주장하지 않는다.

어떻게 학습하고 비교했나
기존100단어 공동학습 가중치를 이어서 시간 합성곱+GRU와 모양 출력을 미세조정했다. 새 방향/순서 출력도 함께 지도학습했다.
A 결합: 팔·손목의 상대 움직임과 공유 신경망 특징을 함께 본다.
B 분리: 이동 방향에는 팔·손목 흐름만 넣어, 손가락 모양이나 공유 특징이 방향 판단을 직접 바꾸지 못하게 한다.
양쪽 모두 같은 자료·학습회수·3개 고정 초기값을 사용했다. 각12회, 총6학습 완료; 학습 계산 합계{seconds/60:.1f}분. 결과가 가장 좋은 초기값을 골라 보고하지 않았다.
역순 대조는 순서/방향을 구별하기 위한 오답 대조다. 뒤집힌 동작을 올바른 수어라고 가르치지 않았다.
현재 웹의 구간 찾기·완료 확인·TCN·DTW는 그대로 두고 새 모양/방향 출력만 격리 연결했다. 순서 출력은 아직 진단용이다.
목표 단어는 기대되는 단계/설명 구조를 고를 때 사용한다. 관측된 손모양과 방향을 정답 단어에 맞춰 강제로 만들지 않는다.

이전→현재: 같은 공개 확인 항목, 대표 초기값42
손모양: 현재웹221/228 → 이전100단어후보226/228 → 이번A227/228, B227/228.
따라서 웹과 비교하면6항목 개선, 이어 학습한100단어후보와 비교하면1항목 개선이다. 228은4표현12영상 속 손모양 확인항목이며 단어 인식률/게임 사용자 전체 정답률이 아니다.
추가8표현 손모양133항목: 이전100단어후보122 → A123, B122. 큰 범위에서 일괄 개선됐다고 말하지 않는다. 확인할 수 없었던83항목은 기존처럼 미확인으로 남는다(전체후보216항목).
방향/유지32구간: 현재기하계산26, 이전별도움직임후보31 → 이번A32, B29. 분리한다고 무조건 더 정확하지 않았다.
신경망 단계 출력: 이전100단어후보71/84 → A/B72/84. 실제 연결에서는 이미 검증한 기존 단계 경로를 사용한다.
정상순서12쌍: 새A/B12/12. 주요구간을 뒤집은 순서12쌍: 새A/B9/12. 아직3쌍을 놓친다.
같은 검토 끝점에서 과거 모델의 저장 단계확률을 읽은 보조 산술비교는 정상12/12, 역순8/12다. 이는 평가 후 추가한 끝점 진단이며, 과거의 자동 전체구간 순서 점수10/12와 다른 지표다. 새 독립시험/확정적 순서성능향상으로 주장하지 않는다.
100단어 보조 구분: 이전244/300 → A258/300, B257/300. 피드백 신경망의 보조 출력이고 기존 TCN 성능 개선 수치가 아니다. 나머지 초기값에서는 모두 좋아진 것이 아니다.

모든 고정 초기값 결과
{chr(10).join(table)}

실제로 구간을 자동으로 찾게 했을 때
69공개 조건×6후보=414비교를 완료했다.69조건은 정상12/느림12/빠름12/손관절가림12/마지막동작삭제6/마지막관절가림6/모양만변경3/방향만변경3/둘다변경3이다.
같은 공개 원본에 만든 변형을 포함하므로69명의 실제 오답 촬영으로 해석하면 안 된다. 앞60조건은 원시RTMW값부터 최종관측검사까지, 뒤9조건은 저장 편집 관절값부터 검사했다.
정상 전체필수항목 비슷함9/12, 느림8/12, 빠름9/12가 모두 유지됐다. 정상/속도/교차에 새 차이 경고0. 아직 정상3개는 미확인이다.
전체손관절가림12·마지막동작삭제6·마지막관절가림6은 잘했다고 잘못 처리한 경우0을 유지했다. 자료나 실패/미확인을 분모에서 빼지 않았다.
모양만/방향만/둘다변경 각3개에서 차이 표시는 유지됐다. 하지만 모양+방향 동시변경1개에서는 방향 자체를 잘못 지지했다. 모양 경고가 남아 있다고 방향까지 맞았다고 볼 수 없다.
모양만 바꾼 경우 방향을 정상으로 지지한 것은2/3, 나머지1개는 관측 근거 부족으로 보류했다. 기존 기하 방식은3/3이었다.
기존TCN 후보, 입력프레임, DTW경로는 모든원시재생에서 동일했다. 실제 사용자 촬영/새 웹캠/영상 전송/UI는 이번에 시험하지 않았다.

남은 문제를 어디까지 구분했나
방향만 뒤집은 한 조건에서는 비교구간이54~76프레임, 모양도 바꾸면47~78프레임으로 넓어졌다.
넓어진 구간은 앞쪽의 급격한 위치 변화와 뒤쪽의 실제 이동이 섞였다. 새 모델은 깨끗하게 지정해 준 구간에서는 맞혔지만 이 자동 구간에서는 잘못 요약했다.
손가락과 방향 입력을 분리해도, 앞단에서 선택하는 구간이 손모양의 영향을 받는 문제까지 자동으로 없어지지는 않는다.
분리모델이 틀린 일반32구간 중3개는 모두 유지 구간이었다. 작은 위치 변화를 동작으로 취급하지 않는 근거도 더 필요하다.
순서 판정부 역시 정상은 맞지만 뒤집힌 순서 일부는 정상으로 판단한다. 반복 횟수·정확한 접촉·모든 손모양에 대한 교육적 허용 범위는 아직 완성하지 않았다.

바로 보완한 안전 처리
기존 이동 계산은 반대 방향이라고 하는데 새 신경망은 정상 방향이라고 하는 등 근거가 충돌하면, 어느 쪽을 정답으로 강제하지 않고 방향은 미확인으로 남기는 공통 처리를 추가했다.
저장된414조건에서6조건(동일원본1조건×6후보),12카드의 방향 표현만 수정됐다. 정상/속도/가림/누락 결과 변경0.
이는 평가 후 만든 격리 안전보완이며 새 방향 정확도 향상으로 세지 않았다. 모델/정답/점수는 바꾸지 않았고 기본웹에도 넣지 않았다.

속도와 검증
저장된RTMW관절값부터 TCN·추가신경망·DTW·근거검사까지 대표A 중앙값{s['models']['coupled_42']['cached_fullpath_ms_median']:.1f}ms,95백분위{s['models']['coupled_42']['cached_fullpath_ms_p95']:.1f}ms. B 중앙값{s['models']['factorized_42']['cached_fullpath_ms_median']:.1f}ms.
관절검출/촬영/파일읽기/화면표시가 제외된 계산 시간이다. 실제 게임 전체가0.05초에 끝난다는 뜻은 아니다.
학습전 계약264개, 저장출력·분리입력·좌표정규화 검산4413개, 충돌보류후속2913개 통과. 같은Codex의 별도검산이며 독립사람검증 아님.
검산코드 검토 중 기존손모양194항목과추가검토손모양228항목의 예상분모를 혼동한 문장을 수정했다. 첫검산 실행 전에 발견했고 이전코드는 보존했다. 학습·추론·정답·결과 수정0.
원래C/T3-F/현재RTMWTCN/운영웹 파일해시 유지. 사용자·멘토자료 접근/학습/튜닝/후보선택0, 다운로드0, 카메라0, 외부전송0.

다음 작업
개선된 모양 가중치와100단어 공통학습은 보존한다. 공개FIT의 자동 비교구간에 준비/복귀가 섞이는 사례, 같은 움직임의 다른 손모양, 작은 지지손 흔들림을 묶어서 구간 정답과 실제 방향 정답을 별도로 보강한다.
핵심은 학습용으로 정확히 잘라준 구간만 가르치는 데서, 실제 프로그램이 찾은 구간에서도 중요한 이동/순서를 가려내도록 옮기는 것이다. 특정단어예외나DEV점수에맞춘문턱조정은 하지 않는다.
전체 기능의 안정화가 확인되지 않아 새HTML은 생략했다. 준비/학습/비교는 완료됐지만 완성된 피드백AI나 운영교체 완료라는 뜻은 아니다.
'''
 (U/'RESULT_KO.txt').write_text(report,'utf-8')
 overview='100단어899FIT 매회유지·300역사DEV분리, 새RGB움직임31/모양76항목·기존검토순서63쌍 보강. 결합/분리2조건×3seed 학습과414자동경로비교 완료. 대표42 손모양웹221→227/228(이어학습전100후보226), 방향/유지결합32/32·분리29/32, 정상순서12/12·역순9/12. 정상전체9/12·속도/교차·가림/삭제거짓완료0유지,TCN/DTW동일. 모양+역방향1조건에서방향잘못지지잔존: 자동창54~76→47~78확장과경계흐름혼합확인. 기존기하/새신경망충돌은미확인으로남기는후속검증(6후보동일1조건)완료. 264사전/4413산술/2913후속검사PASS,같은Codex. 새모델기본웹미교체·개인자료0·HTML생략.'
 nexttext='완료6학습/414비교재실행금지. 개선된모양/100공통가중치보존. 공개FIT에서자동비교창의준비·복귀혼합/지지손흔들림/모양+방향조합을검토해,시도한사건의범위와사건별방향·순서정답을분리보강한다. 자동창감독조건을FIT만으로고정하고899FIT100공통학습유지,현재단계/완료/TCN/DTW안전경로보존. 역사DEV튜닝/seed선택/단어예외금지. 충돌보류는정확도향상아님. C/T3F·웹8814유지,개인자료금지. NEXT_PLAN.json 참조.'
 write(U/'RESULT.json',dict(status='COMPLETE_SHAPE_GAIN_AUTOMATIC_NORMALS_PRESERVED_MOTION_ORDER_LIMITS',summary=overview,training_runs=6,training_seconds=seconds,automatic_comparisons=414,report=str(U/'RESULT_KO.txt'),default_web_changed=False,private_used=False,new_html=False,next=str(U/'NEXT_PLAN.json')))
 now=datetime.datetime.now().astimezone().isoformat();state=read(O/'STATE.json');state['current_stage']=U.name;state['next_single_action']=nexttext;state['last_updated']=now;state[U.name]=dict(status='COMPLETE_PARTIAL_SHAPE_GAIN_NOT_DEPLOYED',summary=overview,report=str(U/'RESULT_KO.txt'),result=str(U/'RESULT.json'),verification=str(U/'VERIFICATION.json'),followup=str(U/'CONFLICT_GUARD_CHECK.json'),execution=str(U/'CURRENT_EXECUTION.json'),next=nexttext,default_web_changed=False,private_used=False,active=None);write(O/'STATE.json',state)
 p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';text=p.read_text('utf-8-sig');oldtitle='### 2026-11-10 100단어 유지·모양/움직임 분리 공동학습 (진행 중)';newtitle='### 2026-11-10 100단어 유지·세부 정답 보강과 분리 관측 학습 완료'
 for section,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',overview),('## 9. 다음 단일 작업',nexttext)]:
  pos=text.index(oldtitle,text.index(section));end=text.index('\n### ',pos+len(oldtitle));text=text[:pos]+newtitle+'\n\n'+body+' 근거 '+str(U/'RESULT_KO.txt')+'\n\n'+text[end:]
 p.write_text(text,'utf-8')
 (ROOT/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n최신 승인:100단어 공통학습 유지·부족한조합/세부정답보강·4표현우선피드백검증·학습/기존비교.\n\n마지막 완료\n'+overview+'\n\n다음 한 작업\n'+nexttext+'\n\n모든 학습·평가 프로세스 종료. 현재 웹/C·T3F/RTMWTCN 변경 없음. 새 HTML 없음.\n결과 '+str(U/'RESULT_KO.txt')+'\n검산 '+str(U/'VERIFICATION.json')+'\n원인 '+str(U/'CAUSE_AUDIT.json')+'\n계획 '+str(U/'NEXT_PLAN.json')+'\n','utf-8')
 hist=ROOT/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt'
 with hist.open('a',encoding='utf-8') as f:f.write('\n\n2026-11-10 — 100단어 공통학습과4표현 피드백 검증의 범위 분리 확정\n사용자는4표현만학습하는방향이아니라100단어공통학습을유지하며세부정답/조합을보강하도록명시했다. 기존100단어분류보조학습이세부동작정답완비와다름을구분. '+overview+'\n판단:넓은공통학습을버리지않고정답/출력/실제자동창을분리검증한다. 모양분리만으로자동창의정렬오류까지해결되지는않는다. 반복실험을새독립성능이라고하지않으며운영채택과학습개선을구별.\n근거 '+str(U/'RESULT_KO.txt')+'\n')
 goal=ROOT/'Docs/FEEDBACK_GOAL_AND_PLAN_KO.txt';prior=goal.read_text('utf-8-sig');goal.write_text('2026-11-10 최신 실행 범위:100단어899공개FIT 공통학습 유지, 세부정답은12표현모양/10표현움직임/4표현순서에서보강,4표현을초기피드백검증범위로둔다. 단어분류보조학습과세부피드백정답완비는구별한다. 이번손모양부분개선/자동정상유지,방향·순서잔존문제로웹미교체. 다음은FIT자동창감독과공통동작조합보강. 근거 '+str(U/'RESULT_KO.txt')+'\n\n'+prior,'utf-8')
 write(U/'CURRENT_EXECUTION.json',dict(status='COMPLETE',pid=None,training_complete=True,comparison_complete=True,verification_complete=True,default_web_changed=False,updated=now));event('finalized',report=str(U/'RESULT_KO.txt'))
if __name__=='__main__':main()
