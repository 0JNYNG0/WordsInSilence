from window_core import *
from datetime import datetime
from collections import Counter
import re
main=read(U/'SUMMARY.json');v=read(U/'VERIFICATION.json');follow=read(U/'ORDER_ONLY_VERIFY.json');adapter=read(U/'ORDER_GUARD_VERIFIED.json');assert v['passed'] and follow['passed'] and adapter['passed']
rows=read(U/'ORDER_ONLY_COMPLETE.json')['rows'];remaining=[]
for r in rows:
 if r['kind']!='reversed_core_auto_order' or 'difference' in r['after']['display']:continue
 remaining.append(dict(clip_id=r['clip_id'],state='similar_without_warning' if r['after']['all_required_similar'] else 'uncertain',result=r['result']))
scores=read(U/'scores/learned_core_42.json');errors=[r for r in scores if r['task']=='movement' and r['variant']=='normal' and r['padding']=='clean' and r['pred']!=r['target']]
write(U/'CAUSE_AUDIT.json',dict(observed=[dict(issue='Clean windows regressed despite supervised padding',count=len(errors),errors=[{k:r[k] for k in ['clip_id','axis','target','pred']} for r in errors]),dict(issue='Whole-context mixed direction predicted inside a single upward interval',count=sum(r['pred']==3 for r in errors)),dict(issue='Maintain position misread as movement',count=sum(r['target']==1 for r in errors)),dict(issue='Optional missing query span can be supplied only after a direction contradiction; old/new windows not always equal',evidence='First ORDER_ONLY assertion and3edited mismatches; corrected by exact-window validation/new order queries, no test case removed')],inference='Window emphasis alone is insufficient to distinguish each motion event from neighbouring events. Classifying an expanded window as a single mixed-direction class did not preserve clean hold/single-stroke recognition. This is an interpretation of saved errors and architecture, not a causal proof from a separate ablation.',retained='Preserve Pcoupled42 shape/direction. New primary42 ordering used only as negative conflict evidence; never fills missing events or certifies completion.',remaining=remaining,comparability='PreviousP reverse-main component controls used different reversal bounds in8of9clips. Do not describe previous9/12->new12/12 as a like-for-like improvement. The new9whole-path controls have exact same-input before/after baselines.',first_error_logs=[str(p) for p in U.glob('*FAILURE*.txt')]))
next_text='완료6학습·468자동비교·78후속검증 재실행금지. Pcoupled42 모양/방향과100단어공통기반보존. 새order_guard는순서충돌전용격리후보이며기본웹미교체. 공개FIT에서핵심동작의시작/방향전환/끝과유지구간을사건별로연결한시간정답을보강하고,넓은창한개를혼합방향한값으로판정하는대신관측된사건순서와각사건방향을따로감독하는조건을먼저동결한다. 기존좋은방향판정부를교체하지않고그입력범위선택을분리검증한다. 미확인/미배정사건도분모와근거에남기고짧은반대동작삭제금지. 추가검증은새로분리한공개자료가가능한지먼저감사;역사DEV튜닝/seed선택/사용자자료금지. C/T3F·현재RTMWTCN·웹8814보존.'
write(U/'NEXT_PLAN.json',dict(next=next_text,reason='Attention-only replacement lost normal holds/single strokes; order-only composition gave bounded improvement without changing existing observation.',automatic_remaining=remaining,mandatory_scope='100word foundation retained;4word service-validation scope, not claim of100word detailed feedback',no_new_download_or_deployment_authorization=True))
primary=main['models']['learned_core_42'];table=[]
for name,m in main['models'].items():
 c=m['component']['by_variant'];table.append(f"{name}: 정상의 지정구간 방향 {c['normal']['clean']['movement']['correct']}/32, 앞뒤8프레임확장 {c['normal']['both8']['movement']['correct']}/32, 정상순서 {c['normal']['clean']['order']['correct']}/12, 새역순대조 {c['reversed_main']['clean']['order']['correct']}/12;100단어보조출력 {m['component']['aux_word100']['correct']}/300")
report='''이번 작업 결과: 순서 충돌을 알리는 부분은 일부 개선, 새 방향 판정부 전체 교체는 보류

무엇을 바꿨나
100단어899공개학습영상은 유지했습니다. 기존에 개선된 손모양 모델도 고정해 보존했습니다.
학습용 자동 구간에도 준비·마무리가 섞이는지 확인했고, 예를 들어 주요 동작79~104프레임 주변을 자동으로71~109프레임까지 잡는 사례를 재확인했습니다.
그 넓은 구간 안에서 핵심 움직임을 찾도록 시간합성곱+GRU와 작은 판정부를 학습했습니다. 모든 시점을 고르게 보는 방식과 핵심 시점에 비중을 주는 방식을 각각3회 비교했습니다.
한 창 안에 실제 올라감과 내려감이 둘 다 있으면, 하나를 버리거나 정지로 합치지 않고 함께 다루도록 했습니다. 동작 이름에 따른 임시 예외는 추가하지 않았습니다.

처음 시도한 방법의 한계
대표 후보의 방향 판단은 정확히 지정된 정상32구간에서 이전32개정답→22개정답으로 나빠졌습니다.
10개 오류는 정상 유지5개를 움직임으로 판단, 한 방향3개를 왕복으로 판단, 이동2개를 유지로 판단한 경우입니다.
자동 분석에서도 정상12영상 중 모든 주요카드가 비슷하게 보이는 결과가9→7개로 줄고 확인불가가 늘었습니다. 반대방향만 바꾼3시험 중1개는 기존 경고도 놓쳤습니다.
따라서 이 새 방향 판정부를 그대로 적용하지 않았습니다. 핵심 시점에 비중을 주는 것만으로 이 문제가 해결된다는 가정은 이번 결과로 지지되지 않았습니다.

이어 진행한 보완
더 잘하던 기존 손모양·방향 모델을 유지하고, 새 모델의 순서 검사만 보조로 연결했습니다.
이 보조 검사는 분명한 순서 충돌을 알릴 수만 있습니다. 보이지 않는 행동을 채우거나 완료했다고 만들어 주지 않습니다.
연결 검사 중 이전/새 판정의 비교 구간이 같지 않은3조건을 발견했습니다. 서로 다른 구간의 근거를 그대로 옮기지 않도록 고쳤고, 기존 구간에서 필요한 순서 검사만 다시 계산했습니다. 첫 오류와 중간 결과는 보존했습니다.

최종 보조 조합의 결과
- 정상12영상: 이전과 동일. 차이 경고0, 주요동작이 모두 비슷하게 관측9, 확인불가3.
- 느리게/빠르게 수행한24조건: 이전 결과 그대로 유지. 교차 동작이 있는 조상 정상3영상도 그대로 유지.
- 손모양만/방향만/두가지동시 변경9조건: 기존 차이 경고9개 모두 유지.
- 관절 전체가림12·끝동작관절가림6·끝동작삭제6조건: 정상으로 넘긴 경우0유지. 관측부족을 손모양 정답으로 채우지 않음.
- 추가로 핵심 동작 순서를 뒤집은 공개9조건: 기존 순서 차이 표시0→보완후3개. 경고 없이 주요동작이 모두 비슷하게 표시된 경우4→1개. 나머지5개는 확인불가입니다.
이9조건은 순종·잘하다·이집트 각공개3촬영의 관절 시계열을 인위적으로 뒤집은 시험입니다. 실제 사용자가 틀리게 수행한9영상이나 새 독립 사람9명이 아닙니다.
같은입력의 이전/이후 비교이며, 기존에 완료한69조건은 저장결과를 재사용했습니다. 새로 만든9조건에 대해서만 이전모델 기준을 처음 계산했습니다.
순서 보조에서 정상이라고 출력한 값은 완료 근거로 사용하지 않았습니다. 역순인데도 맞는 순서라고 출력하는 경우가 남아 있기 때문입니다.

남은 문제
역순9조건 중1개는 아직 경고 없이 비슷하게 표시되고5개는 확인불가입니다. 모양과 방향을 함께 바꾼 기존1조건의 방향 불확실성도 남았습니다.
주요동작별 방향, 정상유지, 앞뒤 다른 동작의 영향을 완전히 분리하지 못했습니다. 전체 구간을 하나의 방향으로 요약하기보다 각 동작의 시작·진행·끝을 연결해야 합니다.
반복 횟수나 실제 접촉을 모두 이해했다고 볼 수 없고, 수어교육의 허용범위가 전문가 검증된 것도 아닙니다.
새 순서보조는 실험 후 조합해 확인한 제한적 결과입니다. 기존 개발자료를 반복 사용했으므로 독립적인 최종 성능처럼 설명하지 않습니다.

숫자의 뜻
899=공통학습 영상 수. 31=추가 움직임 구간 정답 수. 76=추가 손모양 특징 정답 항목 수. 63쌍=추가 앞뒤 동작 연결 정답 수.
이전227/228은 순종44+잘하다92+조상92개의 손모양 특징 채점에서227개가 맞았다는 뜻입니다. 이집트의 손끝 오므리기는 이228항목에 포함되지 않았습니다.
32/32와29/32는 같은32개 지정 움직임 구간에 대한 이전 두후보의 결과입니다.32영상·32단어 또는 전체 자동분석32번 성공이라는 뜻이 아닙니다.
더 자세한 설명과 실제 집계 근거: NUMBERS_KO.txt / NUMBERS_CHECK.json.

실행·검증·적용 상태
6고정학습,468자동경로,78조건의 후속조합검증을 끝냈습니다. 사전1698검사·저장결과15802산술검사·후속782검사 통과. 재사용가능한order_guard.py도78저장결과와 일치하고 다른시간창 근거를 거절하는지 검증했습니다.
이 검사는 같은Codex의 코드/산술 검산입니다. 별도 사람이나 농인강사가 정답을 독립 검증한 것으로 설명하지 않습니다.
신규 실험 전체경로의 대표 측정은 캐시관절입력 이후 중앙값약109ms·95백분위약204ms입니다. RTMW검출·카메라·영상전송·화면표시 시간이 빠진 값이며 실제 촬영후 대기시간이 아닙니다. 순서만추가한최종조합의 실제 웹속도는 아직 측정하지 않았습니다.
기존C·T3-F·현재RTMW TCN·기본웹8814는 변경하지 않았습니다. 사용자/멘토자료에 접근하거나 학습·튜닝·후보선택에 사용하지 않았습니다.
제한된 일부 개선이므로 새HTML은 만들지 않았습니다. 새가중치·순서검사부는 격리 보존하며 기본웹에 자동배포하지 않았습니다.

6회 결과 전체(좋은난수만고르지않음)
'''+ '\n'.join(table)+'''
주의: 이전 실험의 역순 대조와 이번 역순 대조는8/9영상에서 뒤집은 경계가 달라9/12→12/12처럼 직접 성능상승으로 비교하지 않습니다. 정확한 같은입력 전후비교는 위78조건 후속 결과입니다.

다음 작업
'''+next_text+'\n'
(U/'RESULT_KO.txt').write_text(report,'utf-8')
status='COMPLETE_LIMITED_ORDER_VETO_GAIN_NEW_MOTION_HEAD_REJECTED_NOT_DEPLOYED';summary='899FIT100단어 유지·기존모양고정, 준비/복귀혼합 감독2조건×3seed·468자동경로완료. 새방향대표32→22/32회귀로미채택. 기존P모양/방향+새순서충돌검사만후속78조건검증: 정상12/속도24/교차/기존오류9/관측부족24유지,새역순9조건경고0→3·경고없는전체비슷함4→1(5보류). 같은DEV사후조합·합성역순이며완전해결/독립성능아님. 사전1698/산술15802/후속782·재사용부78검사PASS(같은Codex). 228은순종44잘하다92조상92이집트0으로정정. 첫구간불일치오류보존·다른창근거전용금지보완. 웹8814/C/T3F/현재TCN보존·개인자료0·HTML없음.'
write(U/'RESULT.json',dict(status=status,summary=summary,report=str(U/'RESULT_KO.txt'),numeric_explanation=str(U/'NUMBERS_KO.txt'),primary_model='learned_core_42',adopted_research_composition='Pcoupled42 preserved shape/direction + primary42 order contradiction-only guard; isolated posthoc',default_web_changed=False,public_only=True,private_access=False,remaining=remaining,components=main['models'],followup=follow['groups'],verification_counts=[1698,15802,782,78],next=next_text))
write(U/'CURRENT_EXECUTION.json',dict(status=status,pid=None,training_complete=6,fullpath_complete=468,followup_complete=78,verification_pass=True,active_processes=[]))
root=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';s=root.read_text('utf-8-sig');oldtag='### 2026-11-10 자동 비교 구간의 방향·순서 학습 진행';newtag='### 2026-11-10 자동 구간 학습·순서 충돌 보조의 제한 개선 검증 완료'
for section,text in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',summary),('## 9. 다음 단일 작업',next_text)]:
 pos=s.index(section);a=s.index(oldtag,pos);b=s.index('\n### ',a+len(oldtag));s=s[:a]+newtag+'\n\n'+text+'\n근거 '+str(U/'RESULT_KO.txt')+'\n\n'+s[b:]
root.write_text(s,'utf-8')
p=O/'STATE.json';d=read(p);d.update(current_stage=U.name,next_single_action=next_text,last_updated=datetime.now().astimezone().isoformat());d[U.name]=dict(status=status,summary=summary,report=str(U/'RESULT_KO.txt'),result=str(U/'RESULT.json'),verification=str(U/'VERIFICATION.json'),followup=str(U/'ORDER_ONLY_VERIFY.json'),execution=str(U/'CURRENT_EXECUTION.json'),next=next_text,default_web_changed=False,private_access=False,active=None);write(p,d)
(ROOT/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n최신 승인: 자동구간의 준비·마무리 혼합에도 주요방향/순서 보완,899/31/76/63/228/32의뜻설명후실행.\n\n마지막 완료\n'+summary+'\n\n다음 한 작업\n'+next_text+'\n\n모든 이번학습/평가프로세스 종료.\n결과 '+str(U/'RESULT_KO.txt')+'\n숫자 '+str(U/'NUMBERS_KO.txt')+'\n검산 '+str(U/'ORDER_ONLY_VERIFY.json')+'\n현재웹8814와기존C·T3-F유지. 사용자자료금지.\n','utf-8')
with (ROOT/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt').open('a',encoding='utf-8') as f:
 f.write('\n\n2026-11-10 — 자동 구간의 중요동작 학습과 기존 관측 보존\n사용자요청:100단어899영상 학습범위와세부정답/평가분모를구분해서설명하고,준비·복귀혼합에강한방향·순서판정을보완.\n'+summary+'\n왜 이렇게 바꿨나: 중요시점 가중학습으로 방향판정까지 모두 바꾸면 정상 유지와 한방향 이동이 회귀했다. 잘하던모양/방향은보존하고 새로운순서판정은충돌만알리는보조로분리했다. 새로운역순9조건에서3경고를추가했으나1비슷함/5보류가남아제품완성으로올리지않았다.\n숫자정정:과거227/228에는이집트손끝닫힘정답0. 과거보고서를소급변조하지않고이번감사로연결.\n다음 '+next_text+'\n근거 '+str(U/'RESULT_KO.txt')+'\n')
print(json.dumps(dict(status=status,report=str(U/'RESULT_KO.txt'),remaining=remaining),ensure_ascii=False))
