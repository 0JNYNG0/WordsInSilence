from pathlib import Path
import json,hashlib
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2]
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
assert not (U/'FINAL_RESULT.json').exists()
r=read(U/'RESULT.json');g=read(U/'GUARD_RESULT.json');v=read(U/'VERIFICATION.json');vg=read(U/'GUARD_VERIFICATION.json');n=read(U/'NEXT_PLAN.json');ca=read(U/'CAUSAL_AUDIT.json')
assert v['passed'] and vg['passed'] and sha(U/'RESULT.json')==v['result_sha256'] and sha(U/'GUARD_RESULT.json')==vg['result_sha256']
final=dict(status='COMPLETE_NO_RELIABLE_GAIN_NO_ADOPTION',first_result=str(U/'RESULT.json'),first_result_sha256=sha(U/'RESULT.json'),guard_result=str(U/'GUARD_RESULT.json'),guard_result_sha256=sha(U/'GUARD_RESULT.json'),retained_baseline=str(O/'word100_event_guide_linkage_20261110/FINAL_RESULT.json'),data=r['data'],training=r['training'],automatic=r['automatic'],reviewed_segments=r['reviewed_segments'],guard_regressions=g['normal_regressions'],candidate_adopted=False,default_changed=False,private_used=False,latency=r['latency'],next_plan=str(U/'NEXT_PLAN.json'),conclusions=['Explicit before/center/after event classification and center-only appearance head trained, but no end-to-end warning gain over retained baseline.','Primary24 reviewed windows:23 old vs22 new; secondary23/23; core appearance20/24 each.','Progress-only positive-withdrawal guard resolves one false similar in primary but demotes normal+bothspeed variants; rejected.','Normal/speed/crossing, old shape/direction and all existing judgments remain retained.','Remaining reversed-core fixtures4warning1similar4uncertain; omission6 uncertain. Not all fixture manipulations are independently verified linguistic errors.'],verification='Codex local implementation; same Codex separate saved-output arithmetic and visual review. Not independent human/expert validation.',limitations=r['limitations']+['New guard was a retrospective development causal experiment, not fresh confirmation.','Parent/child label split is directly observed; its contribution to errors is a hypothesis to isolate, not sole cause proved.','No gain is claimed from a new head, high training fit, or number of checks.'])
write(U/'FINAL_RESULT.json',final)
summary='동작 앞·중간·뒤 사건판정부+중심관측판정부 100단어899공통/36공개FIT251구간,3seed×12회학습·78조건×3비교 완료. 최종경고 개선0: 역순9는4경고1비슷함4보류,누락6보류 유지.24직접관측창은기존23→22/23/23,핵심모습20/24. 진행만보인양성보류후처리는첫seed오류1을보류하나정상1+속도2도보류해미채택. 초기10366·후처리1981저장검산PASS. C/T3F/현재TCN/웹·DTW/기존손모양방향유지,개인자료0. 전체피드백완료/독립평가/성능향상 주장없음.'
nexttext='자세 관측과 행동 역할을 분리한 사건 정답을 보강한다. 이동과 도착을 부모-자식 한 행동으로 연결하고 실제 시작/진행/끝·중단·가림을 별도 확인. 직접확인구간 대 자동구간으로 정답/식별/경계연결 병목을 분리한 후 필요한 부분만 공통학습. 100단어899/기존판정유지,새공개평가 노출감사. 현재3학습·234비교·234후처리재실행금지. 상세NEXT_PLAN.json의4단계는검토완료·미실행.'
text=f'''이번 작업 결과 (2026-11-10)

학습과 비교 검증은 마쳤습니다. 다만 이번 후보가 기존보다 안정적으로 좋아졌다는 근거는 나오지 않아 현재 테스트 화면과 기존 모델은 바꾸지 않았습니다.

1. 무엇을 바꿨나
기존에도 GRU가 앞뒤 프레임을 보았지만, 주석은 주로 순간 자세에 붙어 있었습니다.
이번에는 앞 모습 + 가운데 행동 + 뒤 모습을 묶어 행동 종류를 배우는 판정부를 추가했습니다.
또 가운데 구간에 실제로 보인 핵심 모습은 별도로 판단하게 했습니다. 이 판정부에는 앞뒤 정보가 들어가지 않습니다.
손이 가려진 것을 동작 완료로 가르치지 않았고, 순서를 뒤집은 자료도 별도 대조로 사용했습니다.

2. 어떤 자료인가
공통 학습: 기존 공개100단어899영상, 수행자 기록ID01~09. 매 학습회 모두 포함했습니다.
세부 정답: 공개4표현36영상의 기존 관측 주석에서 만든251구간(배경58,진행94,핵심모습99).
251개는 새 촬영251개가 아니라36개 영상 안에서 나눈 구간이며 일부는 겹칩니다. 정확한 동작 경계를 새로 사람이 모두 붙였다는 뜻도 아닙니다.
비교: 기록ID11~13의 공개12원본에서 정상·속도변경·역순·누락·가림 등78조건. 같은 설정을3개 초기값으로 학습해 비교했습니다. 234개의 새로운 사람 영상이라는 뜻이 아닙니다.
별도 눈 확인: 공개4영상 안의24구간. 같은Codex가 확인했으며 수어전문가의 독립 정답은 아닙니다.

3. 이전 → 이번
정상12조건: 새 잘못된 경고0 → 0, 비슷함9/보류3 유지.
속도 차이24조건: 새 잘못된 경고0 → 0, 기존 판정 유지.
양손 교차3조건: 비슷함3 → 3 (정상12조건에 포함).
기존 손모양/방향 오류 시험9조건: 경고9 → 9.
주요 구간 역순9조건: 경고4/비슷함1/보류4 → 동일.
마지막 동작 삭제6조건: 모두 보류 → 동일. 삭제와 관측 부족을 확실히 구별한 것은 아닙니다.
손 전체 가림12+마지막 구간 가림6: 모두 보류 유지. 보이지 않는 것을 정상으로 올리지 않았습니다.
직접 확인한24구간의 행동종류: 기존23개 → 주비교모델22개. 다른 학습초기값은23개,23개.
핵심모습 구분은20/24였지만, 이것은 행동 전체의 올바른 완료 판정 정확도가 아닙니다.

4. 추가 보완까지 시험한 결과
진행하는 일부 모습만 확인됐는데 전체를 잘 했다고 넘기는 것을 막는 보완도 시험했습니다.
주비교모델에서는 남은 잘못된 비슷함1개를 보류로 바꿨지만, 정상 원본1개와 속도만 바꾼2개도 보류로 바꿨습니다.
다른 초기값에서도 결과가 안정적이지 않아 이 보완도 적용하지 않았습니다. 오류1개를 고쳤다는 이유로 정상3개를 희생하지 않았습니다.
최초 후보 결과와 후처리 결과는 모두 보존했습니다.

5. 왜 아직 막히나
첫째, 얼굴 앞에 있는 손/팔 자세가 나중의 팔 자세와 비슷해, 몸 관계만 사용하는 모델은 앞 행동을 뒤 행동으로 여전히 혼동했습니다.
둘째, '팔을 따라 이동'과 그 끝의 '손목 근처 도착'이 별도 자세 종류로 나뉘어 있습니다. 실제 안내에서는 도착이 앞 행동의 일부인데 학습 정답은 한 행동 전체를 충분히 연결하지 못합니다.
셋째, '핵심 자세가 보였다'와 '필요한 움직임을 끝까지 했다'는 다릅니다. 이번 실험에서 정상 영상도 진행 중으로 판단한 사례가 있어 완료 규칙을 바로 강하게 적용할 수 없습니다.
이 구조와 사례는 확인했지만, 주석 구조 하나가 모든 오류의 유일한 원인이라고 증명한 것은 아닙니다.

6. 다음 작업 계획
① 행동 정답 재연결: 한 행동의 시작·진행·끝을 같은 행동으로 묶고, 그때의 위치와 손모양은 별도 항목으로 둡니다. 이동 끝의 도착은 부모-자식 관계로 연결합니다. 틀린 모양/반대 방향도 시도한 행동 자체는 남깁니다.
② 원인 분리 비교: 직접 확인한 구간을 주었을 때와 AI가 구간부터 찾았을 때를 나눠 검사합니다. 직접 구간도 틀리면 정답·특징 학습을, 직접 구간은 맞는데 자동만 틀리면 구간 연결을 먼저 고칩니다.
③ 필요한 부분만 추가 학습:899개 공통 학습은 유지하며, 확인된 시작·끝·반대방향·중단·가림 정답으로 학습합니다. 기존 손모양·방향 판정은 고정 비교 기준으로 둡니다.
④ 새 자료 검증 후 적용: 공개 파일의 기존 노출과 중복을 감사하고, 정상/속도/교차를 해치지 않으면서 여러 표현에서 개선되는지 확인한 뒤 웹 적용을 판단합니다.
위 계획의 자료 항목과 반례6개를 코드로 점검했습니다. 이는 계획/자료형 검사이며, 다음 학습이 성공했다는 뜻은 아닙니다. 후속 학습은 아직 시작하지 않았습니다.

보존 및 검증
C·T3-F,현재TCN,기존DTW경로,기존손모양·방향,웹 기본은 변경하지 않았습니다. 사용자·멘토 자료 접근/학습/튜닝0,새 다운로드0,카메라0.
같은Codex가 구현하고 별도 저장결과 계산으로 초기10366항목/후처리1981항목을 검산했습니다. 독립 사람 검토와 다릅니다.
추가 판정부/연결 계산 중앙값 {r['latency']['median_seconds']:.3f}초, 최대 {r['latency']['max_seconds']:.3f}초. RTMW·기존특징추출·TCN·DTW·촬영/화면 시간은 제외하므로 전체 피드백 시간이 아닙니다.
HTML은 명확한 성능 개선이 없어 생략했습니다.

근거: FINAL_RESULT.json, VERIFICATION.json, GUARD_VERIFICATION.json, CAUSAL_AUDIT.json, NEXT_PLAN.json.
'''
(U/'RESULT_KO.txt').write_text(text,'utf-8')
planlines=['다음 작업 상세 계획 — 검토 완료, 실행 전\n']
for s in n['steps']:
 planlines += [f"{s['step']}. {s['title']}", '근거: '+s['reason'],*['- '+x for x in s['work']],'완료/분기 기준: '+s['gate'],'']
(U/'NEXT_PLAN_KO.txt').write_text('\n'.join(planlines),'utf-8')
p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';txt=p.read_text('utf-8-sig')
for heading,value in [('## 8.',summary),('## 9.',nexttext)]:
 i=txt.index('\n',txt.index(heading))+1;txt=txt[:i]+'\n### 2026-11-10 앞·중간·뒤 사건 맥락 후보 비교 완료·미채택\n\n'+value+'\n근거 '+str(U/'RESULT_KO.txt')+'\n\n'+txt[i:]
p.write_text(txt,'utf-8')
s=read(O/'STATE.json');s.update(current_stage=U.name,next_single_action=nexttext);s[U.name]=dict(status=final['status'],summary=summary,next=nexttext,result=str(U/'FINAL_RESULT.json'),verification=[str(U/'VERIFICATION.json'),str(U/'GUARD_VERIFICATION.json')],execution=str(U/'CURRENT_EXECUTION.json'),candidate_adopted=False,private_used=False);write(O/'STATE.json',s)
(ROOT/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+summary+'\n\n다음 한 작업: '+nexttext+'\n\n현재단위 '+str(U)+'\n학습세션89358/평가35512/후처리41873 모두 종료코드0. 실행 중 명령 없음. 현재 웹 미변경. 첫RESULT와GUARD_RESULT보존,최종FINAL_RESULT/RESULT_KO참조.\n','utf-8')
p=ROOT/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt'
with p.open('a',encoding='utf-8') as f:
 f.write('\n\n[2026-11-10 앞·중간·뒤 사건 맥락 보강]\n사용자요청:앞동작을뒤동작으로착각하는문제와순서·누락보완,100공통유지.\n'+summary+'\n계획변천:순간 신체관계 판정부에서 사건 앞/중간/뒤 pooled context와중심관측 독립판정부로 확장했으나 결과향상없음. 진행-only양성보류도정상회귀로미채택. 다음은모델을더붙이기보다행동역할/순간자세/진행완료정답의계층과구간연결을분리검사.\n'+nexttext+'\n증거 '+str(U)+'\n')
write(U/'CURRENT_EXECUTION.json',dict(status='COMPLETE_NO_RELIABLE_GAIN_NO_ADOPTION',pid=None,running=False,training_session=89358,evaluation_session=35512,guard_session=41873,all_exit_codes=0,next=n['status']))
write(U/'CLOSE_RECORD.json',dict(final_sha256=sha(U/'FINAL_RESULT.json'),report_sha256=sha(U/'RESULT_KO.txt'),next_plan_sha256=sha(U/'NEXT_PLAN.json'),docs_synced=True,no_running_process=True))
print(summary)
