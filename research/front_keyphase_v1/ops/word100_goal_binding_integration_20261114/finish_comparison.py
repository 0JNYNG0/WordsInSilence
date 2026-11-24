"""Record completed comparisons and remaining gates without promoting failure."""
from pathlib import Path
from datetime import datetime
import json,hashlib,shutil
from PIL import Image,ImageDraw,ImageFont
U=Path(__file__).resolve().parent;O=U.parent;R=O.parents[2];G=O/'word100_web_completion_graph_20261111'
read=lambda p:json.loads(p.read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):
 with Path(p).open('x',encoding='utf-8')as f:json.dump(d,f,ensure_ascii=False,indent=2)
reports=['INTEGRATION_VERIFIED.json','RUNTIME_EXTENSION_VERIFIED.json','FULL_GRAPH_VERIFIED.json',
 'VISIBILITY_VERIFIED.json','OWNERSHIP_VERIFIED_V2.json','ASSOCIATION_SCOPE_VERIFIED.json']
verified={n:read(U/n)for n in reports}
assert all(r['passed']for r in verified.values())
assert read(U/'relational_change/VERIFIED.json')['implementation_checks_pass']
for p,h in read(G/'PROTECTIONS.json').items():assert sha(Path(p))==h,p
full=read(U/'FULL_GRAPH_COMPARISON.json');cov=read(U/'COVERAGE_COMPARISON.json');vis=read(U/'VISIBILITY_VERIFIED.json')['summary']
now=datetime.now().astimezone().isoformat()
checks={n:d.get('total',len(d['checks']))for n,d in verified.items()}
review=dict(at=now,implementation='Codex local',verification='Same Codex separate code/arithmetic review; not independent KSL expert',
 status='NORMAL_ERROR_COMPARISON_COMPLETE_RELEASE_GATES_REWORK',
 completed=['Automatic movement/arrival evidence connected without deleting actions',
 'All independent role cores retained for order; entire candidates retained for shape/direction',
 'Missing-hand direction eligibility corrected without changing underlying measurements',
 'Mutual action ownership and posture-only evidence distinguished from verified temporal action',
 'All82 exposed scenarios x3 seeds compared; separate93 web-lineage conditions preserved'],
 checks=checks,protected_files=477,
 result_scope={'public_source_recordings':12,'source_scenarios':82,'selector_seeds':[42,43,44],
  'cases':246,'web_lineage_cases':93,'web_normal_cases':60,'new_role_calls':12,
  'new_training_runs':6,'new_training_scope':'4-parameter physical-change calibration; rejected as completion model',
  'no_retraining_old_runs':True,'no_final8':True,'no_user_data':True},
 comparison={'normal_arm_goal_before':8,'normal_arm_goal_after':12,'normal_arm_goal_cases':15,
  'normal_separation_before':8,'normal_separation_after':14,'normal_separation_cases':15,
  'normal_speed_order_pair_observations':81,'new_normal_order_regressions':0,
  'missing_hand_direction':vis,'last_action_edits':read(U/'ASSOCIATION_SCOPE_COMPARISON.json')['summary']},
 release={'G2':'REWORK','G3':'PREFLIGHT_COMPARE_COMPLETE_NOT_PASS','G4':'NOT_RUN','G5':'NOT_RUN','G6':'NOT_RUN',
  'web_changed':False,'whole_project_complete':False},
 why_not_release=['Some action identity/terminal relationship remains uncertain; preserving or withholding evidence is not completed-action accuracy',
  'Normal public variation disproves mandatory downward palm displacement as a general action target',
  'Initial hand/body relations, alternation and final fingertip requirements have explicit capability gaps',
  'Edited public controls are not natural learner error ground truth; sealed final and actual-web gates not yet met'],
 next_single_action='Prepare directly reviewed, non-exclusive hand-closing/body-region/uprighting target correspondence on public FIT. Preserve normal variants and unknowns. Do not repeat the rejected scalar-displacement calibration or tune on exposed probes.',
 evidence={n:sha(U/n)for n in reports+['JOINT_SAVED_COMPARISON.json','ASSOCIATION_SCOPE_COMPARISON.json','FULL_GRAPH_COMPARISON.json']})
write(U/'RESULTS_REVIEW.json',review)
text='''보완·정상/오류 비교 결과 | 2026-11-14

이번 보완과 고정 비교는 끝났다. 전체 그래프와 웹 피드백 완성은 아직 아니다.
정상과 오류 비교 결과에서 통과하지 못한 의미 판단을 완료로 올리거나, 봉인 평가를 먼저 열지 않았다.

이전 → 현재
1. 움직임과 도달 구간이 따로 표시되어 마지막 상태를 놓침 → 같은 설명에 속한 이동/도달 관측을 함께 전달.
   공개 정상3영상×5조건에서 반대팔 손목 쪽으로 이동한 상태 변화 관측은8/15→12/15,
   양손 가까움→벌어짐 관측은8/15→14/15. 같은 영상의 처리 조건을 포함한 수치이며 수어 정답률이 아니다.
2. 두 행동의 가장자리가 겹쳐 순서를 모호하게 판단 → 각 행동이 실제로 관측된 부분의 시간 순서를 별도로 확인.
   손모양·방향은 좁은 부분만 검사하지 않고 기존 전체 후보 구간으로 계속 비교한다.
   정상/느리게/빠르게의81쌍 관측에서 기존 순서 지지를 잃지 않았다.
   81은9개 행동쌍×3속도조건×3모델이며81개의 독립 영상이 아니다.
3. 손 관절을 가렸는데 팔의 이동값으로 손동작 방향을 지지/교정 → 필요한 손이 보였는지 먼저 구분.
   기존 방향 지지17건·반대 방향7건→모두 관측부족. 팔 이동 측정값 자체는 보존했다.
   정상·속도 조건과 의도적으로 바꾼 손모양/방향의 기존 근거는 유지됐다.
4. 마지막 행동을 지우자 앞의 비슷한 자세를 대신 연결 → 구간 소유 충돌·한 순간 자세·근거 없음 분리.
   공개6영상×삭제/가림2조건×3모델=36조건에서31은해당행동미확인,4는서로다른행동연결충돌,
   1은순간자세만관측으로구분됐다. 어떤 경우도마지막행동전체확인으로올리지않았다.
   36건을전부실제사용자오류로맞혔다는뜻은아니다. 가림과실제누락은여전히다른문제다.

어떤 비교를 했나
- 공개12원본의82파생조건×3초기값=246조건을 모두 대조했다. 정상·속도·손모양변경·반대방향·두오류동시·순서편집·끝삭제·관측가림을 포함한다.
- 현재 웹 계보의 별도93조건에서는 정상60조건의 기존 상태와 기존교정9사례가 보존됐다. 두 계보의 점수를 한 정확도로 합치지 않았다.
- 동작순서 블록교환 시험은4개이며 새로 필요한 고정role판정부12호출만 수행했다. 이전234출력·손모양·방향 점수는 재추론하지 않고 그대로 재사용했다.
- 짧은정상,일시가림,긴가림,시작자세만,끝자세만,역순의연결계약을 검사했다. 이 일부는 코드 합성시험이며 실제사용자성능으로 보고하지 않는다.
- 기존477보호파일 해시와 저장 출력/원본 시점 대응을 별도 코드로 확인했다. 동일Codex의별도검산이며 독립전문가검증은 아니다.

학습으로 확인한 것
기존공통100단어신경망은바꾸지않았다. 기존공개FIT자료에서손목대신손바닥을볼때의변화와팔뚝방향변화를구분하는
작은4가중치판정부를2조건×3초기값으로학습했다. 하지만같은정상동작도손을닫고팔이바뀌는동안손중심높이는거의
그대로일수있었다. 따라서‘손이충분히아래로이동해야완료’라는목표자체가맞지않아완료판정부로채택하지않았다.
이첫결과와문제있는평가해석을보존했고,비교정답을바꿔다시정확도를높였다고보고하지않았다.

전체 그래프에서 남은 핵심
G0 범위/보호:완료. G1 기존설명/정답정리:완료(모든특징정답완비라는뜻아님).
G2 행동-설명-도달상태:보완중. 이번에연결/관측혼동은개선했지만가슴쪽관계·손닫힘·팔세움의전체수행을동시에확인하는근거는아직부족하다.
G3 정상/오류 종합비교:이번고정비교는완료. 전체기능합격은아님. 일부핵심특징은미지원/보류이고실제누락을확정하지못한다.
G4 새공개봉인평가:G2/G3통과전미실행. G5 통과항목웹반영·색상/문장검증:미실행. G6 실제촬영최종검증:미실행.

다음에는 무엇을 바꿔야 하나
1. 공개FIT에서손닫힘,손과가슴/얼굴의관계,전완방향변화를서로겹칠수있는관측으로연결한다.
   한정상예를기준으로몇도/얼마나아래로움직여야한다는규칙을만들지않는다.
2. 실제진행이확인된짧은정상과시작자세만있는예,마지막관측이없는예를따로평가한다.
   이번에통과한구간보존·손관측구분·구간충돌구분을유지한다. 같은물리변위보정학습은반복하지않는다.
3. 전체구간의손모양/방향차이가비교에서빠지지않는지,정상·교차·속도조건을해치지않는지다시고정비교한다.
4. G2/G3를실제로통과한뒤봉인평가→웹문장/강조→사용자가시작하는촬영검증으로간다.

적용/보호
이번보완은연구용연결부와검증결과에반영했다. 운영웹·C·T3-F·공통100모델은교체하지않았다.
사용자/멘토자료접근·학습·튜닝·선택없음. 신규다운로드/외부전송/Git/카메라/서버재시작없음.
완료된학습이나평가의중복실행없음. 이보고시실행중인학습/평가프로세스없음.

오류 이력
원래전역guide번호를카드배열인덱스로읽은첫오류,상태라벨명검사오류,3개모델의출력이같다고가정한첫검산오류를별도보존했다.
특히마지막검산은알고리즘이나점수변경없이실제모델별출력을각각확인하도록수정했다(충돌4건,관측없음/순간자세각각구분).
근거: RESULTS_REVIEW.json, FULL_GRAPH_VERIFIED.json, VISIBILITY_VERIFIED.json, OWNERSHIP_VERIFIED_V2.json,
ASSOCIATION_SCOPE_VERIFIED.json, COVERAGE_COMPARISON.json, relational_change/TARGET_REVIEW.json.
'''
(U/'RESULT_KO.txt').write_text(text,'utf-8')
graph=dict(at=now,whole_graph_complete=False,nodes=[
 dict(id='A',title='동작 구간 보존·도달 근거 연결',status='LIMITED_VERIFIED'),
 dict(id='B',title='관측 부족·구간 혼동 보완',status='VERIFIED_ON_FIXED_PUBLIC_CONTROLS'),
 dict(id='C',title='정상·오류 고정 비교 및 검산',status='COMPLETED_NOT_GLOBAL_PASS'),
 dict(id='D',title='설명 요구의 실제 완료 판단',status='REWORK'),
 dict(id='E',title='새 공개 최종평가',status='PENDING_G2_G3'),
 dict(id='F',title='웹 연결·실제 촬영 확인',status='PENDING')],
 edges=[['A','B'],['B','C'],['C','D'],['D','C'],['D','E'],['E','F']],
 note='Completed comparisons are not the same as all release gates passed.')
write(U/'GRAPH.json',graph)
im=Image.new('RGB',(1700,1050),'#f6f8fb');d=ImageDraw.Draw(im)
font=lambda s,b=False:ImageFont.truetype('C:/Windows/Fonts/malgunbd.ttf'if b else'C:/Windows/Fonts/malgun.ttf',s)
navy='#192b43';muted='#53647a';line='#bdc8d5'
d.text((70,45),'피드백 기능 · 현재 작업 위치',font=font(48,True),fill=navy)
d.text((72,117),'보완과 고정 비교는 완료  /  전체 피드백 완성은 아직',font=font(29),fill=muted)
xs=[70,625,1180];w=450
items=[('1  구간 연결','이동 + 도달 근거를 함께 전달','비교할 행동은 지우지 않음'),('2  관측·혼동 구분','손이 안 보임 ≠ 방향이 틀림','서로 다른 행동의 구간 충돌 표시'),('3  정상·오류 비교','정상·속도·손모양·방향·순서','기존 정상 근거 유지 확인')]
for x,(a,b,c)in zip(xs,items):
 d.rounded_rectangle((x,220,x+w,445),radius=18,fill='white',outline=navy,width=3)
 d.text((x+25,242),a,font=font(31,True),fill=navy)
 d.text((x+25,312),b,font=font(24),fill=navy)
 d.text((x+25,355),c,font=font(23),fill=muted)
 d.text((x+25,400),'이번 비교에서 확인',font=font(21,True),fill=muted)
for x in [545,1100]:
 d.line((x,325,x+60,325),fill=navy,width=5);d.polygon([(x+60,325),(x+45,315),(x+45,335)],fill=navy)
d.text((72,500),'남은 핵심: 행동을 찾는 것과 끝까지 했다고 판단하는 것은 다름',font=font(31,True),fill=navy)
left=70;d.rounded_rectangle((left,570,810,855),radius=18,fill='#e6edf5',outline=navy,width=3)
d.text((95,595),'4  설명이 요구하는 마지막 상태',font=font(31,True),fill=navy)
for i,s in enumerate(['손을 닫았는가 + 신체 관계가 맞는가','짧아도 수행한 것인가 / 중간에 멈춘 것인가','모르는 부분을 오답이나 완료로 만들지 않기']):
 d.text((95,665+i*51),s,font=font(27),fill=navy)
d.line((840,710,910,710),fill=muted,width=4);d.polygon([(910,710),(895,700),(895,720)],fill=muted)
for y,title,body in [(570,'5  새 공개 최종평가','앞 단계 통과 뒤 진행'),(730,'6  웹·실제 촬영 검증','통과한 안내만 화면에 연결')]:
 d.rounded_rectangle((945,y,1630,y+125),radius=15,fill='white',outline=line,width=2)
 d.text((970,y+17),title,font=font(29,True),fill=muted);d.text((970,y+69),body,font=font(26),fill=muted)
d.text((72,935),'기존 C·T3-F / 운영 웹 유지    ·    개인 촬영은 학습·튜닝에 사용하지 않음',font=font(26),fill=muted)
im.save(U/'PROGRESS_GRAPH.png')
print(json.dumps(dict(review='RESULTS_REVIEW.json',report='RESULT_KO.txt',graph='PROGRESS_GRAPH.png',checks=checks,graph_complete=False),ensure_ascii=False))
