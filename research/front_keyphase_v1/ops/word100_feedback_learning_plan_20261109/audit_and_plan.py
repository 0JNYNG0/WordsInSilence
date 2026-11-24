from pathlib import Path
import json,hashlib,collections,re,datetime
import numpy as np
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2]
PREV=O/'word100_gru_semantic_finetune_20261109';M=O/'word100_motion_components_20261109';D=O/'word100_action_feedback_demo_20261108';J=O/'word100_rtmw_joint_learning_20261108'
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
 assert not (U/'AUDIT.json').exists()
 rows=read(PREV/'READY.json')['rows'];counts={};checks=0
 for maskkey in ['ym','old_ym','direct_ym']:
  counts[maskkey]={}
  for role in ['fit','development']:
   bins=collections.defaultdict(lambda:dict(labels=0,people=set(),words=set(),clips=set()))
   for r in rows:
    if r['role']!=role:continue
    assert sha(r['path'])==r['sha256'];checks+=1;z=np.load(r['path']);target=z['old_y' if maskkey=='old_ym' else 'y']
    for t,j in zip(*np.where(z[maskkey]>0)):
     q=bins[f'{j}:{int(target[t,j])}'];q['labels']+=1;q['people'].add(r['person']);q['words'].add(r['word']);q['clips'].add(r['clip_id'])
   counts[maskkey][role]={k:{kk:sorted(vv) if isinstance(vv,set) else vv for kk,vv in v.items()} for k,v in bins.items()}
 assert counts['ym']['fit']['2:1']['labels']==7;assert counts['ym']['development']['2:1']['labels']==1;checks+=2
 auto=read(M/'AUTO_WINDOW_CHECK.json')['rows'];failure=[q for q in auto if q['clip_id']=='NIA_SL_WORD1133_REAL11_F' and q['card']=='brush_arm' and q['variant']=='closed_hand_and_reversed'];assert len(failure)==1;failure=failure[0]
 audit=read(O/'word100_shape_direction_audit_20261109/AUDIT.json')['rows'];intervention=next(q['main_indices'] for q in audit if q['clip_id']==failure['clip_id'] and q['kind']==failure['variant'])
 assert failure['span']==[47,78] and intervention==[49,80];checks+=1
 source_files=[PREV/'READY.json',M/'PUBLIC_DECISION.json',M/'AUTO_WINDOW_CHECK.json',J/'feedback_model.py',D/'model.py',D/'consumer.py',D/'evidence.py',O/'word100_shape_direction_audit_20261109/audit.py']
 observed=dict(status='READ_ONLY_AUDIT_COMPLETE_PLAN_ONLY',annotation_counts=counts,annotation_unit='frame-attribute supervision cells, not independent videos or correctness ratings',automatic_case=dict(clip=failure['clip_id'],synthetic=True,automatic_span=failure['span'],intervention_span=intervention,new_prediction=failure['state'],original_geometry=failure['old_state'],trajectory=failure['evidence']['trajectory'],interpretation='Automatic span starts two frames before edited action. The sampled relation jumps from about -0.17 to2.96 then decreases. Window contamination and synthetic boundary discontinuity are confirmed; relative causal contributions to classifier error require controlled comparisons. Do not call this a measured natural learner-error rate.'),runtime=dict(input='RTMW body-normalized2D coordinates/local hand geometry/masks/time, no SignRep or RGB network inference',learned='Conv+bidirectionalGRU observation attributes, target-linked stage outputs, additional learned phase/difference heads',description='Structured keywords/guide IDs and requirements, not unrestricted text/video-language understanding',missing_output='No separately trained within-hand fingertip convergence output',dtw='Playback alignment after evidence decisions; cannot complete missing evidence'),checks=checks,source_pins={str(p):sha(p) for p in source_files},training=False,inference=False,camera=False,private_access=False,web_changed=False)
 write(U/'AUDIT.json',observed)
 plan=dict(status='PLAN_READY_NOT_EXECUTED',objective='With current RTMW coordinate stream and Conv+GRU, learn visible shared handshape, movement and event sequence; provide grounded target-guide feedback without word-specific numerical exceptions',constraints=['Keep C/T3-F/current TCN unchanged','No user/mentor learning, tuning or candidate selection','No new detector, RGB model or downloads in this plan','Do not require exact all-tip contact labels before learning visible flexion','No claim of unrestricted natural-language understanding or all-word validation'],steps=[
 dict(step=1,title='Shared target coverage and causal audit',actions=['Inventory each visible attribute by hand, person, expression, positive/negative event, observation status and provenance','RGB-review additional existing FIT public videos for visible extension/flexion and shape changes; label hidden thumb/contact separately','Bind source clause variants to observed examples without inferring shape from the sign name','Using cached public controls, compare reviewed vs automatic windows and normal/shape-only/reversed/combined conditions; inspect intervention boundaries before treating a failure as model incapability'],deliverables=['FEATURE_TARGETS.json','PUBLIC_WINDOW_DIAGNOSIS.json','ANNOTATION_COVERAGE.json'],gate='Train only claims with visible positive and contrasting examples. Aim for at least3FIT IDs and2expressions per reusable new feature; this is a feasibility minimum, not statistical sufficiency. Sparse features remain separately unvalidated, not blockers for supported ones. Full video context includes return-to-rest negatives.'),
 dict(step=2,title='Coordinate observability check and frozen comparison protocol',actions=['Match RGB labels to RTMW visibility/masks and relative finger geometry','If RGB is clear but coordinates lack the distinguishing information, separate it from missing supervision; do not force an incorrect coordinate-only training target','Use FIT-person inner folds for architecture/training choices; retain historicalDEV as regression','Audit existing additional public IDs/source SHA/exposure before reserving a final check; if no clean set exists report development-only','Freeze2candidate conditions,3seeds,primary seed,epoch/loss budget and decision gates before development scores'],gate='Label-to-frame mapping, person/source separation, left/right semantics, time resampling, missing masks and component scope checked. No same-DEV threshold search.'),
 dict(step=3,title='Coherent supervised learning and fine-tuning',actions=['Keep existing Conv+GRU baseline and good attributes; add separate visible-finger/shape-transition and movement outputs','Supply shared body/hand-relative motion measurements independently of appearance-like latent features','Learn original untrimmed sequences with main-event boundaries and before/middle/after context; preparation/return cannot satisfy a required event','Compare frozen-trunk+new heads against low-rate joint fine-tuning with retention losses on existing labels','If trunk features change, train/validate dependent runtime phase/difference heads with that same trunk checkpoint; do not combine new latent with stale old heads','Target word/structured description selects expected components and order; observation heads do not receive expected shape/direction as answers','DTW pseudo-boundaries are weak labels, distinct from reviewed event truth; missing events are never filled by warping'],gate='Genuine weight changes and train-only losses verified. Include normal, shape-only, direction-only, combined, omitted/short action and missing evidence controls. Coordinate perturbations remain synthetic controls and are not natural learner-error truth.'),
 dict(step=4,title='Full-video regression and research-web adoption',actions=['Evaluate automatically selected intervals on complete public videos, separately from known-interval component accuracy','Report normal false corrections, important-error misses, wrong-time cues, supported coverage and abstention with full denominators, by hand/person/feature','Keep natural repeated motion, small body sway, crossings and speed variation; test missing short events and final handshape omission','Freeze candidates before any final held-out check; audit unfamiliar expressions only as reusable-feature diagnostics, not an open-vocabulary product requirement','Keep capture-time RTMW reuse and time cached-coordinate neural work separately from full replay stop-to-feedback median/p95','Apply passing coherent model/checkpoint set only to separate research demo; highlight exact component and span, distinguish unsupported model feature from unobservable input'],gate='No additional normal false correction or lost core evidence on fixed shared regression; previously handled shape+direction combinations remain handled; no missing-input or absent-event automatic success; lower targeted misses without merely increasing abstention. Additional latency within current replay budget, end-to-end~3s target measured not assumed. If labels cannot support natural error claims, release remains a research demo, not validated teaching correctness.')],branches=dict(clear_rgb_good_coordinates='Target coverage and model learning',clear_rgb_bad_coordinates='Observation/input repair or scoped abstention before learning',good_known_interval_bad_auto='Phase/interval selection and event coverage',bad_known_interval='Attribute/motion representations and supervision',only_synthetic_failure='Inspect intervention artifacts and realistic equivalents before changing core model',no_clean_test='Report honest DEV improvement and retain final-test limitation'),validation_kind='Same Codex code/label/artifact audit; not independent expert assessment',training_executed=False)
 write(U/'PLAN.json',plan)
 text='''현재 판단과 다음 학습 계획 — 2026-11-09

결론
학습 실행은 여러 번 끝났지만, 목표로 한 세부 피드백 기능의 학습 범위와 자동 영상 평가 검증은 아직 완성되지 않았다. 단순히 학습 횟수를 늘릴 문제가 아니다. 아직 출력/정답이 없는 특징, 정답 불균형, 자동 구간 선택과 모델 연결 문제를 함께 해결해야 한다.

현재 구현을 확인한 사실
1. 실제 연구 웹은RTMW 관절흐름→시간합성곱+GRU→손특징/동작구간→설명별 비교→DTW재생이다. 현재SignRep/RGB신경망으로영상을읽는경로가아니다. 공개RGB는 사람이볼수있는정답을확인할때사용한다.
2. 손특징 출력7개는 검지/긴손가락펴기·닫기/양손높이 등 일부다. 각손의손끝모으기 전용 출력과검증된감독정답은없다. 문장전체를자연어로이해하는LLM이아니며 가이드키워드·단계ID와시각정답을연결한다.
3. 최근추가학습입력(ym)에서왼손펴기양성은FIT7시점(ID02/05/07),DEV1시점(ID13)이다. 7영상이나7독립시도가아니라프레임별학습정답7개다. 추가직접RGB검토에서는이양성을보강하지않았다. 모든7출력을충분히배웠다고할수없다.
4. 최근움직임32구간26→31개정답은진전이지만 사람이시점을준시험이다. 자동구간복합오류사례에서는수정구간49~80보다앞선47~78을골랐고,관계값이약-0.17에서2.96으로뛴뒤감소했다. 합성역순연결의불연속과구간변화가함께있다. 이것만으로모델용량이부족하다거나실제학습자에서동일비율로실패한다고단정하지않는다.

1단계: 가르칠 정답과 실패 원인을 함께 정리
현재 공개4단어와 이미보유한추가표현에서 손펴짐/굽힘/손끝모임/방향/순서의필요한양성·대조사례를선별한다. 모든영상모든프레임을주석달지않고 주요상태,변화,유지,손내림을구분할구간을묶어서검토한다. 손날에서보이는굽힘은학습가능하며엄지접촉미확인때문에함께버리지않는다. 각필요특징별최소3FIT기록ID와가능하면2개이상표현을확보하는것을착수조건으로삼되성능보장조건으로오해하지않는다.
동시에저장공개시험으로 사람이확인한구간/자동구간 × 정상/손모양만변경/방향만변경/복합을나눠본다. 합성연결이만든순간점프와실제주요동작을분리해구간선택문제인지판정부문제인지확인한다.

2단계: 보이는 정보가 실제 모델 입력에도 있는지 확인
RGB에서는굽힘이보이지만RTMW손가락좌표가이를구분못하는경우와,좌표는좋은데학습정답이부족한경우를구분한다. 전자는관측보완/부분보류,후자는감독정답보강으로처리한다. 없어진좌표를단어정답에맞춰채우지않는다.
FIT기록ID내분할에서학습조건을고르고,기존DEV는회귀검사로유지한다. 별도공개자료는중복SHA와기존노출이력을확인한후최종시험용으로분리한다. 새독립자료가없으면개발결과라고명시한다. 같은DEV점수에맞춘문턱/가중치반복조정은하지않는다.

3단계: 기존 신경망을 목적에 맞게 지도학습·미세조정
새거대모델을찾는대신 현재시간합성곱+GRU를기반으로한다. 손특징과몸기준이동방향을분리해서출력하고전체영상에서주요동작시작/진행/완료를배운다. ‘손내리며굽힘’을‘주요동작중굽힘’과같게학습시키지않는다. 몸통동결+새출력부학습과낮은학습률의전체연결미세조정2조건을고정비교한다.
기존몸통가중치를바꾸면그특징을받는웹단계/차이판정부도같은특징으로학습·검증한다. 새몸통과옛판정부를그대로조합하지않는다. 기존에맞추던손모양/순서를유지하는학습항목도넣는다. 목표단어는무엇을확인할지선택하고관측답을미리주지않는다. DTW가주요동작누락을성공으로채우지않게판정근거와재생정렬을분리한다.

4단계: 전체 영상에서 검증하고 통과한 경우에만 웹 적용
정상속도차이·작은몸흔들림·교차·정상손내림과 손모양오류/역방향/복합/짧은주요동작생략/마지막모양누락/관측부족을함께검사한다. 구간을미리준점수와자동으로구간을찾은점수는따로보고한다. 정상동작의오경고가늘지않고,기존에잡던중요오류를놓치지않으며,보류만늘리지않고목표오류를더잡아야한다. 오류가난시간과설명의해당부분이제대로연결되는지도확인한다.
촬영중RTMW추출재사용을유지하고종료후전영상재검출을추가하지않는다. 좌표준비후계산시간과전체재생시험의종료~화면결과중앙값/상위지연을분리한다. 약3초이내응답목표는실측으로확인하고0.043ms판정부시간을전체시간처럼표현하지않는다.
기존C·T3-F/현재TCN은유지한다. 사용자촬영은학습/튜닝/후보선택금지. 사람의실제잘못된수어정답이없는조건은합성기능검사라고명시하며교육적교정완성이라고하지않는다. 일부특징정답이부족하면그특징만분리하고나머지학습을무기한중단하지않는다.

이번에 실행한 범위
현재코드읽기,저장학습정답재집계,저장실패경로확인,조건별대응계획작성만했다. 새학습·성능재평가·웹교체·개인자료접근·다운로드없음. 같은Codex의기록/코드검토이며여러독립전문가검증을받았다는뜻이아니다.
'''
 (U/'PLAN_KO.txt').write_text(text,encoding='utf-8')
 note='''### 2026-11-09 피드백 학습 완성도 감사와 다음4단계 계획

사용자요청은구체적검증·검토후계획수립. word100_feedback_learning_plan_20261109 완료(계획만,새학습/추론/웹교체0). 코드확인:RTMW관절전용Conv+GRU7속성+별도단계/차이출력이며현재SignRep/RGB/LLM추론없음. 전체자연어이해나모든수형학습완성아님. 최근추가학습묶음왼손펴기양성FIT7프레임(ID02/05/07)/DEV1(ID13),추가직접주석양성0확인. 최근자동복합실패는개입49~80보다앞선구간47~78·합성연결급변이포함되어구간/판정부/개입인공성을분리해야함;모델단독원인확정아님.
계획1공통보이는특징정답보강+저장실패요인분리,2RGB정답/RTMW입력정보·학습평가분리검증,3현Conv+GRU의새출력부/낮은학습률연결미세조정2조건,4전체영상자동구간/정상·단일·복합오류/누락·관측부족·속도검사후연구웹적용. 손끝정확접촉부족이보이는굽힘학습전체를막지않음. 몸통변경시의존단계출력부재학습필수. 학습조건은FIT내선택,기존DEV회귀전용,추가공개봉인자료가능성은노출·중복감사후결정. 세부근거AUDIT.json/PLAN.json/PLAN_KO.txt. 기존수치/8814/모델불변,사용자자료접근0.
'''
 nxt='''### 2026-11-09 검토한 피드백 학습계획의 다음 작업

다음은PLAN1~2:이미보유한공개FIT에서손별보이는펴짐/굽힘/모임/변화정답의부족한양음성과수행자·표현다양성을확보하고,저장자료로검토구간대자동구간·단일대복합·합성경계영향을나누어확인한다. 관측정보부족과학습정답부족을구분한후학습조건을고정한다. 보이는손굽힘을엄지/전손끝접촉정답확보까지보류하지않음. ‘손내림중굽힘’을핵심모양변화로대체금지. 이후학습은현재Conv+GRU와연결된단계/차이출력부의일관된묶음으로진행하고새거대모델도입없음. 기존DEV반복튜닝/개인학습튜닝선택/자동카메라금지,C·T3F/현재TCN유지. 현재는계획완료이며새학습없음.
'''
 p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';s=p.read_text('utf-8-sig')
 for n,b in [(8,note),(9,nxt)]:
  m=re.search(r'^## '+str(n)+r'\. .*$',s,re.M);assert m;s=s[:m.end()]+'\n\n'+b+s[m.end():]
 p.write_text(s,encoding='utf-8');(ROOT/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+note+'\n'+nxt+'\n최근실행완료는word100_motion_components_20261109:움직임지정구간31/32,복합오류회귀로새방향모델미적용.8814는기존TCN/Conv+GRU+공통확인범위보완유지.이전실행재학습/재채점금지.\n',encoding='utf-8')
 p=O/'STATE.json';s=read(p);s={U.name:dict(status='PLAN_REVIEWED_NOT_TRAINED',plan=str(U/'PLAN_KO.txt'),audit=str(U/'AUDIT.json'),verification=str(U/'VERIFIED.json'),last_executed_unit=M.name,active=None,next=nxt.split('\n\n',1)[1].strip()),**s};s['current_stage']='FEEDBACK_LEARNING_PLAN_REVIEWED_NO_NEW_TRAINING';s['updated_date_kst']='2026-11-09';s['updated_at']=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat();write(p,s)
 for p,h in observed['source_pins'].items():assert sha(p)==h;checks+=1
 assert len(plan['steps'])==4 and all(q['gate'] for q in plan['steps']);checks+=1
 assert read(O/'STATE.json')[U.name]['status']=='PLAN_REVIEWED_NOT_TRAINED';checks+=1
 for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md']:assert U.name in p.read_text('utf-8');checks+=1
 write(U/'VERIFIED.json',dict(passed=True,checks=checks,scope='Existing input hashes/counts/code/saved failure path and plan constraints; no new accuracy or model inference',same_operator=True,plan_sha256=sha(U/'PLAN.json'),audit_sha256=sha(U/'AUDIT.json'),runtime_source_unchanged=True))
 print('PLAN_AUDITED',checks)
if __name__=='__main__':main()
