"""Close this analysis unit and synchronize handoff; no runtime/model changes."""
import hashlib
import json
from pathlib import Path
import shutil
from datetime import datetime

U=Path(__file__).resolve().parent
OPS=U.parent
ROOT=OPS.parents[2]
def read(p): return json.loads(Path(p).read_text('utf-8-sig'))
def write(p,d): Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    assert not (U/'COMPLETION.json').exists()
    a=read(U/'ANALYSIS.json'); v=read(U/'INDEPENDENT_CHECK.json'); b=read(U/'BUDGET.json')
    assert v['status']=='PASS'
    now=datetime.now().astimezone().isoformat()
    visual=dict(status='REVIEWED_WITH_SEMANTIC_LIMITATIONS',reviewer='Codex visual inspection of four contact sheets, seven timestamps per clip, eight training clips; no sign-language expert',
        views='Full frame thumbnails; not metric hand-angle measurements and not continuous human playback.',
        cases=[
            dict(code='WORD0077',names=['배추국','배춧국'],observation='Both sampled sequences show a hand near the face followed by a two-hand gesture around the trunk; timing/hand postures differ.',decision='Spelling-variant hypothesis, not certified equivalent form. Keep on review queue; no automatic mapping.'),
            dict(code='WORD0085',names=['된장찌개','돼지고기'],observation='Both sampled sequences show a movement near the opposite forearm followed by a two-hand gesture. Gross similarity despite conflicting words does not identify the correct gloss.',decision='Possible annotation conflict. Neither majority label nor sampled appearance establishes which word; keep unresolved.'),
            dict(code='WORD0111',names=['유리그릇','백설기'],observation='Both samples show a hand approaching the other hand, then palms separated around the trunk. These sparse frames do not prove full sequence or meaning equivalence.',decision='Mismatch confirmed in metadata, gloss correctness unresolved; do not relabel REAL08 by majority.'),
            dict(code=['WORD1185','WORD1194'],names=['알다','알다1'],observation='Same training signer REAL01: WORD1185 shows a hand contacting chest and moving down; WORD1194 shows a hand near mouth/chin. Clearly different visible places of articulation in sampled frames.',decision='Do not merge just because Korean label matches or remove numeric suffixes. Meaning/variant relationship needs lexical/source adjudication.')],
        approved_semantic_merges=0,approved_spelling_corrections=0,
        safe_format_overlay='Outer whitespace only, 14 training annotations; raw originals preserved.')
    write(U/'VISUAL_REVIEW.json',visual)
    temporal=dict(status='PREPARATION_CONTRACT_FIXED_NOT_MODEL_IMPLEMENTED',version='word_core_context_v1',
        purpose='Recognize the signing action without requiring every user to reproduce the dataset starting/ending rest pose.',
        preserve='Full decoded native-PTS observation timelines, original masks, raw intervals and raw video ZIPs. No camera or source edits.',
        annotation='One training core interval per candidate, weak temporal supervision rather than verified semantic truth.',
        core_definition='All consecutive frames whose timestamps fall in the annotation interval, including stationary holds and internal missing detections; not motion peaks/keypoints only.',
        short_view='Core interval plus 0.30 seconds before/after, clipped to media boundary. This is a preregistered initial margin, not a proven optimum. Outside annotation is not labelled useless.',
        proposed_control='Same fresh AIHub-only TCN280, whole observed-hand span, ordinary masked temporal mean. Padding mask remains separate from observation masks.',
        proposed_candidate='Same TCN280 and optimizer/data/schedule: for each clip alternate whole-span and short-view across epochs, clip parity from stable SHA256. One presentation per clip per epoch in both arms. Short view uses its own normalization and dt first row=0, then original timestamp differences; intersect with normal input support rules.',
        candidate_rationale='Reduce dependence on long rest periods, retain full-flow examples, preserve static handshape and transitions near annotated boundaries. Does not add an automatic semantic boundary detector or guarantee higher accuracy.',
        inference='Same existing observation-quality gates and active-hand-span adapter in both arms. No ground-truth word or core interval supplied for main development/final scores. No manual core crop for headline result.',
        web_entry='Hands absent before first valid observation are not proof of camera failure. Existing outer active-span trim may remove leading/trailing fully absent hands; retain any internal missing interval and original elapsed time. Do not demand a particular starting pose.',
        evaluation='Full recorded/observed-span main score; oracle core crop diagnostic only, labelled separately. Common fixed vocabulary and preprocessing; report top1/macro/signer metrics, failed input coverage, paired improvements/regressions. Do not silently remove failed clips from accuracy denominator.',
        boundary_check='Before bulk extraction, inspect first300 training-only clips and fixed boundary contact sheets; if annotation timing mismatch is systematic pause to correct input contract before any model score, not by tuning on final results.',
        preparation_now='Saved full/core/context time ranges for22388 candidate clips; reused12 observations to verify634 core rows all retained within850 context rows; no features or models changed.',
        deferred=['Learned attention/temporal localization if justified by this simple controlled comparison','RGB fusion comparison after common skeleton baseline','No sparse frame dropping/interpolated hand synthesis in this unit'],
        training_contract_pending=['Training-only bounded compute benchmark','Final class/label map and unresolved evaluation-label accounting','Exposure audit of proposed final-test signer groups','Exact optimizer/epoch budget fixed before any new development comparison'])
    write(U/'TEMPORAL_CONTRACT.json',temporal)
    split=dict(status='CURRENT_ROLES_UNCHANGED',roles={'train':'01-09','development':'11-13','reserved':'14-16','official_validation_reserved':'17-18','unassigned':'10'},
        why='Development people guide method/epoch selection without gradient updates. Final test people must not guide choices. Holding out clips from the same learned signer is not an unseen-signer test.',
        caveat='Reserved is a policy label, not proof of zero historic exposure. Audit prior arrays/scores before calling any group untouched. Dataset official Validation name differs from project role.',
        use_more_data='After method and training duration are fixed, train+development refit can use12 people while preserving separate final test. This is a proposed later model, not done here; its development score is no longer independent.',
        signer10='Existing unassigned role retained; no evidence in current artifacts establishes this performer is unusable. Original exclusion rationale not established by this unit. Exposure/identity audit before changing role.',
        alternatives='Signer-group cross-validation uses data more broadly but multiplies training cost; not necessary to launch before a first fixed baseline.',
        correction='8/9 refers to valid per-code metadata/video links, not8 correct performers versus1 poor performer. Seventy problematic training-folder rows include55 filename/folder-ID conflicts beyond numeric filename IDs01-09.',
        eight_person_policy='Use eligible8 rows of label-consistent noncolliding codes; do not reconstruct missing ninth data or change quality checks. Revised metadata candidate list2494/22388 includes58 eight-person codes.')
    write(U/'SPLIT_POLICY.json',split)
    research=dict(checked='2026-10-13',sources=[
        dict(url='https://scikit-learn.org/stable/modules/cross_validation.html',finding='Separate method selection from final testing; group-based splits prevent subject overlap. Repeated method selection can overfit validation.'),
        dict(url='https://arxiv.org/abs/2010.04002',title='Watch, read and lookup: learning to spot signs from multiple supervisors',finding='Sign spotting/localization is distinct from treating an entire video as an isolated sign; dictionary/continuous examples can differ.',limit='BSL research, not validation of our0.30s margin or TCN performance.')],
        local_inference='Core+context/full-view paired training is a simple testable project hypothesis, not a claim that cited papers prove this recipe.',
        no_claim='No new classification accuracy or real-user generalization measured in this unit.')
    write(U/'RESEARCH_NOTES.json',research)
    report=f'''AI Hub 단어 학습: 사람 분리·이름 충돌·시간 구간·처리 예산
작성 {now} / Codex 로컬 분석·작성, 별도 SQL·원본 재구성 검산 PASS

1. 개발용 사람을 왜 따로 두나
학습용은 모델의 값을 바꾸는 문제집, 개발용은 학습 방법/학습 횟수를 고르는 모의시험, 최종 시험용은 선택을 모두 끝내고 보는 마지막 시험이다.
예시: 학습 사람에게99%인 방법이 새 사람에게60%, 학습 사람에게96%인 방법이 새 사람에게85%일 수 있다. 뒤 방법을 고르기 위해 중간 점검도 학습하지 않은 사람이 필요하다. 이 수치는 설명용 가상 예시다.
같은 사람의 다른 영상만 빼두면 손 크기·동작 습관에는 이미 익숙할 수 있어 사람 단위로 나눈다. 개발 점수를 보면서 방법을 계속 고르면 개발용에도 간접적으로 맞출 수 있어 마지막 시험용을 별도로 둔다.
현재 역할:01~09 학습,11~13 개발,14~16 보류,17~18 공식Validation도학습제외,10미배정. 9/3/5/1이 최적 비율이거나 반드시 지킬 보편 법칙은 아니다. 이번에 역할을 변경하지 않았다.
방법/횟수 확정 후 학습+개발12명으로 최종 재학습하는 방안은 가능하다. 이때 개발 점수를 새 사람 시험이라고 보고하지 않으며 별도 최종시험을 남긴다. 보류 그룹의 과거 노출 감사도 먼저 해야 한다. ID10의 미배정은 불량 판정이 아니며 이번에 최초 사유를 확정하지 못했다.

2. 8명과9명의 의미 및 복구
9명 중1명이 부정확한 사람이었다는 뜻이 아니다. 일부 단어에서 영상과 주석의 연결이 불완전하다. 학습 폴더01~09의 문제70행 중 폴더02/파일이름REAL25처럼 식별자가 충돌한 경우도 있다. 폴더나 다수결만 믿고 이름을 바꾸지 않는다.
기존 후보2427개는9명전부연결이라는 보수적 규칙을 썼다. 8명도 정상 학습 표본이므로 정상8행을 살리고, 표기 앞뒤 줄바꿈/공백만 파생 목록에서 정리했다.
수정 후보2494코드·22388정면 영상 =9명완비2436코드 +8명정상58코드. 기존 대비67코드/545영상추가. 58코드는8명자료복구,9코드는줄바꿈형식정리로포함됐다. 총14주석의 앞뒤공백을 파생표기에서만 제거했다.
나머지506코드는 이름/표현 검토 큐다. 후보는 의미클래스 확정이나 새 인식률이 아니다. 원본ZIP/주석/이전2427명세는 그대로 보존한다. 개발/보류자료를 빼거나 변조해도 이 후보 결정은 같다는 것을 검산했다.

3. 실제 이름·표현 충돌 검토
교통법교육: 한 주석 뒤 줄바꿈2개. 내용 단어를 바꾸지 않는 형식 정리 가능.
배추국/배춧국(WORD0077): 철자 차이 후보. 두 영상의 얼굴 근처→몸통 앞 양손 흐름은 비슷해 보였으나 수형 전체의 동등성을 확정하지 않았다.
된장찌개/돼지고기(WORD0085): 같은 코드 안에 다른 단어 표기. 일부 영상의 큰 흐름이 비슷해 보여도 어떤 이름이 옳은지 영상만으로 결정할 수 없어 유지보류.
유리그릇/백설기(WORD0111): REAL08만백설기. 큰 양손 흐름은 비슷해 보이나 다수결로유리그릇으로수정하지 않았다.
알다(WORD1185/1194): 같은REAL01에서 한쪽은가슴, 다른쪽은입·턱 근처동작이보인다. 이름만으로두코드를합치거나알다1의숫자를지우지않는다. 뜻/변이관계는별도확인대상이다.
8개학습영상만완전디코딩해각7시점총56장면의연속사진4장을검토했다. 원본선택멤버CRC확인. 수어전문가검수·연속재생에의한의미판정은아니다. case_1~4.jpg와VISUAL_REVIEW.json에근거보존.

4. 핵심 동작 중심으로 바꾸는 방향
현재TCN은정답영상과매프레임일대일비교하지않는다. 시계열특징을학습하고마지막에유효시간특징을평균한다. DTW의시간정렬비교와다르다. 그래서길게남는준비/정지구간이결과에영향을줄여지가있다.
새후보22388영상총길이26.40시간(주석기준), 핵심구간은전체의{a['core_fraction']*100:.2f}%, 핵심앞뒤0.30초를포함하면{a['context_fraction']*100:.2f}%. 나머지모두쓸모없다는뜻은아니다. 들기/접촉직전/내리는과정도뜻을구별하는맥락일수있다.
기존12관절캐시재사용:전체1444프레임,핵심634프레임,앞뒤0.30초포함850프레임. 핵심634개전부포함/중간프레임삭제0/시각보존검산. 손이보이는구간은12개모두전체라손관측만으로핵심을찾을수없었다.
확정한준비방식:
- 원본과관절좌표는영상전체시간을보존하고,핵심/문맥구간인덱스만따로저장한다. 움직임이작은정적손모양도핵심에포함한다.
- 첫비교후보는TCN구조와280입력을유지하고학습구간만바꾼다. 각영상에대해전체흐름과핵심+앞뒤0.30초를번갈아보여준다. 기준과후보는영상당같은학습노출횟수로맞춘다.
- 웹/주평가는둘다정답구간을알려주지않는실제입력으로한다. 핵심만잘라낸정답률은별도진단일뿐대표성능으로쓰지않는다.
- 사용자의손이처음화면밖이어도특정준비자세를강요하지않는다. 앞뒤완전미관측구간은기존웹관측규칙으로처리하고동작중누락은시간/마스크를보존한다. 없는손을임의로채우거나빠른부분만추리는변경은없다.
0.30초는처음비교할고정설계값이지최적값이아니다. 잘못된주석경계는첫입력점검에서살피고실험전에정정한다. 새자동핵심검출기/attention은이번에추가하지않는다. 먼저이간단한시간정책하나의효과를분리해볼계획이다. 모델학습/운영반영은아직없다.

5. 처리 예산
현재W여유약{b['disk_snapshot']['W:/']['free']/1e12:.2f}TB. 수정후보학습영상원본크기약102.90GB이나ZIP을보존하고정면영상을모두중복풀지않는다.
12개샘플의실측추출속도로선형계산시학습분관절추출약34.4시간,좌표+280채널캐시약5.65GB. 30fps가정이며다른사람/긴영상/디스크상태에따라달라진다. 개발·최종평가분추출과모델학습시간은별도다.
모든프레임손RGB를저장하면학습분만약382GB/기존파이프라인약57.8시간으로추정된다. 지금전량생성하지않는다.
다음실행단위상한:학습쪽정면300개또는2시간중먼저도달시점,추출작업1개. 학습용초기자료만으로길이/사람범위를고르게구성하고성과에따른선별은하지않는다. 완료캐시는이어받고처음부터반복하지않는다.
전체좌표캐시상한20GB,임시원본캐시2GB,RGB임시캐시2GB,외장여유500GB이상유지. 상한은점검중단기준이며원본삭제지시가아니다. 300개실측으로전체시간예측을갱신한다.
학습횟수/옵티마이저/최종실행시간은입력검증및학습용계산시간측정후첫개발점수를보기전에동결한다. 현재단위에서는학습0회다.

6. 완료와 다음 작업
완료:표기/8명자료파생목록2494개,구체적충돌8영상검토,22388개시간구간표,12캐시시간보존검산,처리예산과비교방향문서화.
미완료:506코드의수어의미/변이확정,전영상무결성/관절추출,개발측표기충돌처리,최종시험노출감사,새학습/새정확도.
다음단일작업:이계약의재개가능한300개정면입력준비기와고정목록을만들어원본/시각/주석경계/관측마스크를검사하고실제처리속도를측정한다. 첫12개캐시는중복추출하지않는다. 모델학습은입력과분할계약검증후별도시작한다.

근거:INDEPENDENT_CHECK.json / TRAINING_CANDIDATES_V2.jsonl / TEMPORAL_CONTRACT.json / BUDGET.json / SPLIT_POLICY.json
공식분할설명:https://scikit-learn.org/stable/modules/cross_validation.html
수어시간위치연구(우리설계효과보장은아님):https://arxiv.org/abs/2010.04002
'''
    (U/'RESULT_KO.txt').write_text(report,'utf-8')
    before=U/'before'; before.mkdir()
    paths=[ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',OPS/'STATE.json']
    for p in paths: shutil.copy2(p,before/p.name)
    completion=dict(status='COMPLETE_VERIFIED_PLANNING_AND_INPUT_AUDIT_NOT_NEW_ACCURACY',at_kst=now,
        candidates=2494,training_clips=22388,remaining_semantic_review_codes=506,
        verified='INDEPENDENT_CHECK.json',video_review='VISUAL_REVIEW.json',temporal='TEMPORAL_CONTRACT.json',budget='BUDGET.json',
        author='Codex',verification='Codex separate SQL/source reconstruction; not separate agent or sign expert',
        new_model_training=0,new_detector_inference=0,production_changes=0,source_changes=0,ongoing_processes=[],
        next='Bounded300 training-clip preparation and throughput/boundary audit; no final-test/model changes.',
        source_pins=a['source_hashes'])
    write(U/'COMPLETION.json',completion)
    state=read(OPS/'STATE.json')
    state['updated_date_kst']=now
    state['current_stage']='WORD_LABEL_TIME_CONTRACT_VERIFIED_BOUNDED_PREPARATION_NEXT'
    state['word_label_time_contract_20261013']=dict(completion,unit='ops/'+U.name,report='ops/'+U.name+'/RESULT_KO.txt')
    state['next_single_action']='동결 계약의 학습쪽 정면300개 재개가능 입력준비·시각/주석경계/마스크·처리속도 검산. 기존12캐시 재사용. 최종시험·학습·운영변경 없이 입력 준비를 먼저 완료.'
    write(OPS/'STATE.json',state)
    context=paths[0].read_text('utf-8-sig')
    def insert_section(text,num,block):
        marker=f'## {num}. '
        pos=text.index('\n',text.index(marker))+1
        return text[:pos]+'\n'+block+'\n'+text[pos:]
    block8='''### 2026-10-13 이름·시간구간·예산 검토 완료 — 8명자료 복구 포함2,494 후보
`ops/word_label_time_contract_20261013` RESULT_KO.txt / COMPLETION / INDEPENDENT_CHECK PASS.
기존2427목록보존,앞뒤공백만파생정리14행·정상8명자료활용→2494코드22388학습영상(9명2436/8명58),67코드추가·기존후보제거0. 의미검토506남음. 개발/보류무관선정SQL검산.
학습8영상56시점시각검토:같은알다1185/1194의가슴vs입·턱위치차이확인;0077철자,0085돼지고기/된장찌개,0111유리그릇/백설기는의미미확정·자동병합/다수결수정0. 수어전문가검수아님.
전체원관절시간보존,핵심+앞뒤0.30초파생구간계약. 기존12캐시1444중핵심634/문맥850·핵심보존/중간행삭제0. 신규2494학습시간총26.40h,핵심37.87%/문맥52.00%. 웹은정답구간미제공,동일TCN의전체vs전체/문맥교대학습비교계획이며미구현·미학습.
예산다음300개/2h·추출1작업,전체좌표20GB/임시영상2GB/RGB2GB,전량RGB생성없음. 학습분추출34.4h/좌표5.65GB추정(12개30fps외삽),학습시간별도미측정. 분할유지·추후train+dev재학습가능성기록,보류노출감사/ID10사유확인남음.
Codex분석/별도검산완료;원본·기존모델·웹·Git변경0,신규MediaPipe0,학습0. 실행종료.
'''
    block9='''### 2026-10-13 다음 — 동결 시간계약의 정면300개 입력준비
`ops/word_label_time_contract_20261013`의TRAINING_CANDIDATES_V2/TEMPORAL_CONTRACT/BUDGET을따른다.
학습01~09만으로300개고정목록·재개가능준비기구성,기존12캐시재사용. W원본ZIP보존,전체시각/관측마스크와핵심/문맥인덱스저장. 주석경계표본확인·동일웹좌표규격·처리량/저장검산후대량예측갱신.
2h또는300개상한도달시완료분보존. 모델학습·자동최종시험·운영기본변경시작금지. 정확한학습계수/횟수는입력검증·노출감사·학습용계산시간측정후개발점수전동결. 506의미충돌은자동라벨변경하지않는다.
'''
    context=insert_section(context,8,block8)
    context=insert_section(context,9,block9)
    paths[0].write_text(context,'utf-8')
    handoff=f'''# WordsInSilence 최신 인수인계
갱신:{now}. 시작:AGENTS.md/PROJECT_CONTEXT 최신§3·§8·§9/WORKFLOW_HANDOFF.md.

## 마지막 완료
`ops/word_label_time_contract_20261013` RESULT_KO.txt/COMPLETION/INDEPENDENT_CHECK PASS.
사람별개발/최종시험분리설명,8명정상자료재활용·앞뒤공백14행파생정리→2494코드22388영상(2436×9+58×8). 기존2427불변보존. 67코드추가,의미충돌506보류. 개발/보류성적·라벨로선택안함.
8학습영상56시점/4사진표시각검토. 알다1185vs1194가슴vs입·턱차이;배추국/배춧국,된장찌개/돼지고기,유리그릇/백설기미확정. 다수결·숫자제거·의미병합0. 전문가검수아님.
시간전체보존,핵심+앞뒤0.30초인덱스저장.12기존캐시1444/핵심634/문맥850검산. 손관측=핵심아님. 전체/문맥교대TCN학습과전체입력평가의비교계획고정,모델변경/학습미실행.
学習22388합26.40시간,핵심37.87%/문맥52.00%. 좌표추출34.4시간/5.65GB예상은12영상30fps외삽으로확정소요시간아님. RGB전량캐시382GB추정,실행안함.
Codex구현·별도SQL검산. 검사기첫이름파싱오류및검산분모정정기록보존,최종PASS. 원본/운영/Git변경0·MediaPipe신규0·분류추론/학습0. 진행프로세스없음.

## 다음 단일 작업
TEMPORAL_CONTRACT.json/BUDGET.json/TRAINING_CANDIDATES_V2.jsonl 따라학습쪽정면300개고정목록·재개가능준비기구성및입력검증.
추출1작업,300개또는2h먼저도달시마감. W좌표캐시전체20GB,임시영상2GB/RGB2GB,여유500GB. 전체정면중복압축해제/전량RGB저장금지. 기존12MediaPipe캐시재사용.
주석경계/시간/마스크/웹입력일치·실측처리속도검산. 새학습계수/횟수/학습시간은별도동결전미확정. 모델/최종시험은아직실행하지않음.

## 보존과 분할
01~09학습/11~13개발/14~16보류/17~18공식Validation학습제외/10미배정유지. 9명이최적/충분하다는보장은없음. 후속method확정후train+dev12명최종재학습가능,기존dev는독립평가아님. 보류과거노출감사와10최초미배정사유확인남음.
8/9는단어별연결완전성. 학습폴더문제70행:ID폴더2에56행(파일25등충돌포함),나머지다른폴더;불량수어촬영자판정아님.
NIKL18/KSL은노출된외부참고,새AIHub-only학습제외. 사용자/멘토자료학습·튜닝·선택금지. TCN832/웹기본/원본ZIP보존.

## 반복 금지
word_vocabulary_manifest_20261013:3000코드/9명2930·8명70/이전2427후보검산완료. 이번v2로복구하되이전결과수정안함.
word_pilot12_inputs_20261013:REAL01정면12·1444프레임/2801손사진87누락/105.39초·관절62.78초. W캐시재사용.
word_dataset_plan_20261013:33ZIP1.181TB/270000MP4/3000코드×18ID×5시점. F54000·248.59GB,정상연결53917. 전영상디코딩/CRC완료아님.
'''.replace('学習','학습')
    paths[1].write_text(handoff,'utf-8')
    assert read(OPS/'STATE.json')['current_stage']=='WORD_LABEL_TIME_CONTRACT_VERIFIED_BOUNDED_PREPARATION_NEXT'
    assert all(sha(p)==h for p,h in a['source_hashes'].items())
    files={p.name:sha(p) for p in U.iterdir() if p.is_file()}
    write(U/'COMPLETION_CHECK.json',dict(status='PASS',root_documents_synchronized=True,source_pins_preserved=True,unit_file_hashes=files,at_kst=now))
    print(json.dumps(completion,ensure_ascii=False))

if __name__=='__main__': main()
