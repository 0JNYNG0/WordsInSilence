from common import *
from html import escape as E
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

a=read(U/'ANALYSIS.json');v=read(U/'INDEPENDENT.json');s=read(U/'SCOPE.json')
assert v['analysis_sha256']==sha(U/'ANALYSIS.json') and v['status']=='PASS'
rows=read(U/'GUARDED_GEOMETRY.json')['rows']
font_manager.fontManager.addfont('C:/Windows/Fonts/malgun.ttf');plt.rcParams['font.family']='Malgun Gothic';plt.rcParams['axes.unicode_minus']=False
conditions=['other_signer_same_word','omit_middle','swap_first_two','repeat_middle']
names=['다른 사람 · 같은 단어\n799쌍','중간 1/3 삭제\n100쌍','앞 두 구간 교환\n100쌍','중간 1/3 반복\n100쌍']
fig,ax=plt.subplots(figsize=(10,4.2),layout='constrained')
data=[[r['metrics']['global_R']['ordered'] for r in rows if r['view']=='sensor' and r['condition']==c and r['metrics']['global_R']['ordered'] is not None] for c in conditions]
bp=ax.boxplot(data,tick_labels=names,patch_artist=True,showfliers=False)
for patch,color in zip(bp['boxes'],['#398d9d','#d89d4a','#d89d4a','#8196b0']):patch.set_facecolor(color);patch.set_alpha(.6)
ax.set_ylabel('순서에 맞춰 비교한 오른손 위치 차이\n(어깨 기준 정규화 좌표, 낮을수록 비슷)')
ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.set_title('영상 평균 차이만으로 동작 누락·순서 오류를 판정할 수 있을까?',pad=15)
fig.savefig(U/'distance_distribution.png',dpi=170);plt.close(fig)

notes={
 '수어':'두 손을 번갈아 움직이는 표본이 반복되어 보임. 반복 횟수를 프레임 반복과 혼동하지 말아야 함. 허용 반복 횟수는 미확정.',
 '남아':'가슴 앞 양손 접촉에 가까운 자세에서 양손을 얼굴 양옆으로 벌리는 변화가 보임. 위치 변화만으로 올바른 손 모양까지 확인한 것은 아님.',
 '냉커피':'양손 가슴 부근 → 입 부근 한손 → 가슴 앞 동작의 위치 전환이 보임. 반대손 내림이나 전환 중 화면 이탈을 곧바로 단계 누락이라 부르면 안 됨.',
 '후원':'머리 옆에 실제 손이 보이는 표본에서 저장 관절이 없는 순간이 확인됨. 영상 동작 부재와 관절 미검출은 다름.',
 '사육장':'가슴 앞 양손 동작 뒤 입 부근과 전방 쪽 한손 동작으로 바뀌는 표본이 보임. 균등 3등분 경계와 언어적 단계 경계는 다를 수 있음.',
 '어린이날':'얼굴 양옆에서 가슴 앞을 거쳐 다시 얼굴 양옆으로 가는 표본. 처음과 끝 자세가 비슷해도 중간 단계를 없애도 된다는 뜻은 아님.',
 '성남시수어통역센터':'초반 가슴 앞 양손, 입 부근 한손, 뒤쪽 양손/팔 접촉 표본이 보임. 전주시 표현과 공통된 후반 형태가 보이나 도시 구별의 정확한 경계·필수 손 모양은 미확정.',
 '전주시수어통역센터':'초반 양손에서 입 부근 한손, 뒤쪽 양손/팔 접촉으로 바뀌는 표본. 성남시와 공통 부분이 있어 보이지만 어느 구간이 도시 이름인지 글자만 보고 분할하지 않음.'}
reviews={r['label']:r for r in read(AUD/'VISUAL_REVIEW.json')['reference_reviews']}
manifest=read(AUD/'VISUAL_MANIFEST.json');frames={r['clip_id']:r['frames'] for r in manifest['samples'] if not r['risk']}
scene=[]
for label,note in notes.items():
 r=reviews[label];scene.append({'label':label,'clip_id':r['clip_id'],'sheet':str(AUD/r['sheet']),'row':r['row'],'sampled_frames':frames[r['clip_id']],
 'observable_note':note,'human_method':'Codex inspected the existing eight-frame contact-sheet row. Not continuous full-video or sign-language expert review.',
 'semantic_phase_boundaries':None,'mandatory_events':None,'allowed_repeat_count':None,'expert_reviewed':False})
write(U/'SCENE_REVIEW.json',{'status':'OBSERVABLE_SCENE_REVIEW_NOT_SEMANTIC_ANNOTATION','clips':8,'frames':64,'items':scene})

criteria_rows=[
 ('속도 차이','같은 필요한 동작들이 같은 순서로 유지되는지','초 단위 길이·프레임 수만으로 감점하지 않음. 시간 정렬 후 비교'),
 ('잠깐 멈춤','한 동작 상태가 오래 유지됐는지','동일 상태의 여러 관측은 새 동작 횟수로 세지 않음'),
 ('실제 반복','A→B→A→B처럼 움직임 주기가 더 생겼는지','단어별로 검토된 허용 반복만 인정. 규칙 미확정이면 판단 보류'),
 ('동작 누락','필수라고 확인된 동작이 충분히 보이는 영상에서 빠졌는지','영상 구간 삭제만으로 오류 정답을 만들지 않음. 필수 단계 검토 필요'),
 ('순서 변경','먼저 해야 한다고 확인된 두 동작의 순서가 뒤집혔는지','모든 순서 변화가 잘못된 수어라고 가정하지 않음. 선후 관계를 확인'),
 ('손 미검출','해당 단계에 필요한 손·손가락이 실제로 관측됐는지','안 보임은 안 함과 다름. 화면 밖·가림·좌우 불확실은 근거 부족'),
 ('준비·마무리','손을 들고 내리는 구간인지, 단어에 필요한 동작인지','시작·끝의 전체 손 관측을 일괄 요구하지 않고 필수 단계별로 확인')]
next_plan={'status':'CRITERIA_REVIEW_BEFORE_ADDITIONAL_TRAINING','training_authorized_now':False,'automatic_adoption':False,
 'next_single_action':'Review these criteria and prepare a public-video phase-label pilot: first mark observable event candidates without model predictions, then obtain competent KSL review of required/optional events, precedence and allowed cycles before assigning semantic error labels.',
 'steps':[{'step':1,'work':'Use100-word draft catalog and8reviewed visual examples. Assign visible event timestamps, active/support hands and detection uncertainty. Do not derive phases from equal thirds or Korean spelling.'},
 {'step':2,'work':'Independent KSL-competent review of necessary/optional phases and permissible repeat/speed variations; disagreements stay unknown. Keep normal original and real cross-signer variants.'},
 {'step':3,'work':'Freeze criterion labels and scorer before inspecting model scores. Measure normal false-alarm rate, reviewed omission/order sensitivity, and abstention separately with signer/source disjointness; all derivatives remain with their source video.'},
 {'step':4,'work':'Only after result review and separate approval, design a single auxiliary phase/order objective using reviewed public training labels; compare unchanged C/T3F and use no user footage.'}],
 'limits':['Current100word classifier does not recognize all untrained words. Shared evaluation structure can extend with new labels.','No new training/model inference/threshold selection this unit.','No guaranteed accuracy gain or complete signing correctness.','Existing ID14-16 unused holdout remains untouched.'],
 'failure_prevention':['Never force missing hand tozero difference.','Never choose classifier/head weights using tester footage.','No universal duration cutoff, fixed repeat count or Korean-label substring credit.','No synthetic edit called an incorrect sign without review.']}
write(U/'NEXT_PLAN.json',next_plan)
conclusion={'status':'COMPLETE_CRITERIA_TECHNICAL_VALIDATION_SEMANTIC_LABELS_PENDING','at':now(),'public_words':100,'public_clips':899,
 'pair_views':3398,'clock_controls':600,'dwell_controls':200,'observed_state_invariance_verified':True,'semantic_phase_contracts_reviewed':0,
 'semantic_error_detection_accuracy':None,'new_training_runs':0,'new_model_inference':0,'user_recordings_read':0,'heldout_raw_read':0,
 'existing_models_changed':False,'production_changed':False,'finding':'Geometry averages and identity alone cannot establish omission/order correctness; preserve observation provenance and reviewer-declared allowed cycles.',
 'first_probe_issue':'Local origin-only geometry allowed false zero distances; first result preserved and guarded same-side complete-landmark diagnostic separately verified.'}
write(U/'CONCLUSION.json',conclusion)
def tr(row):return '<tr>'+''.join('<td>'+E(str(x))+'</td>' for x in row)+'</tr>'
def table(head,rows):return '<div class="scroll"><table><thead>'+tr(head)+'</thead><tbody>'+''.join(tr(r) for r in rows)+'</tbody></table></div>'
ref_by_label={r['label']:r for r in s['references']}
videos=''.join(f'<article><h3>{E(label)}</h3><video controls preload="none" src="{E(Path(ref_by_label[label]["video_path"]).as_uri())}"></video><p>{E(notes[label])}</p></article>' for label in ['성남시수어통역센터','전주시수어통역센터'])
summary_table=[]
for axis,label in [('global_R','오른손 위치'),('local_R','오른손 로컬 형태'),('global_L','왼손 위치'),('local_L','왼손 로컬 형태')]:
 for cond,cname in [('omit_middle','중간 구간 삭제'),('swap_first_two','앞 두 구간 교환')]:
  item=next(x for x in a['overlap'] if x['view']=='sensor' and x['axis']==axis and x['condition']==cond)
  summary_table.append((label,cname,f"{item['edited_le_same_word_person_median']} / {item['eligible_words']}", '편집본 차이 ≤ 같은 단어 다른 사람 차이의 중앙값'))
html=f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>동작 단계·순서 평가 기준과 공개 자료 검증</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#edf3f5;color:#183348;font:17px/1.8 "Malgun Gothic",sans-serif}}main{{max-width:1120px;margin:auto;padding:32px 20px 80px}}header,section{{padding:32px;border-radius:20px;margin-bottom:22px;background:white;border:1px solid #d8e4e9}}header{{background:#133746;color:white}}h1{{font-size:34px;line-height:1.4;margin:10px 0}}h2{{font-size:25px;margin:0 0 18px}}h3{{font-size:19px;margin:12px 0}}p{{margin:10px 0}}.pill{{display:inline-block;background:#fde8bd;color:#765010;border-radius:20px;padding:5px 14px;font-weight:bold}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}}.card{{background:#eff5f7;padding:20px;border-radius:12px}}.num{{font-size:29px;color:#087b73;font-weight:bold}}.notice{{border-left:5px solid #d89b36;padding:14px 20px;background:#fff5e5;margin-top:18px}}.good{{border-left-color:#168679;background:#edf8f5}}table{{border-collapse:collapse;width:100%;font-size:15px}}td{{padding:13px 12px;border-bottom:1px solid #dbe5ea;vertical-align:top}}thead{{background:#edf4f7;font-weight:bold}}.scroll{{overflow:auto}}img{{max-width:100%;height:auto}}video{{width:100%;background:#101d27;aspect-ratio:16/9}}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}a{{color:#006c95}}details{{margin-top:15px}}summary{{cursor:pointer;font-weight:bold}}.flow{{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap}}.flow span{{background:#eef4f7;padding:12px 18px;border-radius:12px}}ol li,ul li{{margin:8px 0}}@media(max-width:650px){{main{{padding:14px 10px}}header,section{{padding:22px 18px}}h1{{font-size:27px}}.grid,.pair{{grid-template-columns:1fr}}td{{min-width:120px}}}}
</style></head><body><main>
<header><p>WORDS IN SILENCE · 공개 자료 평가 연구 · 2026.10.19</p><h1>‘무슨 단어인가’와<br>‘필요한 동작을 했는가’를 나눠 평가합니다.</h1><p>속도와 반복 때문에 정상 동작을 틀렸다고 하지 않으면서,<br>필수 동작의 누락·순서 변경을 확인하기 위한 기준입니다.</p><span class="pill">기준·기술 검증 완료 / 수어 단계 정답 검토 필요 / 추가 학습 없음</span></header>
<section><h2>이번에 확인한 범위</h2><div class="grid"><div class="card"><div class="num">100단어 · 899영상</div><p>공개 TRAIN 촬영자 9명<br>사용자·멘토 촬영 사용 없음</p></div><div class="card"><div class="num">3,398쌍</div><p>두 관측 조건의 비교<br>시간 변화·편집·사람 차이</p></div><div class="card"><div class="num">학습 0회</div><p>C·T3-F·웹·DTW 유지<br>새 모델 추론·임계값 조정 없음</p></div></div>
<p>기존100개 참고 영상과 나머지799개 같은 단어의 다른 촬영자를 비교했습니다. 같은 원본의 변형들이 포함되어 있어3,398개의 독립 영상이나 새 사용자 시험을 뜻하지 않습니다.</p><div class="notice">평가 도구가 계산 규칙대로 작동하는지는 검증했습니다. 하지만 어떤 단계가 수어의 의미상 필수인지 확인한 정답표는 아직 없으므로, ‘잘못된 수어를 몇% 찾아냈다’는 성능은 산출하지 않았습니다.</div></section>
<section><h2>평가 기준</h2>{table(['구분','확인할 것','처리 기준'],criteria_rows)}
<div class="flow"><span>어느 단어인가<br>기존 TCN</span><b>＋</b><span>필수 단계·순서를 보였는가<br>검토된 동작 구조</span><b>＋</b><span>판단할 만큼 보였는가<br>관측 근거</span></div><p>동작 구조와 맞더라도 모든 손 모양·방향·표정까지 올바르다고 단정하지 않습니다. 정보가 부족하거나 허용 변형의 기준이 없으면 확인 필요로 남깁니다.</p></section>
<section><h2>속도·멈춤·반복을 구분했습니다</h2>{table(['검증 항목','결과','해석'],[
('시계만0.7배/1.3배·구간별 변경','600비교에서 관측 상태·거리 불변','절대 시간 때문에 차이가 커지지 않음. 새 촬영속도에서 MediaPipe가 잘 잡는지는 별도'),
('같은 관측 프레임을 더 유지','200비교에서 기존 거리·보류 상태 유지','잠깐 멈춤/샘플 중복을 새 동작으로 세지 않음'),
('허용 반복 A→B를 여러 번 수행','구조검사 예제에서 허용·정상','이것은 검토된 반복 규칙을 입력한 합성 예제. 실제 수어 반복 횟수는 미확정'),
('관측이 없는 A→?→C','오류 대신 근거 부족','보이지 않는 B를 안 했다고 결론 내리지 않음')])}
<p>구조검사12개와 단계 이름을 바꾼300개 검사, 거리192개 별도 계산 및192개 작은 정렬 경로 전수 검산을 통과했습니다. 손이 없어 원래 계산하지 못한 항목은 시간 변경 후에도 보류됐습니다.</p><div class="notice">반복된 ‘같은 자세의 관측’과 ‘움직임 주기의 반복’은 다릅니다. A→B→A→B가 허용된 단어라면 정상일 수 있지만, 허용 규칙이 없는 단어에 임의로 적용하지 않습니다. <a href="https://aclanthology.org/L12-1057/">ASL 연구</a>에서도 반복이 언어적 역할을 가질 수 있음을 다룹니다. 이 논문을 한국수어의 개별 반복 규칙으로 대신 사용하지 않았습니다.</div></section>
<section><h2>평균 차이 하나로는 누락·순서를 판정하기 어렵습니다</h2><img src="distance_distribution.png" alt="같은 단어 사람 간 차이와 기계적 구간 편집 차이 분포"><p>주관측(sensor)에서 관측 가능한 오른손 위치의 비교입니다. 상자=중앙50%,가운데선=중앙값. 이상점은 그림에서만 생략했고 원자료에는 모두 남겼습니다. 중간1/3 삭제·순서 교환은 시간으로 만든 탐침이며, 실제로 잘못된 수어라는 정답은 아닙니다.</p>
<p><b>중간 구간을 삭제한 자기 영상이 다른 사람의 같은 단어 영상보다 더 비슷하게 보일 수 있었습니다.</b> 오른손 위치는100개 단어 모두, 순서 교환은100개 중80개에서 해당 편집본 차이가 같은 단어 사람 간 차이의 중앙값 이하였습니다. 이 점수에 공통 합격선을 넣는 것만으로는 충분하지 않습니다.</p>
<details><summary>좌우·위치·손 로컬 형태별 확인</summary>{table(['항목','탐침','해당 단어 수','비교 의미'],summary_table)}<p>왼손은완전한관절근거를비교할수있는67개단어만수치분모에포함했습니다.나머지33개를오류로세지않았습니다.몸기준위치와손로컬형태는별도단위이며합산정답점수가아닙니다.두관측조건전체값은 <a href="ANALYSIS.json">ANALYSIS.json</a>에있습니다.</p></details>
<div class="notice">중간 구간을 지운 경우와 같은 구간의 관절을 안 보이게 한 경우는, 관측된 부분의 거리만 계산하면 같은 값이 나왔습니다. 그래서 거리와 함께 ‘관측이 끊긴 구간’을 반드시 보존해야 합니다. 이것은 만들어 본 조건의 특성이며 실제 오류 발생률이 아닙니다.</div></section>
<section><h2>손목만 보고 손 모양이 같다고 하지 않도록 했습니다</h2><p>처음 만든 탐색용 거리에서는 손목의 로컬 좌표가 항상(0,0)이라는 점을 잘못 이용할 수 있었습니다. masked의 다른 사람 비교799쌍 중534쌍에서 손 모양 차이가0으로 나왔고, 직접 확인한 표본에서는 공통 손목 한 점만 대응시킨 것이 원인이었습니다.</p><p>첫 결과를 보존하고, 새 진단에서는 <b>좌우를 분리하고 로컬 손목을 제외한20개 관절의 근거</b>를 확인했습니다. 근거가 없으면0점으로 채우지 않습니다. 전체관절을요구한이번조건은보수적인진단용이며부분관측을사용할최적기준은아닙니다.</p><div class="notice good">이 문제는 이번에 만든 탐색용 평가 거리의 한계입니다. 기존 C·T3-F나 운영 DTW의 오류를 입증한 것이 아니며, 모델 입력·제품 기능은 변경하지 않았습니다.</div></section>
<section><h2>성남시와 전주시 사례는 어떻게 볼까?</h2><div class="pair">{videos}</div><p>후반에 비슷한 위치 변화가 보이므로 공통 부분의 영향이라는 해석은 가능한 가설입니다. 하지만 도시 이름을 구별하는 동작까지 맞혔다는 뜻은 아닙니다. 앞으로는 ‘공통 부분 수행’과 ‘구별 단계 수행’을 각각 확인해야 하며, 한글 명칭을 잘라 수어 단계로 간주하지 않습니다.</p><details><summary>공개8영상·64표본 관측 검토</summary>{table(['표현','관측 사실 / 유의점'],list(notes.items()))}<p>기존8프레임접촉시트를검토한결과이며전체연속영상의수어전문가검토가아닙니다.근거: <a href="SCENE_REVIEW.json">SCENE_REVIEW.json</a></p></details></section>
<section><h2>다음 작업: 학습보다 평가 정답부터</h2><ol><li><b>공개 영상의 관측 가능한 단계 후보 표시.</b> 어느 손이 어디에서 어떤 상태로 바뀌었는지, 가림은 언제였는지 기록합니다.100단어 검토표와8개 장면 예시를 준비했습니다.</li><li><b>필수·선택 단계와 반복 허용 범위 검토.</b> 수어를 아는 검토자가 선후 관계와 반복을 확인하고, 의견이 다르면 보류합니다.</li><li><b>평가 기준 고정.</b> 정상 동작을 잘못 지적하는 비율, 확인된 누락·순서 오류를 찾는 비율, 근거 부족으로 보류하는 비율을 나눠 봅니다. 같은 원본에서 만든 변형은 같은 자료 분할에 둡니다.</li><li><b>결과 확인 후 추가 학습 설계.</b> 확인된 공개 단계·순서 정보로 보조 학습을 설계하고 C·T3-F와 비교합니다. 지금은 학습하지 않았습니다.</li></ol><div class="notice">공통 평가 구조를 목표로 하지만, 모든 수어의 필수 단계·반복 규칙을 이미 확인한 것은 아닙니다. 사용자·멘토 촬영은 실사용 시험 전용으로 보존했습니다.</div></section>
<section><h2>검증 근거</h2><p><a href="CONCLUSION.json">결론</a> · <a href="INDEPENDENT.json">별도 수치 검산</a> · <a href="CONTRACT_TESTS.json">기준 사례 검사</a> · <a href="PHASE_REVIEW_CATALOG.json">100단어 검토표</a> · <a href="NEXT_PLAN.json">다음 계획</a> · <a href="FIRST_PROBE_FINDING.json">초기 탐색 문제 보존</a></p><p>저장13,592개 손·항목 지표를 점검했고 별도 스칼라 계산과의 최대 차이는 {v['max_numeric_error']:.2e}였습니다. 같은 Codex가 별도 계산으로 검산했으며 독립 에이전트나 수어 전문가의 검토는 아닙니다.</p><p>시간 정렬 배경: <a href="https://jeffe.cs.illinois.edu/teaching/compgeom/refs/Sakoe-Chiba-DTW.pdf">Sakoe–Chiba</a>. 수어 분할 배경: <a href="https://aclanthology.org/2023.findings-emnlp.846/">Linguistically Motivated Sign Language Segmentation</a>. 이론적 배경이며 본 프로젝트의 수어 오류 판정 정확도를 보증하지 않습니다.</p></section>
</main></body></html>'''
(U/'RESULT_KO.html').write_text(html,encoding='utf8')
(U/'RESULT_KO.txt').write_text('''동작 단계·순서 평가 기준 및 공개 자료 기술 검증 완료. C/T3-F·웹·DTW 유지, 추가학습0/새모델추론0/사용자·멘토촬영접근0.
100단어899TRAIN영상,100고정참고와799다른사람영상,2관측조건 총3398쌍. 속도시계변경600쌍·동일관측유지200쌍에서기존거리/보류상태불변. 실제프레임속도변경의검출안정성을검증한것은아님.
관측근거와필수단계·선후관계·반복허용을분리. 절대길이나반복횟수만으로오류단정금지. 미검출은동작누락과다름. 소스단어라벨만으로필수동작/허용반복정답을만들지않음.
공개sensor오른손위치에서기계적중간삭제100/100,앞두구간교환80/100이같은단어다른사람차이중앙값이하. 삭제/교환은기계적탐침이며잘못된수어정답아님. 평균거리공통합격선불충분근거.
최초탐색거리masked로컬799쌍중534쌍0:공통손목원점만골라생긴평가기준문제.기존모델버그아님.원결과보존후좌우분리/로컬손목제외/전체관절근거진단을추가검증.보수적인관측요건이며최적기준미확정.손을0으로복원하거나기존입력을바꾸지않음.
삭제와미검출마스킹은관측부분거리같을수있어마스크·시간·관측근거따로유지.준비/마무리미검출을일괄실패로판정하지않고필수단계별로확인해야함.
합성구조12사례+이름변경300검사,별도스칼라192·정렬경로192검산PASS. 저장13592지표확인. 사람/전문가독립검토아님.
공개8영상64장면관측검토,100단어단계검토표초안준비. 수어전문가검토필수단계정답0개,따라서오류검출정확도미산출. 성남/전주후반유사형태는보이지만도시구분정확한단계미확정.
다음:관측가능단계후보→수어검토로필수/선택/선후/반복정답확정→정상오판/누락순서검출/보류별평가동결→결과함께확인한후에만추가학습계획. 사용자촬영은학습·튜닝·후보선택에미사용.
''',encoding='utf8')
print('Saved criteria report,100word catalog,8scene review and next plan; no semantic correctness claims or training.')
