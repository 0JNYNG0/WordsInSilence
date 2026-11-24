from pathlib import Path
import json,statistics as st,html
U=Path(__file__).resolve().parent
def read(n):return json.loads((U/n).read_text(encoding='utf8'))
def save(n,x):
    p=U/n;assert not p.exists();p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def main():
    a=read('ANALYSIS.json');v=read('INDEPENDENT.json');assert v['status']=='PASS'
    visual=read('VISUAL_EVIDENCE.json')
    plan={
      'status':'PROPOSED_ONLY_WAIT_USER_REVIEW_NO_TRAINING',
      'goal':'공통으로 동작 단계의 전환·순서를 더 명시적으로 보존하는지 검증. 믿다/사육장 전용 규칙 없음.',
      'next_single_step':'공개 TRAIN899·100단어에서 기존 모델의 단계/순서 민감도와 입력 보존 상태를 진단. 변형 입력은 정답 수어·오답 수어로 자동 라벨링하지 않고 민감도 확인에만 사용. 사용자 자료 및 보류 ID14–16 접근 없음.',
      'candidate_after_review':{
        'name':'고정 T3-F + 작은 순서 요약 보조 후보 (가칭 F-G16)',
        'frozen':'C 특징 추출부·기존 분류부·T3-F의 기존 phase_weight까지47236개와 정규화 버퍼 전부 고정. 운영 파일/기존 체크포인트 불변.',
        'new_part':'C 평균 전의 프레임별32차원 특징을 시간순으로 1층 단방향 GRU(hidden16)에 입력. 실제 마지막 프레임의 상태→bias 없는 Linear(16,100)→기존 T3-F logits에 더함.',
        'parameter_budget':'GRU2400 + 출력층1600 = 새 학습 변수4000개. 출력층0초기화로 최초 결과는 T3-F와 일치하도록 검증.',
        'missing_and_time':'원래280채널과 모든 실제 미검출 프레임·시간을 유지. 패딩만 제외하고 누락을 지워서 앞뒤 단계를 붙이지 않음. 가짜 손 복원, 순서뒤집기/부분영상의 정답 학습 없음.',
        'pretrain_gates':['초기 logits T3-F 일치','기존 가중치·BN 버퍼 동결','패딩 불변·실제 미검출 유지','공개 입력/촬영자 분리·누락 없음','새 부분에만 기울기/갱신'],
        'comparison':'승인 시 같은 TRAIN899/DEV300, 같은3seed·관측/전체핵심 일정·선택 규칙. 기존 C/T3-F 저장 결과와 비교. 정상 전체100단어 정확도·top5·CE·사람별/미검출별 회귀를 우선 보고.',
        'adoption_gate':'공개 sensor 평균 정확도 상승과3seed중2이상 상승, masked/top5 및 사람·미검출 집합 회귀 없음. 충족해도 실제 사용 개선 보장이나 자동 운영 교체 아님.',
        'sequence_claim_limit':'GRU의 시간순 처리는 수단일 뿐 성능 개선 보장이 아니다. 파라미터 증가 효과도 있어 순서만의 인과를 주장하려면 같은 크기 순서제거 대조가 추가로 필요.',
        'completeness_limit':'단어 이름만 맞추는 손실로는 모든 필수 단계 수행 여부가 보장되지 않는다. 이를 제품 판정으로 쓰려면 전문가가 확인한 정상/생략/순서오류 공개 사례와 별도 기준 검증이 필요.',
        'holdout':'ID17/18은 이미 평가에 사용. 반복 DEV를 새로운 독립검증으로 부르지 않음. 정상 ID14–16 총299개는 후보/조건 확정 후 별도 최종평가를 위한 미사용 상태로 유지.'},
      'deferred':['초 단위 하드 컷오프','단어별 가산점/감점','사용자 영상 맞춤 보강','재검출/좌우 규칙 동시 변경','임의 순서변형을 수어 오류 정답으로 학습'],
      'literature':[
        {'title':'PyTorch 2.12 GRU 공식 문서','url':'https://docs.pytorch.org/docs/2.12/generated/torch.nn.GRU.html','use':'시간순 상태 갱신과 구성/파라미터의 근거. 수어 성능 개선의 증거 아님.'},
        {'title':'Bai et al., An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling','url':'https://arxiv.org/abs/1803.01271','use':'TCN도 순서를 처리하며 GRU가 본질적으로 우수한 것은 아니라는 배경. 실제 분석은 프로젝트 코드와 저장 자료 기준.'}],
      'training_authorized_this_turn':False,'user_mentor_data_used':False}
    save('NEXT_PLAN.json',plan)
    conclusions={
      'finding':'부분 동작·일부 순서 변화에도 높은 단어 점수가 남는 한계 확인. 공개 정상 입력에서 믿다↔사육장 오답은 재현되지 않음. 실사용 혼동의 단일 원인 미확정.',
      'public_TRAIN_clips':18,'views':2,'visual_TRAIN_clips':6,'visual_sampled_frames':96,
      'normal_TRAIN_top1_per_model':36,'normal_TRAIN_views_total':36,
      'saved_DEV_clips':6,'saved_DEV_predictions_top1':72,
      'training_runs':0,'user_recordings_read':0,'production_changes':0,
      'exact_visual_description':'표본에서 믿다는 양손을 가슴 앞에 모아 한손의 손가락을 다른 손에 대고 유지하는 모습. 사육장은 양손을 따로 움직이는 앞부분, 가슴 앞에서 두손이 가까워지는 중간, 한손을 앞으로 내미는 뒷부분이 구분됨. 비전문가의 장면 기술이며 언어학적 단계 수 정답 아님.',
      'limits':['학습 자료의 높은 점수는 새 사람 일반화 근거가 아님','변형 입력은 인위적 단절 포함, 실제 수어의 옳고 그름 판정 아님','관측 마스크는 관절 위치의 정확성을 보장하지 않음','미검출을 이번 혼동의 원인으로 특정하지 않음','별도 산술 검산은 같은 Codex의 다른 코드, 독립 사람/다른 에이전트 검토 아님'],
      'correction':'첫 reverse_thirds의 내부 dt0 혼입을 발견하고 최초 증거 보존. 좌표/관측 채널만 재배열하고 원래 시간은 유지하여 해당36쌍만 검증 복구. 보고는 복구 결과만 사용.',
      'next':'NEXT_PLAN.json 검토 후 공개100단어의 순서 진단 범위부터 확정. 추가학습/배포는 현재 미실행.'}
    save('CONCLUSION.json',conclusions)
    def summary(view,label,cond):return next(r for r in v['summaries'] if r['view']==view and r['label']==label and r['condition']==cond)
    duration=[]
    for lab in ('믿다','사육장'):
        rows=[r for r in a['metadata'] if r['label']==lab and r['view']=='sensor']
        vals=[r['active_seconds'] for r in rows];full=[r['full_seconds'] for r in rows]
        duration.append((lab,min(vals),st.median(vals),max(vals),min(full),max(full)))
    durhtml=''.join(f'<div class="duration"><b>{lab}</b><span>{lo:.2f}–{hi:.2f}초</span><div class="track"><i style="width:{mid/4*100:.1f}%"></i></div><small>활동 입력 중앙값 {mid:.2f}초 · 공개 TRAIN 9개</small></div>' for lab,lo,mid,hi,_,_ in duration)
    conditions=[('original','원래 입력'),('prefix_half','앞 절반만'),('suffix_half','뒤 절반만'),('reverse_thirds','뒤→중간→앞 순서'),('double_sequence','전체 동작을 두 번 연결')]
    def table(view):
        body=''
        for label in ('믿다','사육장'):
            for cond,desc in conditions:
                s=summary(view,label,cond)
                body+=f"<tr><td>{label}</td><td>{desc}</td><td>{s['C']['original_label_top1']}/9</td><td><b>{s['T3F']['original_label_top1']}/9</b></td><td>{s['T3F']['target_score_median']:.2f}</td></tr>"
        return '<div class="scroll"><table><thead><tr><th>단어</th><th>진단 입력</th><th>C 원래 단어 1위</th><th>T3-F 원래 단어 1위</th><th>T3-F 점수 중앙값</th></tr></thead><tbody>'+body+'</tbody></table></div>'
    visualhtml=''
    for signer in (1,5,9):
        visualhtml+=f'<details {"open" if signer==1 else ""}><summary>공개 TRAIN 촬영자 ID{signer:02} · 두 단어 비교</summary><div class="twocol">'
        for code,label in [('WORD1139','믿다'),('WORD0196','사육장')]:
            stem=f'NIA_SL_{code}_REAL{signer:02}_F'
            visualhtml+=f'<article><h3>{label}</h3><video controls preload="metadata" src="public_visuals/{stem}.mp4"></video><a href="public_visuals/{stem}.jpg" target="_blank"><img src="public_visuals/{stem}.jpg" alt="{label} 시간순 공개 장면 16개"></a></article>'
        visualhtml+='</div></details>'
    text='''믿다·사육장: 길이보다 단계와 순서 확인이 필요한 이유
공개 TRAIN18영상·두 관측 조건36개 입력과 기존 DEV6영상의 저장예측72개를 분석했다. 새 학습0, 사용자 촬영 읽기0, C/T3-F/웹 변경0.

확인한 결과
두 단어는 실제로 구분되는 동작 흐름을 가진다. 표본 RGB6영상의 시간순96장에서 믿다는 가슴 앞에서 두손이 만나 유지되는 부분이 중심이며, 사육장은 앞의 양손 움직임과 중간의 접촉에 가까운 자세, 뒤의 한손을 앞으로 내미는 단계가 보였다. 이것은 장면 설명이며 전문가가 확인한 언어학적 단계 수가 아니다.
sensor 활동 입력은 믿다1.27~1.77초(중앙값1.50), 사육장2.73~3.37초(3.03)였다. 전체 파일의 준비·마무리 시간을 포함하면 믿다2.90~4.80초, 사육장4.03~5.83초로 겹친다(프레임 타임스탬프 차이 기준). 시간만으로 구분하는 규칙은 사람별 수행 속도·대기시간에 취약하다.
정상 TRAIN18영상×2관측에서 C와 T3-F는 모두36/36을 맞혔다. 관련 공개 DEV6영상의 기존3seed×2모델×2관측 저장72예측도 전부1위다. 따라서 공개 정상 자료에서 이번 두 단어의 실제 혼동은 재현되지 않았다. 이 높은 수치로 새 사용자 성능을 주장하지 않는다.

구분 동작을 다 했는지 확인하는 능력은 별도다
공개 sensor 입력을 세 덩어리로 나눈 뒤 덩어리 순서만 뒤집으면, T3-F는 사육장9/9를 계속 사육장1위로 출력했다(최소98.72점, 중앙99.41점). 믿다는8/9가 여전히1위였다. 단어 정답 판정 실험이 아니라 순서 변화에 대한 진단이다.
사육장의 앞 절반만 남겨도5/9, 뒤 절반만 남겨도6/9가 사육장1위였다. 다만 뒤 절반 masked 조건에서는0/9로 달라져 관측 조건 영향도 크다. 절반에 이미 핵심 특징이 남을 수 있으므로 이를 전부 틀린 수행이라고 자동 규정하지 않는다.
전체 동작을 두 번 연결한 경우 sensor18/18, masked18/18이 원래 단어1위였다. 반복 횟수나 완료 여부를 정확히 검증했다는 증거로 단어 점수를 사용할 수 없다.
T3-F는 순서를 전혀 무시하는 것은 아니다. 덩어리 순서를 바꾸면 정답-상대단어 내부 점수차가 중앙값 약8만큼 줄었다(C는약0.5). 하지만 정상 학습 예시의 점수차가 크므로1위가 유지될 수 있다. softmax 점수는 ‘그 단어의 필수 동작을 몇% 수행했는가’가 아니다.
시간간격만2배/0.5배로 바꾸는 입력 진단에서는 모든 입력에서 정답 단어의1위가 유지됐다. T3-F 정답 점수 최대 변화는0.000067점 미만. 좌표의 프레임 배열은 그대로 둔 검사라 실제 속도·프레임률 변화 실험과 같지 않다.

구조상 이유와 한계
C는 각 구간의 시간 특징을 만든 후 전체 평균을 낸다. T3-F는 같은 특징의 상대적 앞·중간·뒤 요약을 더한다. 이것은 ‘첫 행동, 다음 행동, 마지막 행동이 모두 존재한다’는 규칙과 다르다. 서로 다른 길이의 입력도 시작0~끝1로 정리되며, 손 관절이 겹쳐 잘못 잡히면 어느 요약 방식도 사라진 사실을 저절로 복원하지 못한다.
시간 정보 dt는 원래 입력에 있으므로 길이를 전혀 못 본다고 설명하면 부정확하다. 시간 합성곱도 순서에 반응한다. 반면 100단어 가운데 상대적으로 가장 높은 것을 고르는 학습만으로 필수 단계의 누락·순서오류를 거절하는 기능이 보장되지는 않는다.
공개 자료에서도 일부 손 관측 누락은 있었지만 정상 입력에서는 두 모델 모두 정답이었다. 이번 혼동을 미검출·좌우 배정 하나의 문제로 단정하거나 사용자의 동작이 틀렸다고 판단하지 않는다. 공개 자료만으로 실사용 사례의 직접 원인은 아직 분리되지 않았다.

다음 계획
1. 먼저 같은 순서 진단을 공개 TRAIN100단어 범위로 넓혀 공통 현상인지 확인하고, 구별 단계를 포함한 입력이 남아 있는지 점검한다. 아직 정상/오류 수어라는 자동 라벨은 붙이지 않는다.
2. 순서 보완 후보는 기존 T3-F 전체를 고정하고, C가 만든 프레임별 특징을 시간순으로 읽는 작은 GRU16 보조부만 추가하는 방안이다. 새 변수4000개, 출력0초기화로 시작 결과는 T3-F와 동일하게 만들고 동결·패딩·미검출 처리부터 검증한다. 기존 C나 T3-F를 교체하지 않는다.
3. 이후 승인한 조건으로 TRAIN899/DEV300·3회 비교한다. 믿다·사육장 두 단어만이 아니라 전체100단어, 사람별, 손 누락별 개선과 퇴보를 함께 본다. 시간순 처리만 추가해도 개선되는지는 미확정이다. 단어 라벨만으로 모든 단계 완료를 보장할 수 없어, 수행 완성도 판정에는 전문가가 확인한 정상·단계 생략·순서 오류 사례가 별도로 필요하다.
이미 평가한 ID17/18을 새 독립시험으로 재사용하지 않고, 미사용 정상 ID14~16은 최종 후보 고정 전 열지 않는다. 사용자 촬영은 계속 평가 전용이다. 이 계획은 아직 구현·학습하지 않았다.

검증
최초 순서변형 진단에서 시간 채널까지 이동해 내부dt0이 들어가는 문제를 찾아 최초 결과를 보존하고, 원래 시간을 유지하도록 해당36쌍만 재계산했다. 보고는 수정된 순서 진단만 사용했다. 정상 평가·완료된 DEV를 재채점하지 않았다.
저장64800개logits,36개입력 통계,36개시간요약 기여도를 별도 산술 코드로 검산했다. 입력·모델·웹104개 파일과 공개RGB6개 해시 유지. 동일 Codex의 다른 코드 검산이며 독립 사람/다른 에이전트·수어 전문가 검토는 아니다.
'''
    (U/'RESULT_KO.txt').write_text(text,encoding='utf8')
    doc='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>믿다·사육장 공개 자료 진단</title><style>
    *{box-sizing:border-box}body{margin:0;background:#eef2f6;color:#15233a;font-family:"Malgun Gothic",sans-serif;line-height:1.65}main{max-width:1120px;margin:auto;padding:42px 24px 80px}h1{font-size:36px;line-height:1.3;margin:12px 0 22px}h2{font-size:25px;margin:0 0 18px}h3{font-size:19px}p{margin:12px 0}section{background:white;padding:30px;border-radius:18px;margin:22px 0;border:1px solid #dce3ed}.eyebrow{font-size:13px;letter-spacing:1px;color:#365b94;font-weight:bold}.lead{font-size:20px;max-width:950px}.badge{display:inline-block;padding:7px 12px;border-radius:8px;background:#e4f1ee;color:#155e53;margin:4px 6px 0 0}.twocol{display:grid;grid-template-columns:1fr 1fr;gap:24px}.duration{padding:18px;background:#f2f6fb;border-radius:12px}.duration b{font-size:22px}.duration span{float:right;font-size:22px;color:#245cba}.track{height:13px;background:#dae4f2;border-radius:7px;margin:16px 0}.track i{height:100%;display:block;background:#3276d4;border-radius:7px}small{font-size:14px;color:#4d6079}table{width:100%;border-collapse:collapse;font-size:15px}th,td{padding:12px;text-align:left;border-bottom:1px solid #e2e7ef}th{background:#edf3fb;font-size:13px}video,img{width:100%;border-radius:8px}img{margin-top:10px}details{margin:14px 0;padding:14px 0}summary{cursor:pointer;font-weight:bold;color:#285f9d}.callout{padding:18px 22px;border-left:4px solid #e5a84f;background:#fff8e9;border-radius:5px}.flow{display:flex;flex-wrap:wrap;gap:12px;align-items:center}.flow b{padding:14px;border-radius:10px;background:#eaf1fb}.scroll{overflow:auto}li{margin:12px 0}a{color:#225ba7}.minor{color:#4d6079;font-size:14px}.step{padding:20px;border:1px solid #e0e7f0;border-radius:10px;margin:14px 0}.step b{color:#225ba7} @media(max-width:700px){.twocol{grid-template-columns:1fr}h1{font-size:29px}main{padding:24px 14px}section{padding:22px 18px}.duration span{float:none;display:block}}
    </style><main><div class="eyebrow">WORDS IN SILENCE · 공개 자료 원인 진단 · 2026.10.19</div><h1>비슷한 부분이 있어도,<br>동작 전체는 다를 수 있습니다.</h1><p class="lead">믿다와 사육장의 구분 단계·순서·길이를 확인했습니다. 현재 점수는 <b>단어 후보 점수</b>이며, 모든 구분 동작을 끝까지 올바르게 했다는 판정은 아닙니다.</p><div><span class="badge">공개 TRAIN 18영상</span><span class="badge">추가 학습 0</span><span class="badge">사용자 촬영 읽기 0</span><span class="badge">C · T3-F 보존</span></div>
    <section><h2>실제 동작 길이와 단계는 다릅니다</h2><div class="twocol">'''+durhtml+'''</div><p>자른 화면에서 검출한 sensor 조건의 활동 입력 길이입니다. 준비·마무리를 포함한 전체 영상 길이는 두 단어 사이에 겹치므로, ‘길면 사육장’ 같은 규칙으로 해결하지 않습니다.</p><p><b>믿다</b> 표본은 가슴 앞에서 두 손이 만나 유지되는 장면이 중심입니다. <b>사육장</b> 표본은 앞부분의 양손 움직임, 중간의 두 손이 가까워지는 자세, 뒷부분의 한 손을 앞으로 내미는 변화가 보입니다. 아래 장면은 공개 자료이며 사용자 촬영이 아닙니다.</p>'''+visualhtml+'''<p class="minor">6개 영상에서 미리 정한 96장을 시각 검토했습니다. 단계 설명은 장면 관찰이며 수어 전문가가 확정한 단계 수·정오 판정이 아닙니다. 영상을 직접 재생해 전체 흐름도 확인할 수 있습니다.</p></section>
    <section><h2>순서를 바꿔도 같은 단어가 1위로 남았습니다</h2><p>원래 입력을 고정 모델에 넣은 뒤, 앞/뒤 절반·구간 순서 변경·전체 반복을 별도 진단했습니다. 아래는 <b>원래 단어가 여전히 1위였던 개수</b>이며, 변형 동작을 올바르게 수행했다는 정답률이 아닙니다.</p>'''+table('sensor')+'''<div class="callout"><b>사육장:</b> ‘뒤→중간→앞’으로 순서를 바꿔도 9개 모두 사육장 1위였고, 점수는 최소 98.72점이었습니다. 이것이 ‘동작 완성도 98.72%’라는 뜻은 아닙니다.</div><details><summary>전체 영상 검출 후 범위를 제한한 masked 조건도 보기</summary>'''+table('masked')+'''<p>사육장 뒤 절반은 sensor에서6/9, masked에서0/9로 결과가 다릅니다. 한 조건만으로 전부 같은 현상이라고 일반화하지 않습니다.</p></details><p>구간 순서 변경은 인위적 단절을 포함합니다. 일부 구간만으로도 단어의 특징을 알아볼 수 있으므로, 이 검사만으로 올바른 수어·틀린 수어를 자동 지정하지 않습니다.</p></section>
    <section><h2>왜 이런 점수가 나올 수 있나</h2><div class="flow"><b>프레임별 움직임 특징</b><span>→</span><b>C: 전체 평균</b><span>＋</span><b>T3-F: 앞·중간·뒤 요약</b><span>→</span><b>100개 단어 중 순위</b></div><p>현재 구조는 ‘어느 단어의 특징에 가까운가’를 학습합니다. <b>첫 단계와 다음 단계, 마지막 단계가 전부 들어 있어야 통과</b>라는 별도 규칙은 없습니다. 서로 다른 길이도 시작부터 끝까지의 상대적인 위치로 요약합니다.</p><p>그렇다고 T3-F가 순서를 전혀 못 보는 것은 아닙니다. 순서를 바꾸면 정답과 상대 단어의 내부 점수차는 중앙값 약8 감소했습니다(C는약0.5). 하지만 원래 점수차가 크면 최종1위는 그대로 남습니다. 손 모양과 위치가 비슷한 부분만으로도 후보 점수가 생길 수 있으며, 화면의 참고영상 전체와 직접 일치율을 계산한 값도 아닙니다.</p><p>시간 정보만2배·절반으로 바꿨을 때는 T3-F 정답 점수 변화가 최대0.000067점 미만이었습니다. 시간 정보가 입력에서 빠진 것은 아니며, 이 확인은 실제 촬영 속도 변경 실험과도 다릅니다.</p></section>
    <section><h2>확인한 것과 아직 모르는 것</h2><div class="twocol"><article><h3>확인</h3><ul><li>정상 공개 TRAIN18영상×2관측은 C·T3-F 모두36/36 정답.</li><li>관련 DEV6영상의 기존72개 예측도 전부 정답. 새로 평가한72영상이 아닙니다.</li><li>일부 구간·순서 변경에도 높은 단어 점수가 남는 경우가 존재.</li><li>손 누락이 있는 공개 예시도 정상 입력에서는 두 모델 모두 정답.</li></ul></article><article><h3>미확정</h3><ul><li>공개 정상 자료에서 믿다→사육장 오답 자체는 재현되지 않음.</li><li>실사용 혼동의 단일 원인이 좌우·미검출·순서 중 무엇인지는 미확정.</li><li>학습 자료의 높은 점수는 새 사람 일반화 증명이 아님.</li><li>순서 보완을 추가했을 때 전체 인식률이 오를지는 새 비교가 필요.</li></ul></article></div></section>
    <section><h2>다음은 구분 단계와 순서를 보존하는 방향</h2><div class="step"><b>1 · 공개100단어에서 공통 현상인지 확인</b><p>두 단어에만 맞추지 않고 TRAIN 범위에서 단계·순서 민감도와 실제 구별 동작이 입력에 남는지 점검합니다. 사용자 촬영이나 미사용 최종평가 자료는 이 설계에 넣지 않습니다.</p></div><div class="step"><b>2 · 기존 T3-F는 고정, 작은 순서 보조부만 후보로</b><p>평균 내기 전 프레임 특징을 순서대로 읽는 작은 GRU16을 붙이는 방안을 제안합니다. ‘지금까지 어떤 특징을 거쳐 왔나’를 이어받는 부품입니다. 새 학습 변수4000개, 기존 가중치와 통계는 전부 고정하고 최초 출력은 T3-F와 같도록 검증합니다.</p></div><div class="step"><b>3 · 검토 후3회 비교, 전체100단어의 개선·퇴보 확인</b><p>기존 공개 TRAIN899/DEV300 분리와 선택 규칙을 유지합니다. 믿다·사육장만 좋아졌는지가 아니라 전체 정확도·상위5·사람별·손 누락별 결과를 함께 봅니다. 아직 구현·학습하지 않았으며 성능 향상을 보장하는 제안은 아닙니다.</p></div><div class="callout">‘어떤 단어인가’와 ‘필수 동작을 다 올바르게 했나’는 다른 검증입니다. 후자까지 판정하려면 공개 자료의 정상·단계 생략·순서 오류를 전문가가 확인한 별도 시험이 필요합니다. 길이 하드 컷오프나 사용자 맞춤 점수 보정은 제안하지 않습니다.</div><p class="minor">GRU 구조: <a href="https://docs.pytorch.org/docs/2.12/generated/torch.nn.GRU.html">PyTorch 공식 문서</a>. TCN도 순서 모델이라는 배경: <a href="https://arxiv.org/abs/1803.01271">Bai 외, 2018</a>. 문헌은 이번 프로젝트의 개선 증거가 아니며 실험으로 확인해야 합니다.</p></section>
    <details><summary>검산·정정·근거 파일</summary><p>별도 코드로 저장64,800개 logits,36개 입력 통계와36개 시간 요약 기여도를 검산했습니다. 모델·입력·웹104파일과 RGB6영상 해시 보존. 같은 Codex의 별도 산술 코드이며 독립 사람/에이전트 검토는 아닙니다.</p><p>최초 순서변형에서 시간 채널까지 이동한 문제는 최초 증거를 보존하고 해당36쌍만 수정했습니다. 보고의 순서 결과는 원래 시간간격을 유지한 복구 결과입니다. 기존 완료평가 재실행과 사용자 자료 접근은 없습니다.</p><p><a href="RESULT_KO.txt">전체 설명</a> · <a href="NEXT_PLAN.json">다음 계획</a> · <a href="INDEPENDENT.json">검산</a> · <a href="ANALYSIS.json">최초 진단 기록</a> · <a href="ORDER_CLOCK_CORRECTION.json">순서 진단 정정</a></p></details></main></html>'''
    (U/'RESULT_KO.html').write_text(doc,encoding='utf8')
    print('Report and proposed-only plan ready.')
if __name__=='__main__':main()
