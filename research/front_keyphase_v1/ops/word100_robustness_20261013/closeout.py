"""Generate evidence figures and synchronize the approved unit's final state."""
from pathlib import Path
import sys
from datetime import datetime
import numpy as np
U=Path(__file__).resolve().parent
sys.path.insert(0,str(U.parent/'word100_training_20261013'))
from common import BASE,ROOT,read,write,sha

def main():
    assert not (U/'COMPLETION.json').exists()
    assert read(U/'INDEPENDENT_CHECK.json')['status']=='PASS'
    assert read(U/'dtw96/INDEPENDENT_CHECK.json')['status']=='PASS'
    c=read(U/'COMPARISON.json');d=read(U/'dtw96/SUMMARY.json');rows=read(U/'dtw96/PAIRS.json')
    original=read(U/'ORIGINAL_GUARD.json');assert all(sha(p)==h for p,h in original.items())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':12})
    fig,ax=plt.subplots(figsize=(9,4.6),layout='constrained')
    vals=[277/3,290/3,c['metrics']['candidate']['accuracy_percent']]
    labels=['기존 전체 구간 학습\n(앞선 실험)','현재 전체·핵심 혼합\n(웹 모델 유지)','속도 보강 후보\n(이번 실험·미채택)']
    bars=ax.barh(labels,vals,color=['#8A96A5','#166B80','#BD5151'],height=.58)
    ax.set_xlim(0,107);ax.set_xticks([0,20,40,60,80,100]);ax.set_xlabel('1순위 분류 정확도 (%) · 동일한 공개 개발 자료 300건')
    for b,v,n in zip(bars,vals,[277,290,284]):ax.text(v+.8,b.get_y()+b.get_height()/2,f'{v:.2f}% ({n}/300)',va='center',fontsize=11)
    ax.invert_yaxis();ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('AI Hub 100단어 TCN 비교',fontweight='bold')
    fig.text(.5,-.045,'동일 개발 자료로 학습 시점도 선택함 · 새 사용자 최종 시험 정확도가 아님',ha='center',fontsize=10)
    fig.savefig(U/'ACCURACY_COMPARISON.png',dpi=160,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(10,4.3),layout='constrained')
    for ax,key,label,unit in zip(axes,['handshape','location','movement'],['손가락 굽힘','손 위치','이동 궤적'],['차이 (2D 각도, °)','거리 차이 (어깨너비 %)','거리 차이 (어깨너비 %)']):
        v=[next(m['value'] for m in r['result']['metrics'] if m['key']==key) for r in rows]
        ax.boxplot(v,showfliers=False,widths=.4,patch_artist=True,boxprops={'facecolor':'#DCECF0'},medianprops={'color':'#B52F3B','linewidth':2})
        ax.scatter(np.random.default_rng(0).uniform(.82,1.18,len(v)),v,s=12,color='#166B80',alpha=.38)
        ax.set_xticks([1],[label]);ax.set_ylabel(unit);ax.set_ylim(bottom=0);ax.spines[['top','right']].set_visible(False)
        ax.set_title(f"중앙값 {np.median(v):.2f}");ax.grid(axis='y',alpha=.15)
    fig.suptitle('같은 단어의 사람 간 DTW 차이 · 12단어 × 8명 = 96쌍',fontweight='bold')
    fig.text(.5,-.055,'단어당 동일한 1인 참조와 비교 · 올바름이 개별 검증된 자료가 아님 · 합격선으로 사용하지 않음',ha='center',fontsize=10)
    fig.savefig(U/'DTW_VARIABILITY.png',dpi=160,bbox_inches='tight');plt.close(fig)
    now=datetime.now().astimezone().isoformat()
    report='''WordsInSilence 성능 보강 비교 결과

결론
이번 속도 보강 후보는 채택하지 않는다. 현재 웹 모델은 e39d8ac4… 그대로 유지한다.
현재 모델 290/300 = 96.67%, 속도 보강 284/300 = 94.67%로 2.00%p 하락했다.
오답을 고친 경우 6건, 새로 틀린 경우 12건. 촬영자 11:98→96, 12:92→93, 13:100→95.
상위5개 안에 정답 포함: 현재 299/300=99.67%, 후보 297/300=99.00%.
확률값이 사용자 동작의 정확도라는 뜻은 아니며, 전체100분류의 정답 순위 통계다.

1. 실행한 작업
AI Hub 정면100단어, 학습899건(01~09), 개발300건(11~13)의 저장 입력1199건을 감사했다.
입력은 MediaPipe를 통일 적용한 몸·양손 좌표, 몸/손 기준 좌표, 마스크, 시간간격280채널이다.
시간 간격은 모든 입력에서 약33.333ms로 일정했고 어깨 기준 좌표 누락은 없었다.
정답290건의 손 내부 미관측 프레임 수 중앙값은7, 오답10건은6이었다.
정답/오답 모두 양손 동시 미관측 비율 중앙값0. 오답만 손 누락이 많다는 근거는 없었다.
한 손이 보이지 않는 이유는 휴식·화면이탈·가림·검출실패 중 구별하지 못하므로 모두 오류라고 하지 않는다.
오답8/10은 촬영자12에 몰려 있지만 시간 간격·누락만으로 그 원인을 특정할 수 없다.
상관관계 점검이며 동작 구간·추정 오차가 인식에 영향을 전혀 주지 않는다는 증명은 아니다.

2. 무엇을 바꾸었나
현재 전체/핵심±0.30초 혼합 절차를 유지하고, 학습 입력 절반에 속도0.8~1.25배 변형만 추가했다.
기존 관절 프레임을 시간순 최근접 선택하므로 손모양을 새로 그리거나 빈 좌표를 보간하지 않는다.
관측 마스크를 동일 프레임에서 복사하고 시간간격을 합성된 수행 시간에 맞춘다.
빠르게 할 때 일부 프레임 생략, 느리게 할 때 반복이 발생한다. 아주 짧은 세부 동작 손실 가능성은 있다.
학습자료 내부 같은 단어의 핵심동작 길이가 중앙값 대비0.47~1.80배로 달라 속도 강건성 가설을 시험했다.
0.8~1.25와50%는 사전 고정한 설계값이며 수어의 의미 보존을 전문가가 인증한 범위는 아니다.
참고 논문: Jiang et al. (CVPRW2021), Skeleton Aware Multi-Modal Sign Language Recognition.
https://openaccess.thecvf.com/content/CVPR2021W/ChaLearn/papers/Jiang_Skeleton_Aware_Multi-Modal_Sign_Language_Recognition_CVPRW_2021_paper.pdf
이 연구의 시간 표집/자료 증강 원리를 참고했지만 모델·방법·자료가 다르며 그 논문의 효과를 본 결과로 대체하지 않는다.

3. 공정한 비교와 결과
동일 구조37,636매개변수·초기seed·초기가중치·자료순서·옵티마이저·시점선택 규칙을 유지했다.
학습 전5394개 변형에 대해 원 좌표/마스크 보존·양의 시간간격·양끝 유지·입력 불변을 검증했다.
변형을 끈 배치는 기존 whole_context 배치와 정확히 같다.
후보 학습은1회,70epoch·2030step,약130초. 사전 규칙대로 개발CE가 최소인58epoch 모델을 선택했다.
기존 기준은 앞선72epoch 중60epoch. 동일 선택 절차 비교이지 동일 step에서 변형만의 인과효과 증명은 아니다.
통과 조건은 전체정답>290이며 촬영자별98/92/100보다 감소하지 않는 것이었고 불통과했다.
0.8배 합성시험: 현재288/300=96.00%, 후보283/300=94.33%.
1.25배 합성시험: 현재290/300=96.67%, 후보285/300=95.00%.
같은300건을 변형한 것이므로600개의 독립 실사용 자료가 아니다.
현재 모델도 이 범위의 합성 속도 변화에는 비교적 안정적이었다. 이번 속도 보강을 반복 탐색할 근거는 약하다.
왜 회귀했는지는 단정할 수 없다. 프레임 표집에 따른 세부 정보 손실·최적화 경로 변화 등이 가능하지만 분리 실험하지 않았다.
결과 후 계수·seed·epoch 선택 기준을 다시 찾지 않았다. 현재 가중치/웹 코드/사용자 촬영 원본은 해시까지 보존했다.

4. DTW 사람 간 비교
웹 연습12단어에 대해 같은 AI Hub 라벨의 학습자02~09를 현재 참고01과 비교했다(총96쌍).
96쌍 모두 관측근거를 충족해 계산 가능했다. 이는96개 동작이 모두 올바르다고 판정했다는 뜻이 아니다.
현재 웹과 동일한 DTW 시간정렬·손가락굽힘/위치/궤적 수치를 사용했다.
굽힘 차이: 중앙값9.91°, 사분위8.05~12.82°, 범위4.79~17.16°.
손 위치: 중앙값13.87, 사분위10.96~16.92, 범위7.56~25.50(어깨너비%=단위).
궤적 차이: 중앙값9.71, 사분위8.01~13.19, 범위4.38~27.08(어깨너비%=단위).
관측된 대응쌍 비율은 중앙값97.13%, 최저84.47%. 원 영상 시간 관측률과 다른 지표다.
같은 영상 자체비교12개에서 차이0, 실제96쌍의 양손3축576값을 별도float64 산술로 재계산해 통과했다.
전체 분포는 서로 다른 단어를 합친 기술 통계다. 단어별8개 분포는 dtw96/SUMMARY.json에 별도 보존했다.
실제 옳고 그름 라벨·언어적 변형·정렬의 언어적 적절성은 검증하지 않았다.
따라서 이 숫자를 모든 단어의 합격선이나 자동 교정 기준으로 적용하지 않는다.
세부 관절 피드백은 이번 완료 범위에 넣지 않았다.

5. 검증과 한계
Codex 구현·실행, 별도 코드로 저장logit1800행/메타데이터/70epoch시점선택/학습순서/해시를 검산했다.
다른 에이전트나 외부 전문가가 검토했다고 표현하지 않는다.
모든 인식 수치는 동일 개발자료에서 시점도 선택한 결과. 미공개 최종시험이나 실제 새 사용자 정확도가 아니다.
한seed·촬영자3명·제한된100단어이며, 사용자/멘토 자료는 학습·튜닝·후보선택에 전혀 사용하지 않았다.
추가촬영/카메라/장치탐색/다운로드/보류10·14~18/원본수정/Git변경/운영모델교체 없음.

6. 다음 작업 방향
발표에는 이미 확인된 전체구간277/300=92.33%→전체·핵심혼합290/300=96.67%(+4.33%p)를 사용한다.
이번 후속은 성공적인 성능향상이 아니라 개선 후보 검증 및 부적합 후보 제외로 기록한다.
속도·손누락 계수를 더 찾기보다 기존 오답의 원 영상과 학습 참고 표현을 대조해
손모양/방향/표현변형/라벨 구분 중 실제 정보 차이가 있는지 먼저 확인하는 것이 다음 진단이다.
이는 오답별 예외 규칙을 만드는 작업이 아니다. 공통 개선을 지지할 근거가 없으면 새 학습을 시작하지 않는다.
DTW 정답 기준은 검토된 정상/오류 자료와 사람 분리 검증이 별도 필요하다.
중간보고 문서는 사용자 제공 양식의 상황요약·중간결론·계획변경사항에 이 결과를 반영하는 다음 단위로 둔다.
연구 주제(수어 학습 기능성 게임)는 유지. 속도 보강은 미채택하고 현재 TCN+DTW 프로토타입을 유지한다.
'''
    (U/'RESULT_KO.txt').write_text(report,'utf-8')
    summary=dict(status='COMPLETE_VERIFIED_CANDIDATE_REJECTED',at_kst=now,baseline_correct=290,candidate_correct=284,total=300,delta_pp=-2.,gains=6,regressions=12,dtw_pairs=96,dtw_status='96 numerically usable, not correctness verdicts',training_runs=1,production_replaced=False,report=str(U/'RESULT_KO.txt'),implementer='Codex',verification='separate code; no external review',next='중간보고 결과 정리. 추가 학습은 오답의 실제 표현·손모양·방향·라벨 진단 근거를 확보한 뒤 별도 후보 고정.')
    write(U/'COMPLETION.json',summary)
    p=BASE/'ops/STATE.json';s=read(p);s['updated_date_kst']=now;s['current_stage']='WORD100_ROBUSTNESS_REJECTED_DTW96_VERIFIED'
    s['next_single_action']=summary['next'];s['word100_robustness_20261013']=dict(summary,unit='ops/word100_robustness_20261013',run='runs/word100_temporal_robustness_v1',baseline_hash=read(U/'SCOPE.json')['baseline_hash'],running_processes_this_unit=[])
    write(p,s)
    p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';text=p.read_text('utf-8')
    head='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인'
    text=text.replace(head,head+'\n\n### 2026-10-13 속도 보강 후보 비교·DTW96 검증 완료\n현재96.67%(290/300) 대비 속도보강94.67%(284/300), -2.00%p·개선6/회귀12로 미채택. 촬영자11/12/13:98/92/100→96/93/95. 같은 공개 개발자료·epoch선택 포함이며 새 사람 최종정확도 아님. 단일후보70epoch(선택58) 종료, 계수 재탐색 없음. 기존 웹e39d8ac4/사용자파일/운영코드 해시 보존. AIHub1199입력 감사에서 오답만의 시간간격/손누락 과다 증거 없음. DTW12단어×8명=96쌍 계산 및576축값 검산 PASS; 굽힘중앙9.91°, 위치13.87/궤적9.71어깨너비%. 같은라벨 변이 통계이며 합격선/교정정답 아님. Codex 구현·별도코드 검산(다른 에이전트/외부전문가 검토 아님). 근거 ops/word100_robustness_20261013/RESULT_KO.txt, COMPARISON/INDEPENDENT_CHECK/dtw96.\n',1)
    head='## 9. 다음 단일 작업';text=text.replace(head,head+'\n\n### 2026-10-13 다음 — 검증된 결과를 중간보고에 반영\n사용자 양식 상황요약·중간결론·연구계획변경사항에 현재 TCN100+DTW 흐름, 앞선92.33→96.67% 개발개선과 이번94.67% 미채택, DTW96정상변이 예비값/한계를 정리한다. 보고서파일은 아직 만들지 않았고 실험근거·그림만 준비. 다음 기술진단은 공개 오답의 원영상/학습참고 표현·손모양·방향·라벨 대조; 속도계수/빈손보간 자동재탐색 금지. 사용자학습/보류시험/자동모델교체 없음. 이번 승인단위 완료, 진행중학습0.\n',1)
    p.write_text(text,'utf-8')
    handoff=f'''# WordsInSilence 최신 인수인계
갱신 {now}. AGENTS/CONTEXT 최신§3·§8·§9/WORKFLOW 우선.

## 현재 목표와 마지막 완료
사용자가 개선 우선순위(관측 감사→보강 후보→DTW96)를 승인하여 모두 완료했다.
단위 `ops/word100_robustness_20261013`, RESULT_KO.txt/COMPARISON/INDEPENDENT_CHECK/COMPLETION 및 dtw96 참고.
속도0.8~1.25 단일 보강: 현재290/300=96.67%→후보284/300=94.67%(-2pp), 개선6/회귀12, 미채택.
실행1회70epoch·2030step/선택58epoch, 약130초. 새 학습 중0. 완료실험 재실행금지.
공개개발300은epoch선택에도 사용했으므로 독립최종시험으로 부르지 않는다.
AIHub1199캐시 감사: 시간간격일정, 오답10에서 손누락이 특히 많다는 증거없음. 근본원인확정 아님.
합성속도시험0.8:현재288/후보283,1.25:현재290/후보285. 같은300변형이라 독립600표본아님.
DTW12단어×학습자02~09대비01참고96쌍 완료. 계산가능96, 실제정답96 뜻아님.
중앙굽힘9.91°, 손위치13.87/궤적9.71어깨너비%. 단어당8개, 합격선/새사람정확도/교정정답 미검증.
12개자체비교차이0/96쌍576측정값독립산술검산PASS. 모델/운영코드/사용자파일해시보존.
Codex가 구현·별도코드 검산. 다른에이전트나외부전문가검토 아님.

## 다음 한 작업
첨부된 중간보고 양식의 상황요약·중간결론·연구계획변경사항에 검증된 결과를 정리한다.
보고서는 아직생성안함. 실험결과텍스트와 ACCURACY_COMPARISON.png/DTW_VARIABILITY.png만 준비.
기술후속은 공개오답의 원영상/참고 표현·손모양·방향·라벨 대조 근거를 먼저 확보.
속도계수·seed재탐색, 사용자자료튜닝, 최종보류시험, 자동모델교체는 하지않음.

## 웹 / 학습기준
http://127.0.0.1:8765/practice.html 서버 PID11368 마지막확인; 재개때실제확인.
현재연습모델 `runs/word100_tcn_v1/whole_context/best.pt` SHA e39d8ac46f5a5f8442c48a141498dc810ccbfad242e3f104e1b8633988515c2f.
AIHub-only100/280채널/37,636매개변수.899학습01~09,300개발11~13;10/14~18 미사용.
앞선전체277/300=92.33%→전체·핵심혼합290/300=96.67%(+4.33pp) 실제개발개선 유지.
NIKL/KSL/사용자/멘토 학습·튜닝·후보선택없음. W:/WordsInSilence/data/AIHub_103/word100_inputs_20261013 연결필요.
참고영상100개 ID01기준, 연습메뉴12·분류100. UI2×2 +top5막대+DTW3축막대+상위3영상+기존촬영기록.
DTW requested참고를시간정렬해2D손가락굽힘/위치/궤적차이표시. 합격선/상세관절교정 없음.
표시최대각180°/거리100어깨너비%는그래프눈금, 통과기준 아님.

## 보존할 기존 촬영
ses_79977f57893d CLOSED: 수어1회/감사2회,확인메모2개지만3회모두원결과·관절·capture보존.
tri_22160c4af551 수어1위; tri_9171304d585b 감사2위; tri_cc75cf631a1f 감사1위.
수정·학습·임계값튜닝에 사용하지 않음. 이번단위 ORIGINAL_GUARD 보존.
새촬영/카메라/장치탐색/다운로드/운영교체/Git변경 없음.
'''
    (ROOT/'NEW_CHAT_HANDOFF.md').write_text(handoff,'utf-8')
    print(summary)

if __name__=='__main__':main()
