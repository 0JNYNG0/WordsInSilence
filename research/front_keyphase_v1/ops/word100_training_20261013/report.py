"""Close out the already-verified run; never changes or reruns model selection."""
from datetime import datetime
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from common import U,BASE,ROOT,OUT,RUN,read,write,sha,pin_check

def main():
    f=pin_check();check=read(U/'METRICS_CHECK.json');assert check['status']=='PASS'
    inputs=read(U/'INPUT_CHECK.json');assert inputs['status']=='PASS'
    assert read(U/'MEDIA_CHECK.json')['status']=='PASS'
    assert not (U/'COMPLETION.json').exists(), 'Use the completed report'
    guards=read(U/'PRODUCTION_GUARD.json')
    assert all(sha(p)==h for p,h in guards.items())
    comp=read(RUN/'COMPARISON.json');a=comp['arms']['whole'];b=comp['arms']['whole_context']
    ma,mb=a['metrics'],b['metrics'];delta=comp['percentage_point_difference']
    assert check['correct']=={'whole':ma['correct'],'whole_context':mb['correct']}
    verdict='핵심 구간을 섞은 조건의 개발 정답 수가 늘었습니다.' if delta>0 else ('두 조건의 개발 정답 수가 같습니다.' if delta==0 else '핵심 구간을 섞은 조건의 개발 정답 수가 줄었습니다.')
    per=[]
    for x,y in zip(ma['by_class'],mb['by_class']):
        assert x['code']==y['code']
        per.append(dict(code=x['code'],label=x['label'],development_total=3,whole_correct=x['correct'],
            context_correct=y['correct'],difference=y['correct']-x['correct']))
    write(U/'WORD_RESULTS.json',per)
    records=[read(p) for p in (OUT/'records').glob('*.json')]
    sizes=Counter()
    for r in records:
        sizes['observations']+=Path(r['observation']).stat().st_size
        if r['channel_file']:sizes['channels']+=Path(r['channel_file']).stat().st_size
    now=datetime.now().astimezone().isoformat()
    people='\n'.join(f"  촬영자 {s}: 전체 {ma['by_signer'][str(s)]['correct']}/100 → 핵심 혼합 {mb['by_signer'][str(s)]['correct']}/100" for s in [11,12,13])
    result=f'''WordsInSilence — AI Hub 정면 100단어 학습·개발 검증 결과
완료 시각: {now}

결론
전체 동작 학습: {ma['correct']}/300, {ma['accuracy_percent']:.2f}%
전체 + 핵심 구간 혼합 학습: {mb['correct']}/300, {mb['accuracy_percent']:.2f}%
차이: {delta:+.2f}퍼센트포인트. {verdict}
새로 맞힌 영상 {len(comp['improved'])}개, 새로 틀린 영상 {len(comp['regressed'])}개입니다.
기존 웹 모델은 그대로 유지했습니다. 이 결과는 개발(검증) 성능이며 최종 시험 성능은 아닙니다.

1. 어떤 자료로 진행했나
- AI Hub REAL/WORD의 정면(F) 영상, 사전에 고정한 100단어만 사용했습니다.
- 학습: 촬영자 ID 01~09의 899개. 99단어는 9개씩, 1단어는 정상 주석이 있는 8개입니다.
- 개발 검증: 학습에 포함하지 않은 ID 11~13의 300개, 단어마다 3개입니다.
- 입력 통과: 학습 {inputs['counts'].get('train_candidate',0)}개, 개발 {inputs['counts'].get('development_only',0)}개.
- 입력 실패: 학습 {inputs['counts'].get('train_candidate_failed',0)}개, 개발 {inputs['counts'].get('development_only_failed',0)}개. 개발 실패는 주 지표의 300개 분모에 남겼습니다.
- 국립국어원·KSL·사용자·멘토 영상과 기존 모델 가중치는 새 학습에 사용하지 않았습니다.
- ID 10 및 보류 ID 14~18 영상은 이번 추출·학습·평가에 사용하지 않았습니다.

2. 영상이 모델 입력이 되는 과정
정면 영상 → 기존 웹과 같은 MediaPipe 추출기 → 원본 시각과 관측 여부를 보존한 75점 기록
→ 몸 기준 XY, 손목 기준 손 XY, 관측 마스크, 시간 간격을 담은 280채널 → 100개 단어의 점수.
원본 영상의 중간 프레임을 삭제하거나 미검출 손을 임의로 채우지 않았습니다.
모델 입력은 손이 처음 관측된 시점부터 마지막 관측 시점까지이며, 손을 내린 상태도 관측되면 포함됩니다.
추출 기록은 {sum(r['frames'] for r in records):,}프레임입니다. 관절·채널 파일 합계는 {sum(sizes.values())/1e6:.1f}MB입니다.
원본 ZIP을 풀어 전량 복제하지 않고 선택한 정면 영상만 임시로 읽었습니다. 원본 ZIP은 보존했습니다.

3. 무엇을 비교했나
두 모델 모두 같은 MaskedSignTCN 구조, 같은 초기 가중치, 같은 100단어와 학습 순서를 사용했습니다.
A: 전체 관측 구간만 학습.
B: 같은 영상을 볼 때 전체 구간과 주석 핵심 구간 앞뒤 0.30초를 포함한 구간을 교대로 학습.
검증에서는 둘 다 전체 관측 구간만 입력했습니다. 정답 단어와 정답 핵심 구간을 모델에 알려주지 않았습니다.
구간별 정규화는 기존 입력 생성기를 그대로 사용하므로, 구간 선택 시 그 구간의 크기 기준도 다시 계산됩니다.
학습 계수와 횟수 선택 규칙은 점수를 보기 전에 고정했습니다. 최대 80회, 개발 손실이 12회 연속 개선되지 않으면 종료합니다.
A 실행 {a['epochs_executed']}회 / 선택 {a['selected_epoch']}회, B 실행 {b['epochs_executed']}회 / 선택 {b['selected_epoch']}회.
총 모델 매개변수 {a['parameter_count']:,}개. 본 학습 2조건 각 1회. 별도 사전 실행 점검은 폐기한 4단계 학습이며 성능 결과에 포함하지 않았습니다.

4. 사람별 개발 결과
{people}
단어별 결과는 WORD_RESULTS.json에 모두 기록했습니다. 단어마다 3개뿐이므로 개별 단어의 일반화 성능을 확정할 수 없습니다.

5. 무엇을 검증했나
- 전체 입력을 다른 계산 코드로 재구성하여 좌표 정규화·관측 마스크·실제 시간 간격·행 보존을 대조했습니다.
- 사전에 고정한 학습 영상 6개의 원본 바이트와 디코딩 시각을 캐시와 별도로 대조했습니다.
- 저장된 300개 예측 점수를 NumPy로 재집계하여 정확도·사람별/단어별 집계·개선/회귀를 확인했습니다.
- 학습 기록에서 사전에 정한 최저 개발 손실 선택 규칙과 체크포인트 해시를 확인했습니다.
작성·실행·별도 검산 담당은 Codex입니다. 다른 사람이나 수어 전문가의 판정 검증을 받았다는 의미는 아닙니다.

6. 이번 결과로 아직 말할 수 없는 것
개발 300개는 학습 횟수를 고르는 데도 사용했습니다. 처음 보는 최종 시험이나 실제 새 사용자 성능이라고 부르지 않습니다.
이번 100단어 과제와 기존 93단어 과제는 단어·영상·입력이 달라 정확도 숫자만 직접 비교할 수 없습니다.
단어를 가장 가깝게 고르는 분류 성능입니다. 틀린 손모양이나 수어가 아닌 동작을 거절하는 능력은 별도 평가가 필요합니다.
단일 seed 결과이며, 핵심 혼합의 효과가 모든 데이터와 사람에서 반복된다고 보장하지 않습니다.

7. 다음 단계
개발 결과와 오답 분포를 검토한 뒤, 실제 웹 시험 연결 여부와 최종 평가 범위를 결정합니다.
더 많은 단어로 확장할 때 이번 100단어의 관절 캐시는 재사용합니다. 새 단어의 출력층을 확장하고 기존·새 학습 자료를 함께 사용하는 별도 모델을 학습합니다.
기존 18개 참조는 새 모델 어휘와 겹치는 부분을 확인한 뒤 외부 진단용으로 다룹니다. 현재 결과에 섞지 않았습니다.
운영 기본 모델 교체와 보류 시험은 자동으로 실행하지 않았습니다.
'''
    (U/'RESULT_KO.txt').write_text(result,'utf-8')
    font=FontProperties(fname='C:/Windows/Fonts/malgun.ttf')
    fig,ax=plt.subplots(figsize=(8,4.7),layout='constrained')
    names=['전체 동작 학습','전체 + 핵심 구간 혼합'];values=[ma['accuracy_percent'],mb['accuracy_percent']]
    bars=ax.bar(names,values,color=['#315984','#b43d40'],width=.5)
    ax.set_ylim(0,110);ax.set_yticks([0,20,40,60,80,100]);ax.set_ylabel('개발 정확도 (%)',fontproperties=font)
    ax.set_title('AI Hub 정면 100단어 · 학습에 없는 촬영자 3명',fontproperties=font,pad=16)
    for lab in ax.get_xticklabels():lab.set_fontproperties(font)
    for bar,v,n in zip(bars,values,[ma['correct'],mb['correct']]):
        ax.text(bar.get_x()+bar.get_width()/2,v+2,f'{v:.2f}%\n({n}/300)',ha='center',fontproperties=font)
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.text(.5,.01,'학습 횟수를 선택한 개발 자료의 결과 · 독립 최종 시험과 구분',ha='center',fontproperties=font,fontsize=9)
    fig.savefig(U/'ACCURACY.png',dpi=160);plt.close(fig)
    done=dict(status='COMPLETE_DEVELOPMENT_COMPARISON',at_kst=now,classes=100,training_rows=899,development_rows=300,
        correct={'whole':ma['correct'],'whole_context':mb['correct']},accuracy_percent={'whole':ma['accuracy_percent'],'whole_context':mb['accuracy_percent']},
        difference_pp=delta,improved=len(comp['improved']),regressed=len(comp['regressed']),
        checkpoints={arm:comp['arms'][arm]['checkpoint_sha256'] for arm in comp['arms']},
        implementation='Codex local',independent_verification='Codex separate algorithms; no external reviewer',
        final_blind_test=False,production_replaced=False,report=str(U/'RESULT_KO.txt'),input_check=str(U/'INPUT_CHECK.json'),metrics_check=str(U/'METRICS_CHECK.json'),
        next='Review development result; no automatic heldout evaluation or production replacement.')
    write(U/'COMPLETION.json',done);write(U/'STATUS.json',done)
    statepath=BASE/'ops/STATE.json';state=read(statepath);state['updated_date_kst']=now
    state['current_stage']='WORD100_COMPLETE_DEVELOPMENT_COMPARISON';state['word100_training_20261013']=done
    state['next_single_action']='100단어 개발 비교 결과 검토. 웹 연결·최종 평가 범위 결정 전 기존 운영 모델 유지. 완료된 추출·학습·점수 재실행 금지.'
    write(statepath,state)
    contextpath=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';text=contextpath.read_text('utf-8-sig')
    def replace(text,num,block):
        start=text.index('\n',text.index(f'## {num}. '))+1
        old=text.index('### 2026-10-13',start);end=text.index('\n### ',old+4)
        assert '100단어' in text[old:old+120]
        return text[:start]+'\n'+block+'\n'+text[end:]
    text=replace(text,8,f'''### 2026-10-13 AI Hub 정면 100단어 학습·개발 검증 완료
`ops/{U.name}/RESULT_KO.txt`, COMPLETION / INPUT_CHECK / METRICS_CHECK 참조. 상태 COMPLETE_DEVELOPMENT_COMPARISON, {now}.
학습 899개(ID01~09), 개발 300개(ID11~13), AIHub-only 랜덤 초기화 MaskedSignTCN 100클래스. 기존 웹 MediaPipe / 280채널 재사용.
전체 학습 {ma['correct']}/300 ({ma['accuracy_percent']:.2f}%), 전체·핵심 혼합 {mb['correct']}/300 ({mb['accuracy_percent']:.2f}%), {delta:+.2f}pp. 개선 {len(comp['improved'])}, 회귀 {len(comp['regressed'])}. 개발 자료로 epoch를 선택했으므로 독립 최종 시험 아님.
전체 입력 별도 재구성, 6개 원본 영상 시각 감사, 저장 점수·선택 기록 검산 PASS. Codex 작성·실행·별도 코드 검산; 수어 전문가/다른 에이전트 검수 아님.
원본 ZIP·기존 운영 모델 보존. NIKL/KSL/사용자/멘토 자료 및 보류 ID14~18·미배정10 미사용. 기존93클래스 성적과 직접 비교 금지.
캐시 W: word100_inputs_20261013, 모델 runs/word100_tcn_v1. 완료된 실행 반복 금지. 상세 단어별 결과 WORD_RESULTS.json.
''')
    text=replace(text,9,f'''### 2026-10-13 다음 — 100단어 개발 결과 검토
`ops/{U.name}/RESULT_KO.txt`와 WORD_RESULTS.json을 바탕으로 웹 시험 연결 여부 및 최종 평가 범위를 결정한다.
관절 캐시는 다음 어휘 확장에 재사용한다. 기존100+새 단어를 함께 학습할 계획을 고정한 뒤 별도 버전으로 진행한다.
이번 단위의 추가 튜닝·단어 교체·보류 평가·웹 기본 교체를 자동 시작하지 않는다. 기존 결과는 개발 점수이며 새 사용자/동작 정확성 검증과 구분한다.
''')
    contextpath.write_text(text,'utf-8')
    (ROOT/'NEW_CHAT_HANDOFF.md').write_text(f'''# WordsInSilence 최신 인수인계
갱신: {now}. 먼저 AGENTS / CONTEXT 최신 §3·§8·§9 / WORKFLOW를 읽는다.

## 마지막 완료 — AI Hub 정면 100단어 학습·개발 검증
사용자의 100단어 진행 승인으로 `ops/{U.name}` 완료. RESULT_KO.txt / COMPLETION / INPUT_CHECK / METRICS_CHECK 참조.
학습 899개, 개발 300개(학습에 없는 ID11~13), fresh MaskedSignTCN / 280채널 / 100출력.
전체 학습 {ma['correct']}/300 ({ma['accuracy_percent']:.2f}%), 핵심 혼합 {mb['correct']}/300 ({mb['accuracy_percent']:.2f}%), {delta:+.2f}pp.
개선 {len(comp['improved'])}, 회귀 {len(comp['regressed'])}. 같은 개발 자료로 epoch 선택: 독립 최종 시험 아님. 기존93클래스 점수와 직접 비교 금지.
Codex 로컬 작성·실행·별도 계산 검산. 첫6 입력 및 4단계 폐기 학습 점검 후, 전체 입력/6원본/저장 점수/선택 이력 검산 완료.

## 보존과 재개
캐시: {OUT}. 모델: {RUN}. 완료된 추출·학습·개발 채점·검산 재실행 금지.
원본 ZIP과 운영 체크포인트 보존. 사용자/NIKL/KSL/멘토 자료, 보류14~18/미배정10 미사용.
다음: 결과와 오답 분포 검토 → 웹 연결·최종 평가 범위 결정. 자동 추가 튜닝·운영 기본 교체 없음.
어휘 확장은 기존 관절 캐시 재사용 가능. 새 클래스 출력층 확장과 기존+새 자료 별도 학습 필요.

## 이전 완료 (반복하지 않음)
word_label_time_contract_20261013: 2494 후보/22388 학습 행, 의미충돌506 보류. 시간구간 정책 고정.
word_pilot12_inputs_20261013: 12영상/1444프레임 MediaPipe·RGB 입력 점검.
word_dataset_plan_20261013: 33ZIP/1.181TB/270000영상, 정면54000, 3000코드×18ID×5시점.
''','utf-8')
    assert read(statepath)['word100_training_20261013']==read(U/'COMPLETION.json')
    assert all(Path(done[k]).exists() for k in ['report','input_check','metrics_check'])
    print(result[:600])

if __name__=='__main__':main()
