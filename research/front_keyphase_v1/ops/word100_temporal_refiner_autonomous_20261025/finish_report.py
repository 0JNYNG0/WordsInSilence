from pathlib import Path
from datetime import datetime
import json,hashlib,html
import numpy as np

U=Path(__file__).resolve().parent;ROOT=U.parents[3];B=U.parent/'word100_temporal_public_bundle_20261025'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,value): (U/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    c=read(U/'COMPARISON.json');control=read(U/'CONTROL_RESULTS.json');d=read(U/'downstream/RESULT.json')
    v=read(U/'SAVED_SCORE_VERIFICATION.json');ind=read(U/'DOWNSTREAM_INDEPENDENT.json')
    assert v['status']=='PASS' and ind['status'].startswith('PASS'),ind
    assert not (U/'RUNNING.json').exists()
    frozen=read(U/'FROZEN_PROTOCOL.json')
    for p,h in frozen['source_pins'].items():assert sha(p)==h,(p,'frozen source changed')
    protected=read(B/'PROTECTED_BEFORE.json')
    for p,h in protected.items():assert sha(p)==h,(p,'protected source changed')
    base=control['test']['linear_interpolation'];ag=c['aggregate']
    rows=[]
    for condition,name in [('linear','앞뒤 좌표를 직선으로 잇기'),('B2','관절 + 시간 학습'),('B3','RGB + 관절 + 시간 학습')]:
        if condition=='linear':mean=lo=hi=base['mean_euclidean']
        else:
            x=ag[condition]['test']['mean_euclidean'];mean=x['mean'];lo=x['min'];hi=x['max']
        rows.append(dict(condition=condition,name=name,error=mean,low=lo,high=hi,relative_change_percent=(mean/base['mean_euclidean']-1)*100))
    now=datetime.now().astimezone().isoformat()
    result=dict(status='COMPLETE_BOUNDED_EXPERIMENT_NEW_REFINERS_NOT_ADOPTED',at=now,
        stages={'public_bundle':'COMPLETE_61_CLIPS_24_EXPRESSIONS_9_PEOPLE','robust_proxy_training':'COMPLETE_B2_B3_3_SEEDS_EACH','fixed_classifier_and_DTW_diagnostic':'COMPLETE','optional_handshape_supervision':'NOT_ELIGIBLE_NO_INDEPENDENT_HANDSHAPE_LABELS_IN_CURRENT_BUNDLE'},
        inputs={'clips':61,'expressions':24,'people':9,'frames':7647,'refiner_train_clips':39,'refiner_dev_clips':7,'refiner_test_clips':15,'synthetic_train_cases':213,'synthetic_dev_cases':39,'synthetic_test_cases':81},
        proxy_results=rows,normal_coordinate_preservation_checks=v['preserved_observation_slots_checked'],
        classification={'C':'61/61 in all8 conditions','T3-F':'61/61 in all8 conditions','improvement_pp':0,'regressions':0,'unseen_user_accuracy':None,'limit':'All classifier people previously seen; refiner split does not make them unseen for classifier'},
        DTW={'pairs':176,'supported':110,'withheld':66,'queries_without_reference':17,'observed_residual_checks':660,'new_estimates_per_candidate':1994,'observed_promotion':0,'correct_feedback_or_false_alarm_improvement':'not established'},
        decision='Keep existing C/T3-F and production DTW/web. Do not adopt B2/B3. Valid features, separate evidence and evaluation tools retained.',
        known_remaining=['V05 existing wrist error survives quality rules and preserved-coordinate refiner','No independent hidden-hand or21point truth; synthetic missing pixels-visible task differs from real occlusion','Short1/3frame proxy holes worsened; B2 seven-frame holes improved modestly','RGB small pilot does not show generalization benefit','Coarse visible36frames are diagnostic, not expert anatomical labels'],
        next_recommendation='Before more learned fusion, obtain an independent visible-hand recovery benchmark and trustworthy hand-pose supervision; compare public-frame redetection and transparent interpolation. Do not expand this failed RGB candidate to all words.',
        user_recording_access=0,training_on_user=False,production_changed=False,protected_unchanged=len(protected))
    write('FINAL_RESULT.json',result)
    # Standard plotting, no generated or altered source photographs.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    colors=['#94a3b8','#157f87','#db7c47']
    means=np.array([r['error']*100 for r in rows]);lower=means-np.array([r['low']*100 for r in rows]);upper=np.array([r['high']*100 for r in rows])-means
    axes[0].bar(['Interpolation','Pose + time','RGB + pose + time'],means,color=colors,yerr=np.vstack([lower,upper]),capsize=5)
    for i,m in enumerate(means):axes[0].text(i,m+.18,f'{m:.2f}',ha='center',fontsize=12)
    axes[0].set_title('Held-out refiner proxy error (lower is better)');axes[0].set_ylabel('Distance / shoulder width x 100');axes[0].set_ylim(0,7)
    for cond,name,col in [('linear','Interpolation',colors[0]),('B2','Pose + time',colors[1]),('B3','RGB + pose + time',colors[2])]:
        vals=[control['test']['linear_interpolation']['by_hole_frames'][str(h)]['mean_euclidean']*100 if cond=='linear' else ag[cond]['test']['by_hole_frames'][str(h)]['mean_euclidean']['mean']*100 for h in (1,3,7)]
        axes[1].plot([1,3,7],vals,'o-',label=name,color=col)
    axes[1].set_title('Effect of missing interval length');axes[1].set_xlabel('Artificially hidden frames');axes[1].set_ylabel('Proxy error / shoulder width x 100');axes[1].set_xticks([1,3,7]);axes[1].legend(fontsize=9)
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    fig.savefig(U/'proxy_comparison.png',dpi=150);plt.close(fig)
    tr=''.join(f"<tr><td>{html.escape(r['name'])}</td><td>{r['error']*100:.3f}</td><td>{r['low']*100:.3f}–{r['high']*100:.3f}</td><td>{'기준' if r['condition']=='linear' else f'+{r["relative_change_percent"]:.2f}% 오차 증가'}</td></tr>" for r in rows)
    holes=''.join(f"<tr><td>{h}프레임</td><td>{base['by_hole_frames'][str(h)]['mean_euclidean']*100:.3f}</td><td>{ag['B2']['test']['by_hole_frames'][str(h)]['mean_euclidean']['mean']*100:.3f}</td><td>{ag['B3']['test']['by_hole_frames'][str(h)]['mean_euclidean']['mean']*100:.3f}</td></tr>" for h in (1,3,7))
    body=f'''<header><p class="eyebrow">WORDS IN SILENCE · RESEARCH CHECK · 2026.10.25</p><h1>RGB·관절·시간 보완<br>학습과 프로젝트 비교 완료</h1><p class="lead">6회 학습을 마쳤지만, 이번 새 후보의 전반적인 성능 향상은 확인하지 못했습니다.<br>기존 C·T3-F와 운영 DTW·웹은 그대로 유지합니다.</p><span class="pill">후보 미채택</span> <span class="pill light">사용자 촬영 접근·학습 0</span></header>
    <section><h2>이번에 어디까지 했나</h2><div class="steps"><div><b>01 자료 준비</b><p>61영상 · 24표현 · 9명<br>7,647프레임 RGB+관절</p></div><div><b>02 보완기 학습</b><p>관절+시간 / RGB 추가<br>각 3회, 총 6회 완료</p></div><div><b>03 프로젝트 영향</b><p>고정 C·T3-F 분류<br>여러 참고 DTW 비교 완료</p></div><div><b>04 손 모양 감독학습</b><p>독립 손 모양 정답 미확보<br>조건 미충족으로 미실행</p></div></div><p>4번은 신뢰할 수 있는 주석이 있을 때 진행하는 선택 단계입니다. Gloss는 단어·참고 연결에 사용했고, 손가락 정답으로 대신하지 않았습니다. 이번 실험 완료가 프로젝트 전체 피드백 기능의 검증 완료를 뜻하지는 않습니다.</p></section>
    <section><h2>학습 전 문제를 어떻게 다뤘나</h2><p>기존 MediaPipe 좌표를 모두 정확한 정답으로 믿으면 기존 오류를 새 모델도 따라 배울 수 있습니다. 원래 좌표는 보존하고, 팔·손목 연결의 큰 불일치나 다른 이상을 동반한 순간적인 튐에는 낮은 학습 가중치를 주었습니다. 손 교차나 빠른 동작만으로 좌표를 삭제하지 않았습니다.</p><p>기준 계산은 학습 사람 01~06만 사용했습니다. 유효 손 좌표 125,787개 중 삭제는 0개이며, 113,029개는 가중치 1, 12,707개는 0.75, 51개는 0.5입니다. 큰 좌표 오차의 학습 영향을 완화하는 손실도 적용했습니다.</p><div class="note"><b>해결된 것으로 처리하지 않은 문제</b><br>기존 V05 손목 위치 의심 사례는 이 공통 규칙으로 걸러지지 않았습니다. 팔과 손의 추정이 서로 비슷하게 틀릴 수 있기 때문입니다. 따라서 이번 학습은 <b>불완전한 좌표를 대상으로 한 제한적 복원 시험</b>이며 실제 손 관절 정답을 확보했다는 뜻이 아닙니다.</div></section>
    <section><h2>누락된 좌표를 더 잘 복원했나</h2><p>보완기 학습에 사용하지 않은 사람 08·09의 15영상에서, 정상적으로 기록된 손 좌표를 1·3·7프레임씩 가리고 복원했습니다. 총 81시험입니다. RGB 영상에는 손이 보이는 합성 누락 시험으로, 실제로 손이 가려진 상황과는 구분합니다.</p><img class="chart" src="proxy_comparison.png" alt="3조건의 복원 오차와 누락 길이별 오차 그래프"><table><thead><tr><th>방법</th><th>평균 오차</th><th>3회 범위</th><th>직선 보간 대비</th></tr></thead><tbody>{tr}</tbody></table><p class="unit">단위: 어깨 폭을 100으로 보았을 때의 평균 2D 거리. 낮을수록 저장된 대리 정답과 가깝습니다. 정답률·수어 수행 정확도 %가 아닙니다. 직선 보간은 학습하지 않는 기준 방법입니다.</p><table><thead><tr><th>가린 길이</th><th>직선 보간</th><th>관절+시간</th><th>RGB 추가</th></tr></thead><tbody>{holes}</tbody></table><p>관절+시간 후보는 7프레임 누락에서 약 1.79% 오차 감소가 있었지만, 1·3프레임에서는 오차가 증가했습니다. 가중 손실만 보면 평균 2.25% 좋아졌지만, 원래 거리·손목·상대 손가락 지표가 함께 개선된 결과는 아닙니다. RGB 후보는 3회 모두 전체 원거리 오차가 증가했습니다.</p></section>
    <section><h2>TCN과 T3-F의 단어 인식은?</h2><div class="stats"><div><b>61 / 61</b><span>기존 C · 모든 조건</span></div><div><b>61 / 61</b><span>T3-F · 모든 조건</span></div><div><b>0%p</b><span>인식률 변화</span></div></div><p>B0 기존 검출, B1 재검출 통합, B2·B3 각 3회 결과를 같은 61영상에 적용했습니다. 후보 입력은 31개 영상에서 실제로 달라졌고 모델 점수도 변했지만 1위는 유지됐습니다.</p><div class="note"><b>새 사용자 인식률 100%라는 뜻이 아닙니다.</b><br>이 사람들은 C·T3-F의 기존 학습에 이미 포함돼 있습니다. 보완기에서만 학습·검증·평가 사람을 분리했습니다. 이미 전부 맞추는 묶음이라 추가 인식 개선을 입증할 수 없는 천장효과가 있습니다. 기존 라이브 전처리를 통째로 재현한 시험도 아니며, 시간 범위·몸/손 크기 기준을 고정한 입력 효과 비교입니다.</div></section>
    <section><h2>DTW와 관측 근거는 어떻게 달라졌나</h2><ul><li>같은 단어를 수행한 다른 학습 사람의 참고 <b>176쌍</b>을 각각 비교했습니다. 본인·같은 사람 참고는 제외했습니다.</li><li><b>110쌍</b>은 정렬 근거가 충분했고 <b>66쌍</b>은 근거 부족으로 수치를 보류했습니다. 17개 영상은 이 작은 묶음 안에 다른 사람 참고가 없었습니다.</li><li>후보마다 64개 짧은 구간에서 <b>1,994개 관절-프레임</b>을 추정했습니다. 이는 1,994개의 오류를 고쳤다는 뜻이 아닙니다.</li><li>기존 관측 기반 위치·각도·팔 차이는 <b>660개 비교에서 정확히 보존</b>됐습니다. 추정값을 실제 관측으로 바꿔 표시한 경우는 0입니다.</li><li>손이 사라져도 전체 시간 분모는 줄이지 않았고, 비교 근거가 없으면 0오차 대신 ‘근거 부족’으로 남겼습니다. 내려놓은 손·한 손 동작 자체를 오류로 판정하지 않았습니다.</li></ul><p>새 추정 구간에서는 RGB 후보의 참고 위치 차이가 작아진 반면 손가락 각도 차이는 커졌습니다. 여러 사람이 수행한 참고 동작과의 차이이므로, 이 수치만으로 복원이 정확해졌거나 사용자가 잘못 수행했다고 판단할 수 없습니다. 잘못된 교정 경고가 줄었다는 효과도 아직 입증하지 못했습니다.</p></section>
    <section><h2>정상 손과 교차를 훼손하지 않았나</h2><p>6회 후보 전체에서 기존 관측 <b>1,681,458개 점·시점 슬롯</b>의 좌표 보존을 독립 검산했습니다. 이는 실제 관절이 모두 맞았다는 뜻이 아니라, 원래 입력을 변경하지 않았다는 확인입니다.</p><p>기존 원본 24장면 외에 좌표를 보기 전에 고른 원본 12장면을 추가 검토했습니다. 새 12장면 중 범위 안의 비교 가능한 손목 11개는 기존·새 조건 모두 고정한 대략적 위치 범위 안에 있었습니다. 새 후보로 추가된 추정 손목은 이 표본에 없으므로 복원 정확도 향상을 검증한 자료로 쓰지 않았습니다. 이 원본 검토는 AI의 대략적 2D 위치 판단이며 전문가의 정밀 21관절 주석이 아닙니다.</p></section>
    <section><h2>왜 RGB를 더했는데 좋아지지 않았을까</h2><p><b>확인된 사실:</b> RGB 후보는 학습 자료와 개발 검증의 대리 오차를 줄였지만, 보완기가 배우지 않은 사람의 평가 자료에서는 오차가 커졌습니다. 기존 좌표의 오류도 여전히 남아 있습니다. 1·3프레임처럼 아주 짧은 누락에서는 단순 보간이 더 안정적이었습니다.</p><p><b>아직 원인 후보인 부분:</b> 39개 학습 영상의 213개 합성 시험은 작은 규모입니다. 사용한 RGB 특징은 일반 이미지용 MobileNet의 고정 특징이며 손가락 위치를 직접 배우도록 훈련된 표현이 아닙니다. 작은 자료에 대한 과적합, 정밀 위치 정보 부족, RGB와 관절의 정답 오류가 영향을 주었을 수 있으나 이번 비교만으로 각각의 원인을 확정할 수는 없습니다.</p></section>
    <section><h2>최종 판단과 다음의 효율적인 방향</h2><p><b>이번 B2·B3를 운영에 넣지 않습니다.</b> 기존 C·T3-F와 웹·DTW는 보존했습니다. 이번 결과가 RGB와 관절 융합 전체의 실패를 뜻하지는 않지만, 이 작은 후보를 모든 단어로 확대할 근거는 없습니다.</p><ol><li>현재 이미지에서 손을 다시 찾는 방법과 단순 시간 보간을 기본 비교 기준으로 유지합니다.</li><li>보완기가 실제로 바꾸는 구간을 대상으로, 좌표를 보지 않고 확인한 손 위치·손 모양 근거를 확보합니다. 사람별 분리와 자연 가림/검출 누락 구분을 유지합니다.</li><li>그 근거로 정밀 손 관측에 적합한 RGB 표현을 검증한 뒤 다음 학습을 설계합니다. Gloss를 손 모양 정답으로 바꾸거나 사용자 촬영에 맞추지 않습니다.</li></ol><p>선택적 손 모양 감독학습은 현재 확보 자료에 독립적인 손 모양·21점 정답이 없어 실행하지 않았습니다. 자료가 없는 상태에서 이름만 바꿔 학습하지 않았다는 의미이며, 외부에 적합한 자료가 전혀 없다는 뜻은 아닙니다.</p></section>
    <section><h2>검증 기록</h2><p>구현·실행: Codex 로컬. 독립 코드·산술·자료 검증: 별도 에이전트. Claude 외부 호출은 사용하지 않았습니다. 학습 결과 재추론 없이 저장 결과를 검산했고 최대 수치 차이는 5.61×10⁻⁹였습니다. 원본·C·T3-F 등 보호 파일 32개 해시가 유지됐고, 사용자·멘토 촬영 접근은 0입니다.</p><p>내부 검토 JSON에 사후 설명을 덧붙여 동결 해시가 달라진 실수를 발견했습니다. 추가본을 별도 보존하고 원 검토 파일을 동결 해시와 바이트 단위로 일치하게 복원했습니다. 학습 코드·모델·실험 조건은 변경하지 않았습니다.</p><div class="links"><a href="FINAL_RESULT.json">최종 요약 JSON</a><a href="COMPARISON.json">6회 비교</a><a href="SAVED_SCORE_VERIFICATION.json">학습 결과 독립 검산</a><a href="downstream/RESULT.json">TCN·DTW 결과</a><a href="DOWNSTREAM_INDEPENDENT.json">downstream 독립 검산</a><a href="VISIBLE_COMPARISON.json">원영상 진단</a><a href="REVIEW_MUTATION_RECOVERY.json">검토 파일 복구 기록</a></div></section><footer>공개 자료의 격리 연구 결과 · 운영 모델 교체 없음 · {now}</footer>'''
    css='''*{box-sizing:border-box}body{margin:0;background:#edf2f7;color:#172636;font:17px/1.8 "Malgun Gothic",sans-serif}main{max-width:1160px;margin:0 auto;padding:38px 30px 70px}header{background:#112c42;color:#fff;padding:48px;border-radius:20px}h1{font-size:42px;line-height:1.35;margin:12px 0 22px;letter-spacing:-1.5px}.eyebrow{font:12px/1.5 sans-serif;letter-spacing:2px;color:#a4d7d7}.lead{font-size:20px;color:#e3eef4}.pill{display:inline-block;background:#f0b66e;color:#34291d;padding:5px 13px;border-radius:30px;font-size:14px;font-weight:bold}.pill.light{background:#29495c;color:#d4eded}section{background:white;border:1px solid #dce5ec;border-radius:16px;padding:32px 38px;margin-top:22px}h2{font-size:25px;line-height:1.4;margin:0 0 20px;letter-spacing:-.7px}p{margin:14px 0}.steps{display:grid;grid-template-columns:repeat(4,1fr);gap:13px}.steps>div{border-radius:12px;background:#f1f6fa;padding:18px;font-size:14px}.steps b{color:#157f87}.note{padding:18px 22px;background:#fff7e9;border-left:4px solid #db9a41;border-radius:6px;margin-top:18px}.chart{width:100%;height:auto;margin:12px 0}table{width:100%;border-collapse:collapse;margin:20px 0}th{text-align:left;background:#eaf2f6;font-size:15px}th,td{padding:13px 14px;border-bottom:1px solid #dbe4eb;font-size:15px}.unit{font-size:14px;color:#526679}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.stats>div{background:#eaf5f4;border-radius:12px;padding:22px;text-align:center}.stats b{display:block;font-size:35px;color:#157f87}.stats span{font-size:14px}.links{display:flex;flex-wrap:wrap;gap:10px}.links a{color:#156676;font-size:14px;background:#eef6f7;padding:8px 12px;border-radius:7px}li{margin:10px 0}footer{font-size:13px;text-align:center;color:#687b89;padding-top:25px}@media(max-width:800px){main{padding:16px}header,section{padding:24px}.steps,.stats{grid-template-columns:1fr}h1{font-size:30px}table{display:block;overflow-x:auto}}'''
    (U/'RESULT_KO.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RGB·관절 보완 검증 결과</title><style>'+css+'</style><main>'+body+'</main></html>',encoding='utf-8')
    text=f'''RGB·관절·시간 보완 자율 비교 완료 ({now})
공개61영상/24표현/9명. 보완기학습39/검증7/평가15영상. B2·B3각3seed80epoch완료.
TEST대리좌표오차(어깨폭100): 선형4.567, B2평균4.681(4.628~4.719), B3평균5.578(5.205~6.150).
선형대비 B2 +2.50%, B3 +22.15% 오차증가. B2가중손실 -2.25%, 7프레임오차 -1.79%만으로전체개선주장금지.
C/T3F 모두8조건61/61, 개선0%p. 이미분류기가본사람이므로새사용자100%아님. 31영상의입력/점수실제변경.
DTW176쌍110지원66보류,17쿼리참고없음. 관측잔차660보존. 각후보1994추정관절프레임은실관측아님.
V05의심손목오류는신뢰가중에서남아있음. 새로운12장원영상진단의ROI내11손목전부기존/후보같으며추정점없음; 복원정확도향상입증아님.
결론: B2/B3새후보미채택. 기존C/T3F/운영DTW웹보존. 사용자촬영접근/학습/튜닝/선택0. 보호32해시유지.
선택적손모양학습은현재자료독립21점/수형정답없어미실행. 다음은실제복원구간독립평가기준/적합한손RGB근거확보; 실패한후보무작정전단어확대금지.
실험점수의개선과실제손관절정확도및새사용자인식률을구분한다. 상세 RESULT_KO.html.
'''
    (U/'RESULT_KO.txt').write_text(text,encoding='utf-8')
    progress=read(U/'PROGRESS.json');progress.update(stage='COMPLETE_NOT_ADOPTED',at=now,execution_session=None,training_runs_completed=6,downstream_complete=True)
    write('PROGRESS.json',progress)
    next_action='현재 승인된 공개 관측보완 학습·B0~B3·고정 C/T3-F·다중참고 DTW 비교 완료. B2/B3미채택, 기존운영유지. 후속은 실제복원구간의 독립영상평가와 신뢰가능손관측정답 확보를 우선하며, 현재 TEST08/09 결과를 보고 같은시험을 튜닝하지 않는다. 선택적수형감독학습은 독립주석 미확보로 실행조건없음. 사용자자료 학습·튜닝·선택·접근 금지와 큰문제외 자율진행 원칙 유지.'
    context=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';ct=context.read_text(encoding='utf-8-sig')
    entry='### 2026-10-25 RGB·관절 보완 자율 학습·프로젝트 비교 완료 — 새후보 미채택\n- 61영상/24표현/9명, B2/B3각3seed80epoch 학습완료. 공통신뢰도loss가중·robust대리학습으로분리, V05실제오류해결주장없음.\n- TEST대리평균거리 선형0.045669/B2 0.046808(+2.50%)/B3 0.055782(+22.15%). B2가중손실/7프레임소폭이득은전체향상아님.\n- 고정C/T3F 8조건모두61/61, 개선0%p; 이미본사람이므로새사용자정확도아님. DTW176쌍110지원66보류/17쿼리참고없음/관측잔차660보존.\n- 6run 저장점수·epoch·checkpoint독립검산, 최대차5.61e-9. 원관측1681458슬롯보존, 추정11964슬롯실관측승격0, 보호32해시유지. 사용자촬영접근0.\n- 새원본12진단: ROI내11손목 기존/후보모두범위내, 추정점0→복원정확도입증아님. 기존24의V05의심남음. 선택적수형학습은독립21점/수형정답없어미실행.\n- 담당Codex+독립에이전트; 기존C/T3F/DTW웹미변경. 근거 `'+str(U.relative_to(ROOT)).replace('\\','/')+'/RESULT_KO.html`. 실행중프로세스없음.\n\n'
    ct=ct.replace('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n','## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n\n'+entry,1)
    ct=ct.replace('## 9. 다음 단일 작업\n','## 9. 다음 단일 작업\n\n### 2026-10-25 현재 비교 완료·후속 근거 우선\n\n'+next_action+'\n\n',1);context.write_text(ct,encoding='utf-8')
    state=U.parent/'STATE.json';s=read(state);s.update(current_stage='TEMPORAL_REFINER_AUTONOMOUS_COMPLETE_NOT_ADOPTED',updated_at=now,next_single_action=next_action)
    s[U.name]=result|dict(unit=str(U),report=str(U/'RESULT_KO.html'),active_execution=None);state.write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf-8')
    handoff=ROOT/'NEW_CHAT_HANDOFF.md';handoff.write_text('# WordsInSilence 최신 인수인계\n\n갱신 '+now+'\n\n마지막 완료: '+text+'\n다음: '+next_action+'\n\n현재 실행/학습잠금 없음. 이전pilot HOLD는역사보존,새단위의robustproxy학습이실제GT통과를뜻하지않음. 학습후REVIEW사후수정사건은원동결SHA정확복원+별도POSTTRAIN기록보존(REVIEW_MUTATION_RECOVERY.json). 동결검토/프로토콜/모델수정금지.\n근거: '+str(U/'RESULT_KO.html')+'\n',encoding='utf-8')
    audit=dict(status='COMPLETE_VERIFIED_NOT_DEPLOYED',at=now,protected_unchanged=len(protected),source_pins_unchanged=len(frozen['source_pins']),training_runs=6,user_recording_access=0,active_execution=None,
        artifacts={p.name:sha(p) for p in U.glob('*.json') if p.name!='FINAL_AUDIT.json'},
        reports={p.name:sha(p) for p in (U/'RESULT_KO.html',U/'RESULT_KO.txt',U/'proxy_comparison.png')},
        root_sync={str(p):sha(p) for p in (context,state,handoff)},
        implementation='Codex root and collaborators',verification='independent code, saved-metric and downstream arithmetic review; no score reruns')
    write('FINAL_AUDIT.json',audit)
    print(json.dumps({'report':str(U/'RESULT_KO.html'),'status':result['status'],'training_runs':6},ensure_ascii=False))
if __name__=='__main__':main()
