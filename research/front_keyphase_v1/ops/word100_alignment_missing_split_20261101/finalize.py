from pathlib import Path
import json,hashlib,statistics,datetime
U=Path(__file__).resolve().parent;R=U.parents[3]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def fmt(value):return f'{value:.4f}' if value is not None else '비교 불가'
def main():
    ev=read(U/'EVALUATION.json');ver=read(U/'review/RESULT_VERIFICATION_FINAL.json');method=read(U/'review/POSTEXECUTION_REVIEW.json')
    compatible=read(U/'READOUT_ONLY_RESULT.json');compatible_review=read(U/'review/READOUT_ONLY_VERIFICATION.json')
    assert ver['status'].startswith('PASS') and method['status'].startswith('PASS')
    assert compatible_review['status'].startswith('PASS') and compatible['record_count']==108
    assert not (U/'COMPLETION.json').exists()
    assert not (U/'RUNNING.json').exists()
    scores=ev['decision']['dev_scores'];avg=statistics.mean(scores)
    base=ev['summary']['development/global_dtw/None']['same_kind_percent']
    prior=statistics.mean(ev['summary'][f'development/joint/{s}']['same_kind_percent'] for s in [42,43,44])
    words=[]
    for wid,label in [('WORD0739','순종'),('WORD1133','잘하다'),('WORD1567','조상'),('WORD2523','이집트')]:
        old=ev['byword'][f'development/{wid}/global_dtw/None']['same_kind_percent']
        vals=[ev['byword'][f'development/{wid}/split/{s}']['same_kind_percent'] for s in [42,43,44]]
        words.append(f'{label}: 기존 전체정렬 {old:.2f}% → 이번 평균 {statistics.mean(vals):.2f}% ({statistics.mean(vals)-old:+.2f}%p)')
    cs=ev['control_summary']
    masks={scope:[cs[f'{scope}/right_hand_unobserved/split/{s}']['mean_original_frame_error'] for s in [42,43,44]] for scope in ['previous16','expanded32']}
    speeds={case:statistics.mean(cs[f'previous16/{case}/split/{s}']['mean_original_frame_error'] for s in [42,43,44]) for case in ['speed_075','speed_125']}
    rows=[s for s in ev['shorts'] if s['seed']==42]
    counts={'cases':len(rows),'actual_changed_cases':sum(s['actual_observed_changes']>0 for s in rows),
            'no_observed_change_cases':sum(s['actual_observed_changes']==0 for s in rows),
            'own_changed_cells':sum(s['actual_observed_changes'] for s in rows),
            'shared_changed_cells':sum(s['new_query']['shared'] for s in rows),
            'missing_counterpart_changed_cells':sum(s['new_query']['own_known_uncompared'] for s in rows)}
    sample=next(s for s in ev['shorts'] if s['alias']=='P033' and s['seed']==42 and s['control']=='short_shape_change_5f')
    now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
    broad=ev['decision']['clear_development_improvement'];tech=ev['decision']['technical_pass']
    report=f'''미관측과 짧은 차이를 분리하는 공통 계산 보완 — 결과
작성 {now}

범용성
특정 두 단어를 고친 것이 아니다. 모든 표현이 같은 계산을 사용하는 공통 부품을 수정했다.
계산에는 단어 이름·단계 정답·사람 ID별 예외가 들어가지 않는다. 정답 단계는 사후 검산에만 사용했다.
다만 모든 한국수어에서 좋아졌다는 뜻은 아니다. 이번 실제 단계 평가는 공개4단어, 기술 검사는 총12단어 범위다.

먼저 결론
정렬 비용을 바꾼 후보는 다른 사람끼리의 비교에서 크게 후퇴해 미채택했다. 기존 정렬 방식은 유지한다.
양쪽 시간축에 차이와 미관측을 남기는 기능은 정렬 계산과 분리해 구현했고, 기존36경로×3seed의108개 연결에서
원래 대응 시점을 하나도 바꾸지 않음을 확인했다. 이 부분은 비교 근거 기록 개선이며, 정렬 정확도 향상은 아니다.

완료한 변경
1. 미관측 벌점을 경로 선택 비용에서 분리했다. 양쪽에 실제 관측된 값의 차이만 같은 고정 가중치로 계산한다.
   안 보이는 손을 정상 손이나 실제 차이0으로 바꾼 것은 아니다. 미관측과 비교 불가는 따로 남긴다.
2. 참고 영상 기준 평균 하나로 끝내지 않고, 양쪽 영상의 각 시간·특징에 평균 차이/최대 차이/미관측/부분 근거를 남겼다.
   예를 들어 5번의 변화가 참고2시점에 몰려도 실제 비교 영상의5시점은 각각 확인할 수 있다.
   최대 차이는 관측 근거이며, 이것만으로 틀린 수어라고 판정하지 않는다.
3. 비용이 정확히 같은 경우만 시간축을 덜 늘이거나 줄이는 대표를 택한다. 여러 가능한 경로의 모호함은 유지한다.

시험 범위
기존 공개4단어16영상36쌍(FIT24/DEV12), 저장 관측값 대조142개에 동일한 조건을 적용했다.
추가 공개8단어32영상에서는 원본 그대로/오른손 미관측/중앙5프레임 오른손각도 변화의96개 대조를 실행했다.
기존 저장3seed 출력 모두 사용: (36+142+96)*3 =822개 새 정렬. 새 모델추론0, 새학습0.
20단어 참고은행 중 저장 원출력이 있는12단어48영상만 해당한다. 원출력 없는8단어32영상은 미지원으로 명시했다.
추가8단어는 계산 성질 검사이며 새로운 의미 정답 평가가 아니다. 기존 결과/경로/점수는 재실행하지 않았다.

기술 계산 결과
고정 기술 조건 전체 통과: {tech}
오른손 정보만 가린 동일 영상의 평균 시점 오차:
  기존16영상: 이전 결합정렬 약2.89프레임 → 이번 {statistics.mean(masks['previous16']):.6f}프레임
  추가32영상: 이번 {statistics.mean(masks['expanded32']):.6f}프레임
손 정보가 없다는 이유만으로 미관측 벌점을 줄이려고 시간을 옮기던 직전 현상은 이 대조들에서 줄었다.
원본 그대로 비교와 양쪽 전체 프레임 보존은 EVALUATION의 기술 조건을 따른다.
속도 대조의 평균 원프레임 오차: 0.75배 {speeds['speed_075']:.4f}, 1.25배 {speeds['speed_125']:.4f}.
속도 대조는 저장 프레임의 반복/생략이며 실제로 다른 속도로 다시 촬영한 영상이 아니다.

짧은 변화 근거 보존(대표seed42; 다른seed도전부저장)
오른쪽/왼쪽/추가오른쪽 합계 {counts['cases']}개 대조 중 실제관측변화 {counts['actual_changed_cases']}개,
미관측 때문에 실제 변경된 관측값이 없던 {counts['no_observed_change_cases']}개는 탐지성공으로 세지 않는다.
바뀐 관측 셀 {counts['own_changed_cells']}개 = 상대와비교된 {counts['shared_changed_cells']}개 + 상대근거부족 {counts['missing_counterpart_changed_cells']}개.
P033 오른손5프레임 사례:
  이전경로 참고기준 평균 {fmt(sample['old_path_reference']['supported_mean'])}, 실제비교영상기준 평균 {fmt(sample['old_path_query']['supported_mean'])}.
  이번경로 참고기준 평균 {fmt(sample['new_reference']['supported_mean'])}, 실제비교영상기준 평균 {fmt(sample['new_query']['supported_mean'])}.
이들은 저장된 정규화 각도특징 차이이며 정확도나 각도 도수가 아니다.
부분관측 셀은 shared라고 적혀 있어도 모든 대응링크가 관측됐다는 뜻은 아니다. 혼합/미관측링크 수를 별도 저장했다.

같은 동작 단계 연결 결과
기존 전체정렬 {base:.2f}%, 직전 미채택 결합정렬 평균 {prior:.2f}%.
이번 평균 {avg:.2f}% (3seed 범위 {min(scores):.2f}~{max(scores):.2f}%), 기존 전체정렬 대비 {avg-base:+.2f}%p.
{chr(10).join(words)}
전체 단계정렬의 명확한 개선 기준 통과: {broad}
이 수치는 개발 비교의360개 참고프레임 기회에서 같은 관측단계 종류에 연결됐는지를 센 것이다.
TCN 단어인식률·실제 교육교정 정확도·정확한 반복회차 정답률이 아니다. 경계/불확실/다른단계도 분모에서 빼지 않았다.
36쌍 전체참고4380/확실단계1080기회를 유지했다. 잘된 일부 표현이나seed만 선택하지 않았다.

후퇴 원인 추가 검산
DEV12쌍 seed42의 기존/새 경로에 똑같은 새 비용식을 적용해 산술 분해했다(재정렬하지 않음).
새 경로는12쌍 모두 실제 공통 관측 근거의 가중량을 줄였다. 평균0.34761→0.16525,
비용0.03977→0.01761로 감소했지만 관측된 부분끼리의 조건부 차이는 평균0.11055→0.14615로 커졌다.
즉 더 잘 맞아서 비용이 작아졌다고 볼 수 없었다. 서로 다른 단계로 연결된 가중량도360중약176.67이었다.
관측이 적은 곳으로 연결하면 비용이 작아질 수 있다는 사전 반례가 실제 개발 비교에서도 문제를 일으켰다.
다른 구성요소의 기여까지 모두 분리한 실험은 아니며, 이 결과로 새 가중치를 튜닝하지 않았다.

아직 남은 문제
미관측 벌점을 제외해도, 값이 없는 시간대에 변화가 연결되는 또 다른 정렬 오류는 남는다.
사전 반례 q=[0,1,0,0], r=[0,0,미관측,0]에서 변화1이 상대의 빈 시점으로 연결돼 경로비용0이 된다.
이번에는 그 변화시점을 비교불가/null로 남기며 정상이라고 판정하지 않는다. 이 반례의 시점 정렬 자체를 해결한 것은 아니다.
즉 모든프레임을보존하거나 작은계산점수를 얻는것만으로 같은동작단계를 맞췄다고 확신하면 안 된다.
정상 손내림·교차 입력을 지우거나 좌우를 임의로 바꾸지는 않았지만, 실제 학습자에게 잘못된 경고가 없는지의 교육정답 시험은 아니다.
공개 정상영상·합성관측값 대조, 같은Codex물리단계주석, 이미본개발자료의 한계를 유지한다.

적용 및 다음 방향
새 정렬 계산은 미채택으로 보존했다. 사용할 수 있는 별도 부품은 readout_only.py의 compare_existing_path다.
기존 정렬 결과를 입력받아 양축 근거만 덧붙인다. 호환 시험에서는 align/solve/cost 호출을 강제로 막고
기존36개 경로의 모든 대응쌍을 그대로 유지한108개 기록을 검산했다. 기존82.16%는 저장 점수 그대로이며 재채점하지 않았다.
실패한 새 후보를 이 점수로 바꾸거나 성공으로 포장하지 않는다. 운영 웹/DTW와 C·T3-F는 변경하지 않았다.
이번에는 비용/집계의 수학적 문제를 확인한 것이므로 추가학습하지 않았다. 사용자·멘토촬영 접근/학습/튜닝/선택0.
다음은 기존 정렬을 출발점으로, 근거가 없는 구간의 대응을 확정할 수 있는지 앞뒤 관측을 확인해 구분하는 작업이다.
이를 정상적인 손내림/속도차이와 구분해서 검증해야 하며, 현재 후보가 모든 단어의 단계정렬을 해결했다고 주장하지 않는다.
미관측 벌점과 가중치를 계속 바꾸기보다, 기존 대응이 모호한 곳을 근거 없이 확정하지 않는 데 우선 집중한다.
새모델검색이나특정단어예외보다 관측이있는앞뒤연결과단계대응의모호함을 우선검증한다. 후속후보는아직실행하지않았다.

검증/보존
순수구현 semantic_feedback, 실행·통합 root, 방법검토 semantic_model, 독립산술검산 semantic_data_audit.
합성151검사 통과, 알려진 미해결 반례 보존. 실제822결과의 별도검산과 사후방법검토는 review를 참조한다.
첫 검산에서 확률 차이260건이 불일치했으나 검산 코드만32비트로 차이를 계산한 원인이었다.
최초FAIL을 보존하고 원구현과 같은64비트 계산으로 해당 부분만 재검산해 오류0을 확인했다. 모델/경로/평가 결과는 바꾸지 않았다.
같은Codex의 별도 문맥 검산이지 독립 한국수어 전문가의 의미 정답 평가는 아니다.
원본/과거동결결과/C·T3-F/운영보호핀 유지. 다운로드/외부전송/카메라/Git/운영교체0.
명확한 전체 성능 개선이 없어 HTML은 생략했다.
근거: EVALUATION.json, AXIS_ACCOUNTING.json, FROZEN.json, SELFTEST.json, review/RESULT_VERIFICATION_FINAL.json,
review/POSTEXECUTION_REVIEW.json, review/SHARED_SUPPORT_DIAGNOSIS.json, READOUT_ONLY_RESULT.json, review/READOUT_ONLY_VERIFICATION.json.
'''
    (U/'RESULT_KO.txt').write_text(report,encoding='utf8')
    status='COMPLETE_NEW_ALIGNMENT_REJECTED_ADDITIVE_READOUT_VERIFIED'
    stable=['PLAN_KO.txt','FROZEN.json','TASKS.json','AUTHORIZATION.json','SELFTEST.json','RESULT_MANIFEST.json','EVALUATION.json','AXIS_ACCOUNTING.json','RESULT_KO.txt','review/RESULT_VERIFICATION.json','review/RESULT_VERIFICATION_FINAL.json','review/POSTEXECUTION_REVIEW.json','review/SHARED_SUPPORT_DIAGNOSIS.json','readout_only.py','READOUT_ONLY_CONTRACT.json','READOUT_ONLY_RESULT.json','review/READOUT_ONLY_VERIFICATION.json']
    completion=dict(status=status,at_kst=now,technical_pass=tech,clear_development_improvement=broad,dev_mean=avg,dev_range=[min(scores),max(scores)],
        baseline=base,delta_pp=avg-base,paths=822,unchanged_path_readout_records=108,new_training=0,new_model_inference=0,private_used=False,production_changed=False,html_created=False,active=None,
        new_alignment_adopted=False,additive_readout_compatible=True,unresolved_missing_island=True,artifacts={str(U/f):sha(U/f) for f in stable})
    write(U/'COMPLETION.json',completion)
    write(U/'CURRENT_EXECUTION.json',dict(status=status,active=None,completion=str(U/'COMPLETION.json'),session=49205,exit_code=0))
    statepath=U.parent/'STATE.json';state=read(statepath)
    nxt='Keep existing global alignment and additive two-axis readout; distinguish unsupported temporal correspondence using before/after evidence. Do not repeat ad-hoc missing-penalty/weight search; validate normal lowering/speed and preserve full scope.'
    state.update(updated_date_kst=now,current_stage=status,next_single_action=nxt)
    state[U.name]={k:v for k,v in completion.items() if k!='artifacts'}|dict(unit=str(U),result=str(U/'RESULT_KO.txt'),completion=str(U/'COMPLETION.json'),next=nxt)
    write(statepath,state)
    p=R/'WordsInSilence_PROJECT_CONTEXT.md';s=p.read_text(encoding='utf-8-sig')
    entry=f'''### 2026-11-01 미관측 비용·양축 차이 공통보완 완료 — 연구후보 검증

`{U.name}` 완료. 단어이름예외없는 shared residual 경로 + 양축mean/max/missing/partial근거. 기존4단어36쌍/142대조 및 추가8단어32영상96대조, 저장3seed822경로. 추가학습/모델추론0. 기술조건PASS={tech}, 오른손마스킹 동일영상 기존16+추가32 평균오차 {statistics.mean(masks['previous16']):.4f}/{statistics.mean(masks['expanded32']):.4f}프레임. 짧은변화양축셀보존/모르는값null, 비용0≠정상. DEV같은관측단계 기존82.16/직전80.23→이번{avg:.2f}%(범위{min(scores):.2f}~{max(scores):.2f}), 전체개선기준PASS={broad}, 새정렬미채택. 실제공통관측량감소가비용을낮춘기전DEV12쌍저장경로분해확인. TCN/교육정오정확도아님. 기존global경로는유지하고readout_only로양축근거만추가하는108호환기록검산,기존82.16점수재사용/재채점X. Missing-island반례의정렬자체미해결. 첫검산float32차이260건은검산dtype수정후영향부분PASS,원FAIL보존/결과불변. 운영/C·T3F/개인/원본보존,HTML없음. RESULT/COMPLETION·별도검산참조, 모든실행종료.

'''
    s=s.replace('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인','## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n\n'+entry.rstrip(),1)
    nextentry='''### 2026-11-01 다음 단일 작업 — 미관측 구간의 대응 모호함과 차이 분리

missing_split 완료822경로/점수재실행X. 새shared-only정렬미채택, 기존global경로+readout_only 양축근거를기준으로유지한다. 미관측벌점/가중치의추가순회를멈추고앞뒤관측연결로대응가능/모호/비교불가범위를구분하는한정검증부터. 108호환기록은정렬정확도개선아닌경로불변근거기록개선. 빈구간에변화가연결되는정렬반례는미해결로유지하며변화시점을null로남기는것만으로중요동작대응완성/교육오류탐지성공주장X. 정상손내림·속도차이분모유지. 새모델순회·단어예외·사용자튜닝X. 이번자료는개발/기술대조이며새독립의미평가X. 목표·변천사유지.

'''
    s=s.replace('## 9. 다음 단일 작업','## 9. 다음 단일 작업\n\n'+nextentry.rstrip(),1);p.write_text(s,encoding='utf8')
    (R/'NEW_CHAT_HANDOFF.md').write_text(f'''# WordsInSilence 최신 인수인계

현재 `{U.name}` 완료, 실행중없음. 사용자: 두계산문제를특정단어아닌범용보완으로진행요청. 모든표현공통코드,검증범위와보편성구분.

공통관측잔차정렬(미관측벌점분리/고정D그룹가중치/exacttie최소warp)+양축mean/max/missing/partial근거 구현. 기존4단어16영상36쌍+142대조, 추가8단어32영상96대조. 저장3seed822경로, 새추론/학습0. 20단어은행에서original3없는8단어32영상미지원그대로.
기술조건PASS={tech}, rightmask동일영상 기존16/추가32평균오차 {statistics.mean(masks['previous16']):.4f}/{statistics.mean(masks['expanded32']):.4f}. DEV같은관측단계 기존82.16/직전80.23→{avg:.2f}%({min(scores):.2f}~{max(scores):.2f}), 전체개선기준PASS={broad}. TCN/교육정오정확도아님. 모든원기회분모유지, 실제변형0반례는탐지성공X. 합성151검사와최종검산JSON참조.

새shared-only정렬은미채택. DEV12쌍seed42공통관측가중량.34761→.16525로줄어비용낮춘원인확인. 기존global경로36×3seed를그대로둔 readout_only108호환replay완료,정렬검색/점수재계산0. 양축근거부품만분리활용가능,운영미적용. 알려진미해결: q=[0,1,0,0],r=[0,0,?,0]에서실제변화가빈시점으로정렬될수있음. 차이0성공이아닌NOT_COMPARABLE/null로남기지만정렬자체해결아님. 다음은기존global+양축근거에서앞뒤관측으로미관측대응모호함/보류범위를정상속도·손내림과구분검증. 미관측벌점/가중치계속변경하는후보순회X. 후속후보아직미실행. 현재완료결과재정렬/재채점/튜닝X.

결과 `{U/'RESULT_KO.txt'}`; COMPLETION/EVALUATION/AXIS_ACCOUNTING/FROZEN/SELFTEST와review/RESULT_VERIFICATION_FINAL·POSTEXECUTION_REVIEW·READOUT_ONLY_VERIFICATION. 최초RESULT_VERIFICATION은검산float32뺄셈260불일치FAIL보존,영향learned3를float64로만바꿔재검산PASS,허용오차/원결과/점수불변. exec49205/27934/87099종료0,잠금없음. 구현semantic_feedback/실행root/방법semantic_model/검산semantic_data_audit. 동일Codex검산≠수어전문가정답.
운영/C·T3F/개인·멘토촬영/원본·기존동결보존. 카메라·다운로드·외부전송·Git·운영교체0. 목표이력 Docs/PROJECT_DIRECTION_HISTORY_KO.txt. HTML생성여부는결과보고기록참조.
''',encoding='utf8')
    hist=R/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt'
    with hist.open('a',encoding='utf8') as f:f.write(f'\n\n2026-11-01 두 계산문제의 공통 보완 검증\n미관측벌점과실제관측차이를분리하고양쪽시간축짧은변화의평균/최대/미관측근거를남겼다. 4단어단계평가+추가8단어기술대조,822경로 완료. 기술조건PASS={tech}, DEV단계연결{base:.2f}→{avg:.2f}%,전체개선기준PASS={broad}로새정렬미채택. 실제공통관측량을줄여비용이낮아지는문제를확인했다. 양축근거만기존global경로에덧붙이는108호환기록은대응시점불변으로검산완료;정렬개선아닌근거기록개선이다. 특정단어규칙X,새학습/추론/운영적용X. 빈구간으로변화가연결되는반례는여전히정렬불확실성으로남아교육정오완성을뜻하지않는다. 다음은비용가중치순회보다기존경로에서앞뒤관측과보류근거를검증한다. 근거 {U / "RESULT_KO.txt"}\n')
    docs=[R/'WordsInSilence_PROJECT_CONTEXT.md',R/'NEW_CHAT_HANDOFF.md',statepath,hist]
    for path,h in completion['artifacts'].items():assert sha(path)==h,path
    assert read(U/'STATUS.json')['done']==822 and read(statepath)['current_stage']==status
    write(U/'FINAL_DOCUMENT_VERIFICATION.json',dict(status='PASS',at_kst=now,documents={str(p):sha(p) for p in docs},completion_sha256=sha(U/'COMPLETION.json')))
    print(json.dumps({k:v for k,v in completion.items() if k!='artifacts'},ensure_ascii=False))
if __name__=='__main__':main()
