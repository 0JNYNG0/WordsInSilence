"""Close this bounded verification only if unchanged readiness fails and separate audit passes."""
from pathlib import Path
import json,hashlib,datetime
N=Path(__file__).resolve().parent;O=N.parent;F=N/'full24_native'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
    result=read(F/'FULL24_AUDIT.json');native=read(N/'NATIVE_DIAGNOSIS.json')
    support=read(N/'support_inventory/SUPPORT_RESULT.json')
    prior=read(O/'word100_interval_motion_20261101/PRESENCE_PILOT_AUDIT.json')
    assert result['status']=='REPAIR_PRESENCE_PILOT_BEFORE_BROAD_LABELS_OR_TRAINING'
    verify_paths=[N/'review'/name for name in ['EXPORT_VERIFICATION.json','DIAGNOSIS_VERIFICATION.json',
        'SUPPORT_SELECTION_EXPORT_VERIFICATION.json','SUPPORT_OUTCOME_VERIFICATION.json',
        'REMAINING_EXPORT_VERIFICATION.json','FULL24_OUTCOME_VERIFICATION.json']]
    for p in verify_paths:assert read(p)['status'].startswith('PASS'),p
    protection=read(O/'word100_core_feature_split_20261101/SELECTION.json')['protected']
    # Preserve the exact protected map; source formats are checked before execution.
    if isinstance(protection,dict):
        for p,h in protection.items():assert sha(p)==h
    else:
        for r in protection:assert sha(r['file'])==r['sha256']
    failed=[k for k,v in result['checks'].items() if not v]
    before={k:prior['metrics'][k] for k in prior['metrics']}
    after=result['metrics']
    def counts(mm):
        return dict(agreed_known=sum(v['agreed_positive']+v['agreed_negative'] for v in mm.values()),
            known_conflicts=sum(v['known_conflicts'] for v in mm.values()),
            known_unknown=sum(v['known_unknown_mismatch'] for v in mm.values()),
            both_unknown=sum(v['both_unknown'] for v in mm.values()))
    out=dict(status='VERIFICATION_COMPLETE_TRAINING_NOT_READY',at=datetime.datetime.now().astimezone().isoformat(),
        implementation='Codex root',visual_review='Codex A/B; A remaining15 split into disjoint9+6, raw rows preserved',
        separate_verification='Separate Codex source pixels and saved-output recount; not independent human/KSL truth',
        computer_restart_recovery='Completed native9 export reused with SHA checks; only interrupted review resumed',
        left_six_resolved=native['left_six_resolved'],left_six_unresolved=native['left_six_unresolved'],
        same24_before=counts(before),same24_native=counts(after),same24_metrics=after,failed_checks=failed,
        support12=support['metrics'],support_diversity=support['support'],
        conditional_training_started=False,actual_new_training=0,actual_new_model_inference=0,
        performance_improvement=None,html_created=False,protected_files=len(protection),
        originals_C_T3F_web_DTW_unchanged=True,user_mentor_material_access=False,
        next_single_action='New prospective FIT60/180 training-support contract: fixed full population, same target/unknown rules and class diversity minima; check feasible observation support, then bounded complete labels only if justified. Do not rerun small pilots, lower gates, or mix support12 into old24. Training still conditional on validated full readiness.',
        evidence={str(p):sha(p) for p in [F/'FULL24_AUDIT.json',N/'NATIVE_DIAGNOSIS.json',N/'support_inventory/SUPPORT_RESULT.json']+verify_paths})
    write(N/'COMPLETION.json',out)
    text=[
        '원본 몸통·팔 화면 재검토 결과',
        '',
        f'컴퓨터 재부팅 후 완료된 자료를 해시 확인해 재사용하고, 중단됐던 판독을 이어서 마쳤습니다. 애매했던 왼손6장면 중 {native["left_six_resolved"]}장면은 두 검토가 같은 확정 판단을 했고, N008_W1은 둘 다 미확인으로 남겼습니다.',
        f'같은24구간·96개 항목의 두 검토 확정일치: {out["same24_before"]["agreed_known"]} → {out["same24_native"]["agreed_known"]}. 이는 원본을 읽는 판독 결과이며 모델 정확도가 아닙니다.',
        '각 항목 집계(24구간 고정):']
    for key,v in after.items():
        text.append(f'- {key}: 있음일치{v["agreed_positive"]}, 없음일치{v["agreed_negative"]}, 반대확정{v["known_conflicts"]}, 한쪽미확인{v["known_unknown_mismatch"]}, 둘다미확인{v["both_unknown"]}; 입력범위 내 적격없음{v["eligible_agreed_negative"]}.')
    text.extend(['',
        '별도12구간에서 상반신 입력범위 안에 손이 계속 움직여 유지구간이 없는 사례를 확인했습니다. 오른손4구간(3수행자ID/2표현), 왼손1구간(1ID/1표현)입니다. 두 검토가 반대로 확정한 경우0, 한쪽만 미확인인 경우1을 그대로 보존했습니다. 이12구간은 원24검사의 분모나 지원수에 합치지 않았습니다.',
        '남은 준비기준 실패: '+', '.join(failed),
        '쉽게 말해, 원24구간에서 양손의 유지구간이 없다고 확인한 각1구간은 손이 모델의 입력범위 밖으로 내려갔습니다. 그래서 이24구간 안에는 학습에 쓸 수 있는 반대 사례가 양손 모두0입니다.',
        '정답 기준·자료 지원이 충분하다는 조건을 충족하지 못해 새 학습과 모델 성능 비교는 실행하지 않았습니다. 기존C·T3-F, 운영웹·DTW, 관절입력·정상교차 자료는 변경하지 않았습니다. 사용자·멘토촬영은 접근·학습·튜닝·선택에 쓰지 않았습니다.',
        '',
        '후속 제안은 작은 표본과 기준을 계속 바꾸는 작업이 아니라, 기존 공개FIT60영상·180구간 전체에서 필요한 정답이 충분한지 한 번에 확인하는 별도 준비계약입니다. 표적별 양/음성 최소12구간·6영상·4표현·3수행자 지원과 관측 신뢰도를 함께 점검해야 합니다. 지금 결과만으로 그 조건을 충족했다고 말할 수 없습니다. 이 새 계약과 전수 판독은 아직 실행하지 않았습니다.',
        '모든 검토는 별도 Codex 판독/코드 검산입니다. 수어 전문가의 독립 정답이나 교육적 정오 판정이 아니며, 현재자료 전체·실사용자의 인식률 향상을 입증하지 않습니다. 명확한 모델 개선이 없어 HTML은 만들지 않았습니다.'])
    with (N/'RESULT_KO.txt').open('x',encoding='utf8') as f:f.write('\n'.join(text)+'\n')
    (N/'CURRENT_EXECUTION.json').write_text(json.dumps(dict(status=out['status'],active_execution=None,active_agents=[],training=0,
        completion=str(N/'COMPLETION.json'),next_single_action=out['next_single_action']),ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({k:out[k] for k in ('status','same24_before','same24_native','failed_checks')},ensure_ascii=False))
if __name__=='__main__':main()
