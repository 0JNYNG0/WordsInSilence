"""Summarize saved predictions only; no model loading or scoring."""
import numpy as np
from train_local import U,O,read,write,sha

def main():
    d=read(U/'DATASET_FROZEN.json');e=read(U/'EVAL_RESULT.json');tr=read(U/'TRAIN_RESULT.json')
    assert read(U/'EVAL_FINISHED.json')['result_sha256']==sha(U/'EVAL_RESULT.json')
    metrics=[]
    for t in d['supported_targets']:
        selected=[g for g in e['rows'] if g['role']=='development' and g['target_index']==t]
        native=[b for g in selected for b in g['bags'] if b['kind']=='weak_positive_bracket']
        local=[b for g in selected for b in g['bags'] if b['kind']=='reviewed_local']
        pos=[b for b in local if b['label']==1];neg=[b for b in local if b['label']==0];unk=[b for b in local if b['label'] is None]
        high=lambda b:bool(b.get('high_confirmation',False))
        native_high=sum(map(high,native));false_high=sum(map(high,neg))
        denominator=len(native)
        metrics.append(dict(target_index=t,native_positive_denominator=denominator,native_positive_high=native_high,
            native_positive_raw=sum(b.get('raw_positive',False) for b in native),
            local_positive_denominator=len(pos),local_positive_high=sum(map(high,pos)),
            local_negative_denominator=len(neg),local_negative_high_false_positives=false_high,
            local_unknown_denominator=len(unk),unavailable_bag_denominator=sum(b['probability'] is None for b in native+local),
            broad_anchor_in_reviewed_positive=sum(g['broad']['anchor_inside_reviewed_positive_bracket'] is True for g in selected),
            broad_anchor_denominator=len(selected),
            broad_monotonic=sum(g['broad']['same_score_broad_monotonic'] for g in selected),
            prototype_research_gate=denominator==3 and native_high>=2 and len(neg)>0 and false_high==0))
    ag=read(O/'word100_context_episode_query_20261103/RESULT.json');beforeafter=[]
    for old in ag['rows']:
        row=next(r for r in e['rows'] if r['clip_id']==old['clip_id'] and r['target_index']==0)
        a=np.load(row['prediction']['path']);lo,hi=old['frames'];ids=a['frame_id'];idx=np.flatnonzero((ids>=lo)&(ids<=hi)&a['motion_anchor'])
        assert len(idx)
        at=int(idx[np.argmax(a['probabilities'][idx,0])]);v=float(a['probabilities'][at,0])
        beforeafter.append(dict(clip_id=old['clip_id'],role='fit',scope=old['scope'],frames=old['frames'],
            previous_net_direction_probability=old['observation']['probability'],new_local_occurrence_probability=v,
            new_anchor_frame=int(ids[at]),same_old_weights_rerun=False,
            limitation='Same original interval, different trained question and temporal context. Diagnostic FIT only, not a fair new-person accuracy comparison.'))
    times=np.asarray([r['seconds'] for r in e['timing']]);gate=all(m['prototype_research_gate'] for m in metrics)
    result=dict(status='SINGLE_OCCURRENCE_PILOT_COMPLETE_PENDING_SAVED_REVIEW',
        training=dict(fit_clips=36,registered_fit_ids=list(range(1,10)),parameters=39013,steps=tr['steps'],epochs=80,
            geometry_projection_finetuned=tr['geometry_projection_changed'],rgb_projection_finetuned=tr['rgb_projection_changed'],backbone_finetuned=False),
        development=dict(clips=12,registered_ids=[11,12,13],previously_exposed=True,per_target=metrics),
        gate_pass=gate,live_demo_changed=False,old_models_changed=False,
        cached_head_latency_seconds=dict(median=float(np.median(times)),p95=float(np.percentile(times,95)),first=float(times[0]),total=float(times.sum())),
        latency_scope=e['latency_scope'],saved_prior_scope_diagnostic=beforeafter,
        limitations=e['limitations']+['A maximum inside a labelled interval supplies weak occurrence evidence, not precise onset/offset labels','Full-sequence argmax can be outside known positive brackets; reported separately','No speed, handshape, cross-hand identity, source clause or real-user feedback improvement is implied by this pilot'],
        whole_goal_complete=False,pins={str(U/n):sha(U/n) for n in ['DATASET_FROZEN.json','TRAIN_RESULT.json','EVAL_RESULT.json','EXECUTION_FROZEN.json','summarize_local.py']})
    write(U/'RESULT.json',result)
    lines=['앞뒤 과정 속 짧은 움직임을 찾는 학습 시험', '',
        '이전 판정부는 잘라 준 구간의 전체 이동방향을 물었다. 이번 후보는 영상의 각 시점 주변에서 해당 움직임이 있었는지 배우고, 같은 점수열로 좁은 구간과 넓은 구간을 읽는다.',
        f'공개4표현, 등록ID01~09의36영상을 사용해 39,013개 파라미터 판정부를80회차/{tr["steps"]}단계 학습했다. 기존 관절·RGB 투영 가중치의 복사본을 미세조정했다. SignRep 본체와 C·T3-F, 기존웹은 바꾸지 않았다.',
        '개발평가는 등록ID11~13의12영상이다. 과거에도 개발에 사용된 자료이며 처음 보는 최종시험은 아니다. 검토자는 Codex이며 독립 수어 전문가 정답은 아니다.', '',
        '방향별 결과: 기존 직접 확인한 이동 찾기 / 새 직접 확인한 국소 비발생 구간의 잘못된 이동 표시 / 전체 흐름 최고점이 확인한 이동구간 안에 있는지']
    names=['반대팔을 따라 손목 쪽으로 쓸기','양손 벌리기','오른손의 상대적 상승','얼굴 앞에서 몸통 쪽으로 이동','오른손의 상대적 하강']
    for m in metrics:
        lines.append(f'{names[m["target_index"]]}: {m["native_positive_high"]}/{m["native_positive_denominator"]} / {m["local_negative_high_false_positives"]}/{m["local_negative_denominator"]} / {m["broad_anchor_in_reviewed_positive"]}/{m["broad_anchor_denominator"]}')
    lines.extend(['',f'사전 고정한 소규모 연구후보 기준: {"통과" if gate else "미통과"}. 웹 적용은 아직 하지 않았다.',
        f'캐시된 입력에서 작은 판정부 계산 중앙값 {np.median(times)*1000:.2f}ms. 영상 관절 추출·SignRep·웹 전체 대기시간은 포함하지 않는다.',
        '움직임 근거가 있는 시점은 다른 구간의 누락 때문에 지우지 않는다. 반대로 관측이 빠진 범위에서 점수가 낮다고 정상 또는 움직임 없음으로 확정하지 않는다.',
        '이 시험은 중요한 차이와 정상 동작을 구분하는 전체 피드백 데모의 일부다. 올바른 수어인지, 반복이 충분한지, 짧은 구별 동작 누락을 찾는지, 새 사용자 속도가 좋아졌는지는 아직 증명하지 않았다.',
        '저장 예측의 별도 산술 검산 이후 결과와 다음 연결 범위를 확정한다. 사용자·멘토 자료는 학습·튜닝·선택에 사용하지 않았다.'])
    with (U/'RESULT_KO.txt').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    print('\n'.join(lines))
if __name__=='__main__':main()
