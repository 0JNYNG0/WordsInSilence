"""Independent arithmetic replay of saved labels/metrics; no inference or imports from producer."""
import copy
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

U = Path(__file__).resolve().parent.parent
P = U.parent / 'word100_temporal_shape_context_20261027'
FEATURES = ('index_only', 'open_four')

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def direct_or_temporal(values, times, i):
    if values[i] is not None:
        return {'value': values[i], 'kind': 'direct', 'support_indices': [i]}
    before = [j for j,v in enumerate(values) if j < i and v is not None and times[i]-times[j] <= 200001]
    after = [j for j,v in enumerate(values) if j > i and v is not None and times[j]-times[i] <= 200001]
    if len(before) < 2:
        return dict(value=None,kind='unknown',reason='past_insufficient',support_indices=before)
    if any(values[j] != values[before[0]] for j in before):
        return dict(value=None,kind='unknown',reason='past_conflict',support_indices=before)
    if len(after) < 2:
        return dict(value=None,kind='unknown',reason='future_insufficient',support_indices=before+after)
    if any(values[j] != values[before[0]] for j in after):
        return dict(value=None,kind='unknown',reason='direction_conflict',support_indices=before+after)
    return dict(value=values[before[0]],kind='temporal_estimate',reason='bidirectional_agreement',support_indices=before+after)

def independent_baseline(states,times,i):
    out = {f: direct_or_temporal(states[f],times,i) for f in FEATURES}
    if out['index_only']['value'] == 1 and out['open_four']['value'] == 1:
        for f in FEATURES:
            if out[f]['kind'] == 'temporal_estimate':
                out[f].update(value=None,kind='unknown',reason='feature_conflict')
    return out

def edge_status(edge,cal):
    if not (edge['rgb']['valid'] and edge['pose']['valid']):
        return 'insufficient_evidence'
    kinds=[]
    for k in ('rgb','pose'):
        assert math.isfinite(edge[k]['score'])
        if edge[k]['score'] > cal[k+'_threshold']:
            kinds.append(k)
    return 'change_or_viewpoint:'+','.join(kinds) if kinds else 'no_change_signal'

def independently_guard(base,i,edges,cal):
    out=copy.deepcopy(base)
    out['checked_edges']=[]
    if base['kind'] != 'temporal_estimate':
        return out
    before=[j for j in base['support_indices'] if j<i]
    after=[j for j in base['support_indices'] if j>i]
    assert len(before)>=2 and len(after)>=2
    # Every edge on the entire path between the closest supported anchors.
    a,b=max(before),min(after)
    checked=[]
    for j in range(a,b):
        checked.append({'edge_index':j,'status':edge_status(edges[j],cal)})
    out['checked_edges']=checked
    if all(r['status']=='no_change_signal' for r in checked):
        out['reason']='bidirectional_agreement_with_no_observed_change_signal'
    else:
        out.update(value=None,kind='unknown',reason='change_or_viewpoint' if any(r['status'].startswith('change_or_viewpoint') for r in checked) else 'insufficient_transition_evidence')
    return out

def summarize(rows):
    result={}
    for mode in ('baseline','guard'):
        output=[r[mode]['value'] for r in rows]
        nonmissing=sum(v is not None for v in output)
        match=sum(r[mode]['value'] is not None and r[mode]['value']==r['hidden_draft'] for r in rows)
        result[mode]={'total':len(rows),'estimated':nonmissing,'draft_match':match,'draft_conflict':nonmissing-match,'withheld':len(rows)-nonmissing}
    result['guard_effect']={
        'withheld_previously_matching':sum(r['baseline']['value']==r['hidden_draft'] and r['guard']['value'] is None for r in rows),
        'withheld_previously_conflicting':sum(r['baseline']['value'] is not None and r['baseline']['value']!=r['hidden_draft'] and r['guard']['value'] is None for r in rows),
        'new_wrong_state':sum(r['guard']['value'] is not None and r['guard']['value']!=r['baseline']['value'] for r in rows),
    }
    return result

def main():
    result=read(U/'COMPARISON.json')
    frozen=read(U/'FROZEN.json')
    oldresult=read(P/'COMPARISON.json')
    cal=read(U/'metrics/CALIBRATION.json')
    raw=read(U/'metrics/RAW_EDGES.json')
    paths=set(result['pins'])|set(oldresult['pins'])|{str(U/'COMPARISON.json'),str(U/'review/IMAGE_LABELS_FIRST_PASS.json')}
    before_hashes={p:sha(p) for p in paths}
    for p,h in result['pins'].items(): assert before_hashes[p]==h,(p,'current pin mismatch')
    for p,h in oldresult['pins'].items(): assert before_hashes[p]==h,(p,'old pin mismatch')
    assert sha(U/'review/IMAGE_LABELS_FIRST_PASS.json')==sha(U/'review/IMAGE_LABELS.json')
    assert result['calibration']==cal
    assert cal['calibration_case_ids']==['T07','T10']
    assert cal['threshold_rule']=='maximum_valid_score_no_floor_no_search'
    rawmap={(r['case_id'],r['edge_index']):r for r in raw['rows']}
    assert len(rawmap)==len(raw['rows'])==432
    for metric in ('rgb','pose'):
        scores=[r[metric]['score'] for r in raw['rows'] if r['case_id'] in ('T07','T10') and r[metric]['valid']]
        assert len(scores)>=12 and len(scores)==cal[metric+'_valid']
        assert cal[metric+'_threshold']==max(scores)
        assert math.isfinite(cal[metric+'_threshold'])
    indices=read(U/'visual_inputs/INDEX.json')['rows']+read(P/'visual_inputs/INDEX.json')['rows']
    labels=read(U/'review/IMAGE_LABELS.json')['rows']+read(P/'review/IMAGE_LABELS.json')['rows']
    labelmap={(r['case_id'],r['image_key']):r for r in labels}
    assert len(labelmap)==len(labels)==468
    cases={s['case_id']:s for s in result['sequences']}
    masks={(r['case_id'],r['feature'],r['index']):r for r in result['masked_trials']}
    unknowns={(r['case_id'],r['feature'],r['index']):r for r in result['real_unknown_trials']}
    assert len(cases)==len(indices)==36
    assert len(masks)==len(result['masked_trials']) and len(unknowns)==len(result['real_unknown_trials'])
    mask_keys=set();unknown_keys=set();known=Counter();endpoints=Counter();checked=0;estimates=0;events=[]
    for item in indices:
        cid=item['case_id'];fresh=cid.startswith('N');group='new' if fresh else 'old'
        frames=sorted(item['frames'],key=lambda r:r['actual_pts_us'])
        assert len(frames)==13
        times=[r['actual_pts_us'] for r in frames]
        assert all(b>a for a,b in zip(times,times[1:]))
        seq=cases[cid]
        assert len(seq['frames'])==13 and seq['person_id']==item['recorded_person_id']
        values={f:[labelmap[(cid,r.get('image_key',r.get('neutral_image_id')))][f] for r in frames] for f in FEATURES}
        snapshot=copy.deepcopy(values)
        edges=[rawmap[(cid,j)] for j in range(12)]
        for j,e in enumerate(edges):
            assert e['from_frame_id']==frames[j]['frame_id'] and e['to_frame_id']==frames[j+1]['frame_id']
            assert seq['edges'][j]['status']==edge_status(e,cal)
            assert seq['edges'][j]['from_pts_us']==times[j] and seq['edges'][j]['to_pts_us']==times[j+1]
        for i,frame in enumerate(frames):
            saved=seq['frames'][i]
            assert saved['frame_id']==frame['frame_id'] and saved['pts_us']==times[i]
            assert saved['direct']=={f:values[f][i] for f in FEATURES}
            expected=independent_baseline(values,times,i)
            assert saved['baseline']==expected
            for f in FEATURES:
                g=independently_guard(expected[f],i,edges,cal)
                assert saved['guard'][f]==g,(cid,i,f,'real guard mismatch')
                checked+=len(g['checked_edges']);estimates+=bool(g['checked_edges'])
                if values[f][i] is None:
                    key=(cid,f,i);unknown_keys.add(key)
                    assert unknowns[key]['baseline']==expected[f] and unknowns[key]['guard']==g
                else:
                    known[group]+=1
                    assert expected[f]['kind']==g['kind']=='direct'
                    assert expected[f]['value']==g['value']==values[f][i]
                    if i in (0,12):
                        endpoints[group]+=1
                    else:
                        key=(cid,f,i);mask_keys.add(key)
                        trial=copy.deepcopy(values)
                        trial[f][i]=None
                        assert sum(trial[z][j]!=values[z][j] for z in FEATURES for j in range(13))==1
                        b=independent_baseline(trial,times,i)[f]
                        mg=independently_guard(b,i,edges,cal)
                        actual=masks[key]
                        assert actual['hidden_draft']==values[f][i]
                        assert actual['baseline']==b and actual['guard']==mg,(key,'mask result mismatch')
                        assert actual['frame_id']==frame['frame_id']
                        assert mg['value'] is None or mg['value']==b['value']
                        checked+=len(mg['checked_edges']);estimates+=bool(mg['checked_edges'])
            assert values==snapshot,'source state mutated or recursively filled'
        for f in FEATURES:
            observed=[i for i,v in enumerate(values[f]) if v is not None]
            for a,b in zip(observed,observed[1:]):
                if b==a+1 and values[f][a]==values[f][b]:continue
                events.append({'case_id':cid,'fresh_dense':fresh,'feature':f,'from_index':a,'to_index':b,'from_pts_us':times[a],'to_pts_us':times[b],
                               'from_value':values[f][a],'to_value':values[f][b],
                               'kind':'observed_state_change_interval' if values[f][a]!=values[f][b] else 'same_endpoints_unresolved_middle',
                               'gap_frames':b-a-1,'semantic_importance':'not_determined','intermediate_exact_shape':None})
    assert mask_keys==set(masks) and unknown_keys==set(unknowns)
    assert dict(known)==result['known_direct_preserved']=={'new':411,'old':334}
    assert events==result['events']
    fresh=[r for r in masks.values() if r['fresh_dense']]
    old=[r for r in masks.values() if not r['fresh_dense']]
    assert summarize(fresh)==result['fresh_summary']
    assert summarize(old)==result['old_diagnostic_summary']
    assert summarize([r for r in fresh if r['person_id']!=1])==result['fresh_different_recorded_ID_summary']
    for p in (1,2,3):assert summarize([r for r in fresh if r['person_id']==p])==result['fresh_by_person'][str(p)]
    reals={}
    for label,isfresh in [('new',True),('diagnostic_old',False)]:
        rows=[r for r in unknowns.values() if r['fresh_dense']==isfresh]
        reals[label]={'original_unknown':len(rows),**{m:sum(r[m]['value'] is not None for r in rows) for m in ('baseline','guard')}}
    assert reals==result['real_unknown_summary']
    assert result['training'] is False and result['production_change'] is False
    assert {p:sha(p) for p in paths}==before_hashes,'read-only inputs changed during review'
    report={
        'status':'PASS_INDEPENDENT_SAVED_ARITHMETIC_REPLAY',
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'reviewer':'Codex ksl_source_audit; separate implementation in same AI system',
        'implementation_imported_from_producer':False,
        'detector_or_model_inference_performed':False,'labels_modified':False,'threshold_search_performed':False,
        'comparison_sha256':sha(U/'COMPARISON.json'),'verification_code_sha256':sha(__file__),
        'checks':{
            'pins_all_unchanged':True,'first_pass_labels_unchanged':True,
            'fixed_calibration_max_verified_from_saved_scores':True,
            'single_cell_independent_masks_no_recursive_fill':True,
            'known_interior_denominators_exact':True,
            'nearest_past_future_anchor_every_intervening_edge_checked':True,
            'all_direct_known_states_preserved':True,
            'endpoints_and_uncertain_transition_events_retained':True,
            'per_person_and_actual_unknown_summaries_match':True,
        },
        'counts':{'new_mask_trials':len(fresh),'old_diagnostic_mask_trials':len(old),
                  'direct_preserved':dict(known),'known_endpoint_cells_excluded_from_mask_only':dict(endpoints),
                  'baseline_estimates_with_edge_checks':estimates,'checked_edge_records':checked,
                  'raw_edges':len(rawmap),'sequences':len(cases),'observed_event_intervals':len(events)},
        'fresh_summary':result['fresh_summary'],'old_diagnostic_summary':result['old_diagnostic_summary'],
        'real_unknown_summary':reals,
        'interpretation':[
            '새 348개 마스크 시험에서 기존 일치 259개 중 220개를 보류하여 일치는 39개로 감소했다. 새 시험에는 원래 충돌이 0개여서 새 오답 감소를 입증하지 않았다.',
            '과거 진단 279개에서는 불일치 1개를 보류했으나 기존 일치 102개도 보류했다. 이는 과거 진단에서의 보수적 거절이며 새 정확도 향상 증거가 아니다.',
            '실제 미확인 새 57개 값은 기존 추정 7개가 모두 보류되어 후보가 새로 복원한 손모양은 0개다.',
            '411개 새 직접 관측 초안과 334개 과거 직접 관측 초안은 값 그대로 보존되었다. 초안 자체의 진실성은 검증하지 않았다.',
            '동일 AI 이미지 주석을 기준으로 한 일치/충돌 산술 검산이다. 독립 사람 정답, TCN 인식률, 교육 피드백 정확도, 실제 손가락 3D 정답을 검증한 것이 아니다.',
            'RGB/2D 관절 변화 점수의 원시 추출 산술·물리적 타당성은 이번 검산 범위가 아니며 저장 점수에 대한 고정 규칙 적용만 검산했다.',
        ],
        'cautions':[
            '새 시험의 넓은 보류 증가는 복원 기회 포기이며 완성된 복원기로 채택할 성능 근거가 아니다.',
            '평가자는 이 새 RGB 주석 작성자와 동일 AI 계열이므로 사람 정답 독립성과 별도 구현 검산을 구분해야 한다.',
            '두 안정 구간의 최대값은 일반 사용자 허용 범위로 검증된 임계값이 아니다.',
        ],
        'input_hashes_verified':before_hashes,
    }
    dest=Path(__file__).with_name('GUARD_VERIFY.json')
    with dest.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps({'status':report['status'],'counts':report['counts'],'path':str(dest)},ensure_ascii=False))

if __name__=='__main__':main()
