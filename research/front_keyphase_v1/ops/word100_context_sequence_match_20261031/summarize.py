from pathlib import Path
import json,sys
from collections import Counter,defaultdict
import numpy as np
from run import U,OLD,read,write,sha,load

def main():
    manifest=read(U/'RESULT_MANIFEST.json');rows={r['alias']:r for r in read(U/'AUTHORIZATION.json')['rows']}
    controls=defaultdict(list);pairs=[];cases=[]
    for item in manifest['results']:
        assert sha(item['file'])==item['sha256'];d=read(item['file']);baseline=read(d['baseline_path']);oldanchors={a['segment_id']:a for a in baseline['free_order']['anchors']}
        a,b=d['query'],d['reference'];oa=load(rows[a]['input']);ob=load(rows[b]['input'])
        inds=baseline.get('metadata',{}).get('original_indices',list(range(d['query_frames'])))
        audit=[]
        if d['kind']=='control':
            for x in d['numerical_correspondence']:
                old=oldanchors[x['segment_id']];rlo,rhi=old['reference_core'];orig=inds[old['query_center']]
                audit.append(dict(**x,old_center=old['query_center'],old_center_original=orig,old_center_in_original_core=rlo<=orig<=rhi,old_supported=old['supported'],old_unique=old['unique']))
            controls[d['control']].append(dict(name=d['name'],alias=a,role=rows[a]['role'],stages=audit,new_states=dict(Counter(x['state'] for x in d['anchors'])),old_reverse=sum(x['state']=='reverse' for x in baseline['free_order']['relations']),new_reverse=sum(x['state']=='reverse_candidate' for x in d['relations']),new_overlaps=sum(x['state']=='overlapping_occurrences' for x in d['relations']),seconds=d['seconds']))
        else:
            pairs.append(dict(query=a,reference=b,role=rows[a]['role'],seconds=d['seconds'],states=dict(Counter(x['state'] for x in d['anchors'])),relations=d['relations'],old_relations=baseline['free_order']['relations']))
        amap={x['segment_id']:x for x in audit}
        anchors=[]
        for x in d['anchors']:
            old=oldanchors[x['segment_id']]
            anchors.append(dict(**x,old_center=old['query_center'],old_window=old['query_window'],numerical=amap.get(x['segment_id'])))
        cases.append(dict(id=d['name'],title=f"{rows[a]['label']} {a} / "+(d.get('control') or f'참고 {b}'),kind=d['kind'],query=a,reference=b,role=rows[a]['role'],control=d.get('control'),query_pts=(load(OLD/'controls'/(d['name']+'.npz'))['timestamp_us']/1e6).tolist() if d['kind']=='control' else (oa['timestamp_us']/1e6).tolist(),original_query_pts=(oa['timestamp_us']/1e6).tolist(),reference_pts=(ob['timestamp_us']/1e6).tolist(),original_indices=inds,anchors=anchors,relations=d['relations'],outside_core_candidates=d['outside_core_candidates'],seconds=d['seconds']))
    metrics={}
    for name,ds in controls.items():
        stages=[x for d in ds for x in d['stages']]
        metrics[name]=dict(cases=len(ds),stages=len(stages),old_center_hits=sum(x['old_center_in_original_core'] for x in stages),new_center_hits=sum(x['center_in_original_core'] for x in stages),new_mean_core_recall=float(np.mean([x['recall'] for x in stages])),new_mean_core_precision=float(np.mean([x['precision'] if x['precision'] is not None else 0. for x in stages])),old_supported_hits=sum(x['old_center_in_original_core'] and x['old_supported'] for x in stages),new_fully_supported_hits=sum(x['center_in_original_core'] and x['state']=='supported_candidate' for x in stages),new_states=dict(sum((Counter(d['new_states']) for d in ds),Counter())),old_reverse_relations=sum(d['old_reverse'] for d in ds),new_reverse_relations=sum(d['new_reverse'] for d in ds),new_overlaps=sum(d['new_overlaps'] for d in ds))
    times=defaultdict(float)
    for p in pairs:
        if p['role']!='fit':times[p['query']]+=p['seconds']
    # An explicit bounded decision, not selection of a production replacement.
    clean=['identity','speed_075']
    raw_gate=all(metrics[n]['new_center_hits']>=metrics[n]['old_center_hits'] and metrics[n]['new_reverse_relations']<=metrics[n]['old_reverse_relations'] for n in clean)
    coverage_gate=all(metrics[n]['new_fully_supported_hits']>=metrics[n]['old_supported_hits'] for n in clean)
    summary=dict(status='COMPLETE_PENDING_INDEPENDENT_VERIFICATION',controls=metrics,person_pairs=len(pairs),development_pair_seconds_by_query=dict(times),new_person_pair_states=dict(sum((Counter(p['states']) for p in pairs if p['role']!='fit'),Counter())),gates=dict(raw_same_source_no_regression=raw_gate,supported_correct_coverage_no_regression=coverage_gate,not_enough_for_semantic_accuracy=True),adopted_production=False,neural_training=0,all_original_inputs_preserved=True)
    write(U/'SUMMARY.json',summary);write(U/'CONTROL_EVALUATION.json',dict(controls=controls));write(U/'PAIR_EVALUATION.json',dict(pairs=pairs));write(U/'REPORT_DATA.json',dict(summary=summary,cases=cases))
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
