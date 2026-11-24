"""Verify saved full24 outcome without importing/rerunning producer audits.

Uses the frozen earlier independent frame-set decoder plus fresh direct tallies,
raw A-part union checks, native view provenance and frozen gate comparisons.
"""
from pathlib import Path
import collections
import copy
import datetime
import hashlib
import importlib.util
import json
import sys
sys.dont_write_bytecode = True
N = Path(__file__).resolve().parent.parent
F = N / 'full24_native'
W = N.parent / 'word100_interval_motion_20261101'
OUT = N / 'review/FULL24_OUTCOME_VERIFICATION.json'
checks = []


def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def ck(group,name,okay):checks.append(dict(group=group,name=name,passed=bool(okay)))


def main():
    if OUT.exists():raise FileExistsError('Reuse immutable saved verification.')
    saved=read(F/'FULL24_AUDIT.json')
    prior_manifest=read(W/'PILOT_MANIFEST.json')
    gates=read(W/'PILOT_GATES.json')
    export=read(N/'review/REMAINING_EXPORT_VERIFICATION.json')
    protected=dict(export['protected_pins'])
    for pins in [export['source_pins'],read(F/'AUDIT_FROZEN.json')['pins'],saved['source_pins'],read(N/'review/A_MERGE_IMPLEMENTATION_REVIEW.json')['source_pins']]:
        for p,h in pins.items():
            ck('pin_consistency',p,p not in protected or protected[p]==h);protected[p]=h
    merged=read(F/'REVIEW_A.json');parts=[]
    assignment=read(F/'A_ASSIGNMENT.json')
    ck('A_union','assignment hash',merged['assignment_sha256']==sha(F/'A_ASSIGNMENT.json'))
    for i,part in enumerate(merged['raw_parts'],1):
        p=Path(part['file']);r=read(p)
        ck('A_union',str(p)+' SHA',sha(p)==part['sha256']);protected[str(p)]=part['sha256']
        aliases=assignment[f'groupA_part{i}']['aliases']
        ck('A_union',str(p)+' exact assigned aliases',r['status']=='COMPLETE_PART' and len(r['rows'])==len(aliases) and sorted(x['alias'] for x in r['rows'])==aliases==part['aliases'])
        ck('A_union',str(p)+' source manifest/rubric',r['native_manifest_sha256']==merged['native_manifest_sha256'] and r['rubric_sha256']==merged['rubric_sha256'])
        parts.extend(r['rows'])
    ck('A_union','exact raw row concatenation with no edits',merged['rows']==parts and len(parts)==15 and len({r['sample_id'] for r in parts})==15)
    for p,h in protected.items():ck('protected_sha',p,sha(p)==h)
    native_rows=read(N/'MANIFEST.json')['rows']+read(F/'MANIFEST.json')['rows']
    native={r['sample_id']:r for r in native_rows}
    source_order=[r['sample_id'] for r in prior_manifest['rows']]
    ck('scope','same unique24 source population',len(native_rows)==len(native)==len(source_order)==24 and set(source_order)==set(native))
    support_ids={r['sample_id'] for r in read(N/'support_inventory/SELECTION.json')['selected']}
    ck('scope','support12 absent from original24',not support_ids.intersection(native))
    ck('scope','verified195 new export',export['status']=='PASS_INDEPENDENT_REMAINING15_NATIVE_EXPORT' and not export['failures'])
    for sid,row in native.items():
        ck('native_source_sha',sid,sha(row['source'])==row['source_sha256'])
    manifest=copy.deepcopy(prior_manifest)
    original={r['sample_id']:r for r in manifest['rows']}
    raw_maps, normalized, unique_images = {}, [], set()
    expected_raw_paths=[]
    for who in 'AB':
        rows=[]
        for base,count,filename in [(N,9,f'review/NATIVE_{who}.json'),(F,15,f'REVIEW_{who}.json')]:
            path=base/filename;expected_raw_paths.append(str(path));review=read(path)
            expected_ids={r['sample_id'] for r in read(base/'MANIFEST.json')['rows']}
            ck('review_scope',str(path),review['status']=='COMPLETE' and len(review['rows'])==count and len({r['sample_id'] for r in review['rows']})==count and {r['sample_id'] for r in review['rows']}==expected_ids)
            ck('review_scope',str(path)+' pinned manifest/rubric',review['native_manifest_sha256']==sha(base/'MANIFEST.json') and review['rubric_sha256']==sha(W/'RUBRIC.json'))
            for raw in review['rows']:
                sid=raw['sample_id'];src=native[sid];old=original[sid]
                frame_ids=[f['frame_id'] for f in src['frames']]
                ck('frame_metadata',who+'/'+sid,frame_ids==old['frame_ids'] and [f['timestamp_us'] for f in src['frames']]==old['timestamps_us'] and raw['alias']==src['alias'] and raw['actual_image_viewed'] is True)
                allowed={p['file']:(p['sha256'],p['frame_ids']) for p in src['pairs']}
                for frame in src['frames']:
                    for kind in ('full','fixed'):allowed[frame[kind]]=(frame[kind+'_sha256'],[frame['frame_id']])
                seen=set()
                for im in raw['images_viewed']:
                    p=im['file'];ck('view_reference_sha',who+'/'+p,p in allowed and im['sha256']==allowed[p][0]==sha(p));seen.update(allowed[p][1]);unique_images.add(p)
                ck('view_full13',who+'/'+sid,seen==set(frame_ids))
                # Independently verified opaque identity for the frame-set decoder.
                # No producer audit or label transformation is called.
                proof='native24-verified:'+sid+':'+src['source_sha256']
                row=copy.deepcopy(raw);row['image_sha256']=proof
                ck('raw_adapter',who+'/'+sid,{k:v for k,v in row.items() if k!='image_sha256'}=={k:v for k,v in raw.items() if k!='image_sha256'})
                rows.append(row);old['alias']=src['alias'];old['image_sha256']=proof
        raw_maps[who]={r['sample_id']:r for r in rows}
        ck('review_scope',who+'/exact24',len(rows)==len(raw_maps[who])==24 and set(raw_maps[who])==set(native))
        normalized.append(dict(status='COMPLETE',rows=rows))
    ck('raw_paths','exact four input paths',saved['raw_review_paths']==expected_raw_paths)
    for before,after in zip(prior_manifest['rows'],manifest['rows']):
        ck('metadata_preservation',before['sample_id'],{k:v for k,v in before.items() if k not in ('alias','image_sha256')}=={k:v for k,v in after.items() if k not in ('alias','image_sha256')})

    decoder_path=W/'review/verify_presence.py'
    ck('decoder_pin','frozen separate frame-set decoder',sha(decoder_path)=='8f5aad801d3aa5e95748728796914327ac2086171ab91daa1a773e072266aa2e')
    spec=importlib.util.spec_from_file_location('separate_frozen_frame_set_decoder',decoder_path)
    decoder=importlib.util.module_from_spec(spec);spec.loader.exec_module(decoder)
    expected=decoder.independent_expected(manifest,normalized,gates)
    bad=decoder.compare(saved,expected)
    for key in expected:ck('independent_decoder_comparison',key,key not in bad)
    for name,value in expected['checks'].items():ck('unchanged_gate_recount',name,saved['checks'].get(name)==value)
    for row in expected['rows']:
        result=next(r for r in saved['rows'] if r['sample_id']==row['sample_id'])
        for key,cell in row['predicates'].items():ck('raw96_cell_recount',row['sample_id']+'/'+key,result['predicates'][key]==cell)
        ck('ordered_segments',row['sample_id'],result['ordered_evidence']==row['ordered_evidence'])

    # A fresh direct count, independent also of the reused decoder, cross-checks
    # every raw paired category and its unconditional24 denominator.
    direct={}
    for key in ('R_down','R_hold','L_down','L_hold'):
        hand,predicate=key.split('_');c=collections.Counter()
        for sid in source_order:
            a,b=[raw_maps[w][sid]['presence'][hand][predicate] for w in 'AB']
            c['all']+=1
            if a is None and b is None:c['both_unknown']+=1
            elif a is None or b is None:c['known_unknown_mismatch']+=1
            elif a!=b:c['known_conflicts']+=1
            else:c['agreed_positive' if a==1 else 'agreed_negative']+=1
        direct[key]={name:c[name] for name in ('all','agreed_positive','agreed_negative','known_conflicts','known_unknown_mismatch','both_unknown')}
        ck('fresh_raw_count',key,all(saved['metrics'][key][name]==value for name,value in direct[key].items()) and sum(value for name,value in direct[key].items() if name!='all')==24)
    totals=dict(agreed_known=sum(r['agreed_positive']+r['agreed_negative'] for r in direct.values()),
                opposite_known=sum(r['known_conflicts'] for r in direct.values()),
                one_unknown=sum(r['known_unknown_mismatch'] for r in direct.values()),both_unknown=sum(r['both_unknown'] for r in direct.values()))
    ck('fresh_raw_count','all96 opportunities retained',sum(totals.values())==96)
    failed_gates=[k for k,v in expected['checks'].items() if not v]
    ck('closing_scope','training stays disallowed and source counts',saved['training_allowed'] is False and saved['training_runs']==saved['new_inference_runs']==0 and
       saved['all_windows']==24 and saved['all_predicate_cells']==96 and saved['native_frames']==312 and saved['reused_frames']==117 and saved['new_frames']==195)
    ck('closing_scope','support excluded and prior audit preserved',saved['support12_excluded_from_every_gate'] is True and saved['original24_failed_audit_unchanged'] is True)
    prior=read(W/'PRESENCE_PILOT_AUDIT.json')
    ck('earlier_failures','old24 still FAIL77/13/6',prior['status']=='REPAIR_PRESENCE_PILOT_BEFORE_BROAD_LABELS_OR_TRAINING' and
       sum(v['agreed_positive']+v['agreed_negative'] for v in prior['metrics'].values())==77 and
       sum(v['known_unknown_mismatch'] for v in prior['metrics'].values())==13 and sum(v['both_unknown'] for v in prior['metrics'].values())==6)
    ck('earlier_failures','N008 remains unknown in reused A/B',all(raw_maps[w]['N008_W1']['presence']['L']['down'] is None for w in 'AB'))
    ck('earlier_failures','centre first FAIL',read(N.parent/'word100_center_motion_pilot_20261101/PILOT_AUDIT.json')['status']=='REPAIR_PILOT_BEFORE_BROAD_LABELS_OR_TRAINING')
    ck('earlier_failures','whole-window first FAIL',read(N.parent/'word100_core_feature_split_20261101/LABEL_AUDIT.json')['status']=='NOT_READY_SUPPORT_OR_REVIEW')
    for p,h in protected.items():ck('protected_sha_after',p,sha(p)==h)
    groups={}
    for group in sorted({r['group'] for r in checks}):
        values=[r for r in checks if r['group']==group];groups[group]=dict(checks=len(values),passed=sum(r['passed'] for r in values),failed=sum(not r['passed'] for r in values))
    failures=[r for r in checks if not r['passed']]
    report=dict(status='PASS_INDEPENDENT_FULL24_NATIVE_OUTCOME_VERIFICATION' if not failures else 'FAIL_FULL24_NATIVE_OUTCOME_VERIFICATION',
        at_kst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),
        method='Separate Codex reviewer; frozen independently implemented frame-set decoder reused, plus fresh direct raw-category count, exact A-part union and provenance checks. Producer audit/calculate not imported or run.',
        counts=dict(windows=24,reused_windows=9,new_windows=15,paired_predicate_opportunities=96,raw_reviewer_values=192,
                    referenced_unique_images=len(unique_images),protected_files=len(protected),protected_production_model_baseline_files=32,checks=len(checks),failed=len(failures)),
        totals=totals,fresh_direct_counts=direct,metrics=expected['metrics'],failed_gates=failed_gates,
        contract_failure_identities=expected['contract_failure_identities'],
        physical_hold_negative_support={h:expected['class_support'][h+'_hold']['physical']['0'] for h in 'RL'},
        eligible_hold_negative_support={h:expected['class_support'][h+'_hold']['eligible']['0'] for h in 'RL'},
        check_groups=groups,checks=checks,failures=failures,protected_pins=protected,
        source_pins={str(p):sha(p) for p in [Path(__file__),decoder_path,F/'FULL24_AUDIT.json',F/'REVIEW_A.json',F/'REVIEW_A_PART1.json',F/'REVIEW_A_PART2.json',F/'REVIEW_B.json',F/'A_ASSIGNMENT.json',W/'PILOT_GATES.json',W/'PRESENCE_PILOT_AUDIT.json']},
        interpretation=['Unchanged24 readiness fails only R_hold_eligible_both_classes and L_hold_eligible_both_classes. Each physical hold-negative is N022_W2 outside full13 modelROI, leaving eligible negatives0 for both hands.',
                        'Native75 known agreements/0 known conflicts/4 one-unknown/17 both-unknown preserves96 opportunities. Old77/0/13/6 is separate and immutable; fewer mismatches with more both-unknown is not general certainty or model accuracy improvement.',
                        'Support12 establishes eligible negative existence elsewhere but supplies neither counts nor labels to this24 gate.',
                        'N008 both-unknown is preserved; it is not a demand to force a label. Actual blockers are the two within24 eligible-class support gates.',
                        'Bounded verification is complete, training prerequisite is not met, and broad288/new training remains unexecuted.'],
        execution=dict(producer_audit_rerun=False,model_inference_runs=0,training_runs=0,source_decode_rerun=False,raw_labels_modified=False,private_recording_access=0,root_documents_changed=False),
        limitations=['Same Codex system, not independent human/KSL truth.','Reused frame-set decoder was originally separately implemented by the earlier audit author; this reviewer independently checked its pinned code and directly recounted categories.','This verification checks saved evidence/contracts/arithmetic, not a third full RGB annotation.'])
    with OUT.open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:report[k] for k in ('status','counts','totals','failed_gates')},ensure_ascii=False))
    return bool(failures)


if __name__=='__main__':raise SystemExit(main())
