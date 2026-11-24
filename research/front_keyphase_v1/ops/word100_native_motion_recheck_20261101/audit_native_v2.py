"""Compare nine native-view research labels with preserved small-view labels.

No scoring, inference, label repair, or replacement of the failed24-window audit.
"""
from pathlib import Path
import sys, json, hashlib, importlib.util, copy
sys.dont_write_bytecode=True
N=Path(__file__).resolve().parent
W=N.parent/'word100_interval_motion_20261101'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def imp(p):
    spec=importlib.util.spec_from_file_location('unchanged_presence_contract',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def main():
    freeze=read(N/'AUDIT_FROZEN_v2.json')
    for p,h in freeze['pins'].items():assert sha(p)==h,p
    manifest=read(N/'MANIFEST.json'); plan=read(N/'PLAN_FROZEN.json')
    sources={r['sample_id']:r for r in manifest['rows']};assert len(sources)==9
    reviews=[read(N/'review'/f'NATIVE_{a}.json') for a in 'AB']
    previous=[read(W/'review'/f'PRESENCE_{a}.json') for a in 'AB']
    prevmaps=[{r['sample_id']:r for r in v['rows']} for v in previous]
    maps=[]; issues=[];m=imp(W/'audit_presence.py')
    for who,review in zip('AB',reviews):
        assert review['status']=='COMPLETE' and review['rubric_sha256']==sha(W/'RUBRIC.json')
        assert review['native_manifest_sha256']==sha(N/'MANIFEST.json')
        assert len(review['rows'])==9, 'Exactly9rawrows; do not collapse duplicate reviews'
        rows={r['sample_id']:r for r in review['rows']};assert len(rows)==9 and rows.keys()==sources.keys()
        for sid,r in rows.items():
            src=sources[sid]; allowed={p['file']:(p['sha256'],p['frame_ids']) for p in src['pairs']}
            for f in src['frames']:
                for kind in ('full','fixed'):allowed[f[kind]]=(f[kind+'_sha256'],[f['frame_id']])
            seen=set()
            for item in r['images_viewed']:
                p=item['file'];assert p in allowed,(who,sid,p)
                h,frames=allowed[p];assert item['sha256']==sha(p)==h,(who,sid,p)
                seen.update(frames)
            assert seen=={f['frame_id'] for f in src['frames']},('all13actuallyviewed',who,sid)
            assert r['alias']==src['alias'] and r['actual_image_viewed'] is True
            # The existing interval validator expects one source-image pin. Here it
            # is an explicit aggregate proof, after every actually viewed image is
            # checked above. Raw records are never changed or resaved.
            proof=hashlib.sha256(json.dumps(src['frames'],sort_keys=True).encode()).hexdigest()
            normalized=copy.deepcopy(r);normalized['image_sha256']=proof
            source=dict(alias=src['alias'],image_sha256=proof,frame_ids=[f['frame_id'] for f in src['frames']])
            issues.extend(dict(reviewer=who,sample_id=sid,**i) for i in m.validate_record(normalized,source))
        maps.append(rows)
    rows=[]
    for sid,src in sources.items():
        row=dict(sample_id=sid,predicates={})
        for hand in 'RL':
            for pred in ('down','hold'):
                k=hand+'_'+pred
                new=[d[sid]['presence'][hand][pred] for d in maps]
                old=[d[sid]['presence'][hand][pred] for d in prevmaps]
                elig=[m.eligible(d[sid],hand,pred) for d in maps]
                same=new[0] is not None and new[0]==new[1]
                row['predicates'][k]=dict(old=old,new=new,eligible=elig,
                    agreed=new[0] if same else None,
                    known_conflict=all(v is not None for v in new) and new[0]!=new[1],
                    known_unknown_mismatch=(new[0] is None)!=(new[1] is None),
                    both_unknown=all(v is None for v in new))
        rows.append(row)
    by={r['sample_id']:r for r in rows}
    six=[]
    for sid in plan['six_left']:
        c=by[sid]['predicates']['L_down']
        six.append(dict(sample_id=sid,**c,resolved_on_native_view=c['agreed'] is not None,
                        note_A=maps[0][sid]['notes'],note_B=maps[1][sid]['notes']))
    supports={h:[] for h in 'RL'}
    for sid in plan['hold_support']:
        for h in 'RL':
            c=by[sid]['predicates'][h+'_hold']
            if c['agreed']==0 and all(c['eligible']):supports[h].append(sid)
    summary={}
    for k in ('R_down','R_hold','L_down','L_hold'):
        cells=[r['predicates'][k] for r in rows]
        summary[k]=dict(all=9,agreed_positive=sum(c['agreed']==1 for c in cells),
            agreed_negative=sum(c['agreed']==0 for c in cells),
            known_conflicts=sum(c['known_conflict'] for c in cells),
            known_unknown_mismatch=sum(c['known_unknown_mismatch'] for c in cells),
            both_unknown=sum(c['both_unknown'] for c in cells))
    unresolved=[r['sample_id'] for r in six if not r['resolved_on_native_view']]
    ready_for_reassessment=not issues and not unresolved and all(supports.values()) and not any(v['known_conflicts'] for v in summary.values())
    out=dict(status='SOURCE_REPAIR_SUPPORTED_REASSESS_FULL_READINESS' if ready_for_reassessment else 'SOURCE_DIAGNOSIS_COMPLETE_TRAINING_NOT_READY',
        selected_retrospective_diagnosis=True,unchanged_failed24_denominator=True,windows=9,native_frames=117,
        summary=summary,rows=rows,left_six=six,left_six_resolved=6-len(unresolved),left_six_unresolved=unresolved,
        eligible_hold_absent_support=supports,annotation_contract_failures=issues,
        further_readiness_assessment_supported=ready_for_reassessment,training_allowed=False,
        trained_models=0,model_accuracy_improvement=None,
        limitations=['Same-Codex fresh-context viewing, not independent human ground truth.',
            'Retrospective selected9 cannot replace original24 reliability estimate.',
            'More visible pixels do not prove hidden motion or semantic importance.',
            'Existing-model preservation is not a learned non-regression result.'],
        source_pins={str(p):sha(p) for p in [Path(__file__),N/'AUDIT_FROZEN_v2.json',N/'MANIFEST.json',
            N/'review/NATIVE_A.json',N/'review/NATIVE_B.json',W/'PRESENCE_PILOT_AUDIT.json',W/'RUBRIC.json']})
    write(N/'NATIVE_DIAGNOSIS.json',out)
    print(json.dumps({k:out[k] for k in ['status','summary','left_six_resolved','left_six_unresolved','eligible_hold_absent_support','annotation_contract_failures']},ensure_ascii=False))
if __name__=='__main__':main()
