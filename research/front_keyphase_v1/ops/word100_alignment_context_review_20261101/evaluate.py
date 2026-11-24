"""Structural/context evaluation only. Does not recompute recognition or DTW scores."""
from pathlib import Path
import collections,sys
sys.dont_write_bytecode=True
import run
U=Path(__file__).resolve().parent

def status(value):
    return value['status'] if isinstance(value,dict) else value

def main():
    run.checkpins();manifest=run.read(U/'RESULT_MANIFEST.json')
    total=collections.Counter();temporal=collections.Counter();refcontext=collections.Counter();axes=collections.defaultdict(collections.Counter)
    rows=[];examples=[]
    for rec in manifest['contexts']:
        assert run.sha(rec['path'])==rec['sha256']
        out=run.read(rec['path']);source=run.read(out['source']);assert run.sha(out['source'])==out['source_sha256']
        assert run.digest(source)==out['original_evidence_digest'] and out['original_path_exact']
        a=out['annotation'];row=collections.Counter()
        for group,byaxis in a['groups'].items():
            for axis,info in byaxis.items():
                original=source['evidence']['groups'][group][axis]
                counts=original['counts']
                assert info['possible']==counts['possible']
                assert info['direct_complete_cells']==counts['matched_support']-counts['mixed_support_cells']
                assert info['direct_complete_cells']+info['needs_context_cells']==info['possible']
                assert sum(e['cell_count'] for e in info['episodes'])==info['needs_context_cells']
                for key in ('possible','direct_complete_cells','needs_context_cells'):total[key]+=info[key];row[key]+=info[key];axes[axis][key]+=info[key]
                for e in info['episodes']:
                    assert e['cell_count']==e['end']-e['start']+1
                    t,r=status(e['temporal_context']),status(e['reference_context'])
                    temporal[t]+=e['cell_count'];refcontext[r]+=e['cell_count'];total['episodes']+=1
                    if len(examples)<12 and t=='BOUNDED_BY_DISTINCT_OBSERVATIONS' and axis=='query':
                        examples.append(dict(query=out['query'],reference=out['reference'],seed=out['seed'],group=group,axis=axis,episode=e,
                            source=rec['path'],not_an_error_verdict=True))
        rows.append(dict(query=out['query'],reference=out['reference'],seed=out['seed'],counts=dict(row),path_original=True))
    assert len(rows)==108
    assert total['possible']==1926120 and total['direct_complete_cells']==732837 and total['needs_context_cells']==1193283
    avcounts=collections.Counter();avrows=[]
    for rec in manifest['availability']:
        assert run.sha(rec['path'])==rec['sha256']
        out=run.read(rec['path']);av=out['availability'];cnt=collections.Counter()
        for group,g in av['groups'].items():
            assert sum(r['cell_count'] for r in g['runs'])==g['possible']
            avcounts['possible']+=g['possible'];cnt['possible']+=g['possible']
            for r in g['runs']:
                n=r['cell_count'];assert n==r['end']-r['start']+1
                if r['direct_refs']:category='at_least_one_direct_reference'
                elif r['partial_refs']:category='partial_references_only'
                else:category='no_shared_reference'
                avcounts[category]+=n;cnt[category]+=n
                if r['covered_despite_other_reference_missing']:
                    assert r['direct_refs'] and r['unsupported_refs']
                    avcounts['direct_reference_with_other_unsupported']+=n;cnt['direct_reference_with_other_unsupported']+=n
        avrows.append(dict(query=out['query'],seed=out['seed'],counts=dict(cnt),reference_count=len(out['source_records'])))
    assert len(avrows)==48 and avcounts['possible']==437640
    assert sum(avcounts[k] for k in ('at_least_one_direct_reference','partial_references_only','no_shared_reference'))==437640
    out=dict(status='ADDITIVE_CONTEXT_ACCOUNTING_COMPLETE_NOT_ACCURACY_TEST',source_records=108,unique_query_seed_groups=48,
        total=dict(total),axis_counts={k:dict(v) for k,v in axes.items()},temporal_context_cells=dict(temporal),reference_context_cells=dict(refcontext),
        availability=dict(avcounts),contexts=rows,availability_rows=avrows,examples=examples,
        decision=dict(raw_evidence_path_scope_unchanged=True,no_interpolation=True,no_query_truth_used=True,
            new_alignment=False,new_scores=False,new_inference=False,new_training=False,educational_correctness=None,
            alignment_improved=False,observational_context_added=True,production_changed=False),
        limits=['Historical4word16clip inputs with3saved model outputs; context counts repeat masks acrossseeds',
            'DIRECT means measured-value availability, not correct same-stage alignment',
            'Before/after anchors never establish the hidden intermediate handshape',
            'FIT reference physical core drafts are not mandatory linguistic truth',
            'Crossing and handlowering protection is synthetic and source preservation, not annotated real educational accuracy'])
    run.new(U/'EVALUATION.json',out)
    run.new(U/'EXAMPLES.json',dict(examples=examples,selection='first12query-axis bounded contexts in frozen manifest; illustrations not metric selection',automatic_correction=None))
    print(dict(total));print('temporal',dict(temporal));print('reference',dict(refcontext));print('availability',dict(avcounts))

if __name__=='__main__':main()
