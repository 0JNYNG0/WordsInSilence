"""Connect saved, target-blind observer outputs to audited source contracts.

Fit/development only; no detector/backbone/head inference or training.
"""
from pathlib import Path
import sys,json,hashlib,time,copy,traceback
sys.dont_write_bytecode=True
import numpy as np
from engine import *
U=Path(__file__).resolve().parent;OPS=U.parent;OLD=OPS/'word100_stage_feedback_cycle_20261031'
sys.path.insert(0,str(OLD));import feedback as fb

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)

def freeze():
    info=read(OLD/'LEARNING_INPUTS.json');rows=[r for r in info['rows'] if r['role'] in ['fit','development']]
    assert len(rows)==48
    files=[Path(__file__),U/'engine.py',U/'AUTHORIZATION.json',U/'source_audit/SOURCE_CONTRACTS.json',OLD/'LEARNING_INPUTS.json',OLD/'feedback.py',OLD/'learning.py',OLD/'DEVELOPMENT_COMPARISON.json',OLD/'runs/pose_capacity_matched_42/probabilities.npy']
    for r in rows:files.extend(Path(r[k]) for k in ['input','targets_file','observation'])
    protocol=dict(status='FROZEN_BEFORE_LINKED_DEV_OUTPUTS',rows=rows,targets=info['targets'],target_indices=info['target_indices'],
      observer='Existing pose_capacity_matched seed42 probabilities; no new inference/selection',
      alignment='Previously saved pose_capacity_matched development DTW paths, unchanged. New order checks use original time order.',
      source='source_audit/SOURCE_CONTRACTS.json; prose-derived hypotheses, partial form links, no expert-verified acceptable bounds',
      anchors='Fit-only draft RGB known positive labels inside original token interval; preserve every anchor including reference/query missing or unmappable',
      thresholds=dict(confidence=CONFIDENCE,min_consecutive_order_frames=MIN_RUN_FRAMES,min_references_same_query_frame=MIN_REFERENCES),
      support='Shape needs wrist plus all12 nonthumb projected angles; other fixed supports retained. Coordinate support is not image truth.',
      prohibitions=['no held/fresh score tuning','no user/mentor data','no new downloads/training','no production replacement','no missing=absent','no open_four0=fist','no numerical match=correct KSL'],
      pins={str(p):sha(p) for p in files},protected=read(OLD/'PROTOCOL.json')['protected'])
    write(U/'FROZEN.json',protocol)
    return protocol

def load_data(frozen):
    info=read(OLD/'LEARNING_INPUTS.json');probs=np.load(OLD/'runs/pose_capacity_matched_42/probabilities.npy',mmap_mode='r')
    arrays={};cases={};observations={};index={r['alias']:i for i,r in enumerate(info['rows'])}
    for row in frozen['rows']:
        alias=row['alias'];a=dict(np.load(row['input'],allow_pickle=False));arrays[alias]=a
        p=np.array(probs[index[alias],:len(a['x'])],copy=True)
        case=fb.prepare_case(row,a,p);cases[alias]=case
        support=feature_support(a,row['feature_names'],info['targets'],case['full_support'])
        obs=dict(alias=alias,prob=p,support=support,frame_ids=a['frame_id'],times_us=a['timestamp_us'],sample_indices=case['indices'],intervals=row['intervals'])
        if row['role']=='fit':
            z=np.load(row['targets_file'],allow_pickle=False);y=z['y'][:,info['target_indices']];m=z['target_mask'][:,info['target_indices']]
            obs['anchors']={f:select_anchor_indices(dict(row,timestamps_us=a['timestamp_us']),j,y,m).tolist() for j,f in enumerate(info['targets'])}
        observations[alias]=obs
    return info,arrays,cases,observations

def observation_digest(obs):
    h=hashlib.sha256()
    for name in ['prob','support','frame_ids','times_us','sample_indices']:
        x=np.asarray(obs[name]);h.update(name.encode());h.update(str(x.dtype).encode());h.update(str(x.shape).encode());h.update(x.tobytes())
    return h.hexdigest()

def main():
    f=read(U/'FROZEN.json')
    for group in ['pins','protected']:
        for p,h in f[group].items():assert sha(p)==h,p
    spec=read(U/'source_audit/SOURCE_CONTRACTS.json');contracts={r['word_id']:r for r in spec['expressions']}
    info,arrays,cases,obs=load_data(f);prior=read(OLD/'DEVELOPMENT_COMPARISON.json')['rows']
    old={r['query']:r for r in prior if r['condition']=='pose_capacity_matched'}
    results=[];timing=[];inputs=[]
    initial={a:observation_digest(o) for a,o in obs.items()}
    for row in f['rows']:
        if row['role']!='development':continue
        alias=row['alias'];contract=contracts[row['word_id']];q=obs[alias]
        refs=[]
        for pair in old[alias]['pairs']:
            ref=copy.copy(obs[pair['reference']]);ref['reference_to_query']=pair['reference_to_query'];refs.append(ref)
        assert len(refs)==3
        t=time.perf_counter();out=run_expression(contract,refs,q,f['targets']);elapsed=time.perf_counter()-t
        out.update(source=row['source'],source_sha256=row['source_sha256'],known_token_interval=row['intervals'],raw_physical_difference_candidates=old[alias]['events'],
          outside_source_contract_candidates=[e for e in old[alias]['events'] if e['feature'] not in {c.get('feature') for c in contract['clauses'] if c['link_status'] in ['partial_supported','proxy_only']}],
          baseline_distance=old[alias]['distance'],baseline_distance_is_not_correctness=True,
          reference_sources=[dict(alias=ref['alias'],source=next(r['source'] for r in f['rows'] if r['alias']==ref['alias'])) for ref in refs])
        results.append(out);timing.append(dict(alias=alias,source_comparison_seconds=elapsed))
        inputs.append(dict(alias=alias,word_id=row['word_id'],refs=[r['alias'] for r in refs],observer_digest=initial[alias]))
    assert initial=={a:observation_digest(o) for a,o in obs.items()}
    # Query labels are read ONLY after every output, solely for descriptive audit.
    audits=[]
    for row,out in zip([r for r in f['rows'] if r['role']=='development'],results):
        z=np.load(row['targets_file'],allow_pickle=False);y=z['y'][:,f['target_indices']];m=z['target_mask'][:,f['target_indices']]
        lookup={int(frame):i for i,frame in enumerate(obs[row['alias']]['frame_ids'])};cells={}
        for clause in out['clauses']:
            if clause.get('feature') not in f['targets']:continue
            j=f['targets'].index(clause['feature'])
            for ref in clause['references']:
                for c in ref['cells']:
                    if c['state'] not in ['SUPPORTED_STATE','OPPOSITE_STATE'] or c.get('query_frame') is None:continue
                    i=lookup[c['query_frame']];key=(c['query_frame'],clause['feature'])
                    cells[key]=dict(frame=c['query_frame'],feature=clause['feature'],state=c['state'],draft_label=int(y[i,j]) if m[i,j] else None)
        vals=list(cells.values());known=[c for c in vals if c['draft_label'] is not None]
        audits.append(dict(alias=row['alias'],evaluated_cells=len(vals),known_draft_cells=len(known),agree=sum((c['state']=='SUPPORTED_STATE')==(c['draft_label']==1) for c in known),cells=vals,scope='Sparse Codex physical draft labels, not independent human or KSL pedagogical accuracy. No new model score.'))
    statuses={s:sum(c['status']==s for r in results for c in r['clauses']) for s in ['GUIDE_RELEVANT_DIFFERENCE_CANDIDATE','DESCRIBED_FEATURE_OBSERVED','INSUFFICIENT_LINKED_EVIDENCE','SOURCE_LINK_WITHHELD']}
    summary=dict(development_videos=len(results),fit_references=36,source_expressions=len(contracts),clause_statuses=statuses,
        order_statuses={s:sum(c['status']==s for r in results for c in r['orders']) for s in ['DESCRIBED_ORDER_PATTERN_OBSERVED','REVERSE_ORDER_CANDIDATE','ORDER_UNCONFIRMED','ORDER_NOT_OBSERVABLE']},
        source_anchored_cells=sum(c.get('expected_anchor_count',0) for r in results for c in r['clauses']),
        raw_difference_candidate_cells=sum(len(r['raw_physical_difference_candidates']) for r in results),
        outside_source_contract_cells=sum(len(r['outside_source_contract_candidates']) for r in results),
        comparison_median_seconds=float(np.median([t['source_comparison_seconds'] for t in timing])),
        new_training=0,new_model_inference=0,educational_accuracy=None,whole_sign_verdict_count=0,linguistic_tolerance_claim_count=0)
    write(U/'RESULT.json',dict(status='COMPLETE_SOURCE_LINKED_DEVELOPMENT_COMPARISON',summary=summary,rows=results,timing=timing,physical_draft_audit=audits,inputs=inputs))
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':
    try:
        if sys.argv[1:]==['freeze']:freeze();print('FROZEN')
        else:main()
    except BaseException:
        p=U/'FIRST_FAILURE.txt'
        if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
        raise
