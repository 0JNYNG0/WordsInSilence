"""Store all twelve finite support-search outcomes; never declares training ready."""
from pathlib import Path
import sys,json,hashlib
sys.dont_write_bytecode=True
N=Path(__file__).resolve().parent;S=N/'support_inventory';W=N.parent/'word100_interval_motion_20261101'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def interval(i,frames):
    a,b=i['first_frame'],i['last_frame']
    assert type(a)is int and type(b)is int and frames[0]<=a<=b<=frames[-1]
    return set(range(a,b+1))
def main():
    freeze=read(S/'AUDIT_FROZEN.json')
    for p,h in freeze['pins'].items():assert sha(p)==h,p
    export=read(N/'review/SUPPORT_SELECTION_EXPORT_VERIFICATION.json')
    assert export['status'].startswith('PASS') and not export['failures']
    src={r['support_alias']:r for r in read(S/'MANIFEST.json')['rows']};assert len(src)==12
    reviews=[read(S/f'REVIEW_{who}.json') for who in 'AB'];maps=[];issues=[]
    for who,d in zip('AB',reviews):
        assert d['status']=='COMPLETE' and d['manifest_sha256']==sha(S/'MANIFEST.json') and d['rubric_sha256']==sha(W/'RUBRIC.json')
        assert len(d['rows'])==12
        rows={r['alias']:r for r in d['rows']};assert len(rows)==12 and rows.keys()==src.keys()
        for a,r in rows.items():
            source=src[a];frames=[f['frame_id'] for f in source['frames']]
            assert r['sample_id']==source['sample_id'] and r['target_hand']==source['target_hand']
            assert r['actual_image_viewed'] is True
            allow={p['file']:(p['sha256'],p['frame_ids']) for p in source['pairs']}
            for f in source['frames']:
                for k in ('full','fixed'):allow[f[k]]=(f[k+'_sha256'],[f['frame_id']])
            seen=set()
            for im in r['images_viewed']:
                h,ids=allow[im['file']];assert h==im['sha256']==sha(im['file']);seen.update(ids)
            assert seen==set(frames)
            v=r['hold_present'];assert v is None or type(v)is int and v in (0,1)
            for k in ('full13_position_observable','full13_roi'):assert r[k] is None or type(r[k])is bool
            unknown=set()
            for i in r['uncertain_intervals']:unknown|=interval(i,frames)
            segments=r['hold_segments'];assert type(segments)is list
            for i in segments:
                span=interval(i,frames);assert i['in_roi'] is None or type(i['in_roi'])is bool
                if len(span)<3 or span&unknown:issues.append(dict(alias=a,reviewer=who,reason='short_or_unknown_positive_witness'))
                if i['in_roi'] is False and r['full13_roi'] is True:issues.append(dict(alias=a,reviewer=who,reason='contradictory_ROI'))
            if v==1 and not segments:issues.append(dict(alias=a,reviewer=who,reason='positive_without_witness'))
            if v!=1 and segments:issues.append(dict(alias=a,reviewer=who,reason='nonpositive_with_witness'))
            if v==0 and (r['full13_position_observable'] is not True or unknown):issues.append(dict(alias=a,reviewer=who,reason='negative_without_complete_observation'))
        maps.append(rows)
    paired=[]
    for a,source in src.items():
        rr=[m[a] for m in maps];values=[r['hold_present'] for r in rr]
        same=values[0] is not None and values[0]==values[1]
        eligible=[(any(s['in_roi'] is True for s in r['hold_segments']) if r['hold_present']==1 else
                   r['hold_present']==0 and r['full13_position_observable'] is True and r['full13_roi'] is True) for r in rr]
        paired.append(dict(alias=a,sample_id=source['sample_id'],target_hand=source['target_hand'],person=source['person'],word_id=source['word_id'],
            raw=values,agreed=values[0] if same else None,eligible=eligible,
            eligible_agreed_absence=same and values[0]==0 and all(eligible) and not any(i['alias']==a for i in issues),
            notes=[r['note'] for r in rr]))
    metrics={};support={}
    for h in 'RL':
        rows=[r for r in paired if r['target_hand']==h]
        metrics[h]=dict(all=len(rows),agreed_absent=sum(r['agreed']==0 for r in rows),agreed_present=sum(r['agreed']==1 for r in rows),
            known_conflicts=sum(all(v is not None for v in r['raw']) and r['raw'][0]!=r['raw'][1] for r in rows),
            known_unknown_mismatch=sum((r['raw'][0] is None)!=(r['raw'][1] is None) for r in rows),
            both_unknown=sum(all(v is None for v in r['raw']) for r in rows),
            eligible_agreed_absence=sum(r['eligible_agreed_absence'] for r in rows))
        yes=[r for r in rows if r['eligible_agreed_absence']]
        support[h]=dict(windows=[r['sample_id'] for r in yes],people=sorted({r['person'] for r in yes}),expressions=sorted({r['word_id'] for r in yes}))
    out=dict(status='FINITE_SUPPORT_INVENTORY_COMPLETE_NOT_TRAINING_READINESS',metrics=metrics,support=support,rows=paired,
        annotation_contract_failures=issues,all_candidates_retained=True,opportunities_screened=360,reviewed_windows=12,
        candidate_replacements=0,original24_denominator_unchanged=True,native9_result_unchanged=True,
        training_allowed=False,model_runs=0,model_accuracy_improvement=None,
        conclusion='Observed support can establish existence in selected publicFIT cases only. Full same-rule labels, class support, controls and learning contract remain separate; no candidate ranking or agreement is educational ground truth.',
        pins={str(p):sha(p) for p in [Path(__file__),S/'AUDIT_FROZEN.json',S/'PLAN_FROZEN.json',S/'SELECTION.json',S/'MANIFEST.json',
            S/'REVIEW_A.json',S/'REVIEW_B.json',N/'NATIVE_DIAGNOSIS.json',N/'review/SUPPORT_SELECTION_EXPORT_VERIFICATION.json',W/'PRESENCE_PILOT_AUDIT.json']})
    write(S/'SUPPORT_RESULT.json',out)
    print(json.dumps({k:out[k] for k in ['status','metrics','support','annotation_contract_failures']},ensure_ascii=False))
if __name__=='__main__':main()
