"""Retrospective paired evaluation of stored user results only; no model imports."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import Counter
import hashlib, json, math
import numpy as np

U=Path(__file__).resolve().parent
LAB=U.parents[1]
S=LAB/'runs/ses_067e0348ff32'
C='word100_tcn_roi_c_v1';T='word100_tcn_t3f_v1'
CP='283a54bd9866c97959f9d615ebc38d126ee25688821960b208cf9412389eeaab'
TP='b0faa0df5055204295c60b78c596a0b490cd63ef14f30225582a85ea39855846'
def read(p):return json.loads(p.read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False),'utf-8')

def main():
    assert not (U/'ANALYSIS.json').exists(),'Completed analysis must not be overwritten'
    session=read(S/'session.json');cfg=read(LAB/'config/word100_practice.json')
    assert session['state']=='CLOSED' and not session['camera']
    ids=session['trial_ids'];files=list((S/'trials').glob('*.result.json'))
    assert len(ids)==len(set(ids))==len(files)==12
    assert {p.stem.removesuffix('.result') for p in files}==set(ids)
    reviews=[json.loads(x) for x in (S/'trial_reviews.jsonl').read_text('utf-8').splitlines() if x]
    counts=Counter(x['trial_id'] for x in reviews)
    assert counts==Counter({i:1 for i in ids})
    reviews={r['trial_id']:r for r in reviews}
    pins={str(p.relative_to(LAB)):sha(p) for p in sorted(S.rglob('*')) if p.is_file()}
    web=read(LAB/'ops/word100_t3f_web_20261019/RESULT.json')
    for p,h in web['changed_files'].items():assert sha(LAB/p)==h,p
    fixed=[LAB/'ops/word100_body_boundary_20261017/training_C_20261013_best.pt',
           LAB/'ops/word100_frozen_c_temporal_20261019/T3F_20261013_best.pt']
    assert [sha(p) for p in fixed]==[CP,TP]
    pins.update({str(p.relative_to(LAB)):sha(p) for p in fixed})
    dump(U/'SCOPE.json',dict(session_id=S.name,selection='all12 trial_ids in closed post-deployment session, not best attempts',
        trial_ids=ids,expected_practice_count=12,source_hashes=pins,frozen_C=CP,frozen_T3F=TP,
        design='retrospective paired analysis of pre-fixed models; not a preregistered human study',
        allowed='read stored logits, reviews, observation arrays, existing DTW summaries and hashes; write isolated report',
        forbidden=['training','tuning','model selection','threshold changes','re-inference','camera','source mutation','external transmission']))
    rows=[]
    for tid in ids:
        base=S/'trials'/tid;d=read(base.with_suffix('.result.json'));o=d['practice100'];b=d['comparison']['baseline'];p=d['comparison']
        assert d['session_id']==S.name and d['trial_id']==tid and d['origin']=='camera'
        assert (o['model_key'],o['checkpoint_sha256'])==(T,TP)
        assert (b['model_key'],b['checkpoint_sha256'])==(C,CP)
        assert d['research_model']['checkpoint_sha256']==TP
        assert p['same_capture'] and p['same_frames'] and p['same_input']
        assert o['input_sha256']==b['input_sha256']==p['input_sha256']
        assert o['input_shape']==b['input_shape']==p['input_shape']
        assert d['data_use']=='evaluation_only_never_training_tuning_or_model_selection'
        assert sha(base.with_suffix('.npz'))==d['observation_sha256']
        assert sha(base.with_name(tid+'.candidate_c.npz'))==d['candidate_observation']['observation_sha256']
        review=reviews[tid];assert review['actual_word_id']==d['requested_word_id'] and review['followed_guidance']=='yes'
        assert not review['is_expert_verification']
        target=cfg['word_codes'].index(d['requested_word_id'])
        row=dict(trial_id=tid,word_id=d['requested_word_id'],label=cfg['labels'][target],same_input_sha256=p['input_sha256'],models={})
        for name,r in [('C',b),('T3F',o)]:
            assert r['model_status']=='READY'
            v=r['logits'];assert len(v)==100 and all(math.isfinite(x) for x in v)
            order=sorted(range(100),key=lambda i:(-v[i],i));den=sum(math.exp(x-max(v)) for x in v)
            probs=[math.exp(x-max(v))/den for x in v]
            rank=order.index(target)+1
            assert rank==r['requested_rank_all100']
            assert [x['word_id'] for x in r['ranking']]==[cfg['word_codes'][i] for i in order[:5]]
            assert all(math.isclose(x['model_score'],probs[i],rel_tol=1e-12,abs_tol=1e-14) for x,i in zip(r['ranking'],order[:5]))
            row['models'][name]=dict(rank=rank,correct=rank==1,top5=rank<=5,
                target_model_score_percent=100*probs[target],top1_label=cfg['labels'][order[0]],
                top2_label=cfg['labels'][order[1]],top2_model_score_percent=100*probs[order[1]],
                first_second_gap_score_points=100*(probs[order[0]]-probs[order[1]]),
                target_logit=v[target],top1_logit=v[order[0]],elapsed_ms=r.get('elapsed_ms'),
                top5_readings=[dict(rank=j+1,label=cfg['labels'][i],model_score_percent=100*probs[i]) for j,i in enumerate(order[:5])])
        row['outcome']='gain' if not row['models']['C']['correct'] and row['models']['T3F']['correct'] else 'regression' if row['models']['C']['correct'] and not row['models']['T3F']['correct'] else 'both_correct' if row['models']['C']['correct'] else 'both_wrong'
        rival=cfg['labels'].index(row['models']['C']['top1_label'])
        row['target_vs_C_winner_logit_margin']={name:r['logits'][target]-r['logits'][rival] for name,r in [('C',b),('T3F',o)]}
        # Per-frame diagnostic counts are observations, not a diagnosis of wrong signing.
        with np.load(base.with_suffix('.npz'),allow_pickle=False) as primary, np.load(base.with_name(tid+'.candidate_c.npz'),allow_pickle=False) as z:
            assert np.array_equal(primary['frame_id'],z['frame_id']) and np.array_equal(primary['timestamp_us'],z['timestamp_us'])
            assert np.all(np.diff(z['timestamp_us'])>0)
            lo=o['support']['active_span_first_index'];hi=o['support']['active_span_last_index']+1
            u=z['usable'][lo:hi];n=len(u);left=u[:,33:54].any(1);right=u[:,54:75].any(1)
            row['observations']=dict(recorded_frames=len(z['frame_id']),active_frames=n,
                left_any_hand_frames=int(left.sum()),right_any_hand_frames=int(right.sum()),
                neither_hand_frames=int((~(left|right)).sum()),both_shoulders_frames=int(u[:,[11,12]].all(1).sum()),
                max_frame_gap_ms=float(np.diff(z['timestamp_us']).max()/1000),
                meaning='Any usable point in hand block; does not establish complete/correct anatomy or reason for absence')
        dp=list((S/'trials').glob(tid+'.role_dtw_*.json'));assert len(dp)==1
        dtw=read(dp[0]);assert dtw['query_observation_sha256']==d['observation_sha256'] and dtw['word_id']==d['requested_word_id']
        row['stored_DTW']=dict(status=dtw['status'],required_sides=dtw['required_sides'],
            metrics=[{k:m[k] for k in ['key','label','unit','value','ratio','status']} for m in dtw['metrics']],
            interpretation='Shared existing DTW, not a C/T3F performance difference or correct-sign verdict')
        rows.append(row)
    assert Counter(r['word_id'] for r in rows)==Counter(x['word_id'] for x in cfg['items'])
    summary={name:dict(top1=sum(r['models'][name]['correct'] for r in rows),top5=sum(r['models'][name]['top5'] for r in rows),n=len(rows)) for name in ['C','T3F']}
    for v in summary.values():v.update(top1_percent=100*v['top1']/v['n'],top5_percent=100*v['top5']/v['n'])
    events=[json.loads(x) for x in (S/'timing_events.jsonl').read_text('utf-8').splitlines() if x]
    outcomes=Counter(r['outcome'] for r in rows)
    out=dict(status='COMPLETE_PENDING_INDEPENDENT_ARITHMETIC',at=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
        session_id=S.name,participant_count=1,attempts=12,words=12,model_comparisons=24,saved_logits_checked=2400,
        attempted_labels='12 intended words, participant self-report; not expert sign-form validation',
        review_count=12,repeat_attempts=0,unscored_session_trials=0,results=summary,
        delta_top1_pp=summary['T3F']['top1_percent']-summary['C']['top1_percent'],paired_outcomes=dict(outcomes),
        rows=rows,event_counts=dict(Counter(x.get('event') for x in events)),
        training=0,tuning=0,model_selection=0,re_inference=0,decoded_user_RGB_frames=0,
        comparison_basis='one stored pair per identical observation/feature-hash; no cross-date capture comparison',
        same_input_verification='Both recorded input hashes and shapes match; original/candidate NPZ hashes and frame/PTS pairs verified; channels not rebuilt',
        limitations=['One known tester, one attempt per expression; not all100 words or population-wide accuracy',
                     'Tester has practised these expressions; not a first-ever naive-user evaluation',
                     'Model scores are not calibrated correctness probabilities',
                     'No causal attribution of occlusion versus gesture error from missing-point counts alone',
                     'DTW was not changed by T3-F; label correctness is not expert-certified execution quality'])
    for p,h in pins.items():assert sha(LAB/p)==h,p
    dump(U/'ANALYSIS.json',out)
    print(json.dumps(dict(results=summary,delta_pp=out['delta_top1_pp'],outcomes=dict(outcomes),source_files_unchanged=len(pins)),ensure_ascii=False))
if __name__=='__main__':main()
