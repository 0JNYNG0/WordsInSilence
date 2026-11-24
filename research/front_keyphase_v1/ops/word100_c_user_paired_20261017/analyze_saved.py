"""Read-only paired evaluation of every initial take; no inference or fitting."""
import hashlib
import json
import sys
from pathlib import Path
from collections import Counter
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SESSION = ROOT / 'runs/ses_b7aef2518a30'
CPIN = '283a54bd9866c97959f9d615ebc38d126ee25688821960b208cf9412389eeaab'
MPIN = 'e39d8ac46f5a5f8442c48a141498dc810ccbfad242e3f104e1b8633988515c2f'
def read(p): return json.loads(p.read_text('utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name, obj): (HERE/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n','utf-8')

def quality(p):
    with np.load(p,allow_pickle=False) as z:
        t=z['timestamp_us']; ids=z['frame_id']; u=z['usable']; xy=z['xy_px']; seen=z['observed']
        assert not (u & ~seen).any() and not np.isfinite(xy[~seen]).any()
        assert np.all(np.diff(t)>0)
        l=u[:,33:54].any(1); r=u[:,54:75].any(1); active=np.flatnonzero(l|r)
        a,b=int(active[0]),int(active[-1]); sl=slice(a,b+1)
        q={'frames':len(t),'duration_s':float((t[-1]-t[0])/1e6),'effective_fps':float((len(t)-1)*1e6/(t[-1]-t[0])),
           'max_gap_ms':float(np.diff(t).max()/1000),'active_first':a,'active_last':b,'active_frames':b-a+1,
           'active_duration_s':float((t[b]-t[a])/1e6),'active_any_hand_fraction':float((l|r)[sl].mean()),
           'active_left_fraction':float(l[sl].mean()),'active_right_fraction':float(r[sl].mean()),
           'active_both_fraction':float((l&r)[sl].mean()),'active_shoulder_fraction':float(u[sl,11:13].all(1).mean()),
           'left_ever':bool(l.any()),'right_ever':bool(r.any()),
           'left_partial_frames':int(((u[:,33:54].sum(1)>0)&(u[:,33:54].sum(1)<21)).sum()),
           'right_partial_frames':int(((u[:,54:75].sum(1)>0)&(u[:,54:75].sum(1)<21)).sum())}
        return q,t.copy(),ids.copy()

def main():
    config=read(ROOT/'config/word100_practice.json'); codes=config['word_codes']; labels=config['labels']
    names={x['word_id']:x.get('display_name',x['label']) for x in config['items']}
    meta=read(SESSION/'session.json'); tids=meta['trial_ids']; assert len(tids)==13
    reviews=[json.loads(x) for x in (SESSION/'trial_reviews.jsonl').read_text('utf-8').splitlines() if x.strip()]
    confirmed={x['trial_id'] for x in reviews}
    contract={'scope':'saved-result evaluation only; no new inference, training, tuning or model selection',
        'session':SESSION.name,'primary_trial_ids':tids[:12],'retry_trial_ids':tids[12:],
        'selection_rule':'first chronological attempt of all12; missing confirmation does not exclude; retry never replaces first attempt',
        'classification_candidates':100,'participant_count':1,'same_capture_comparison':'complete frozen pipelines, not weights-only',
        'not_blind':'participant/environment already informed method design; recordings not used to fit or tune',
        'authorization':'user requests comparison of completed12 first takes and separate 마음에들다 retry, never training/tuning'}
    write('EVALUATION_SCOPE.json',contract)
    rows=[]; sources={}; checks=[]
    for number,tid in enumerate(tids,1):
        base=SESSION/'trials'/tid; p=base.with_suffix('.result.json'); d=read(p)
        assert d['trial_id']==tid and d['session_id']==SESSION.name and d['origin']=='camera'
        assert d['data_use']=='evaluation_only_never_training_tuning_or_model_selection'
        assert d['comparison']['same_capture'] and d['comparison']['same_frames']
        assert d['research_model']['class_word_codes']==codes and d['research_model']['class_labels']==labels
        target=d['requested_word_id']; i=codes.index(target)
        row={'number':number,'trial_id':tid,'group':'first12' if number<=12 else 'retry',
             'word_id':target,'label':names[target],'confirmation_saved':tid in confirmed,'end_reason':d['end_reason']}
        for key,out,pin in [('C',d['practice100'],CPIN),('M0',d['comparison']['baseline'],MPIN)]:
            assert out['checkpoint_sha256']==pin and out['model_status']=='READY'
            v=np.array(out['logits'],dtype=np.float64); assert v.shape==(100,) and np.isfinite(v).all()
            order=np.argsort(-v,kind='stable'); rank=int(np.flatnonzero(order==i)[0])+1
            prob=np.exp(v-v.max()); prob/=prob.sum()
            assert out['requested_rank_all100']==rank
            for j,x in enumerate(out['ranking']):
                k=order[j]; assert x['word_id']==codes[k] and x['rank']==j+1
                assert abs(x['model_score']-float(prob[k]))<1e-12
            row[key]={'rank':rank,'top1':labels[order[0]],'target_model_score':float(prob[i]),
                      'top1_model_score':float(prob[order[0]]),'top5':[labels[k] for k in order[:5]],'support':out['support']}
        assert row['C']['rank']==d['requested_rank_all100']
        mp=base.with_suffix('.npz'); cp=base.with_name(tid+'.candidate_c.npz')
        assert sha(mp)==d['observation_sha256'] and sha(cp)==d['candidate_observation']['observation_sha256']
        mq,mt,mi=quality(mp); cq,ct,ci=quality(cp)
        assert np.array_equal(mt,ct) and np.array_equal(mi,ci)
        row['M0_observation']=mq; row['C_observation']=cq
        row['result_latency_ms']=d['timing']['server_end_candidate_to_result_ms']
        row['capture_frames']=d['diagnostic_capture']['frame_count']
        dp=base.with_name(tid+'.training_dtw_0af5e66be355.json')
        if dp.exists():
            dt=read(dp); assert dt['query_observation_sha256']==d['observation_sha256']
            row['DTW']={'status':dt['status'],'metrics':[{k:x.get(k) for k in ['key','label','value','unit','status','ratio']} for x in dt['metrics']]}
        for fp in [p,mp,cp,mp.with_suffix('.sidecar.json'),cp.with_suffix('.sidecar.json'),dp]:
            if fp.exists(): sources[str(fp.relative_to(ROOT))]=sha(fp)
        rows.append(row); checks.append({'trial_id':tid,'saved_logits_rank_and_softmax':True,'same_frame_ids_and_clock':True,'observation_hashes':True})
    assert len({x['word_id'] for x in rows[:12]})==12
    assert [x['word_id'] for x in rows[:12]]==[x['word_id'] for x in config['items']]
    assert rows[12]['word_id']==rows[1]['word_id']=='WORD0500'
    stats={}
    for group,subset in [('first12',rows[:12]),('retry',rows[12:]),('all13_secondary',rows)]:
        n=len(subset); stats[group]={'n':n}
        for k in ['M0','C']:
            s=stats[group][k]={}
            for top in [1,3,5]:
                count=sum(x[k]['rank']<=top for x in subset); s[f'top{top}_count']=count;s[f'top{top}_percent']=count/n*100
        stats[group]['top1_delta_pp']=stats[group]['C']['top1_percent']-stats[group]['M0']['top1_percent']
        stats[group]['top5_delta_pp']=stats[group]['C']['top5_percent']-stats[group]['M0']['top5_percent']
        stats[group]['top1_gains']=[x['label'] for x in subset if x['C']['rank']==1<x['M0']['rank']]
        stats[group]['top1_regressions']=[x['label'] for x in subset if x['M0']['rank']==1<x['C']['rank']]
        stats[group]['rank_regressions']=[x['label'] for x in subset if x['C']['rank']>x['M0']['rank']]
    checkpoints=[Path(config['checkpoint']),ROOT/'ops/word100_body_boundary_20261017/training_C_20261013_best.pt']
    assert [sha(p) for p in checkpoints]==[MPIN,CPIN]
    for p in checkpoints: sources[str(p.relative_to(ROOT))]=sha(p)
    sources[str((SESSION/'session.json').relative_to(ROOT))]=sha(SESSION/'session.json')
    sources[str((SESSION/'trial_reviews.jsonl').relative_to(ROOT))]=sha(SESSION/'trial_reviews.jsonl')
    result={'schema':'wis_c_user_paired_saved_analysis_v1','scope':contract,'statistics':stats,'rows':rows,
            'verification':checks,'source_hashes':sources,'confirmation_count':len(confirmed),
            'DTW_status_counts':dict(Counter(x.get('DTW',{}).get('status','NOT_SAVED') for x in rows)),
            'new_training':0,'new_inference':0,'tuning':0,'web_changes':0,'camera_access':0}
    write('RESULT.json',result)
    print(json.dumps({'statistics':stats,'rows':[{'label':x['label'],'C':x['C']['rank'],'M0':x['M0']['rank'],'confirmed':x['confirmation_saved']} for x in rows],
                      'DTW_status_counts':result['DTW_status_counts'],'fps_range':[min(x['C_observation']['effective_fps'] for x in rows),max(x['C_observation']['effective_fps'] for x in rows)]},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
