"""Public-only feature and saved-path comparison. No model/detector calls."""
from pathlib import Path
import sys,json,hashlib,time,traceback
from datetime import datetime,timezone,timedelta
from collections import Counter
import numpy as np
import candidate as C
U=Path(__file__).resolve().parent;B=U.parents[1];ROOT=B.parents[1];OLD=U.parent/'word100_multiref_observation_compare_20261019'
sys.dont_write_bytecode=True;sys.path.insert(0,str(B/'src'))
from wis_lab.core import features as F
from wis_lab.core.observation import Observation
from wis_lab.service import word100_analysis as A
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def now():return datetime.now(timezone(timedelta(hours=9))).isoformat()
def frozen():
    f=read(U/'FROZEN.json');assert all(sha(p)==h for p,h in f['protected'].items());assert sha(U/'candidate.py')==f['candidate_sha256'];return f
def freeze():
    assert not (U/'FROZEN.json').exists();old=read(OLD/'FROZEN.json')
    for p,h in old['protected'].items():assert sha(p)==h
    (U/'before').mkdir(exist_ok=True)
    for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',B/'ops/STATE.json']:(U/'before'/p.name).write_bytes(p.read_bytes())
    write(U/'FROZEN.json',{'at':now(),'authorization':'2026-10-20 user approves separated hand-position and per-angle evidence, lowered vs missing intervals, trajectory origin diagnostics on public data only',
      'protected':old['protected'],'previous_frozen_sha256':sha(OLD/'FROZEN.json'),'candidate_sha256':sha(U/'candidate.py'),'runner_sha256':sha(__file__),
      'scope':'1199 public clips, 3 old views, reuse saved DTW paths. No recognition assistance/training/production/private recordings/heldout14-18.',
      'angle_policy':'15 2D flexion triplets; valid observed three points and nonzero bones; local geometry independent of shoulder/wrist/other fingers. No imputation. Per-angle support>=.5 distinct query/reference frames>=8 refs>=5. Whole-hand available only all15 ready; no correctness verdict.',
      'alignment':'Previous saved path, same sampling, old alignment evidence>=.35; no rescoring original results or reoptimizing path.',
      'segment_policy':'View-local observed/missing/arm-hint/internal-gap fields separate from offline full-native geometric witness. No automatic exclusion or semantic hand roles.',
      'lowered_policy':{'full_21_observed':True,'entire_hand_below_ROI':True,'window_us':300000,'min_covered_us':250000,'min_frames':8,'max_gap_us':100000,'stability_sw':.05,'below_shoulders_sw':1.0,'threshold_source':'C .3s/8frames/.1s/.05 stability, existing role height1SW; 250ms tolerance declared for discrete window','deployment':'offline only when full-native is visible'},
      'trajectory':'Same path and valid pairs. Old individual first-observation vs first mutually valid aligned pair. Translation-invariant diagnostic only; no guarantee shared semantic phase; missing prefix remains unassessed.',
      'success':'Preserve previously valid position/angles; hidden joints never affect available scores; no missing->zero/rest; quantify gains/losses and full-hand/partial separately; no accuracy claims.',
      'inputs':old['inputs'],'new_training':0})
    write(U/'STATUS.json',{'status':'FROZEN','at':now()});print('FROZEN',flush=True)
def prepare():
    f=frozen();assert not (U/'PREPARE.json').exists();stats=Counter();beg=time.time()
    for i,row in enumerate(f['inputs']):
        cid=row['clip_id'];recordpath=B/'ops/word100_body_boundary_20261017/data/records'/(cid+'.json')
        assert sha(recordpath)==row['record_hashes'][str(recordpath)]
        rec=read(recordpath);nativepath=rec['views']['native']['observation'];assert sha(nativepath)==rec['outputs'][nativepath]
        native=load(nativepath);nobj=Observation.load(nativepath);origin,scale,_=F.body_frame(nobj)
        for view in ('legacy','masked','sensor'):
            p=row['inputs'][view];assert sha(p)==row['input_hashes'][view];obs=load(p)
            assert np.array_equal(obs['frame_id'],native['frame_id']) and np.array_equal(obs['timestamp_us'],native['timestamp_us'])
            meta=read(OLD/'inputs'/view/(cid+'.json'));sample=load(meta['sample_path']);assert sha(meta['sample_path'])==meta['sample_sha256']
            # Frame IDs, not a guessed span offset, bind new local evidence to old saved paths.
            indices=np.searchsorted(obs['frame_id'],sample['frame_id']);assert np.array_equal(obs['frame_id'][indices],sample['frame_id'])
            data={'frame_id':sample['frame_id'],'timestamp_us':sample['timestamp_us']};info={'view':view,'clip_id':cid,'role':row['role'],'sides':{}}
            states=C.source_states(native,obs,obs['timestamp_us'],rec['calibration']['box_source_pixels'][3],rec['calibration']['source_width'],origin,scale,row['intervals'])
            for side,sl in [('L',slice(33,54)),('R',slice(54,75))]:
                ang,ok=C.angles(obs['xy_px'][:,sl],obs['usable'][:,sl]);oldang,oldvalid=A._group(sample,'bend_'+side)
                delta=np.abs(ang[indices][oldvalid]-oldang[oldvalid,:15].astype(float)*180)
                maximum=float(delta.max()) if delta.size else 0.;assert maximum<.06
                stats['old_angle_values_preserved']+=int(delta.size);stats['angle_frames_added']+=int((ok[indices]&~oldvalid[:,None]).sum())
                pos,pvalid=A._group(sample,'handpos_'+side);data['angles_'+side]=ang[indices];data['position_'+side]=pos;data['position_valid_'+side]=pvalid
                for key,val in states[side].items():data['full_'+side+'_'+key]=val
                data['full_frame_id']=obs['frame_id'];data['full_timestamp_us']=obs['timestamp_us']
                cc=Counter(map(int,states[side]['offline_full_view_state']));gap=states[side]['internal_gap'];inside=states[side]['inside_source_but_view_missing'];core=states[side]['annotated_action']
                info['sides'][side]={'old_angle_max_delta_degrees':maximum,'new_angle_cells':int(ok[indices].sum()),'old_angle_cells':int(oldvalid.sum()*15),'full_frames':len(ang),'offline_full_view_counts':dict(cc),
                  'view_local_no_points':int((~states[side]['any_hand_points']).sum()),'internal_gap_frames':int(gap.sum()),'annotated_internal_gap_frames':int((gap&core).sum()),'source_inside_view_missing':int(inside.sum()),'annotated_source_inside_view_missing':int((inside&core).sum()),
                  'lowered_observed_missing_in_view':int(((states[side]['offline_full_view_state']==3)&~states[side]['any_hand_points']).sum())}
            target=U/'features'/view/(cid+'.npz');target.parent.mkdir(parents=True,exist_ok=True);assert not target.exists();np.savez_compressed(target,**data)
            info['sha256']=sha(target);write(U/'inputs'/view/(cid+'.json'),info);stats['prepared']+=1
        if (i+1)%100==0:write(U/'STATUS.json',{'status':'PREPARING','clips':i+1,'total':1199,'at':now()});print('prepared',i+1,flush=True)
    write(U/'PREPARE.json',{'status':'PASS','counts':dict(stats),'seconds':time.time()-beg});print('PREPARE PASS',flush=True)
def compare():
    f=frozen();assert read(U/'PREPARE.json')['status']=='PASS';assert not (U/'COMPARISON.json').exists()
    stats=Counter();beg=time.time()
    for view in ('legacy','masked','sensor'):
        for wp in sorted((OLD/'words'/view).glob('*.json')):
            old=read(wp);features={q['clip_id']:load(U/'features'/view/(q['clip_id']+'.npz')) for q in old['queries']}
            pairs=[]
            for pair in old['pairs']:
                path=pair['diagnostics'].get('path',[]);q=features[pair['query_clip']];r=features[pair['reference_clip']]
                result={'query_clip':pair['query_clip'],'reference_clip':pair['reference_clip'],'role':pair['role'],'sides':{}}
                for side in old['alignment_sides']:
                    aligned=pair['result']['status']=='OK'
                    aa=C.compared_angles(q['angles_'+side],r['angles_'+side],path,aligned)
                    tr=C.trajectory(q['position_'+side],r['position_'+side],q['position_valid_'+side],r['position_valid_'+side],path)
                    tr['measurement_eligible']=bool(aligned and tr['support']>=.5 and min(tr['query_frames'],tr['reference_frames'])>=8)
                    # Regression: unchanged position and original trajectory values on old valid output.
                    mm=next((m for m in pair['result'].get('metrics',[]) if m['key']=='movement'),None)
                    if mm:
                        prev=next(x for x in mm['sides'] if x['side']==side)['value']
                        if prev is not None:assert abs(prev-tr['old'])<.001;stats['trajectory_parity']+=1
                    result['sides'][side]={'angles':aa,'trajectory':tr}
                pairs.append(result);stats['pairs']+=1
            queries=[]
            for oq in old['queries']:
                subset=[p for p in pairs if p['query_clip']==oq['clip_id']]
                q={'clip_id':oq['clip_id'],'signer_id':oq['signer_id'],'role':oq['role'],'sides':{}}
                for side in old['alignment_sides']:
                    a=C.aggregate_angles([p['sides'][side]['angles'] for p in subset]);tr=[p['sides'][side]['trajectory'] for p in subset if p['sides'][side]['trajectory']['measurement_eligible']]
                    a['trajectory']={k:float(np.median([p[k] for p in tr])) if len(tr)>=5 else None for k in ['old','paired_start','anchor_difference']};a['trajectory']['refs']=len(tr)
                    q['sides'][side]=a
                queries.append(q)
            out={'word_code':old['word_code'],'label':old['label'],'view':view,'role_known':old['role_known'],'required_sides':old['alignment_sides'],'queries':queries,'pairs':pairs,
                 'basis':'Saved path reused. Different-angle support is explicit; whole_hand is all15-angle median-of-references mean, not the legacy frame-mean statistic; never a correctness score.'}
            dest=U/'words'/view/wp.name;assert not dest.exists();write(dest,out)
            if stats['pairs']%100==0:print(view,wp.stem,flush=True)
        print('COMPARED',view,flush=True);write(U/'STATUS.json',{'status':'COMPARING','last_view':view,'pairs':stats['pairs'],'at':now()})
    assert stats['pairs']==29643;write(U/'COMPARISON.json',{'status':'PASS','counts':dict(stats),'seconds':time.time()-beg});print('COMPARE PASS',flush=True)
if __name__=='__main__':
    stage=sys.argv[1]
    try:globals()[stage]()
    except Exception:
        p=U/('FIRST_ERROR_'+stage+'.txt')
        if not p.exists():p.write_text(traceback.format_exc(),'utf-8')
        raise
