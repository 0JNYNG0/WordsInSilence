from pathlib import Path
import sys,time,traceback
B=Path(__file__).resolve().parent
sys.path.insert(0,str(B.parent/'progression_presence_learning'))
from presence_core import *
from transfer_local_timefix_core import encode_windows,normalized_pose

def main():
    assert not(B/'RUNTIME_FEATURES_FROZEN.json').exists();torch.set_num_threads(4);torch.set_num_interop_threads(1)
    previous=N/'evaluation/COMPARISON.json';meta=read(previous);source={}
    for r in meta['predictions']:source.setdefault((r['clip_id'],r['kind']),r)
    assert len(source)==82
    write(B/'RUNTIME_FEATURES_FROZEN.json',dict(at=stamp(),scope='New body-only head input cache for same82 public fixtures; no old-head prediction rerun or editing',
        transform='same normalized pose/relations as previous runtime, erase hand coordinates/masks/hand-root relation before frozen common100 local encoding; preserve body, body relation, timestamp interval',
        candidates='Exact old union span list from saved cached full features; no new proposal or target order input',
        pins={str(p):sha(p)for p in[Path(__file__),previous,PRIOR/'local_rank_core.py',PRIOR/'transfer_local_timefix_core.py']},private_access=False,final8=False,web_changed=False))
    base=old.encoder();out=[]
    for (cid,kind),r in source.items():
        assert sha(r['source'])==r['source_sha256']and sha(r['feature_path'])==r['feature_sha256'];z=np.load(r['source']);prev=np.load(r['feature_path']);sp=prev['spans'];pose=normalized_pose(z['x']);q=z['z'][:,292:392].reshape(-1,10,10);px,qq=old.drop_hand_information(pose,q)
        assert np.array_equal(px[:,:18],pose[:,:18])and np.array_equal(px[:,186:195],pose[:,186:195])and np.array_equal(px[:,279],pose[:,279]);assert np.array_equal(qq.reshape(-1,10,10)[:,:,:4],q[:,:,:4])and np.array_equal(qq.reshape(-1,10,10)[:,:,8],q[:,:,8])
        tick=time.perf_counter();ff=np.concatenate([encode_windows(base,px,qq,sp[a:a+128])for a in range(0,len(sp),128)])if len(sp)else np.empty((0,1160),np.float32)
        ids=np.array([part*290+64+k*10+j for part in range(4)for k in range(10)for j in [0,1,2,3,8]])
        assert np.array_equal(ff[:,ids],prev['x'][:,ids]);dest=B/'runtime_features'/f'{cid}_{kind}.npz';dest.parent.mkdir(exist_ok=True);np.savez_compressed(dest,x=ff,spans=sp)
        out.append(dict(clip_id=cid,kind=kind,path=str(dest),sha256=sha(dest),source=r['source'],source_sha256=r['source_sha256'],original_feature=r['feature_path'],original_feature_sha256=r['feature_sha256'],candidates=len(ff),seconds=time.perf_counter()-tick,explicit_body_relation_equal=True))
        write(B/'RUNTIME_FEATURES_PROGRESS.json',dict(at=stamp(),completed=len(out),total=82));print('runtime body-only features',len(out),82,flush=True)
    write(B/'RUNTIME_FEATURES_COMPLETE.json',dict(at=stamp(),rows=out,model_predictions_computed=False));print('Runtime body-only inputs complete',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:(B/f'RUNTIME_PREP_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
