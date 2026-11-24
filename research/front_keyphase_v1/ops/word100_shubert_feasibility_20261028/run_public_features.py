"""Fixed public development feasibility. No training, no operating model edits."""
from pathlib import Path
import hashlib, json, os, time, traceback
from datetime import datetime, timezone
import numpy as np
import torch
from dino_cpu import load_teacher, extract
from cpu_shubert_adapter import CPUShubertAdapter, CPUShubertConfig, stack_official_features

U=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,obj):
    p.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')
def stamp():return datetime.now(timezone.utc).isoformat()
def as_tensors(streams):return [{k:torch.from_numpy(np.ascontiguousarray(v)) for k,v in streams.items()}]
def encode(model,streams):
    result=model.extract_features(as_tensors(streams),mask=False)
    y=stack_official_features(result)[-1].cpu().numpy()
    assert y.shape==(len(streams['face']),768) and np.isfinite(y).all()
    return y

def main():
    # A running or failed attempt is never silently overwritten.
    lock=U/'FEATURE_RUNNING.json'
    with lock.open('x',encoding='utf-8') as f:json.dump({'pid':os.getpid(),'at':stamp()},f)
    out=U/'features';out.mkdir(exist_ok=False)
    try:
        torch.set_num_threads(4)
        manifest=json.loads((U/'inputs/MANIFEST.json').read_text(encoding='utf-8'))
        verified=json.loads((U/'inputs/VERIFY.json').read_text(encoding='utf-8'))
        assert verified['manifest_sha256']==sha(U/'inputs/MANIFEST.json')
        independent=json.loads((U/'ADAPTER_INDEPENDENT.json').read_text(encoding='utf-8'))
        # Freeze technical controls before observing public model features.
        protocol={'at':stamp(),'threads':4,'batch_size':8,'dtype':'float32','device':'cpu',
          'input_manifest_sha256':sha(U/'inputs/MANIFEST.json'),
          'feature':'official last layer FFN before residual/norm',
          'mask_policy':'provenance saved unchanged; carried crop is NOT current evidence',
          'controls':{'identity_repeat':'V01 only; maximum absolute difference',
             'speed':'timestamp-only factors 0.75,1.25; same frame features and indices',
             'reverse':'reverse ALL four streams, encode again, restore indices, measure contextual response',
             'omit':'remove middle [floor(.45T),ceil(.55T)), keep saved known index map; no semantic error label'},
          'latency_scope':'cached MediaPipe/face observations, prepared RGB crops; not camera-to-feedback total',
          'training':False,'semantic_correctness_threshold':None}
        write(U/'FEATURE_PROTOCOL.json',protocol)
        load_start=time.perf_counter()
        hands,h_audit=load_teacher('hands');face,f_audit=load_teacher('face')
        checkpoint=torch.load(U/'weights/checkpoint_836_400000.pt',map_location='cpu',weights_only=True,mmap=True)
        model=CPUShubertAdapter(CPUShubertConfig.from_mapping(checkpoint['cfg']['model']))
        load_audit=model.load_exact_state_dict(checkpoint['model'])
        model.eval().requires_grad_(False)
        del checkpoint
        load_seconds=time.perf_counter()-load_start
        write(U/'PUBLIC_LOAD_AUDIT.json',{'hands':h_audit,'face':f_audit,'shubert':load_audit,'load_seconds':load_seconds})
        rows=[]
        for row in manifest['rows']:
            alias=row['alias'];folder=U/'inputs'/alias;dest=out/alias;dest.mkdir()
            done=json.loads((folder/'DONE.json').read_text(encoding='utf-8'))
            assert sha(folder/'DONE.json')==row['done_sha256']
            for name,pin in done['output_pins'].items():assert sha(Path(name))==pin,(alias,name)
            streams={};timing={};start=time.perf_counter()
            for name,teacher in [('face',face),('left_hand',hands),('right_hand',hands)]:
                t=time.perf_counter();images=np.load(folder/f'{name}_rgb.npy',mmap_mode='r')
                streams[name]=extract(teacher,images,batch_size=8)
                timing[name+'_dino_seconds']=time.perf_counter()-t
            streams['body_posture']=np.load(folder/'body14.npy')
            np.savez(dest/'streams.npz',**streams)
            t=time.perf_counter();y=encode(model,streams);timing['shubert_seconds']=time.perf_counter()-t
            timing['primary_feature_seconds']=time.perf_counter()-start
            np.save(dest/'last_ffn.npy',y)
            provenance=np.load(folder/'provenance.npz');n=len(y)
            # Keep provenance with each output. Missing evidence remains missing.
            np.savez(dest/'provenance.npz',**{k:provenance[k] for k in provenance.files})
            timestamps=provenance['timestamp_us']
            speed={str(factor):((timestamps-timestamps[0])/factor).tolist() for factor in (.75,1.25)}
            write(dest/'speed_timestamp_maps.json',speed)
            t=time.perf_counter()
            repeat_max=None
            if alias=='V01':repeat_max=float(np.max(np.abs(y-encode(model,streams))))
            reversed_streams={k:np.ascontiguousarray(v[::-1]) for k,v in streams.items()}
            reversed_y=encode(model,reversed_streams)
            lo=int(.45*n);hi=int(np.ceil(.55*n));kept=np.r_[0:lo,hi:n]
            omitted_y=encode(model,{k:np.ascontiguousarray(v[kept]) for k,v in streams.items()})
            np.savez(dest/'controls.npz',reverse_last_ffn=reversed_y,omitted_last_ffn=omitted_y,omitted_keep_indices=kept)
            timing['technical_controls_seconds']=time.perf_counter()-t
            reverse_delta=float(np.linalg.norm(y-reversed_y[::-1])/max(np.linalg.norm(y),1e-12))
            omission_delta=float(np.linalg.norm(y[kept]-omitted_y)/max(np.linalg.norm(y[kept]),1e-12))
            result={'alias':alias,'clip_id':row['clip_id'],'frames':n,'timing':timing,
              'duration_seconds':float((timestamps[-1]-timestamps[0])/1e6),
              'current_hand_crops':int((provenance['source_kind'][:,:2]==1).sum()),
              'carried_hand_crops':int((provenance['source_kind'][:,:2]==2).sum()),
              'repeat_max_abs':repeat_max,'reverse_context_relative_l2':reverse_delta,
              'omission_context_relative_l2':omission_delta,'omitted_indices':[lo,hi],
              'speed_control_note':'timestamps change only; same frame sequence/features by construction, not actual fast-motion validation',
              'output_pins':{p.name:sha(p) for p in dest.iterdir() if p.is_file()},
              'semantic_accuracy_measured':False}
            write(dest/'DONE.json',result);rows.append(result)
            write(U/'FEATURE_STATUS.json',{'at':stamp(),'completed':len(rows),'total':len(manifest['rows']),'last':alias})
            print(json.dumps({'alias':alias,'frames':n,'timing':timing}),flush=True)
        summary={'status':'COMPLETE_TECHNICAL_FEASIBILITY_NOT_KSL_FEEDBACK_ACCURACY','at':stamp(),
          'model_load_seconds':load_seconds,'clips':len(rows),'frames':sum(r['frames'] for r in rows),
          'latency_scope':protocol['latency_scope'],'rows':rows,'training_runs':0,
          'script_sha256':sha(__file__),'adapter_sha256':sha(U/'cpu_shubert_adapter.py'),
          'dino_script_sha256':sha(U/'dino_cpu.py'),'protocol_sha256':sha(U/'FEATURE_PROTOCOL.json')}
        write(U/'PUBLIC_FEATURE_RESULT.json',summary)
        lock.rename(U/'FEATURE_FINISHED.json')
    except BaseException:
        write(U/'FEATURE_FAILURE.json',{'at':stamp(),'traceback':traceback.format_exc()})
        raise
if __name__=='__main__':main()
