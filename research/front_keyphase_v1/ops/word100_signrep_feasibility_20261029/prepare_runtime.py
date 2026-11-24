from common import *
import traceback
def main():
    import torch, numpy as np, cv2, av, importlib.metadata as md
    import openvino as ov
    from augmentation.video.base_video_aug import Transformation
    core=ov.Core()
    devices={d: str(core.get_property(d,'FULL_DEVICE_NAME')) for d in core.available_devices}
    save('RUNTIME.json',dict(python=sys.executable,versions={p:md.version(p) for p in ['torch','torchvision','openvino','timm','albumentations','av']},devices=devices,torch_xpu=torch.xpu.is_available(),cuda=torch.cuda.is_available()))
    event('devices',devices=devices)
    manifest=read(ROOT.parent/'word100_shubert_feasibility_20261028/inputs/MANIFEST.json')
    rows=[]
    for alias in read(ROOT/'FROZEN.json')['public_aliases']:
        row=next(r for r in manifest['rows'] if r['alias']==alias)
        done=read(Path(row['input_dir'])/'DONE.json'); src=Path(row['source_video'])
        assert sha(src)==done['source_video_sha256']
        t=time.perf_counter(); imgs=[]; times=[]; sample=[]
        with av.open(str(src)) as c:
            stream=c.streams.video[0]
            for i,f in enumerate(c.decode(stream)):
                a=f.to_ndarray(format='rgb24')
                if i<3: sample.append(a)
                imgs.append(cv2.resize(a,(224,224),interpolation=cv2.INTER_LINEAR))
                times.append(float(f.pts*f.time_base))
        rgb=np.stack(imgs); decode=time.perf_counter()-t
        assert len(rgb)==done['frames'] and all(b>a for a,b in zip(times,times[1:]))
        official=Transformation().aug_video(sample).numpy()
        own=norm(rgb[:len(sample)])
        assert np.allclose(official,own,rtol=1e-6,atol=1e-6), float(np.max(abs(official-own)))
        seq=starts(len(rgb)); covered=set(i for s in seq for i in range(s,s+16))
        assert covered==set(range(len(rgb)))
        out=ROOT/'inputs'/alias; out.mkdir(parents=True,exist_ok=False)
        np.save(out/'rgb224.npy',rgb,allow_pickle=False)
        rows.append(dict(alias=alias,clip_id=row['clip_id'],source=str(src),source_sha256=sha(src),frames=len(rgb),pts=times,window_starts=seq,tail_window_added=(seq[-1]%2==1),decode_resize_seconds=decode,all_frames_covered=True,input_sha256=sha(out/'rgb224.npy'),official_transform_max_abs=float(np.max(abs(official-own)))))
        event('input',alias=alias,frames=len(rgb),windows=len(seq),seconds=decode)
    save('INPUTS.json',dict(rows=rows,freeze_sha256=sha(ROOT/'FROZEN.json'),data_role='public development, not independent evaluation'))
    model=load_model(); f=feature_wrapper(model); records=[]
    event('model_loaded',parameters=sum(p.numel() for p in model.parameters()))
    with torch.inference_mode():
        for row in rows:
            frames=norm(np.load(ROOT/'inputs'/row['alias']/'rgb224.npy',allow_pickle=False))
            seq=row['window_starts']; ids=[seq[0],seq[len(seq)//2],seq[-1]]
            for s in ids:
                x=torch.from_numpy(batch(frames,[s],1)[0])
                t=time.perf_counter(); a=model(x)['features']; sec=time.perf_counter()-t
                b=f(x)
                record=dict(alias=row['alias'],start=s,full_forward_seconds=sec,**diff(a.numpy(),b.numpy()))
                assert torch.allclose(a,b,rtol=1e-5,atol=1e-5),record
                records.append(record)
    save('WRAPPER_PARITY.json',dict(status='PASS',parameters=sum(p.numel() for p in model.parameters()),checks=records,meaning='Feature equality only, not KSL accuracy; original code unchanged'))
    event('ready_for_export',checks=len(records))
if __name__=='__main__':
    try: main()
    except Exception:
        p=ROOT/'PREP_FIRST_FAILURE.txt'
        if not p.exists(): p.write_text(traceback.format_exc(),encoding='utf-8')
        raise
