"""Preregistered photometric diagnostics on a frozen model; never a selection loop."""
from pathlib import Path
import json
import numpy as np
import torch
from PIL import Image
from run_pair import U,B,OUT,read,load,sha,write,init,load_models,pack
from rgb_bridge import FrozenRGBEncoder


def main():
    c,parent,classes=init();assert (OUT/'EVALUATION_COMPLETE.json').exists()
    assert not (OUT/'APPEARANCE_SCORES.npz').exists()
    model=load_models(parent)['B'];weights=read(B/'ops/rgb_pose_bridge_20261010/WEIGHTS.json')
    encoder=FrozenRGBEncoder(weights['path'],weights['sha256']);variants=['brightness_0p8','brightness_1p2','grayscale']
    samples=[r for r in read(B/'runs/ksl77_rgb_paired_v1/MANIFEST.json') if r['split']=='development34']
    scores=[[] for _ in variants];ids=[];ys=[];photo_count=0
    for row in samples:
        assert sha(row['cache_path'])==row['cache_sha256'];a=load(row['cache_path']);slots=np.argwhere(a['observed_rgb'])
        pixels=a['rgb'][a['observed_rgb']];ids.append(row['sample_id']);ys.append(classes.index(row['label_metadata_only']))
        for v,name in enumerate(variants):
            if name=='grayscale': altered=np.stack([np.array(Image.fromarray(p).convert('L').convert('RGB')) for p in pixels])
            else:altered=np.clip(np.rint(pixels.astype(np.float32)*(.8 if v==0 else 1.2)),0,255).astype(np.uint8)
            x=torch.tensor(altered.transpose(0,3,1,2),dtype=torch.float32)/255
            for ch,(mean,std) in enumerate(zip((.485,.456,.406),(.229,.224,.225))):x[:,ch]=(x[:,ch]-mean)/std
            features=np.zeros((len(a['pose_x']),2,576),np.float32);values=encoder(x).numpy()
            features[slots[:,0],slots[:,1]]=values
            item={'x':a['pose_x'],'features':features,'available':a['observed_rgb'],'y':ys[-1]}
            args,_=pack([item],True)
            with torch.inference_mode():scores[v].append(model(*args).numpy()[0])
            photo_count+=len(pixels)
        print('appearance diagnostic',len(ids),'/34',flush=True)
    assert len(ids)==34
    p=OUT/'APPEARANCE_SCORES.npz'
    with p.open('xb') as f:np.savez_compressed(f,scores=np.array(scores),ids=np.array(ids),y=np.array(ys),variants=np.array(variants))
    write(OUT/'APPEARANCE_COMPLETE.json',{'status':'COMPUTED_FOR_INDEPENDENT_RECOUNT','images_processed':photo_count,'score_sha256':sha(p),'model_sha256':sha(OUT/'checkpoint_B.pt'),'variants':variants,'tuning_performed':False,'limitations':'Synthetic brightness/grayscale diagnostics only; no actual new-person, background or camera generalization proof.'})


if __name__=='__main__':main()
