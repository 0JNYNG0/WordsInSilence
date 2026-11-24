"""CPU-only public handshape model-package comparison. No prior final-test access."""
from pathlib import Path
import argparse, datetime, gc, hashlib, json, os, sys, time, traceback
import numpy as np
import torch
from torch import nn
from PIL import Image, ImageOps

U=Path(__file__).resolve().parent
OLD=U.parent/'word100_feature_learning_pipeline_20261027'
ROOT=U.parents[3]
sys.path.insert(0,str(U/'deps'))
sys.path.insert(0,str(OLD/'deps'))
sys.path.insert(0,str(OLD))
from train_observer import metrics
import timm
from safetensors.torch import load_file

NAMES=['v3','repvit','v4hybrid']
IDS={'repvit':'repvit_m1_0.dist_450e_in1k','v4hybrid':'mobilenetv4_hybrid_medium.ix_e550_r256_in1k'}
SEEDS=[28101,28102,28103]
SIZES={'v3':224,'repvit':224,'v4hybrid':256}
P=U/'FROZEN_PROTOCOL.json'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def write(p,d):
    p=Path(p);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(tmp,p)
def save(p,d):
    p=Path(p);tmp=p.with_name(p.name+'.tmp');torch.save(d,tmp);os.replace(tmp,p)
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
def thash(t):
    a=t.detach().cpu().contiguous();return hashlib.sha256(str(a.dtype).encode()+str(tuple(a.shape)).encode()+a.numpy().tobytes()).hexdigest()
def hashes(m):return {k:thash(v) for k,v in m.state_dict().items()}
def pins():
    d=read(U/'PROTECTED.json');assert all(sha(p)==h for p,h in d.items());return len(d)
def image_tensor(path,size):
    im=Image.open(path).convert('RGB');im=ImageOps.contain(im,(size,size),Image.Resampling.BICUBIC)
    canvas=Image.new('RGB',(size,size),(128,128,128));canvas.paste(im,((size-im.width)//2,(size-im.height)//2))
    x=torch.from_numpy(np.array(canvas,copy=True)).permute(2,0,1).float()/255
    return (x-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
def load_original(name):
    if name=='v3':
        from torchvision.models import mobilenet_v3_small
        m=mobilenet_v3_small(weights=None)
        p=ROOT/'research/front_keyphase_v1/assets/rgb_encoder/mobilenet_v3_small-047dcff4.pth'
        assert sha(p)=='047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f'
        m.load_state_dict(torch.load(p,map_location='cpu',weights_only=True),strict=True)
    else:
        model_id=IDS[name];m=timm.create_model(model_id,pretrained=False)
        info=read(U/'models'/(model_id+'.source.json'));p=Path(info['file']);assert sha(p)==info['sha256']
        m.load_state_dict(load_file(str(p),device='cpu'),strict=True)
    m.eval().requires_grad_(False)
    return m,p
def replace_and_configure(name,m,seed):
    torch.manual_seed(seed)
    if name=='v3':
        m.classifier[-1]=nn.Linear(m.classifier[-1].in_features,45)
        modules=list(m.features)[9:]+[m.classifier]
        new=['classifier.3.weight','classifier.3.bias']
    elif name=='repvit':
        m.head.head.l=nn.Linear(m.head.head.l.in_features,45)
        m.head.head_dist.l=nn.Linear(m.head.head_dist.l.in_features,45)
        modules=[m.stages[-1],m.head]
        new=['head.head.l.weight','head.head.l.bias','head.head_dist.l.weight','head.head_dist.l.bias']
    else:
        m.classifier=nn.Linear(m.classifier.in_features,45)
        modules=list(m.blocks)[3:]+[m.conv_head,m.norm_head,m.classifier]
        new=['classifier.weight','classifier.bias']
    m.requires_grad_(False)
    for x in modules:x.requires_grad_(True)
    m.eval()
    assert all(not x.training for x in m.modules())
    assert all(dict(m.named_parameters())[k].requires_grad for k in new)
    return [k for k,v in m.named_parameters() if v.requires_grad],new
def prefix(name,m,x):
    if name=='v3':return m.features[:9](x)
    if name=='repvit':return m.stages[:-1](m.stem(x))
    return m.blocks[:3](m.bn1(m.conv_stem(x)))
def suffix(name,m,x):
    if name=='v3':return m.classifier(m.avgpool(m.features[9:](x)).flatten(1))
    if name=='repvit':return m.forward_head(m.stages[-1](x))
    return m.forward_head(m.blocks[3:](x))
def check_frozen(m,spec):
    h=hashes(m);bad=[k for k,v in spec['frozen_states'].items() if h[k]!=v]
    assert not bad,('frozen weights/buffers changed',bad[:3])
    return {'unchanged':True,'states':len(spec['frozen_states'])}

def prepare():
    assert not P.exists(),'Already frozen; no silent redefinition'
    src=read(U/'INPUT_MANIFEST.json')
    rows=[]
    for r in src['rows']:
        q=dict(r);q['prior_role']=r['role']
        q['role']='validation' if r['signer'] in ['signer03','signer06','signer07'] else 'fit'
        rows.append(q)
    rows.sort(key=lambda r:(r['role']!='fit',r['signer'],r['id']))
    fit=[r for r in rows if r['role']=='fit'];val=[r for r in rows if r['role']=='validation']
    assert (len(fit),len(val))==(2489,488) and len({r['target'] for r in fit})==45
    assert {r['signer'] for r in fit}=={'signer01','signer02','signer04','signer05'}
    assert {r['signer'] for r in val}=={'signer03','signer06','signer07'}
    assert not ({r['pixel_sha256'] for r in fit}&{r['pixel_sha256'] for r in val})
    assert not ({r['broadcast'] for r in fit}&{r['broadcast'] for r in val})
    for r in rows:assert sha(r['image'])==r['image_sha256']
    write(U/'TRAINING_MANIFEST.json',{'rows':rows,'classes':src['classes'],'fit':len(fit),'validation':len(val),'test':0,'source_manifest_sha256':sha(U/'INPUT_MANIFEST.json')})
    specs={}
    for name in NAMES:
        m,p=load_original(name)
        x=torch.stack([image_tensor(r['image'],SIZES[name]) for r in fit[:2]])
        with torch.no_grad():
            a=m(x);z=prefix(name,m,x);b=suffix(name,m,z)
        assert torch.allclose(a,b,atol=1e-6,rtol=1e-6),'Official pretrained full path mismatch'
        before=float((a-b).abs().max());allowed,new=replace_and_configure(name,m,SEEDS[0])
        with torch.no_grad():a=m(x);b=suffix(name,m,prefix(name,m,x))
        assert a.shape==(2,45) and torch.allclose(a,b,atol=1e-6,rtol=1e-6)
        h=hashes(m)
        specs[name]={'source':str(p),'source_sha256':sha(p),'input_size':SIZES[name],
            'trainable_names':allowed,'new_classifier_names':new,'trainable_parameters':sum(p.numel() for p in m.parameters() if p.requires_grad),
            'total_parameters':sum(p.numel() for p in m.parameters()),'prefix_shape':list(z.shape[1:]),
            'frozen_states':{k:v for k,v in h.items() if k not in allowed},'official_full_split_max_error':before,
            'new_full_split_max_error':float((a-b).abs().max()),'official_head_path_retained':True}
        print('preflight',name,specs[name]['trainable_parameters'],flush=True)
        del m,x,z,a,b;gc.collect()
    protocol={'date_kst':'2026-10-28','scope':'Development-only laptop CPU handshape model-package comparison, not final KSL correctness',
        'models':NAMES,'specs':specs,'seeds':SEEDS,'epochs':20,'batch':32,'backbone_lr':.0002,'new_classifier_lr':.003,'weight_decay':.01,
        'optimizer':'AdamW','loss':'45-class unweighted cross entropy on final model logits','scheduler':'none','augmentation':False,
        'normalization':'ImageNet RGB mean/std; entire crop letterbox bicubic gray128; model input V3/Rep224 Hybrid256',
        'comparison_scope':'Official model packages at documented training resolution; not pure architecture isolation/equal parameter budget',
        'model_mode':'eval (BN and dropout buffers fixed), gradients only listed final stride32 stage + official head path + new classifiers',
        'prefix_cache':'only fit/dev original ordered rows; frozen eval prefix reused exactly',
        'selection':'Mean across 3 seeds of mean(signer03 supported-class macroF1, signer07 supported-class macroF1). signer06 diagnostic only.',
        'latency_tiebreak':'Within 0.01 absolute mean balanced macroF1, choose lower warm single-crop full-model CPU p95; no test-based policy.',
        'learning_epochs_fixed':20,'early_stopping':False,'torch_threads':4,'device':'CPU','new_fit_count':2489,'new_dev_count':488,
        'not_final_evaluation':'Original382excluded. ID03 held out from these new fits but previously used in historical work; development only.',
        'test_predictions':0,'user_mentor_access':False,'manifest_sha256':sha(U/'TRAINING_MANIFEST.json'),
        'code_sha256':sha(Path(__file__)),'metrics_code_sha256':sha(OLD/'train_observer.py'),'protected_count':pins()}
    write(P,protocol);write(U/'STATUS.json',{'state':'PREPARED','training':0,'test_predictions':0,'at':now()})

def cached(name,rows,p):
    assert sha(p['specs'][name]['source'])==p['specs'][name]['source_sha256'],'Frozen source weight changed'
    file=U/'data'/f'{name}_prefix.pt';meta=file.with_suffix('.json')
    if file.exists() or meta.exists():
        assert file.exists() and meta.exists(),'Partial cache preserved'
        d=read(meta);assert d['protocol_sha256']==sha(P) and d['sha256']==sha(file)
        v=torch.load(file,weights_only=True,map_location='cpu');assert v['ids']==[r['id'] for r in rows]
        return v['features']
    m,_=load_original(name);parts=[]
    with torch.no_grad():
        for start in range(0,len(rows),32):
            x=torch.stack([image_tensor(r['image'],SIZES[name]) for r in rows[start:start+32]])
            z=prefix(name,m,x);assert torch.isfinite(z).all();parts.append(z.clone())
            if start%320==0:
                write(U/'STATUS.json',{'state':'PREFIX_CACHE','model':name,'done':start+len(x),'total':len(rows),'at':now()})
                print(name,'prefix',start+len(x),'/',len(rows),flush=True)
    out=torch.cat(parts).contiguous();save(file,{'features':out,'ids':[r['id'] for r in rows]})
    write(meta,{'sha256':sha(file),'protocol_sha256':sha(P),'shape':list(out.shape),'final_test_inputs':0})
    del parts,m;gc.collect();return out

def fit_one(name,seed,rows,p,features):
    stem=f'{name}_{seed}';folder=U/'models';result=folder/f'{stem}.json';cp=folder/f'{stem}.pt';vp=folder/f'{stem}_dev.npz';resume=folder/f'{stem}_resume.pt'
    if result.exists():
        d=read(result);assert d['protocol_sha256']==sha(P) and sha(cp)==d['checkpoint_sha256'] and sha(vp)==d['dev_sha256'];print('reuse',stem,flush=True);return
    spec=p['specs'][name];assert sha(spec['source'])==spec['source_sha256'],'Frozen source weight changed'
    m,src=load_original(name);assert sha(src)==spec['source_sha256']
    allowed,new=replace_and_configure(name,m,seed)
    assert allowed==spec['trainable_names'];check_frozen(m,spec)
    params=dict(m.named_parameters());opt=torch.optim.AdamW([{'params':[v for k,v in params.items() if v.requires_grad and k not in new],'lr':p['backbone_lr']},
        {'params':[params[k] for k in new],'lr':p['new_classifier_lr']}],weight_decay=p['weight_decay'])
    gen=torch.Generator().manual_seed(seed);nfit=p['new_fit_count'];y=torch.tensor([r['target'] for r in rows]);trace=[];start_epoch=0;oldsec=0.
    if resume.exists():
        r=torch.load(resume,map_location='cpu',weights_only=True);assert r['protocol_sha256']==sha(P) and r['name']==name and r['seed']==seed
        m.load_state_dict(r['model'],strict=True);opt.load_state_dict(r['optimizer']);gen.set_state(r['rng']);trace=r['trace'];start_epoch=r['epoch'];oldsec=r['seconds'];check_frozen(m,spec)
    elif cp.exists() or vp.exists():raise RuntimeError('Partial final results without resume; preserve')
    start=time.perf_counter()
    for epoch in range(start_epoch,p['epochs']):
        order=torch.randperm(nfit,generator=gen);seen=0;loss_sum=0.;t=time.perf_counter();m.eval()
        for b in range(0,nfit,p['batch']):
            ix=order[b:b+p['batch']];opt.zero_grad(set_to_none=True)
            logits=suffix(name,m,features[ix]);loss=nn.functional.cross_entropy(logits,y[ix]);assert torch.isfinite(loss)
            loss.backward();assert all(v.grad is None for k,v in params.items() if k not in allowed)
            assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in params.values() if v.requires_grad)
            opt.step();seen+=len(ix);loss_sum+=float(loss.detach())*len(ix)
        assert seen==nfit and len(order.unique())==nfit;check_frozen(m,spec)
        trace.append({'epoch':epoch+1,'fit_loss':loss_sum/seen,'count':seen,'unique':int(len(order.unique())),'order_sha256':sha_bytes(order.numpy().tobytes()),'seconds':time.perf_counter()-t})
        save(resume,{'model':m.state_dict(),'optimizer':opt.state_dict(),'rng':gen.get_state(),'trace':trace,'epoch':epoch+1,'seconds':oldsec+time.perf_counter()-start,'protocol_sha256':sha(P),'name':name,'seed':seed})
        write(U/'STATUS.json',{'state':'FITTING','model':name,'seed':seed,'epoch':epoch+1,'epochs':p['epochs'],'at':now(),'test_predictions':0})
        print(stem,'epoch',epoch+1,'loss',round(loss_sum/seen,4),'sec',round(trace[-1]['seconds'],2),flush=True)
    assert not cp.exists() and not vp.exists(),'Final artifact exists; preserve, no repeated scoring'
    with torch.inference_mode():dev=torch.cat([suffix(name,m,features[b:b+32]) for b in range(nfit,len(rows),32)]).numpy()
    assert dev.shape==(488,45) and np.isfinite(dev).all()
    targets=np.array([r['target'] for r in rows[nfit:]]);people=np.array([r['signer'] for r in rows[nfit:]])
    stats=metrics(targets,dev,read(U/'TRAINING_MANIFEST.json')['classes'])
    per={s:metrics(targets[people==s],dev[people==s],read(U/'TRAINING_MANIFEST.json')['classes']) for s in sorted(set(people))}
    balanced=(per['signer03']['macro_f1_supported']+per['signer07']['macro_f1_supported'])/2
    save(cp,{'model':m.state_dict(),'name':name,'seed':seed,'epoch':20,'protocol_sha256':sha(P)})
    np.savez_compressed(vp,logits=dev,targets=targets,signer=people,ids=np.array([r['id'] for r in rows[nfit:]]))
    write(result,{'name':name,'seed':seed,'protocol_sha256':sha(P),'checkpoint_sha256':sha(cp),'dev_sha256':sha(vp),
        'dev':stats,'per_signer':per,'balanced_selection_macro_f1':balanced,'trace':trace,'epoch':20,
        'frozen_check':check_frozen(m,spec),'seconds':oldsec+time.perf_counter()-start,'test_predictions':0,'source_unchanged':sha(src)==spec['source_sha256']})
    print('complete',stem,round(balanced,4),flush=True);del m,opt;gc.collect()
def sha_bytes(b):return hashlib.sha256(b).hexdigest()

def run():
    p=read(P);assert sha(Path(__file__))==p['code_sha256'] and sha(OLD/'train_observer.py')==p['metrics_code_sha256']
    assert all(sha(s['source'])==s['source_sha256'] for s in p['specs'].values()),'Frozen source weight changed'
    assert sha(U/'TRAINING_MANIFEST.json')==p['manifest_sha256'];rows=read(U/'TRAINING_MANIFEST.json')['rows']
    lock=U/'RUNNING.json';assert not lock.exists(),'Active or unreviewed interrupted execution'
    write(lock,{'pid':os.getpid(),'started':now(),'command':'compare_cpu.py run'})
    try:
        for name in NAMES:
            features=cached(name,rows,p)
            for seed in SEEDS:fit_one(name,seed,rows,p,features)
            del features;gc.collect()
        pins();write(U/'TRAINING_COMPLETE.json',{'runs':9,'test_predictions':0,'protocol_sha256':sha(P),'at':now()})
        write(U/'STATUS.json',{'state':'TRAINING_COMPLETE','runs':9,'test_predictions':0,'at':now()})
    except BaseException as e:
        f=U/'logs'/('ERROR_'+str(time.time_ns())+'.json');write(f,{'type':type(e).__name__,'message':str(e),'trace':traceback.format_exc(),'at':now()});raise
    finally:lock.unlink()

if __name__=='__main__':
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','run']);args=a.parse_args()
    prepare() if args.stage=='prepare' else run()
