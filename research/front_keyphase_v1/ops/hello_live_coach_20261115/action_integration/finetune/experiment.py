"""One fixed, isolated motion-head fine-tuning experiment on public data.

The existing shared Conv+GRU and motion encoder remain frozen. No webcam, private
input, word labels for the greeting, or deployment writes are performed here.
"""
from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import sys
import time
from collections import Counter
import numpy as np
import torch

D = Path(__file__).resolve().parent
U = D.parent.parent
O = U.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(U))
from action_runtime import ActionObserver, _load_model, CHECKPOINT, CHECKPOINT_SHA256


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, d):
    Path(p).write_text(json.dumps(d, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def encode(base, pose):
    with torch.inference_mode():
        h = torch.nn.functional.gelu(base.input(torch.from_numpy(pose[None]))).transpose(1, 2)
        for layer in base.temporal:
            h = layer(h)
        h, _ = base.context(h.transpose(1, 2))
    return h


def head_features(net, h, g, a, b):
    with torch.inference_mode():
        geom = torch.from_numpy(g[None])
        z = (geom-net.gm)/net.gs
        seq = net.motion_conv(z[:, :36].reshape(-1, 9, 4).transpose(1, 2)).transpose(1, 2)
        _, hn = net.motion_gru(seq)
        context = torch.cat([h[:, a], h[:, b], h[:, b]-h[:, a]], -1)
        feature = torch.cat([hn[0], hn[1], z, context], -1)
        direct = net.movement(h, geom, torch.tensor([[0, a, b]], dtype=torch.long))
        assert torch.allclose(net.motion_head(feature), direct, atol=1e-6)
    return feature[0].numpy().copy()


def prepare_old(rc, net):
    buckets = {'old_fit': [], 'old_dev': []}
    for r in rc.public_rows():
        assert r['clip_id'].startswith('NIA_SL_WORD')
        is_fit = r['role'] == 'fit'
        if not is_fit and r['role'] != 'development':
            continue
        for view in (['full', 'upper'] if is_fit else ['full']):
            pose = r['x_'+view].astype(np.float32)
            # Identical unnormalization and geometry to the original training.
            raw_x = rc.prior.physical(pose)
            h = encode(net.shared.base, pose)
            for j, p in enumerate(r['pairs']):
                a, b = int(p['a']), int(p['b'])
                for reverse in (False, True):
                    g, eligible, evidence = rc.prior.geometry(raw_x, r['timestamp_us'], a, b,
                        p['axis'], p.get('moving', 'R'), p.get('anchor', 'L'), reverse=reverse)
                    meta = dict(clip=r['clip_id'], person=int(r['person']), role=r['role'],
                        view=view, pair=j, axis=int(p['axis']), moving=p.get('moving', 'R'),
                        start_frame=int(r['frame_id'][a]), end_frame=int(r['frame_id'][b]),
                        reverse=reverse, target=int(2-p['target'] if reverse else p['target']),
                        eligible=bool(eligible), evidence=evidence,
                        source_input_sha256=sha(r['path']))
                    f = head_features(net, h, g, b if reverse else a, a if reverse else b)
                    buckets['old_fit' if is_fit else 'old_dev'].append((f, meta))
    return buckets


def prefix(rc, net, adapter, raw, selected):
    ch = adapter.convert(raw['xy_px'][selected], raw['score'][selected], raw['timestamp_us'][selected],
        raw['frame_id'][selected], int(raw['source_width']), int(raw['source_height']))
    x = ch['tcn280'].astype(np.float32)
    h = encode(net.shared.base, rc.prior.shared.normalize(x))
    return ch, x, h


def geometry3(rc, x, ts, a, b, axis, moving, anchor):
    obj = ActionObserver.__new__(ActionObserver)
    obj.rc, obj.x = rc, x
    return obj._geometry(ts, a, b, axis, moving, anchor, 3)


def reverse_geom(rc, x, ts, a, b, axis, moving, anchor):
    # The same genuine observations, reversed as a direction contrast. It is not
    # a new normal greeting or a fabricated observation-count increase.
    if b-a+1 >= 4:
        return rc.prior.geometry(x, ts, a, b, axis, moving, anchor, reverse=True)
    xx = x.copy(); xx[a:b+1] = xx[a:b+1][::-1]
    tt = ts.copy(); tt[a:b+1] = ts[a]+ts[b]-ts[a:b+1][::-1]
    return geometry3(rc, xx, tt, a, b, axis, moving, anchor)


def prepare_new(rc, net, adapter, labels):
    with np.load(labels['source_raw'], allow_pickle=False) as z:
        raw = {k:z[k].copy() for k in z.files}
    result=[]
    for item in labels['intervals']:
        for stride in (1, 2):
            keep = np.arange(0, len(raw['frame_id']), stride)
            inside = np.flatnonzero((raw['frame_id'][keep] >= item['start_frame']) &
                                    (raw['frame_id'][keep] <= item['end_frame']))
            assert len(inside) >= 3
            a, b = int(inside[0]), int(inside[-1])
            assert b+3 < len(keep), 'Three received future observations required for matching live delay'
            selected = keep[:b+4]
            ch, x, h = prefix(rc, net, adapter, raw, selected)
            for reverse in (False, True):
                fn = reverse_geom if reverse else geometry3
                g, eligible, evidence = fn(rc, x, ch['timestamp_us'], a, b, item['axis'], item['moving'], item['anchor'])
                assert eligible, (item, stride, evidence)
                meta=dict(id=item['id'], role='supplemental_fit', clip='NIA_SL_SEN0354_REAL01_F',
                    person=1, stride=stride, reverse=reverse, axis=item['axis'], moving=item['moving'],
                    start_frame=int(raw['frame_id'][selected[a]]), end_frame=int(raw['frame_id'][selected[b]]),
                    prefix_last_frame=int(raw['frame_id'][selected[-1]]),
                    target=int(2-item['target'] if reverse else item['target']), eligible=True,
                    evidence=evidence, independent=False,
                    annotation_scope='whole reviewed direction interval; no automatic subspan labels')
                result.append((head_features(net, h, g, b if reverse else a, a if reverse else b), meta))
    return result, raw


def diagnostics(rc, net, adapter, raw):
    # These original controller endpoints are diagnostics, never training labels.
    # The requested direction is not accepted as ground truth for a mixed span.
    result=[]; path=D.parent/'PUBLIC_TRACE.json'
    if path.exists():
        trace=read(path)
        for i,row in enumerate(trace):
            checks=row.get('action_direction_check', [])
            checks=checks if isinstance(checks,list) else [checks]
            if not checks:continue
            ch,x,h=prefix(rc,net,adapter,raw,np.arange(i+1))
            for j,q in enumerate(checks):
                aa=np.flatnonzero(ch['timestamp_us']==q['start_us']);bb=np.flatnonzero(ch['timestamp_us']==q['end_us'])
                if len(aa)!=1 or len(bb)!=1:continue
                a,b=int(aa[0]),int(bb[0])
                g,ok,info=geometry3(rc,x,ch['timestamp_us'],a,b,q['axis'],q['moving'],q['anchor'])
                if not ok:continue
                result.append((head_features(net,h,g,a,b),dict(id=f'controller_{i}_{j}',role='diagnostic',
                    start_frame=a,end_frame=b,prefix_last_frame=i,axis=q['axis'],moving=q['moving'],
                    eligible=True,target=None,evidence=info,independent=False,label='no correctness label')))
    # Archived old baseline first sweep: fixed 15fps uses genuine frames38,40,42.
    selected=np.arange(0,49,2);ch,x,h=prefix(rc,net,adapter,raw,selected)
    a,b=19,21;g,ok,info=geometry3(rc,x,ch['timestamp_us'],a,b,2,'R','L')
    assert ok
    result.append((head_features(net,h,g,a,b),dict(id='controller_15fps_sweep_38_42',role='diagnostic',
        start_frame=38,end_frame=42,prefix_last_frame=48,axis=2,moving='R',eligible=True,
        target=None,evidence=info,independent=False,label='previously observed failure, not independent truth')))
    return result


def tensor(rows):
    return torch.from_numpy(np.asarray([f for f,_ in rows], np.float32))


def summarize(rows, before, after):
    report=[]
    for (_,m),p0,p1 in zip(rows,before,after):
        report.append(dict(**m,before=p0.tolist(),after=p1.tolist(),
            before_class=int(p0.argmax()),after_class=int(p1.argmax()),
            before_confident=bool(p0.max()>=.65),after_confident=bool(p1.max()>=.65)))
    def metric(sub):
        valid=[r for r in sub if r['eligible'] and r['target'] is not None]
        out=dict(total=len(sub),eligible=len(valid))
        for side in ('before','after'):
            cm=np.zeros((3,3),int)
            for r in valid:cm[r['target'],r[side+'_class']]+=1
            out[side]=dict(correct=sum(r[side+'_class']==r['target'] for r in valid),
                confident_correct=sum(r[side+'_class']==r['target'] and r[side+'_confident'] for r in valid),
                confusion=cm.tolist())
        out['regressions']=[{k:r.get(k) for k in ('clip','id','view','pair','axis','moving','target','before_class','after_class','reverse')} for r in valid if r['before_class']==r['target'] and r['after_class']!=r['target']]
        return out
    return dict(all=metric(report),native=metric([r for r in report if not r.get('reverse',False)]),rows=report)


def main():
    assert not (D/'CANDIDATE.pt').exists() and not (D/'FROZEN_PLAN.json').exists(), 'Preserve this single fixed run'
    labels=read(D/'SOURCE_LABELS.json');assert not labels.get('private_training')
    plan=dict(seed=42,steps=400,optimizer='Adam',learning_rate=.001,batch_size=64,
        old_fit_per_batch=32,new_fit_per_batch=32,sampling='each source balanced across decrease/hold/increase with replacement',
        trained_parameters='motion_head only (two Linear layers)',frozen='shared100 Conv+GRU, attributes, stage, prototype, word auxiliary head, motionConv/GRU, normalization, order head',
        base_checkpoint=str(CHECKPOINT),base_sha256=CHECKPOINT_SHA256,
        source_labels_sha256=sha(D/'SOURCE_LABELS.json'),source_raw_sha256=sha(labels['source_raw']),
        code_sha256=sha(__file__),action_adapter_sha256=sha(U/'action_runtime.py'),
        old_supervision='Only public FIT rows; full/upper views and reversed direction contrasts matching original training',
        supplemental_supervision='9 reviewed complete spans at native and fixed stride2, each with explicit reversed direction contrast; no auto-labelled subspans',
        old_features='Existing training normalized full/upper sequence; frozen pretrained encoder',
        supplemental_features='Received raw prefix up to three retained observations after end; normalize each prefix independently',
        evaluation='Old historical DEV normal intervals and reversal diagnostics; same-reference supplemental fit and failed-controller diagnostics are NOT independent accuracy',
        threshold=.65,no_dev_fitting=True,no_private_data=True,no_camera=True,no_deployment=True)
    write(D/'FROZEN_PLAN.json',plan)
    torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42);torch.use_deterministic_algorithms(True)
    rc,net,adapter,digest=_load_model();assert digest==plan['base_sha256']
    started=time.perf_counter();old=prepare_old(rc,net);supp,raw=prepare_new(rc,net,adapter,labels)
    diag=diagnostics(rc,net,adapter,raw);datasets=dict(**old,supplemental_fit=supp,controller_diagnostics=diag)
    np.savez_compressed(D/'FEATURES.npz',**{k:tensor(v).numpy()for k,v in datasets.items()})
    write(D/'FEATURE_INDEX.json',{k:[m for _,m in v]for k,v in datasets.items()})
    print('PREPARED', {k:len(v)for k,v in datasets.items()}, flush=True)
    initial={k:v.detach().clone()for k,v in net.state_dict().items()}
    candidate=copy.deepcopy(net.motion_head);candidate.requires_grad_(True);candidate.train()
    oldfit=[r for r in old['old_fit'] if r[1]['eligible']]
    fitsets=[oldfit,supp];features=[tensor(v)for v in fitsets]
    targets=[torch.tensor([m['target']for _,m in v],dtype=torch.long)for v in fitsets]
    assert all(set(y.tolist())=={0,1,2}for y in targets)
    rng=np.random.default_rng(42)
    def choose(y,n):
        groups=[np.flatnonzero(y.numpy()==i)for i in range(3)]
        cls=np.resize(np.arange(3),n);rng.shuffle(cls)
        return np.asarray([rng.choice(groups[int(c)])for c in cls],np.int64)
    opt=torch.optim.Adam(candidate.parameters(),lr=.001);history=[]
    for step in range(400):
        ii=[choose(y,32)for y in targets]
        x=torch.cat([z[i]for z,i in zip(features,ii)]);y=torch.cat([z[i]for z,i in zip(targets,ii)])
        loss=torch.nn.functional.cross_entropy(candidate(x),y)
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(candidate.parameters(),2.);opt.step()
        if (step+1)%100==0:
            history.append(dict(step=step+1,loss=float(loss.detach())));print('STEP',step+1,float(loss.detach()),flush=True)
    candidate.eval();results={}
    with torch.inference_mode():
        for name,rows in datasets.items():
            f=tensor(rows);before=net.motion_head(f).softmax(-1).numpy();after=candidate(f).softmax(-1).numpy()
            results[name]=summarize(rows,before,after)
    unchanged=all(torch.equal(initial[k],v)for k,v in net.state_dict().items())
    assert unchanged and sha(CHECKPOINT)==CHECKPOINT_SHA256
    assert sha(D/'SOURCE_LABELS.json')==plan['source_labels_sha256']
    torch.save(dict(schema='public_motion_head_only_v1',state=candidate.state_dict(),base_checkpoint=str(CHECKPOINT),
        base_sha256=CHECKPOINT_SHA256,seed=42,steps=400,source_labels_sha256=plan['source_labels_sha256'],
        parameter_scope='motion_head',training_private_data=False),D/'CANDIDATE.pt')
    write(D/'RESULTS.json',dict(results=results,history=history,seconds=time.perf_counter()-started,
        original_all_tensors_unchanged=unchanged,base_sha256=sha(CHECKPOINT),candidate_sha256=sha(D/'CANDIDATE.pt'),
        deployed=False,independent_accuracy=None,threshold_changed=False))
    print(json.dumps({k:{'native':v['native'],'all':{q:v['all'][q]for q in ('total','before','after')}}for k,v in results.items()if k!='controller_diagnostics'},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
