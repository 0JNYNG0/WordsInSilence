"""C6: paired body-context representation diagnostic, FIT people only.

No encoder/old-model forward, no DEV/probe/final/public-user inference.
This is a new scratch-head diagnostic, not replacement of the fine-tuned model.
"""
from pathlib import Path
import datetime, hashlib, json, sys, time, traceback
import numpy as np
import torch
from torch import nn

sys.dont_write_bytecode = True
U = Path(__file__).resolve().parent
H, G, O = U.parent, U.parent.parent, U.parent.parent.parent
BP = H / 'body_progress_witness_learning'
FAMILIES = [2, 3, 4, 5, 7]
PEOPLE = [2, 3, 4, 5, 6, 7]
CONDITIONS = ['absolute_context', 'relative_context']

def read(p): return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stamp(): return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
def write(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
def check(pins):
    for p, h in pins.items(): assert sha(p) == h, p
def stop():
    for d in [U, H, G]:
        for n in ['STOP', 'STOP.txt', 'STOP_REQUESTED', 'STOP_REQUESTED.json']:
            assert not (d/n).exists(), str(d/n)

def relative(x):
    z = np.asarray(x).reshape(-1, 3, 10, 5)
    out = z[:, [1, 0, 2]].copy()
    out[:, 1, :, :4] -= z[:, 1, :, :4]
    out[:, 2, :, :4] -= z[:, 1, :, :4]
    return out.reshape(-1, 150)

def inverse_relative(x):
    z = np.asarray(x).reshape(-1, 3, 10, 5)
    out = z[:, [1, 0, 2]].copy()
    out[:, 0, :, :4] += z[:, 0, :, :4]
    out[:, 2, :, :4] += z[:, 0, :, :4]
    return out.reshape(-1, 150)

def balanced(y, mask, seq_id, rows, allowed):
    w = np.zeros_like(y, dtype=np.float64)
    for j in range(5):
        for value in [0, 1]:
            take = allowed & (mask[:, j] > 0) & (y[:, j] == value)
            ids = np.unique(seq_id[take]).tolist()
            clips = sorted({rows[i]['clip_id'] for i in ids})
            assert clips, (j, value)
            for cid in clips:
                ss = [i for i in ids if rows[i]['clip_id'] == cid]
                for i in ss:
                    ix = take & (seq_id == i)
                    w[ix, j] = 1 / (10 * len(clips) * len(ss) * int(ix.sum()))
    assert np.isclose(w.sum(), 1.)
    assert np.all(w[mask == 0] == 0)
    return w.astype(np.float32)

class Head(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean, dtype=torch.float32))
        self.register_buffer('std', torch.tensor(std, dtype=torch.float32))
        self.head = nn.Sequential(nn.Linear(150, 96), nn.GELU(), nn.Linear(96, 5))
    def forward(self, x): return self.head((x-self.mean)/self.std)

def prepare():
    stop(); assert not (U/'FROZEN.json').exists()
    meta = read(BP/'INPUTS_COMPLETE.json'); rows = meta['sequences']
    assert len(rows) == 144 and {r['person'] for r in rows} == set(PEOPLE)
    assert len({r['clip_id'] for r in rows}) == 24
    assert all(r['split'] == 'train' for r in rows)
    check(meta['protected_pins'])
    files = [Path(__file__), U/'verify.py', BP/'INPUTS_COMPLETE.json', BP/'SOURCE_ADMISSION.json',
             BP/'PRETRAIN_VERIFICATION.json', BP/'prepare.py',
             G/'candidate_description_supervision/source_boundary_core.py',
             G/'candidate_description_supervision/prepare_source_boundary.py',
             G/'candidate_description_supervision/prepare_rank_learning.py',
             O/'word100_candidate_role_learning_20261111/role_core.py',
             O/'word100_episode_joint_linkage_20261111/runs/42/model.pt']
    pins = {str(p): sha(p) for p in files}
    # Read normalization buffers only; no parent parameter initializes a head.
    st = torch.load(files[-1], map_location='cpu', weights_only=True)
    qm, qs = st['qm'].numpy()[:, :4], st['qs'].numpy()[:, :4]
    assert qm.shape == qs.shape == (10, 4) and (qs > 0).all()
    arrays = {k: [] for k in ['absolute_context', 'relative_context', 'y', 'mask', 'observed', 'label_provenance', 'seq_id', 'source_frame_id', 'timestamp_us', 'person']}
    offsets = []; cursor = 0; max_roundtrip = 0.; max_context_error = 0.
    for i, r in enumerate(rows):
        assert sha(r['path']) == r['sha256']; pins[r['path']] = r['sha256']
        with np.load(r['path'], allow_pickle=False) as z: d = {k: z[k].copy() for k in z.files}
        n = len(d['x']); assert n == r['length']
        xx = d['x'].astype(np.float64).reshape(n, 3, 10, 5)
        # Undo upstream affine normalization, including availability-weighted means:
        # mean((q-mu)/sd * m)*sd + mu*mean(m) = mean(q*m).
        raw = xx.copy(); raw[..., :4] = xx[..., :4]*qs + qm*xx[..., 4:5]
        restored = raw.copy(); restored[..., :4] = (raw[..., :4]-qm*raw[..., 4:5])/qs
        max_roundtrip = max(max_roundtrip, float(np.max(np.abs(restored-xx))))
        tt = d['timestamp_us']; current = raw[:, 1]
        lo = np.searchsorted(tt, tt-133333); hi = np.searchsorted(tt, tt+133333, side='right')
        direct = np.stack([np.stack([current[a:t+1].mean(0), current[t], current[t:b].mean(0)]) for t,(a,b) in enumerate(zip(lo,hi))])
        max_context_error = max(max_context_error, float(np.max(np.abs(raw-direct))))
        assert np.max(np.abs(raw-direct)) < 2e-5
        a = raw.reshape(n,150).astype(np.float32); b = relative(a)
        assert np.allclose(inverse_relative(b), a, atol=2e-6)
        assert np.isfinite(a).all() and np.isfinite(b).all()
        assert np.all(d['mask'] <= d['observed'])
        assert np.array_equal(d['label_provenance'] > 0, d['mask'] > 0)
        arrays['absolute_context'].append(a); arrays['relative_context'].append(b)
        for k in ['y','mask','observed','label_provenance','source_frame_id','timestamp_us']: arrays[k].append(d[k])
        for k,v in [('seq_id',i),('person',r['person'])]: arrays[k].append(np.full(n,v,np.int32))
        offsets.append(dict(**r, input_index=i, start=cursor, stop=cursor+n)); cursor += n
    flat = {k: np.concatenate(v) for k,v in arrays.items()}
    assert int((flat['label_provenance']==1).sum()) == 822
    assert int((flat['label_provenance']==3).sum()) == 522
    np.savez_compressed(U/'INPUTS.npz', **flat)
    write(U/'INDEX.json', dict(rows=offsets, families=FAMILIES, total_frames=cursor))
    folds=[]
    for person in PEOPLE:
        tr = flat['person'] != person; te = ~tr
        trids = {r['source_sha256'] for r in rows if r['person'] != person}
        teids = {r['source_sha256'] for r in rows if r['person'] == person}
        assert not trids & teids and len(trids)==20 and len(teids)==4
        w = balanced(flat['y'],flat['mask'],flat['seq_id'],rows,tr)
        assert not w[te].any()
        fp=U/'folds'/str(person); fp.mkdir(parents=True)
        norm={}
        for c in CONDITIONS:
            x=flat[c][tr].astype(np.float64)
            norm[c]={'mean':x.mean(0).tolist(),'std':np.maximum(x.std(0),1e-4).tolist()}
        write(fp/'NORMALIZATION.json',norm); np.savez_compressed(fp/'WEIGHTS.npz',weights=w)
        folds.append(dict(person=person, training_sources=20, evaluation_sources=4,
                          train_frames=int(tr.sum()),eval_frames=int(te.sum()),
                          normalization=str(fp/'NORMALIZATION.json'),weights=str(fp/'WEIGHTS.npz')))
        for p in [fp/'NORMALIZATION.json',fp/'WEIGHTS.npz']: pins[str(p)]=sha(p)
    for p in [U/'INPUTS.npz',U/'INDEX.json']:pins[str(p)]=sha(p)
    write(U/'INPUT_AUDIT.json',dict(status='PASS',sources=24,people=PEOPLE,sequences=144,frames=cursor,
        maximum_upstream_inverse_error=max_roundtrip,maximum_context_reconstruction_error=max_context_error,
        upstream='Only affine-normalized body-relative geometry and masks consumed; invert global affine scaling before fold normalization. No neural hidden features/parent weights consumed.',
        independence='All six recorded IDs have historical research exposure. This is a head-label-held-out diagnostic, not untouched final people or end-to-end product validation.',
        annotation='Existing Codex-reviewed sparse physical observations, not independent KSL teacher truth.',
        no_new_information=True,no_labels_changed=True,no_old_model_forward=True,protected_files=len(meta['protected_pins'])))
    write(U/'FROZEN.json',dict(at=stamp(),status='PREPARED_REQUIRES_PRETRAIN_VERIFICATION',
        authorization='User 2026-11-13 proceed with reviewed stabilization and demo-word readiness plan.',
        hypothesis='Same information, relative current/before-current/after-current coordinates may improve conditioning and head generalization.',
        conditions=CONDITIONS, seeds=[42,43,44], folds=folds,steps=1200,lr=.001,weight_decay=.001,gradient_clip=2,
        architecture='150 ->96 GELU ->5; identical scratch trainable tensors within fold/seed; no trained parent initialization',
        normalization='Undo upstream affine scale; then each condition mean/std from five training recorded IDs only, all their frames; std floor1e-4.',
        loss='Full-batch BCE; each family/class .1, each eligible source/view/reversal/known point balanced. Rows with zero loss weight excluded only from matrix multiply.',
        readout='raw logit>=0 on qualified body observations; all predeclared labels retained; no threshold/epoch/seed selection',
        gate='Per seed pooled motion and own-hold correct counts nondecreasing; every motion family nondecreasing; no held-out fold loses BOTH motion and own-hold; at least one strict gain. Report new individual errors too. Diagnostic gate only, not G2 or deployment.',
        input_scope='FIT recorded IDs02..07 only; 4 expressions,3 views,forward/reverse. Reverse labels are physical movement labels, not correct KSL.',
        limitations='Sparse body labels only; no fingertip/shape/semantic completion/crossing accuracy/normal-speed generalization claim. Historical public exposure remains.',
        no_probe_DEV_final8=True,no_private=True,no_web_change=True,no_common100_change=True,
        pins=pins,protected_pins=meta['protected_pins']))
    print('C6 input prepared:',cursor,'frames, 6 paired folds, 3 seeds',flush=True)

def run():
    stop(); cfg=read(U/'FROZEN.json'); check(cfg['pins']); check(cfg['protected_pins'])
    pv=read(U/'PRETRAIN_VERIFICATION.json'); assert pv['status']=='PASS' and pv['frozen_sha256']==sha(U/'FROZEN.json')
    assert not (U/'STARTED.json').exists()
    write(U/'STARTED.json',dict(at=stamp(),frozen_sha256=sha(U/'FROZEN.json')))
    with np.load(U/'INPUTS.npz',allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    results=[]
    for fold in cfg['folds']:
        p=fold['person'];tr=d['person']!=p;te=~tr
        norm=read(fold['normalization'])
        with np.load(fold['weights']) as z:w=z['weights'].copy()
        active=tr & (w.sum(1)>0)
        y=torch.tensor(d['y'][active],dtype=torch.float32);ww=torch.tensor(w[active])
        for seed in cfg['seeds']:
            initial_hash=None
            for c in cfg['conditions']:
                stop(); torch.manual_seed(seed)
                net=Head(norm[c]['mean'],norm[c]['std'])
                hh=hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in net.head.parameters())).hexdigest()
                if initial_hash is None:initial_hash=hh
                assert initial_hash==hh
                x=torch.tensor(d[c][active]);opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
                losses=[];tick=time.perf_counter();net.train()
                for step in range(cfg['steps']):
                    stop();logits=net(x)
                    loss=(nn.functional.binary_cross_entropy_with_logits(logits,y,reduction='none')*ww).sum()
                    assert torch.isfinite(loss)
                    opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),cfg['gradient_clip']);opt.step()
                    if step%300==0 or step==cfg['steps']-1:losses.append(dict(step=step+1,loss=float(loss.detach())))
                elapsed=time.perf_counter()-tick;net.eval()
                out=U/'runs'/f'{p}_{seed}_{c}';out.mkdir(parents=True,exist_ok=False)
                torch.save(net.state_dict(),out/'model.pt')
                with torch.inference_mode():pred=net(torch.tensor(d[c])).numpy()
                np.savez_compressed(out/'PREDICTIONS.npz',logits=pred)
                rr=dict(person=p,seed=seed,condition=c,initial_trainable_hash=hh,seconds=elapsed,steps=cfg['steps'],loss=losses,
                        model=str(out/'model.pt'),model_sha256=sha(out/'model.pt'),prediction=str(out/'PREDICTIONS.npz'),prediction_sha256=sha(out/'PREDICTIONS.npz'))
                write(out/'COMPLETE.json',rr);results.append(rr)
                print('C6',len(results),'/36',p,seed,c,'seconds',round(elapsed,2),flush=True)
    check(cfg['pins']);check(cfg['protected_pins']);stop()
    write(U/'TRAINING_COMPLETE.json',dict(at=stamp(),status='COMPLETE_DIAGNOSTIC_ONLY',rows=results,
        frozen_sha256=sha(U/'FROZEN.json'),no_model_selection=True,no_old_forward=True,no_web_change=True))

if __name__=='__main__':
    try:
        assert len(sys.argv)==2 and sys.argv[1] in ['prepare','run']
        prepare() if sys.argv[1]=='prepare' else run()
    except BaseException:
        U.mkdir(parents=True,exist_ok=True)
        (U/f'FIRST_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8')
        raise
