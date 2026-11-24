from window_core import *
from collections import Counter
def main():
 assert not (U/'FROZEN.json').exists();torch.set_num_threads(4);torch.set_num_interop_threads(1)
 rr=prior.rows();fit=[r for r in rr if r['role']=='fit'];normold=read(P/'NORMALIZATION.json');frozen=prior.Network('coupled',normold);frozen.load_state_dict(torch.load(P/'runs/coupled_42/model.pt',weights_only=True,map_location='cpu'));frozen.eval();frozen.requires_grad_(False);(U/'cache').mkdir(exist_ok=True)
 gs=[];qs=[[] for _ in range(5)]
 with torch.inference_mode():
  for k in range(0,len(fit),24):
   chunk=fit[k:k+24];cache={r['clip_id']:{} for r in chunk}
   for view in ('full','upper'):
    b=shared.batch(chunk,view);o=frozen.shared(b['x'],b['lengths'],b['word'])
    for i,r in enumerate(chunk):
     raw=prior.physical(r['x_'+view]);g=geometry(raw,r['timestamp_us']);gs.append(g);cache[r['clip_id']]['g_'+view]=g;cache[r['clip_id']]['frozen_h_'+view]=o['latent'][i,:len(raw)].numpy()
     for p in r['pairs']:
      z=relations(raw,p['axis'],p.get('moving','R'),p.get('anchor','L'));qs[p['axis']].append(z[p['a']:p['b']+1])
   for r in chunk:np.savez_compressed(U/'cache'/f"{r['clip_id']}.npz",**cache[r['clip_id']])
   if k%240==0:event('cached_fit',clips=min(k+24,len(fit)))
 g=np.concatenate(gs);gm=g.mean(0);std=g.std(0).clip(.05);gm[22:33]=gm[55:]=0;std[22:33]=std[55:]=1;qm=[];qstd=[]
 for items in qs:
  q=np.concatenate(items);m=q.mean(0);s=q.std(0).clip(.05);m[8:]=0;s[8:]=1;qm.append(m.tolist());qstd.append(s.tolist())
 keys=(frozen.shared.base.guide_encoder(frozen.shared.base.guide_vectors)+frozen.shared.base.stage_embedding).detach().numpy()
 write(U/'NORMALIZATION.json',dict(gmean=gm.tolist(),gstd=std.tolist(),qmean=qm,qstd=qstd,guide_keys=keys.tolist(),fit_only=True))
 review=read(E/'REVIEW_LABELS.json');rechecked=['NIA_SL_WORD1133_REAL04_F','NIA_SL_WORD0739_REAL02_F','NIA_SL_WORD1567_REAL08_F'];write(U/'REVIEW.json',dict(reviewer='Codex RGB recheck, not independent human',rechecked=[r for r in review['rows'] if r['clip_id'] in rechecked],findings=['Actual prepare approach precedes reviewed brush stroke and withdrawal follows it','Open hold -> closing descent -> upright forearm are distinct events','Alternating two hands contains multiple real strokes, not disposable transition'],boundary_scope='Reviewed sparse anchors and motion intervals; interpolated core masks are weak supervision, not dense expert truth.'))
 ready=dict(fit=899,words=100,development=300,motion_intervals=sum(len(r['pairs']) for r in fit),order_pairs=sum(len(r['orders']) for r in fit),fit_ids=sorted(set(r['person'] for r in fit)),development_ids=sorted(set(r['person'] for r in rr if r['role']=='development')),frozen_shape_model=str(P/'runs/coupled_42/model.pt'),all100_cache_count=len(list((U/'cache').glob('*.npz'))))
 assert ready['all100_cache_count']==899;write(U/'READY.json',ready)
 cfg=dict(conditions=['uniform','learned_core'],seeds=[42,43,44],primary_seed=42,epochs=12,steps_per_epoch=38,aux_batch=24,supervised_clips=6,lr=.0005,head_lr=.001,weight_decay=.01,losses=dict(word=.2,motion=1.,core=.2,order=.3,anchor=.1),confidence=.65,coverage=.8,min_observed=4,padding_frames=[0,4,8,12],background_weight=1.,interval_core_weight=.2,fit_reverse_main=True,final_epoch_only=True,
  authorization='User asks improve automatic windows mixed with preparation/finish, explain899/31/76/63/228/32 then proceed; inherited preserve100 and no private data.',
  model='Preserved100-word shape encoder; geometry Conv+BiGRU fine-tuned with100-word auxiliary, local relation stream and core-weighted direction4-way/guide-pair ordering heads. Uniform vs learned core pooling.',
  labels='Use all reviewed same-relation intervals represented in a window. Opposite nonzero directions => mixed class; hold is not mixed. Background only directly reviewed0anchors. Weak interval extent does not become new human gold.',
  adoption_gate='No normal/speed/crossing regression, no missed/hidden important-error interval, no false completion/missing->normal, movement/order improvement in automatic windows; otherwise isolated only.',
  evaluation='32reviewed intervals clean and padded +12order pairs clean/padded/reversed;60cached raw conditions+9edited factorial controls per model;fixed100aux300DEV;historical regression not fresh final test.',
  protections=read(P/'FROZEN.json')['protected'],pins={str(p):sha(p) for p in [P/'FROZEN.json',P/'ADDITIONAL_LABELS.json',P/'runs/coupled_42/model.pt',N/'runs/final_42/model.pt',E/'REVIEW_LABELS.json',U/'FIT_WINDOW_AUDIT.json',U/'NORMALIZATION.json',U/'REVIEW.json']},code_pins={str(U/f):sha(U/f) for f in ['window_core.py','prepare.py','train.py']})
 write(U/'FROZEN.json',cfg);write(U/'CURRENT_EXECUTION.json',dict(status='PREPARED',pid=None));event('prepared',**ready)
if __name__=='__main__':main()
