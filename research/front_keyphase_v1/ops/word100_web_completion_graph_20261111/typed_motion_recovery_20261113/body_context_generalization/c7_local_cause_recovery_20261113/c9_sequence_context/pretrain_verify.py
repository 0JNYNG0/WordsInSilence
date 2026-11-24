from sequence_experiment import *
checks=[]
def ck(n,b):assert b,n;checks.append(n)
c=read(U/'TRAINING_FROZEN.json');check(c['pins']);check(c['protected_pins']);d=np.load(C6/'INPUTS.npz');q=np.load(C7/'conditioned_context/INPUTS.npz')['features'].reshape(-1,3,10,5)[:,0];z=np.load(U/'SEQUENCES.npz');a=z['mean_control'];b=z['ordered'];rows=read(C6/'INDEX.json')['rows']
ck('shape',a.shape==b.shape==(len(q),9,51));ck('finite',np.isfinite(a).all()and np.isfinite(b).all());ck('centres',np.array_equal(a[:,4,:50],b[:,4,:50])and np.array_equal(b[:,4,:50],q.reshape(-1,50)))
ck('masks_equal',np.array_equal(a[:,:,MASK],b[:,:,MASK]));ck('timestamps_equal',np.array_equal(a[:,:,50],b[:,:,50]));ck('centre_timestamp0',not b[:,4,50].any())
for r in rows:
    lo,hi=r['start'],r['stop'];ts=d['timestamp_us'];ck('recordedFIT',r['person']in PEOPLE and r['split']=='train')
    for i in range(lo,hi):
        for position in range(9):
            k=i+position-4
            if lo<=k<hi:
                ck('native_row',np.array_equal(b[i,position,:50],q[k].reshape(50)))
                ck('native_time',abs(b[i,position,50]-(ts[k]-ts[i])/1e6)<1e-6)
            else:ck('padding_not_observed',not b[i,position].any())
    for i in sorted({lo,(lo+hi)//2,hi-1}):
        for slots in [list(range(5)),list(range(4,9))]:
            v=b[i,slots,:50].reshape(-1,10,5);m=v[:,:,4]>0
            for family in range(10):
                avg=v[m[:,family],family,:4].mean(0)if m[:,family].any()else np.zeros(4)
                for p in slots:
                    if p==4:continue
                    ck('mean_control_definition',np.allclose(a[i,p,family*5:family*5+4],avg if b[i,p,family*5+4]>0 else 0,atol=2e-6))
for fold in c['folds']:
    norm=read(fold['normalization']);tr=d['person']!=fold['person'];mu=np.array(norm['mean']);sd=np.array(norm['std'])
    ck('fold_only',norm['training_people']==[p for p in PEOPLE if p!=fold['person']])
    for j in range(10):
        take=tr&(q[:,j,4]>0)
        ck('normalization_mean',np.allclose(mu[j*5:j*5+4],q[take,j,:4].mean(0,dtype=np.float64),atol=1e-9))
        ck('normalization_std',np.allclose(sd[j*5:j*5+4],np.maximum(q[take,j,:4].std(0,dtype=np.float64),1e-4),atol=1e-9))
    ck('unscaled_masks_time',np.all(mu[np.r_[MASK,50]]==0)and np.all(sd[np.r_[MASK,50]]==1))
ts=np.arange(9)*33333;constant=np.zeros((9,10,5),np.float32);constant[...,:4]=2;constant[...,4]=1
ma,ob=sequences(constant,ts);ck('constant_mean_equals_order',np.array_equal(ma,ob))
gap=constant.copy();gap[3:5]=0;ma,ob=sequences(gap,ts);ck('gap_mask_preserved',np.array_equal(ma[:,:,MASK],ob[:,:,MASK]))
pulse=constant.copy();pulse[2,:,:4]+=1;ma,ob=sequences(pulse,ts);ck('short_pulse_native_preserved',np.any(ma[4]!=ob[4]))
torch.manual_seed(42);n1=Net(np.zeros(51),np.ones(51));torch.manual_seed(42);n2=Net(np.zeros(51),np.ones(51));ck('paired_initialization',init_hash(n1)==init_hash(n2));ck('parametercount',sum(p.numel()for p in n1.parameters())==c['trainable_parameters'])
with torch.inference_mode():
    out=n1(torch.from_numpy(b[:3]));ck('untrained_shape',out.shape==(3,5)and torch.isfinite(out).all())
write('PRETRAIN_VERIFICATION.json',dict(status='PASS',checks=len(checks),frozen_sha256=sha(U/'TRAINING_FROZEN.json'),no_trained_model_inference=True,scope='All native rows/masks/timestamps, independent mean arithmetic, FIT-only normalization, synthetic gap/pulse invariants; not linguistic truth.'))
print('C9pretrainPASS',len(checks))
