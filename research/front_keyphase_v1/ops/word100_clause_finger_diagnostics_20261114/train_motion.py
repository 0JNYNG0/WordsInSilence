from pathlib import Path
import sys,json,time,collections
import numpy as np,torch
from motion_core import SymmetricMotion,reverse_geometry
U=Path(__file__).resolve().parent;O=U.parent
sys.path.insert(0,str(O/'word100_guide_requirement_learning_20261111'))
import requirement_core as rc
v2='--v2'in sys.argv;suffix='_V2'if v2 else''
cfg=rc.read(U/'MOTION_FROZEN.json');assert not(U/f'MOTION_TRAINED{suffix}.json').exists()
if v2:assert (U/'REPAIR_FROZEN_V2.json').exists()
for p,h in cfg['pins'].items():assert rc.sha(p)==h
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);old=rc.frozen_observer();data=np.load(U/('motion_arrays_v2.npz'if v2 else'motion_arrays.npz'));g=torch.tensor(data['geometry']);y=torch.tensor(data['targets']);rows=rc.read(U/f'MOTION_INPUTS{suffix}.json')['rows'];fi=np.array([i for i,r in enumerate(rows)if r['role']=='fit'and r['eligible']]);di=np.array([i for i,r in enumerate(rows)if r['role']=='development'])
counts=collections.Counter((rows[i]['axis'],rows[i]['target'],rows[i]['clip_id'])for i in fi);classes=collections.Counter((a,t)for a,t,c in counts);w=np.array([1/(counts[rows[i]['axis'],rows[i]['target'],rows[i]['clip_id']]*classes[rows[i]['axis'],rows[i]['target']])for i in fi]);w/=w.sum()
dest=U/('motion_models_v2'if v2 else'motion_models');dest.mkdir(exist_ok=True);runs=[]
for seed in cfg['seeds']:
 torch.manual_seed(seed);rng=np.random.default_rng(seed);m=SymmetricMotion(old);opt=torch.optim.AdamW(m.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);start=time.perf_counter()
 for step in range(cfg['steps']):
  ids=rng.choice(fi,size=cfg['batch'],replace=True,p=w);xx=g[ids];yy=y[ids];rev=torch.tensor(rng.random(len(ids))<.5);xx=torch.where(rev[:,None],reverse_geometry(xx),xx);yy=torch.where(rev,2-yy,yy)
  loss=torch.nn.functional.cross_entropy(m(xx),yy);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()
 m.eval()
 with torch.inference_mode():p=m(g).softmax(-1).numpy();pr=m(reverse_geometry(g)).softmax(-1).numpy()
 err=float(np.max(np.abs(pr-p[:,[2,1,0]])));assert err<2e-6
 path=dest/f'{seed}.pt';torch.save(m.state_dict(),path);np.savez_compressed(dest/f'{seed}_prob.npz',forward=p,reverse=pr)
 run=dict(seed=seed,model=str(path),sha256=rc.sha(path),seconds=time.perf_counter()-start,fit_rows=len(fi),development_rows=len(di),reversal_equivariance_max_error=err,dev_correct=sum(rows[i]['eligible']and p[i].argmax()==rows[i]['target']for i in di).item(),web_changed=False);runs.append(run);print(json.dumps(run),flush=True)
rc.write(U/f'MOTION_TRAINED{suffix}.json',dict(rows=runs,scope='Additional isolated movement branch fine-tuning; common100/word recognition unchanged; no web promotion'))
