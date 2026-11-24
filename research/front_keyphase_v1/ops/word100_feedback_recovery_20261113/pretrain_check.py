from model_core import *

def main():
    torch.set_num_threads(2)
    rr=rows_load();fit=[r for r in rr if r['role']=='fit'];dev=[r for r in rr if r['role']=='development']
    assert len([r for r in fit if r['origin']=='common100'])==899
    assert {r['person'] for r in fit}<={1,2,3,4,5,6,7,8,9}
    assert {r['person'] for r in dev}<={11,12,13}
    assert not {r['clip_id'] for r in fit}&{r['clip_id'] for r in dev}
    checks=[]
    # Unknown coordinates must not acquire supervised truth or affect the loss.
    y=torch.tensor([[[[0,-1,1,2,-1],[0,1,-1,2,-1]],[[0,1,2,-1,-1],[1,1,2,2,-1]]]])
    torch.manual_seed(1);z=torch.randn(*y.shape,3,requires_grad=True)
    loss=balanced_fine_loss({'fingers':z},{'fy':y});loss.backward()
    assert (z.grad[y<0]==0).all();z2=z.detach().clone();z2[y<0]=1000
    assert torch.equal(loss.detach(),balanced_fine_loss({'fingers':z2},{'fy':y}))
    checks.append('Unknown targets contribute zero loss and zero gradient')
    # Replicating every annotated frame in a clip must not increase clip weight.
    assert torch.allclose(loss.detach(),balanced_fine_loss({'fingers':z.detach().repeat_interleave(3,1)},{'fy':y.repeat_interleave(3,1)}))
    checks.append('Repeated identical annotation density does not change clip/state/hand mean')
    empty=balanced_fine_loss({'fingers':z},{'fy':torch.full_like(y,-1)});assert empty.item()==0
    checks.append('All unknown batch has finite differentiable zero loss')
    r=next(r for r in fit if (r['fy']==1).any());b=batch([r]);torch.manual_seed(42)
    net=FineNet(7);base=torch.load(B/'runs/shared100_42/model.pt',weights_only=True,map_location='cpu');net.load_state_dict(base,strict=False)
    for n,p in net.named_parameters():p.requires_grad=n.startswith('finger')
    net.train();before=net(b['x'],b['lengths'],b['word'],b['hand']);finger_before=before['fingers'].detach().clone()
    opt=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=.001)
    opt.zero_grad();balanced_fine_loss(before,b).backward();opt.step()
    after=net(b['x'],b['lengths'],b['word'],b['hand'])
    for k,v in base.items():assert torch.equal(v,net.state_dict()[k]),k
    for k in ['latent','attributes','stage','word_logits']:assert torch.equal(before[k],after[k]),k
    assert not torch.equal(finger_before,after['fingers'])
    checks.append('One real FIT update changes fingers while frozen common parameters and outputs remain bitwise equal')
    # Source labels remain sparse and are not spread over a word or timeline.
    changed=read(U/'DATA_READY.json');oldrows={r['clip_id']:r for r in read(PREV/'DATA_READY.json')['rows']};newcells=0
    for r in rr:
        if 'parent_cache' not in r:continue
        with np.load(oldrows[r['clip_id']]['path'],allow_pickle=False) as a:
            for k in a.files:
                if k!='fy':assert np.array_equal(a[k],r[k],equal_nan=True),k
            known=a['fy']>=0;assert np.array_equal(a['fy'][known],r['fy'][known]);newcells+=int(((a['fy']<0)&(r['fy']>=0)).sum())
    assert newcells==sum(changed['added_known_cells'].values())
    checks.append('Only new source-reviewed sparse finger targets changed; coordinates and old truth are identical')
    write(U/'PRETRAIN_CHECK.json',dict(status='PASS',checks=checks,rows=len(rr),added_known_cells=newcells,fit_records=sorted({r['person'] for r in fit}),development_records=sorted({r['person'] for r in dev}),production_changed=False))
    print('PASS',len(checks),'checks; new target cells',newcells)
if __name__=='__main__':main()
