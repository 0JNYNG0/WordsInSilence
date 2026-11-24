"""Preserve first diagnostic; repair ONLY its reverse-thirds clock confound."""
from diagnose import *

def run():
    assert not (U/'ORDER_CLOCK_CORRECTION.json').exists()
    original=read(U/'ANALYSIS.json'); scope=read(U/'SCOPE.json')
    for p,h in original['pins'].items(): assert sha(p)==h,p
    write(U/'FIRST_DIAGNOSTIC_CORRECTION.json',{
      'reason':'Reverse-thirds originally reordered all280 channels, leaving an internal zero dt from the original first frame. To isolate ordering, preserve original clock and permute ONLY channels0:279.',
      'first_evidence':{'ANALYSIS.json':sha(U/'ANALYSIS.json'),'diagnose.py':sha(U/'diagnose.py')},
      'scope':'36 reverse-thirds public TRAIN inputs only; original scores and all other diagnostic conditions reused. No training, coefficient change or best-result selection.',
      'rule':'a=x.copy(); a[:,:279]=x[reversed_three_block_indices,:279]; dt remains bitwise original.',
      'invalid_first_condition_not_reported_as_evidence':True})
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    ck=torch.load(T/'T3F_20261013_best.pt',map_location='cpu',weights_only=True)
    cp=torch.load(B/'training_C_20261013_best.pt',map_location='cpu',weights_only=True)
    m=FrozenCTemporal(**ck['model_config']);m.load_state_dict(ck['state_dict']);m.eval()
    c=MaskedSignTCN(**ck['model_config']);c.load_state_dict(cp['state_dict']);c.eval()
    before=[statehash(c),statehash(m)]; labels=ck['classes'];out=[]
    for old in original['predictions']:
        if old['condition']!='reverse_thirds':continue
        r=read(B/'data/records'/f"{old['clip_id']}.json")
        assert r['role']=='train_candidate' and r['clip_id'] in scope['train_clips']
        z=load(r['views'][old['view']]['channel_file']);x=z['x_whole'];n=len(x)
        perm=np.concatenate(np.array_split(np.arange(n),3)[::-1]);a=x.copy();a[:,:279]=x[perm,:279]
        assert np.array_equal(a[:,279],x[:,279]) and np.all(a[1:,279]>0)
        with torch.inference_mode():xx=torch.from_numpy(a[None]);pad=torch.ones(1,n);cc=c(xx,pad)[0].numpy();tt=m(xx,pad)[0].numpy()
        out.append({k:v for k,v in old.items() if k not in ('C','T3F')} | {
          'clock_preserved':True,'C':predict(cc,old['label_index'],old['other_index'],labels),
          'T3F':predict(tt,old['label_index'],old['other_index'],labels)})
    assert before==[statehash(c),statehash(m)]
    for p,h in original['pins'].items():assert sha(p)==h,p
    write(U/'ORDER_CLOCK_CORRECTION.json',{'status':'PASS','predictions':out,'count':len(out),'model_state_unchanged':True})
    print('36 reverse-third pairs corrected; original evidence preserved.')
if __name__=='__main__':run()
