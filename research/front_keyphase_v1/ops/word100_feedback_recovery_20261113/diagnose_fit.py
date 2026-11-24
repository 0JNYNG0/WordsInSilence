"""One new in-sample diagnosis; no DEV rescoring, tuning or new candidate."""
from model_core import *
from collections import defaultdict

def main():
    assert read(U/'RESULT_VERIFIED.json')['status']=='PASS'
    assert not (U/'FIT_DIAGNOSTIC.json').exists()
    names=[f'{c}_{s}'for c in ['frozen','finetune']for s in [42,43,44]]
    protocol=dict(scope='All prepared FIT known finger cells, once, full view. In-sample error diagnosis, not fresh/generalization validation.',
       purpose='Distinguish inability to fit reviewed training states from failure on historical other-record development. No weight/threshold/candidate selection from diagnostic.',
       models={n:sha(U/'runs'/n/'model.pt') for n in names},data_sha256=sha(U/'DATA_READY.json'),script_sha256=sha(__file__),new_training=False,development_inference=False)
    write(U/'FIT_DIAGNOSTIC_PROTOCOL.json',protocol)
    rr=[r for r in rows_load('fit')if(r['fy']>=0).any()]
    added={(r['clip_id'],r['frame'],r['side'],f)for r in read(U/'shape_recovery/EXTRA_LABELS.json')['rows']for f,y in enumerate(r['y'])if y>=0}
    torch.set_num_threads(4);results={}
    for name in names:
        net=FineNet(7);net.load_state_dict(torch.load(U/'runs'/name/'model.pt',map_location='cpu',weights_only=True));net.eval();rows=[]
        with torch.inference_mode():
          for r in rr:
            b=batch([r]);o=net(b['x'],b['lengths'],b['word'],b['hand']);p=o['fingers'][0].softmax(-1).numpy()
            for i,s,f in zip(*np.where(r['fy']>=0)):
                side='L' if s==0 else 'R';frame=int(r['frame_id'][i]);t=int(r['fy'][i,s,f]);pred=int(p[i,s,f].argmax())
                rows.append(dict(clip_id=r['clip_id'],word=r['word_id'],person=r['person'],frame=frame,side=side,finger=int(f),target=t,pred=pred,origin=r['origin'],new_label=(r['clip_id'],frame,side,int(f))in added))
        groups=defaultdict(lambda:dict(total=0,correct=0))
        for r in rows:
          for k in ['all',r['origin'],'new'if r['new_label']else'old','state_'+str(r['target']),'hand_'+r['side'],'word_'+r['word'],'person_'+str(r['person'])]:
            groups[k]['total']+=1;groups[k]['correct']+=r['pred']==r['target']
        assert groups['all']['total']==1061 and groups['new']['total']==200
        results[name]=dict(groups=dict(groups),rows=rows)
    write(U/'FIT_DIAGNOSTIC.json',dict(status='IN_SAMPLE_DIAGNOSTIC_COMPLETE',models=results,scope=protocol['scope'],new_training=False,no_development_rerun=True))
    for n,r in results.items():print(n,{k:r['groups'][k]for k in ['all','new','state_1']})
if __name__=='__main__':main()
