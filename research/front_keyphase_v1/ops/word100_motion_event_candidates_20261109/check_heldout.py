from core import *
def main():
 assert not (N/'HELDOUT_CHECK.json').exists();torch.set_num_threads(4);rows=load_rows();old={(r['clip_id'],r['kind']):r for r in read(U/'INPUTS.json')['rows']};allrows=[]
 for fold in read(N/'FROZEN.json')['folds']:
  name='heldout_'+'_'.join(map(str,fold));ck=torch.load(N/'runs'/name/'model.pt',weights_only=True,map_location='cpu');m=MotionProposal();m.load_state_dict(ck['model']);m.eval();completion=read(N/'runs'/name/'COMPLETE.json');assert not set(completion['fit_people'])&set(fold)
  with torch.inference_mode():
   for r in rows:
    if r['person'] not in fold or r['kind'] not in ['full_normal','upper_normal','actual_crop']:continue
    z=np.load(old[(r['clip_id'],'full_normal')]['path']);gold=z['stage_gold'];g=normalize(r['g'],ck['mean'].numpy(),ck['std'].numpy());p=m(torch.tensor(g[None]),torch.tensor([len(g)]))[0].softmax(-1).numpy();points=[]
    for i in np.flatnonzero(gold>=0):points.append(dict(index=int(i),frame=int(r['frame_id'][i]),target=MAP[int(gold[i])],pred=int(p[i].argmax()),confidence=float(p[i].max())))
    path=N/'heldout_predictions'/f"{name}_{r['clip_id']}_{r['kind']}.npz";path.parent.mkdir(exist_ok=True);np.savez_compressed(path,phase=p)
    allrows.append(dict(model=name,clip_id=r['clip_id'],person=r['person'],kind=r['kind'],records=points,predictions=str(path),sha256=sha(path)))
 summaries={}
 for kind in sorted({r['kind'] for r in allrows}):
  rr=[r for r in allrows if r['kind']==kind];records=[q for r in rr for q in r['records']];summaries[kind]=dict(clips=len(rr),clips_with_direct_gold=sum(bool(r['records']) for r in rr),correct=sum(r['pred']==r['target'] for r in records),total=len(records))
 write(N/'HELDOUT_CHECK.json',dict(rows=allrows,by_kind=summaries,scope='New geometry branch only; project initialization excludes heldout people. Sparse same-Codex anchors, not continuous boundary/expert truth. Old deployed shape branch not person-independent in this cohort.',no_model_selection=True));event('heldout_checked',summary=summaries)
if __name__=='__main__':main()
