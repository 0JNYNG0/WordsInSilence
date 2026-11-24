from hand_core import *
from collections import Counter,defaultdict

def main():
    U.mkdir(exist_ok=True);assert not(U/'INPUTS_COMPLETE.json').exists()
    prop=read(T/'LEARNING_PROPOSAL.json'); review=read(T/'REVIEW_COMPLETE.json')
    sourcepaths=[Path(r['path']) for r in prop['prior_explicit_public_fit_sources']]
    pins={str(p):sha(p) for p in sourcepaths+[T/'LEARNING_PROPOSAL.json',T/'REVIEW_COMPLETE.json',T/'ROOT_VISUAL_REVIEW.json',Path(__file__),U/'hand_core.py']}
    ready_path=OPS/'word100_shared_motion_learning_20261109/READY.json'
    ready={r['clip_id']:r for r in read(ready_path)['rows'] if r['role']=='fit' and 1<=r['person']<=7}
    pins[str(ready_path)]=sha(ready_path)
    cells={};prov=defaultdict(list)
    def put(cid,frame,side,attr,value,origin):
        if cid not in ready:return
        key=(cid,int(frame),side,int(attr))
        assert key not in cells or cells[key]==value,('LABEL_CONFLICT',key,cells.get(key),value)
        cells[key]=float(value);prov[key].append(origin)
    feature_map={0:('R',2),1:('R',0),2:('L',0),3:('R',1),4:('L',1)}
    def old(cid,frame,attrs,source):
        for k,y in attrs.items():
            if int(k) in feature_map:
                side,j=feature_map[int(k)];put(cid,frame,side,j,y,str(source))
    for r in read(sourcepaths[0])['rows']:
        if r['role']=='fit':
            for a in r['shape']:old(r['clip_id'],a['frame'],a['attributes'],sourcepaths[0])
    for r in read(sourcepaths[1])['rows']:old(r['clip_id'],r['frame'],r['attributes'],sourcepaths[1])
    for cid,rows in read(sourcepaths[2])['shape'].items():
        for r in rows:old(cid,r['frame'],r['features'],sourcepaths[2])
    before=len(cells);assert before==434
    dense={}
    for r in review['rows']:
        cid=r['clip_id'];dense[cid]={x['frame']:x for x in r['frame_records']}
        for f,row in dense[cid].items():
            state=row['hand_state']
            if state=='unknown':continue
            vals={'open':{0:1,1:0,3:0},'closed':{0:0,1:1,3:0},'flexing':{0:0,1:0,3:1}}[state]
            for j,y in vals.items():put(cid,f,'R',j,y,str(T/'REVIEW_COMPLETE.json'))
    labels=[]; sequences=[];sources={};obs_audit=[]
    for cid in sorted({k[0] for k in cells}|set(dense)):
        r=ready[cid];assert sha(r['path'])==r['sha256'] and sha(r['channel'])==r['channel_sha256']
        z=np.load(r['path']);ch=np.load(r['channel']);frames=ch['frame_id'];ts=ch['timestamp_us'];x=ch['full']
        assert np.array_equal(frames,z['frame_id']) and np.array_equal(ts,z['timestamp_us'])
        assert np.all(np.diff(frames)==1) and np.all(np.diff(ts)>0)
        pins[r['path']]=r['sha256'];pins[r['channel']]=r['channel_sha256'];sources[cid]=dict(r)
        for side in ('L','R'):
            y=np.zeros((len(x),4),np.float32);mask=np.zeros_like(y)
            inp=hand_input(x,side)
            # Canonical reflection concerns only the hand-local coordinate basis.
            if side=='L':inp[:,:42:2]*=-1
            available=(inp[:,42+np.array([0]+list(range(5,21)))]>.5).all(1)
            for k,v in cells.items():
                if k[0]!=cid or k[2]!=side:continue
                f,j=k[1],k[3];ix=np.flatnonzero(frames==f);assert len(ix)==1;i=int(ix[0])
                y[i,j]=v;mask[i,j]=1
                labels.append(dict(clip_id=cid,person=r['person'],word_id=r['word_id'],frame=f,index=i,side=side,attribute=j,target=v,available=bool(available[i]),provenance=prov[k]))
            if not mask.any():continue
            p=U/'sources'/f'{cid}_{side}.npz';p.parent.mkdir(exist_ok=True)
            np.savez_compressed(p,x=inp,y=y,mask=mask,available=available,frames=frames,timestamp_us=ts)
            sequences.append(dict(clip_id=cid,person=r['person'],word_id=r['word_id'],side=side,path=str(p),sha256=sha(p),source_sha256=r['source_sha256'],label_cells=int(mask.sum()),available_cells=int((mask*available[:,None]).sum())))
            if cid in dense and side=='R':
                for f,rr in dense[cid].items():
                    i=int(np.flatnonzero(frames==f)[0]);pts=inp[i,:42].reshape(21,2)
                    # Descriptive measurements, never shape truth or angle thresholds.
                    tip_mcp=[float(np.linalg.norm(pts[t]-pts[m])) for t,m in [(8,5),(12,9),(16,13),(20,17)]]
                    obs_audit.append(dict(clip_id=cid,person=r['person'],frame=f,index=i,hand_state=rr['hand_state'],body_motion=rr['body_motion'],available=bool(available[i]),tip_mcp_distances=tip_mcp))
    seqmap={(r['clip_id'],r['side']):r for r in sequences};windows=[];controls=[]
    state_value={'open':0,'flexing':1,'closed':2}
    for cid,dm in dense.items():
        seq=seqmap[cid,'R'];z=np.load(seq['path']);lookup={int(f):i for i,f in enumerate(z['frames'])};lo=min(dm);hi=max(dm)
        for a in range(lo,hi+1):
            for b in range(a+3,min(a+24,hi+1)):
                rr=[dm[f] for f in range(a,b+1)];ss=[r['hand_state'] for r in rr]
                known=[state_value[s] for s in ss if s!='unknown'];aidx,bidx=lookup[a],lookup[b]
                if not z['available'][aidx:bidx+1].all():continue
                target=None;scope=None
                if all(s in ('open','closed') and s==ss[0] for s in ss):target=1;scope='direct_stable_handshape'
                elif ss[0]!='unknown' and ss[-1]!='unknown' and 'flexing' in ss and all(v<=w for v,w in zip(known,known[1:])) and known[0]<known[-1]:
                    target=2;scope='full_state_change' if ss[0]=='open' and ss[-1]=='closed' else 'partial_state_change'
                if target is None:continue
                for reverse in (False,True):
                    windows.append(dict(clip_id=cid,person=seq['person'],side='R',span=[aidx,bidx],source_frames=[a,b],reverse=reverse,target=2-target if reverse else target,kind=scope,body_raise_stable=target==1 and any(r['body_motion']=='forearm_raising' for r in rr),source_sha256=seq['source_sha256']))
                if target==2:
                    for anchor in [aidx,(aidx+bidx)//2,bidx]:
                        controls.append(dict(clip_id=cid,person=seq['person'],side='R',span=[aidx,bidx],source_frames=[a,b],reverse=False,target=1,kind='synthetic_repeated_pose',anchor=anchor,source_sha256=seq['source_sha256']))
    # Freeze the plan before any model outputs, including eligibility and folds.
    config=dict(at=stamp(),model='Shared hand-local63 -> spatial48 -> temporalConv32 -> biGRU16 ->4maskedattributes +3sequenceoutputs',
        scope='New isolated observer branch, not modification or replacement of common100/body/motion/oldhandshape; not whole-model fine-tuning.',
        folds=list(range(2,8)),seeds=[42,43,44],steps=600,batch_attributes=24,batch_transitions=24,lr=.001,weight_decay=.001,
        initialization='Deterministic new small branch, same frozen recipe in six leave-person-out diagnostics; all old modules immutable.',
        training='Exclude heldperson across all expressions/sides/windows. Perattribute and positive/negative balanced supervised cells; pertransition class and person balanced, stable real/synthetic equally sampled. BCE masked attributes + CE transition, equal aggregate weights. Final checkpoint only.',
        attribute_context='Each labelled center uses original +/-3 frames, only center attribute losses. Sequence training also receives direct attribute labels inside its own interval; unknown masks0. No full-clip hidden state.',
        normalization='Raw wrist-centred hand-scale coordinates; reflect left local x to right convention, then mean/std from training source rows only; availability unstandardized.',
        sequence_target='Observed hand-state change only: opening0/stable1/closing2. Full/partial kept separate; closed hand during armraise is stable handshape, not still body. Reverse is contrast, never correctword.',
        controls='Repeated identical hand pose with same sequence length, masks; no observed state change. Not new normal KSL recordings.',
        gates='New branch diagnostic only. Before downstream integration: six heldpeople closing/opening controls and closed-hand-armraise must separate; missing-hand emits unknown; old models untouched. Fold scores cannot select thresholds/seed/epoch. Final allFIT training requires reviewed diagnostic first.',
        forbidden_inputs=['word_id','expected_order','body_coordinates','clip_time','guide_expected_shape','outside-window latent'],
        current_hand_masks='Availability not point accuracy. Unavailable labelled cells remain in audit and abstention denominator, excluded from supervised loss.',
        thresholds='Attributes logits0 for masked classification audit; probabilities .65/.35 for optional support/contradiction. Sequence argmax for confusion, maxprob>=.65 and observed endpoints for emitted evidence, otherwiseunknown.',
        upstream_fit_exposed=True,final8=False,private_access=False,web_changed=False,pins=pins)
    assert not(U/'FROZEN.json').exists();write(U/'FROZEN.json',config)
    write(U/'INPUTS_COMPLETE.json',dict(at=stamp(),sequences=sequences,labels=labels,windows=windows,controls=controls,
        counts=dict(source_clips=len(sources),people=sorted({r['person'] for r in sequences}),expressions=len({r['word_id'] for r in sequences}),prior_cells=before,combined_cells=len(labels),unavailable_cells=sum(not r['available'] for r in labels),new206=len(obs_audit),windows=len(windows),controls=len(controls)),
        sources=sources,frame_observation_audit=obs_audit,source_pins=pins,educational_correctness=None))
    print(read(U/'INPUTS_COMPLETE.json')['counts'],flush=True)
    print('window classes',dict(Counter((r['kind'],r['target']) for r in windows)),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:preserve_failure('PREP');raise
