from model_core import *
from collections import Counter

def main():
    assert not (U/'DATA_READY.json').exists(),'Do not replace completed input preparation'
    ordinary=read(B/'READY.json')['rows']
    added=read(A/'INPUTS.json')['rows']+read(U/'INPUTS_NEW.json')['rows']
    allowed={f'WORD{x:04d}' for x in [2632,2633,2634,2635,2636,2637]}|{'WORD0099'}
    added=[r for r in added if r['word'] in allowed and not (r['word']=='WORD0099' and r['person'] not in [1,2,3])]
    # Apple is admitted as a physical-shape example only after the separate RGB audit.
    extra_words=sorted(allowed);rows=[]
    labels=read(U/'annotation_review/ORDINARY_LABELS.json')['rows']
    assert {r['word'] for r in labels}.issubset({r['word_id'] for r in ordinary})
    labels+=read(U/'annotation_review/EXTERNAL_LABELS.json')['rows']
    labels+=[dict(clip_id=r['clip_id'],frame=r['frame'],side='R',y=[{1:0,0:2,-1:-1}[v] for v in r['finger_states']],source_sha256=r['source_sha256']) for r in read(A/'ADMISSION.json')['rows']]
    for num,m in enumerate(ordinary+added):
        extra=num>=len(ordinary);cid=m['clip_id'];path=U/'cache'/f'{cid}.npz'
        if extra:
            z=np.load(m['channels']);x=z['tcn280'].copy();n=len(x)
            # These extra clips have no historical wrist-boundary calibration.
            # Reuse full inputs on both schedules, not a fabricated crop boundary.
            upper=x.copy()
            a=dict(x_full=old.normalize(x),x_upper=old.normalize(upper),raw_x=x,frame_id=z['frame_id'],timestamp_us=z['timestamp_us'],y=np.zeros((n,7),np.float32),ym=np.zeros((n,7),np.float32),sy=np.full(n,-1,np.int64),sw=np.zeros(n,np.float32),ci=np.full(n,-1,np.int64),stage_gold=np.full(n,-1,np.int64))
            meta=dict(clip_id=cid,word=-1,word_id=m['word'],class_index=-1,extra_index=extra_words.index(m['word']),person=m['person'],role=m['role'],source_sha256=m['source_sha256'],has_labels=False,pairs=[],orders=[],origin='selected_extra',channel=m['channels'],channel_sha256=m['channels_sha256'])
        else:
            assert sha(m['path'])==m['sha256'];z=np.load(m['channel']);base=np.load(m['path']);a={k:base[k].copy() for k in base.files};n=len(a['x_full']);meta=dict(m,origin='common100',extra_index=-1)
        assert sha(meta['channel'])==meta['channel_sha256']
        assert np.array_equal(a['frame_id'],z['frame_id'])
        xy=z['raw_xy_px'] if 'raw_xy_px' in z else None
        if xy is None:
            rec=read(J/'records'/f'{cid}.json');raw=np.load(rec['raw_path']);assert sha(rec['raw_path'])==rec['raw_sha256'];xy=raw['xy_px']
        hh=[];vv=[]
        for sl in (slice(91,112),slice(112,133)):
            h,v=hand_features(xy[:,sl],z['point_available'][:,sl]);hh.append(h);vv.append(v)
        a['hand']=np.stack(hh,1);a['hand_valid']=np.stack(vv,1);a['fy']=np.full((n,2,5),-1,np.int64)
        for l in labels:
            if l['clip_id']!=cid:continue
            assert l['source_sha256']==meta['source_sha256'];ix=np.flatnonzero(a['frame_id']==l['frame']);assert len(ix)==1
            a['fy'][int(ix[0]),0 if l['side']=='L' else 1]=l['y']
        path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**a)
        rows.append({k:v for k,v in dict(meta,path=str(path),sha256=sha(path),finger_known=int((a['fy']>=0).sum())).items() if k not in ['raw','video']})
        if num%200==0:event('cache',done=num+1,total=len(ordinary+added))
    write(U/'DATA_READY.json',dict(rows=rows,extra_words=extra_words,counts=dict(Counter((r['origin']+'_'+r['role']) for r in rows)),fine_labels_pending_aux=False,extra_view='full only; no fabricated rest-wrist boundary'))

if __name__=='__main__':main()
