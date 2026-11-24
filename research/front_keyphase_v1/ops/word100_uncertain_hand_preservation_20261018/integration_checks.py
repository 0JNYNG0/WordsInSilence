from run import *
from candidate import HandAssigner
def main():
    c=read(U/'CONTEXT.json');ids={x['clip_id'] for x in read(U/'CHANGES.json')}|{'NIA_SL_WORD2177_REAL09_F','NIA_SL_WORD0072_REAL09_F','NIA_SL_WORD0920_REAL09_F'};counts={'api_order_frames':0,'prefix_frames':0,'observation_order_checks':0}
    for r in c['rows']:
        if r['clip_id'] not in ids:continue
        for v in ('masked','sensor'):
            raw=load(r['views'][v]['raw']);saved=load(U/'results'/f"{r['clip_id']}_{v}.npz");model=HandAssigner();reverse=HandAssigner();prefix=HandAssigner();N=len(raw['raw_count']);nf=[x['index'] for x in read(U/'CHANGES.json') if x['clip_id']==r['clip_id'] and x['view']==v];stop=max(N//2,max(nf,default=0)+1)
            for f,n0 in enumerate(raw['raw_count']):
                n=int(n0);hand=raw['hand_xy_px'][f,:n];labels=raw['raw_label'][f,:n];scores=raw['raw_score'][f,:n];pose=raw['pose_xy_px'][f];usable=raw['pose_usable'][f];tail=(pose,usable,int(raw['timestamp_us'][f]),int(raw['width'][f]),int(raw['height'][f]))
                for q in (hand,labels,scores,pose,usable):q.flags.writeable=False
                a,ad=model.assign(hand,labels,scores,*tail);rev,rd=reverse.assign(hand[::-1],labels[::-1],scores[::-1],*tail)
                sides=[-1 if a[i] is None else a[i] for i in range(n)];assert sides==saved['sides'][f,:n].tolist(),(r['clip_id'],v,f,'api');assert sides==[-1 if rev[i] is None else rev[i] for i in reversed(range(n))]
                for k in ('observed_xy','point_mask','local_xy','local_mask','identity_side','identity_state','suggested_side','effective_side'):
                    assert np.array_equal(ad['observation'][k],saved[k][f,:n],equal_nan=True),(r['clip_id'],v,f,k)
                    assert np.array_equal(ad['observation'][k],rd['observation'][k][::-1],equal_nan=True),(r['clip_id'],v,f,k,'order')
                    counts['observation_order_checks']+=1
                pn=int(raw['raw_count'][f-1]) if f else 0
                assert np.array_equal(ad['observation']['possible_links'][:,:pn],rd['observation']['possible_links'][::-1,:pn][:,::-1])
                if f<stop:
                    pp,pd=prefix.assign(hand,labels,scores,*tail);assert a==pp and np.array_equal(pd['observation']['observed_xy'],ad['observation']['observed_xy'],equal_nan=True);counts['prefix_frames']+=1
                counts['api_order_frames']+=1
    write(U/'INTEGRATION_CHECKS.json',{'status':'PASS','public_clips':len(ids),'streams':2*len(ids),'counts':counts,'inputs_readonly':True,'training':0});print('PASS',counts)
if __name__=='__main__':main()
