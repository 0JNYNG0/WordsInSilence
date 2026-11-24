from short_arrival import *
import copy
def subset(z,ids):
 n=len(z['xy_px']);out={k:(v[ids].copy() if v.ndim and len(v)==n else v.copy()) for k,v in z.items()}
 out['timestamp_us']=np.arange(len(ids),dtype=np.int64)*33333;out['frame_id']=np.arange(len(ids));return out
def main():
 torch.set_num_threads(2);a=FrontalLesson();b=ShortArrivalLesson();rows=[]
 for r in sources():
  if r['word']!='WORD1290' or r['person']not in [1,2]:continue
  raw=dict(np.load(r['raw']));n=len(raw['xy_px'])
  for factor in [1,1.25,1.5,2]:
   ids=np.unique(np.rint(np.arange(0,n-1, factor)).astype(int));z=subset(raw,ids)
   qa=a.analyze(z);qb=b.analyze(z)
   row=dict(id=r['id'],speed=factor,before=len(qa['events']),after=len(qb['events']),events=qb['events'])
   if factor==1.25 and r['person']==2:
    ch=channel(z);m=a.motion.sequence(ch);short=a.motion.sequence(ch,radius=1);s=qa['relation_states']
    print('DETAIL',[(i,s[i],int(m[i]),int(short[i]))for i in range(35,56)],flush=True)
   rows.append(row);print(json.dumps(row),flush=True)
 write(D/'probe/SHORT_FIT.json',rows)
if __name__=='__main__':main()
