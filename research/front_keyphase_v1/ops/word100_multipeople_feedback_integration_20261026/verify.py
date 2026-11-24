"""Separate arithmetic reconstruction; never calls run.pair or consumer.build."""
import run as R
import numpy as np
from collections import Counter,defaultdict

BOUNDS=[(0,15),(15,17),(17,19),(19,21),(21,23)]
NAMES=['finger_angles','palm_position','elbow_position','pose_wrist_position','hand_wrist_position']

def eq(a,b):
 assert np.array_equal(np.asarray(a),np.asarray(b),equal_nan=True)
def close(a,b):
 assert np.allclose(a,b,atol=2e-5,rtol=1e-6,equal_nan=True)
def mean(x):
 x=np.asarray(x,np.float64);good=np.isfinite(x)
 return float(np.where(good,x,0).sum()/good.sum()) if good.any() else None
def number(a,b):
 if b is None:assert a is None
 else:assert a is not None and abs(a-b)<1e-9*max(1.,abs(b))
def intervals(items,mask,t):
 rebuilt=np.zeros(len(mask),bool);last=-2
 for x in items:
  a,b=x['first'],x['last'];assert 0<=a<=b<len(mask) and a>last+1
  assert x['samples']==b-a+1 and x['reference_start_s']==float(t[a]/1e6) and x['reference_end_s']==float(t[b]/1e6)
  rebuilt[a:b+1]=True;last=b
 eq(rebuilt,mask)

def independent(z,ix,lo,hi,domain):
 n=len(ix);v=np.full((n,2,23),np.nan);valid=np.zeros((n,2,23),bool);quality=valid.copy()
 for j,s in enumerate(ix):
  a,b=int(np.floor(s)),int(np.ceil(s));w=s-a
  valid[j]=z['available'][a]&z['available'][b]
  quality[j]=z['quality_eligible'][a]&z['quality_eligible'][b]
  v[j]=np.where(valid[j],(1-w)*z['values'][a]+w*z['values'][b],np.nan)
 desc=np.full((n,2,23,5),np.nan);ok=valid.copy()&False;q=ok.copy()
 for j in range(2,n-2):
  window=slice(int(lo[j-2]),int(hi[j+2])+1)
  ok[j]=np.logical_and.reduce(domain[j-2:j+3])&np.logical_and.reduce(z['available'][window])&valid[j-2]&valid[j]&valid[j+2]
  q[j]=ok[j]&np.logical_and.reduce(z['quality_eligible'][window])&quality[j-2]&quality[j]&quality[j+2]
  desc[j]=np.stack([v[j-2],v[j],v[j+2],v[j]-v[j-2],v[j+2]-v[j]],axis=-1)
 desc[~ok]=np.nan
 return v,valid,desc,ok,q

def main():
 f=R.read(R.U/'FROZEN.json');result=R.read(R.U/'RESULT.json');R.check(f['pins']);R.check(f['protected']);R.check(result['pins']);assert not(R.U/'RUNNING.json').exists()
 assert len(result['rows'])==600 and result['query_count']==300
 samples={x['clip_id']:R.load(x['output']) for x in f['rows']};meta={x['clip_id']:x for x in f['rows']};score={k:{x['clip_id']:{a:x[a] for a in ('status','prediction','top5','correct','top5_correct')} for x in R.read(p)['predictions']['masked']} for k,p in f['scores'].items()}
 count=Counter();totals=defaultdict(Counter);paired=defaultdict(list);seen=set();queryrefs=defaultdict(set);disagreements=[]
 for ri,row in enumerate(result['rows']):
  qid,rid=row['query'],row['reference'];assert (qid,rid) not in seen;seen.add((qid,rid));queryrefs[qid].add(rid)
  assert meta[qid]['word_code']==meta[rid]['word_code']==row['word_code'] and meta[qid]['signer_id']!=meta[rid]['signer_id']
  q,r=samples[qid],samples[rid];z=R.load(row['pair']);receipt=R.read(row['receipt']);n=len(r['core']);path=z['path'];step=np.diff(path,axis=0)
  eq(path[0],[0,0]);eq(path[-1],[len(q['core'])-1,n-1]);assert ((step>=0)&(step<=1)).all() and (step.sum(1)>0).all()
  ix=np.array([np.median(path[path[:,1]==j,0]) for j in range(n)]);lo=np.array([path[path[:,1]==j,0].min() for j in range(n)]);hi=np.array([path[path[:,1]==j,0].max() for j in range(n)])
  new=ix.copy()
  for j in range(2,n-2):new[j]=np.clip(np.dot(ix[j-2:j+3],[1,4,6,4,1])/16,ix[1],ix[-2])
  new=np.minimum(np.maximum(new,ix-1),ix+1);eq(z['phase_old'],ix);eq(z['phase_new'],new);eq(z['span_first'],lo);eq(z['span_last'],hi);assert (np.diff(new)>=0).all()
  domain=np.repeat(np.repeat(r['core'][:,None,None],2,1),23,2);eq(z['domain'],domain)
  ov,op,od,ook,oq=independent(q,ix,lo,hi,domain);nv,npresent,nd,nok,nq=independent(q,new,np.minimum(lo,np.floor(new).astype(int)),np.maximum(hi,np.ceil(new).astype(int)),domain)
  rv,rp,rd,rok,rq=independent(r,np.arange(n,dtype=float),np.arange(n),np.arange(n),domain)
  source=np.zeros(domain.shape,np.uint8);source[ook]=2;source[nok]=1
  rawq=np.where(source==1,nq,oq)&rq;keep=oq&rq&~rawq;source[keep]=2
  ok=(source>0)&rok;quality=np.where(source==1,nq,oq)&rq&ok;point=domain&op&r['available']
  eq(z['source'],source);eq(z['quality_fallback'],keep);eq(z['sequence_old_ok'],ook&rok);eq(z['sequence_ok'],ok);eq(z['sequence_old_quality'],oq&rq);eq(z['sequence_quality'],quality);eq(z['point_ok'],point)
  close(z['point_delta'],np.where(point,abs(ov-r['values']),np.nan));close(z['sequence_old_delta'],np.where((ook&rok)[...,None],abs(od-rd),np.nan));close(z['sequence_delta'],np.where(ok[...,None],abs(np.where((source==1)[...,None],nd,od)-rd),np.nan))
  assert not((ook&rok)&~ok).any() and not((oq&rq)&~quality).any();eq(z['reference_missing'],domain&~r['available']);eq(z['reference_times_us'],r['timestamp_us']);eq(z['query_times_us'],q['timestamp_us'])
  assert receipt['query_id']==qid and receipt['reference_id']==rid and receipt['recognition']=={k:v[qid] for k,v in score.items()}
  assert receipt['whole_gesture_correct'] is None and receipt['skill_score'] is None and receipt['validated_correction'] is None
  assert receipt['research_source']['reference_type']=='different_public_id_same_word' and not receipt['research_source']['anatomical_truth']
  times=r['timestamp_us'];tc=receipt['time_correspondence'];eq(tc['reference_seconds'],times/1e6);close(tc['original_query_seconds'],np.interp(ix,np.arange(len(q['timestamp_us'])),q['timestamp_us']/1e6));close(tc['refined_query_seconds'],np.interp(new,np.arange(len(q['timestamp_us'])),q['timestamp_us']/1e6))
  for h in range(2):
   hand=receipt['hands'][h];assert hand['side']==('L','R')[h]
   for gi,(a,b) in enumerate(BOUNDS):
    g=hand['groups'][gi];ex=domain[:,h,a:b];p=point[:,h,a:b];s=ok[:,h,a:b];qual=quality[:,h,a:b];pd=z['point_delta'][:,h,a:b];sd=z['sequence_delta'][:,h,a:b];old=z['sequence_old_delta'][:,h,a:b];oldok=z['sequence_old_ok'][:,h,a:b]
    assert g['group']==NAMES[gi];counts=[int(v.sum()) for v in (ex,p,s,qual)]
    for key,val in zip(['audit_cells','point_compared_cells','sequence_compared_cells','corroborated_sequence_cells'],counts):assert g[key]==val
    for key,val in zip(['point_coverage','sequence_coverage','corroborated_coverage'],counts[1:]):number(g[key],val/counts[0] if counts[0] else None)
    for key,val in [('point_mean_difference',pd[p]),('sequence_level_mean_difference',sd[s,:3]),('sequence_change_mean_difference',sd[s,3:]),('corroborated_level_mean_difference',sd[qual,:3]),('corroborated_change_mean_difference',sd[qual,3:])]:number(g[key],mean(val))
    src=source[:,h,a:b];assert g['source_cells']==dict(refined=int((s&(src==1)).sum()),retained_original=int((s&(src==2)).sum()))
    for key,mask in [('missing_point_intervals',(ex&~p).any(1)),('sequence_context_unavailable_intervals',(ex&p&~s).any(1)),('uncorroborated_intervals',(s&~qual).any(1)),('reference_missing_intervals',z['reference_missing'][:,h,a:b].any(1))]:intervals(g[key],mask,times)
    tl=g['timeline'];eq(tl['reference_seconds'],times/1e6)
    for key,v in [('audit_channels',ex),('point_channels',p),('sequence_channels',s),('corroborated_channels',qual)]:eq(tl[key],v.sum(1))
    for j in range(n):
     number(tl['point_difference'][j],mean(pd[j,p[j]]));number(tl['sequence_level_difference'][j],mean(sd[j,s[j],:3]));number(tl['sequence_change_difference'][j],mean(sd[j,s[j],3:]))
    assert g['gesture_correct'] is None and g['correction_direction'] is None
    c=totals[NAMES[gi]]
    for key,val in zip(['domain','point','sequence','quality'],counts):c[key]+=val
    c['old_sequence']+=int(oldok.sum());c['old_quality']+=int(z['sequence_old_quality'][:,h,a:b].sum())
    for part,sl in [('level',slice(0,3)),('change',slice(3,5))]:
     for version,arr in [('old',old),('new',sd)]:c[version+'_'+part+'_sum']+=float(arr[oldok,sl].astype(float).sum())
     c[part+'_count']+=int(oldok.sum())*(3 if part=='level' else 2)
    paired[(qid,h,gi)].append(dict(reference=rid,point=mean(pd[p]),sequence_coverage=g['sequence_coverage'],quality_coverage=g['corroborated_coverage']))
   for a,g in enumerate(hand['angles']):
    ex=domain[:,h,a];p=point[:,h,a];s=ok[:,h,a];qual=quality[:,h,a]
    for key,v in [('audit_cells',ex),('point_compared_cells',p),('sequence_compared_cells',s),('corroborated_cells',qual)]:assert g[key]==int(v.sum())
    number(g['point_mean_degrees'],mean(z['point_delta'][:,h,a][p]));number(g['sequence_level_mean_degrees'],mean(z['sequence_delta'][:,h,a,:3][s]));number(g['sequence_change_mean_degrees'],mean(z['sequence_delta'][:,h,a,3:][s]))
    for key,mask in [('missing_point_intervals',ex&~p),('sequence_context_unavailable_intervals',ex&p&~s),('uncorroborated_intervals',s&~qual)]:intervals(g[key],mask,times)
  count['pairs']+=1;count['quality_fallback']+=int(keep.sum());count['timeline_cells']+=n*10
  if (ri+1)%100==0:print('VERIFIED_REAL_PAIRS',ri+1,600,flush=True)
 assert seen=={(x['query'],x['reference']) for x in f['pairs']} and len(queryrefs)==300 and all(len(x)==2 for x in queryrefs.values())
 ranks=Counter()
 for qid,refs in queryrefs.items():
  qr=R.read(R.U/'queries'/(qid+'.json'));assert qr['reference_aggregation']=='none;each preserved' and {x['reference'] for x in qr['references']}==refs and qr['normal_tolerance'] is None and qr['whole_gesture_correct'] is None
  assert qr['recognition']=={k:v[qid] for k,v in score.items()}
  for k,v in qr['recognition'].items():ranks[k+'_correct']+=v['correct'];ranks[k+'_top5']+=v['top5_correct']
 assert ranks['C_correct']==294 and ranks['T3F_correct']==295
 for (qid,h,gi),arr in paired.items():
  assert len(arr)==2
  if all(x['point'] is not None for x in arr):disagreements.append(dict(query=qid,label=meta[qid]['label'],side=('L','R')[h],group=NAMES[gi],unit=('projected_degrees' if gi==0 else 'shoulder_width'),spread=abs(arr[0]['point']-arr[1]['point']),references=arr))
 summaries={}
 for g,t in totals.items():
  d=dict(t)
  for part in ('level','change'):
   for v in ('old','new'):d[v+'_'+part+'_mean']=t[v+'_'+part+'_sum']/t[part+'_count'] if t[part+'_count'] else None
  summaries[g]=d
 R.check(f['pins']);R.check(result['pins']);R.check(f['protected']);R.write(R.U/'VERIFIED.json',dict(at=R.now(),status='PASS_REAL600_EVIDENCE_PRESERVATION',counts=dict(count),groups=summaries,saved_original_predictions_unique300=dict(ranks),reference_disagreements=sorted(disagreements,key=lambda x:x['spread'],reverse=True),
  verdict='Research integration and arithmetic verified; no anatomical/correctness/timing ground truth across people; no improvement claim from lower residual alone; two named references preserved,not pooled.',training=0,inference=0,production_changed=False,verifier=R.sha(__file__),result_sha256=R.sha(R.U/'RESULT.json')));print('VERIFIED_REAL600',dict(count),dict(ranks),flush=True)

if __name__=='__main__':
 try:main()
 except BaseException:
  import traceback
  p=R.U/'VERIFY_FIRST_ERROR.txt'
  if not p.exists():p.write_text(traceback.format_exc(),encoding='utf-8')
  raise
