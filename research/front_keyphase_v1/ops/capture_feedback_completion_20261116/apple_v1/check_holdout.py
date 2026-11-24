from verified_candidate import *
from check_controls import selection,replace_hand
import copy
def main():
 torch.set_num_threads(2);dest=U/'holdout';assert not(dest/'SOURCE_TRUTH.json').exists();truth={16:dict(mouth=[38,53],wipe=[64,72],before=61),17:dict(mouth=[49,62],wipe=[71,75],before=69)}
 write(dest/'SOURCE_TRUTH.json',dict(rows=truth,review='Original cropped sequential panels p16_native/p17_native before candidate prediction. Both normal index-to-palm-wipe sequence; left support hidden during later overlap, individual hidden fingers not labelled.',exact_contact=None))
 net=VerifiedCandidate();rows=[]
 for r in read(dest/'EXTRACTED.json')['rows']:
  raw=dict(np.load(r['raw_path']));n=len(raw['frame_id']);t=truth[r['person']];cases=[('normal',raw,True)]
  for name,end in[('first_only',t['mouth'][1]),('before_descent',t['before'])]:cases.append((name,selection(raw,np.arange(end+1)),False))
  cases.append(('constant',selection(raw,np.full(n,t['mouth'][1])),False));z=copy.deepcopy(raw);z['xy_px'][:,91:133]=np.nan;cases.append(('missing',z,False))
  for sc in[.65,1.6]:
   z=copy.deepcopy(raw);z['timestamp_us']=(z['timestamp_us']*sc).astype(np.int64);cases.append((f'time_{sc}',z,True))
  for name,z,expected in cases:
   q=net.analyze(z);write(dest/f'result_{r["person"]}_{name}.json',q);cs={c['id']:c['status']for c in q['criteria']};full=all(v=='supported'for v in cs.values());rows.append(dict(person=r['person'],case=name,full=full,expected=expected,check=full==expected and ('difference'not in cs.values()if expected else True),criteria=cs,seconds=q['seconds']))
 write(dest/'RESULT.json',dict(rows=rows,passed=sum(r['check']for r in rows),total=len(rows),candidate_changed=False,whole_word_certification=False));print(json.dumps(rows,ensure_ascii=False))
if __name__=='__main__':main()
