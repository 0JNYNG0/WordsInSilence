from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
import json,hashlib,sys
sys.stdout.reconfigure(encoding='utf8')
U=Path(__file__).resolve().parent;O=U.parent;P=O/'word100_point_sequence_split_20261022';F=O/'word100_fractional_phase_alignment_20261026'
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    v=read(U/'VERIFIED.json');assert v['status']=='PASS_EVIDENCE_SELECTION_AND_CANONICAL_METRICS';groups=defaultdict(list)
    for row in v['rows']:
        if row['folder']=='cross' and row['kind'].startswith('skip3'):groups[row['query']+'__'+row['kind']].append(row)
    lost=[];records=[];pins={};combined=Counter()
    for key,rr in groups.items():
        a=sum(x['old_local_detect'] for x in rr);b=sum(x['new_local_detect'] for x in rr)
        if a>=5 and b<5:lost.append(dict(key=key,old_refs=a,new_refs=b))
        total_old=0;total_new=0
        for meta in rr:
            name=meta['name'];cid=meta['query'];rid=meta['reference'];ident='identity__'+cid+'__'+rid
            files=[P/'cross'/(name+'.npz'),P/'cross'/(ident+'.npz'),U/'cases'/('cross__'+name+'.npz'),U/'cases'/('cross__'+ident+'.npz')]
            z,base,n,nb=[load(p) for p in files];pins.update({str(p):sha(p) for p in files})
            removed=np.setdiff1d(np.arange(len(base['source_index'])),z['source_index'].astype(int));refs=np.unique(base['path'][np.isin(base['path'][:,0],removed),1]);local=np.zeros_like(z['expected']);local[refs]=z['expected'][refs]
            ok=local&z['sequence_comparable']&base['sequence_comparable'];oz=z['component_bad'].any(-1);ob=base['component_bad'].any(-1);nz=n['component_bad'].any(-1);nbase=nb['component_bad'].any(-1)
            oldflag=ok&oz&~ob;newflag=ok&nz&~nbase;point=local&z['point_comparable']&base['point_comparable']&z['point_bad']&~base['point_bad']
            total_old+=bool((oldflag|point).any());total_new+=bool((newflag|point).any())
            if not(a>=5 and b<5):continue
            lostcell=oldflag&~newflag
            details=[]
            for j,h,ch in np.argwhere(lostcell):
                why=('variant_difference_no_longer_present' if not nz[j,h,ch] else 'identity_now_already_different')
                details.append(dict(reference_frame=int(j),hand=['L','R'][h],channel=int(ch),group='angle' if ch<15 else 'palm' if ch<17 else 'elbow' if ch<19 else 'wrist',why=why,old_variant_components=z['component_bad'][j,h,ch].tolist(),new_variant_components=n['component_bad'][j,h,ch].tolist(),old_identity_components=base['component_bad'][j,h,ch].tolist(),new_identity_components=nb['component_bad'][j,h,ch].tolist(),variant_source=int(n['source'][j,h,ch]),identity_source=int(nb['source'][j,h,ch])))
            records.append(dict(query=cid,reference=rid,kind=meta['kind'],removed_source_frames=removed.tolist(),local_reference_frames=refs.tolist(),old_sequence_new_channels=int(oldflag.sum()),new_sequence_new_channels=int(newflag.sum()),original_point_new_channels=int(point.sum()),lost_channel_count=int(lostcell.sum()),lost_reasons=dict(Counter(x['why'] for x in details)),lost_groups=dict(Counter(x['group'] for x in details)),details=details,unknown_not_changed=True))
        combined['variants']+=1;combined['old_five']+=total_old>=5;combined['new_five']+=total_new>=5;combined['gained']+=total_old<5 and total_new>=5;combined['lost']+=total_old>=5 and total_new<5
    out=dict(status='SAVED_ARRAY_CAUSE_AUDIT',primary_failure_not_replaced=True,lost_variants=lost,pairs=records,combined_point_sequence_diagnostic=dict(combined),pins=pins,
      interpretation='Separates a genuinely reduced variant discrepancy from a newly higher own-identity discrepancy that hides incremental detection. Neither means proven correct or incorrect signing. The primary frozen sequence nonregression gate remains failed. No class-specific correction or new threshold.',new_inference=0,training=0)
    with (U/'REGRESSION_AUDIT.json').open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
    print('REGRESSION',lost,dict(combined),[(x['reference'],x['lost_reasons'],x['lost_groups'],x['original_point_new_channels']) for x in records],flush=True)
if __name__=='__main__':main()
