"""Separate vector comparison against the immediately preceding candidate."""
from run import *
def main():
    c=read(U/'CONTEXT.json');total=0;seen=[]
    for r in c['rows']:
        for v in ('masked','sensor'):
            old=load(T/'results'/f"{r['clip_id']}_{v}.npz");new=load(U/'results'/f"{r['clip_id']}_{v}.npz");raw=load(r['views'][v]['raw']);total+=int(raw['raw_count'].sum())
            delta=old['sides']!=new['sides']
            for f,i in zip(*np.nonzero(delta)):
                assert i<raw['raw_count'][f]
                seen.append((r['clip_id'],v,int(f),int(i),int(old['sides'][f,i]),int(new['sides'][f,i])))
    assert seen==[('NIA_SL_WORD2177_REAL09_F','sensor',41,0,0,-1)],seen
    # The literal above verifies a reported result; it is not an input to the candidate.
    ann=read(U/'ANNOTATED_COMPARISON.json');prev=read(T/'ANNOTATED_COMPARISON.json');before={(x['clip_id'],x['view'],x['index'],x['raw_index']):x for x in prev}
    normals=[x for x in ann if x['verdict']=='normal_identity'];assert len(normals)==441
    for x in normals:assert before[(x['clip_id'],x['view'],x['index'],x['raw_index'])]['new_side']==x['new_side']==x['actual_side']
    assert all(x['preserved'] for x in read(U/'CROSSING_COMPARISON.json')) and all(x['preserved'] for x in read(U/'PREVIOUS4.json'))
    diagnosis=read(U/'REMAINING_DIAGNOSIS.json');assert diagnosis['threshold']==.5
    remaining=[x for x in ann if x['verdict']=='error_identity' and x['new_side'] is not None and x['new_side']!=x['actual_side']];assert len(remaining)==10
    rr={r['clip_id']:r for r in c['rows']};low_normals=0;low_errors=0;pre_core=0
    for x in normals+remaining:
        r=rr[x['clip_id']];z=load(r['views'][x['view']]['raw']);f=x['index'];low=all(float(z['pose_conf'][f,k])<.5 for k in (15,16))
        if x in normals:low_normals+=low
        else:
            low_errors+=low;sec=float(z['timestamp_us'][f])/1000000;pre_core+=sec<min(a for a,b in r['intervals'])
    assert (low_normals,low_errors,pre_core)==(307,9,9)
    pinscheck(c['pins']);pinscheck(read(U/'OUTPUT_PINS.json'))
    write(U/'DELTA_INDEPENDENT.json',{'status':'PASS','raw_candidates':total,'changed_vs_immediate_previous':len(seen),'withheld':1,'reassigned':0,'normal441_preserved':441,'crossing315_preserved':315,'previous4_preserved':4,'remaining_errors':10,'remaining_errors_both_wrist_low':9,'remaining_errors_before_core':9,'normal_cases_both_wrist_low':307,'limitation':'Same operator; independent numeric calculation, not independent human annotations.'})
    print('delta, preservation and descriptive diagnosis PASS')
if __name__=='__main__':main()
