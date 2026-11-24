from pathlib import Path
import json,statistics
from collections import Counter
import numpy as np
U=Path(__file__).resolve().parent;OLD=U.parent/'word100_multiref_observation_compare_20261019'
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
def dist(v):
    return {'n':len(v),'median':float(np.median(v)) if v else None,'p90':float(np.quantile(v,.9)) if v else None,'max':max(v) if v else None}
def main():
    assert read(U/'VERIFICATION.json')['status']=='PASS';result={};rows=[];queries=[]
    for view in ('legacy','masked','sensor'):
        c=Counter();sidec={'L':Counter(),'R':Counter()};anglec=Counter();traj=[];anchor=[];absolute=[];examples=[];source=Counter()
        for p in sorted((U/'words'/view).glob('*.json')):
            w=read(p);old=read(OLD/'words'/view/p.name);oldq={q['clip_id']:q for q in old['queries']}
            for q in w['queries']:
                if q['role']!='development_only':continue
                row={'view':view,'clip_id':q['clip_id'],'label':w['label'],'word_code':w['word_code'],'sides':{},'role_known':w['role_known']}
                for side,d in q['sides'].items():
                    prior=next(x for x in next(m for m in oldq[q['clip_id']]['metrics'] if m['key']=='handshape')['sides'] if x['side']==side)
                    was=prior['value'] is not None;whole=d['whole_hand'] is not None
                    c['hand_units']+=1;c['old_whole']+=was;c['new_whole']+=whole;c['partial']+=0<d['angles_ready']<15;c['unavailable']+=d['angles_ready']==0
                    c['ready_angles']+=d['angles_ready'];c['angle_units']+=15;c['old_ready_angle_proxy']+=15*was
                    c['whole_gained']+=whole and not was;c['whole_lost']+=was and not whole
                    sidec[side].update(old_whole=int(was),new_whole=int(whole),partial=int(0<d['angles_ready']<15),unavailable=int(d['angles_ready']==0),hand_units=1)
                    for j,val in enumerate(d['angles']):anglec[str(j)]+=val is not None
                    td=d['trajectory'];row['sides'][side]={'old_refs':prior['valid_references'],'new_refs':d['valid_reference_counts'],'angles_ready':d['angles_ready'],'old_whole_measurable':was,'new_whole_measurable':whole,'trajectory':td}
                    if td['old'] is not None:
                        traj.append(td['paired_start']-td['old']);absolute.append(abs(td['paired_start']-td['old']));anchor.append(td['anchor_difference'])
                    rows.append({'view':view,'clip_id':q['clip_id'],'word_code':w['word_code'],'label':w['label'],'side':side,'old':was,'new':whole,'ready_angles':d['angles_ready'],'old_refs':prior['valid_references'],'angle_refs':d['valid_reference_counts']})
                queries.append(row)
        for p in (U/'inputs'/view).glob('*.json'):
            d=read(p)
            for side,x in d['sides'].items():
                pre=d['role']+'_'+side+'_'
                for k in ('full_frames','view_local_no_points','internal_gap_frames','annotated_internal_gap_frames','source_inside_view_missing','annotated_source_inside_view_missing','lowered_observed_missing_in_view'):source[pre+k]+=x[k]
                source[pre+'new_angle_cells']+=x['new_angle_cells'];source[pre+'old_angle_cells']+=x['old_angle_cells']
        result[view]={'DEV':dict(c),'DEV_by_side':sidec,'DEV_by_angle':dict(anglec),'trajectory_common_measurable':{'signed_change':dist(traj),'absolute_change':dist(absolute),'origin_offset':dist(anchor)},'source_diagnostics':dict(source)}
        assert c['hand_units']==591 and c['whole_lost']==0
    write(U/'SUMMARY.json',{'status':'COMPLETE','views':result,'training':0,'private_data_read':0,'production_changed':False,'units':'angle_units=DEV300 x retained hand roles(591) x15=8865; full hand not correctness','limits':['Source rest witness requires full native view; not available to actual cropped webcam','Both old and new rely on estimated MediaPipe points; not anatomical truth','All15 angle output still may omit short distinctive phases; no semantic pass','New whole hand mean of per-angle medians differs from old aggregation; do not compare as accuracy','Paired start diagnostics can erase a true initial displacement; no production adoption','DEV previously used; no fresh holdout']})
    write(U/'DEV_ROWS.json',rows);write(U/'DEV_QUERY_DETAILS.json',queries)
    print(json.dumps({v:x['DEV'] for v,x in result.items()},ensure_ascii=False))
if __name__=='__main__':main()
