"""Source-only diagnosis, never supplied to candidate generation or training."""
from graph import *
import numpy as np

def main():
    dest=U/'boundary_candidate_probe/SOURCE_RECALL_AUDIT.json';assert not dest.exists()
    targets=read(U/'reviews/sequence_target/MOTION_EXTENTS.json')['rows']
    predictions={(r['clip_id'],r['kind']):r for r in read(U/'boundary_candidate_probe/RESULTS.json')}
    out=[]
    for target in targets:
        for kind in ['full_normal','dynamic_complete_blocks']:
            key='normal_query_indices' if kind=='full_normal' else 'permuted_query_indices_in_source_order'
            row=predictions[target['clip_id'],kind]
            for name,axis in [('B',0),('C',1)]:
                truth=target['motions'][name];on=set(truth['onset_uncertainty'][key]);off=set(truth['offset_uncertainty'][key]);witness=set(truth['definite_moving_witness'][key]);matched=[]
                splices=[] if kind=='full_normal' else [j['query_right'] for j in target['splice_joins'] if j['synthetic_discontinuity']]
                for c in row['candidates']:
                    if c['axis']!=axis or c['moving']!='R':continue
                    a,b=c['a'],c['b'];cross=any(a<j<=b for j in splices)
                    inside=witness.issubset(range(a,b+1))
                    if not inside or cross:continue
                    matched.append(dict(span=[a,b],source=c['source'],witness_covered=True,
                        onset_in_reviewed_band=a in on,offset_in_reviewed_band=b in off,
                        boundary_extent_match=a in on and b in off,
                        quality_class=c['quality_class'],quality_confidence=c['quality_confidence']))
                out.append(dict(clip_id=target['clip_id'],kind=kind,motion=name,
                    witnessed_motion_proposal_exists=bool(matched),boundary_extent_match_exists=any(v['boundary_extent_match'] for v in matched),candidates=matched))
    write(dest,dict(status='SOURCE_DIAGNOSTIC_ONLY',rows=out,
        counts={k:dict(motions=sum(r['kind']==k for r in out),witnessed_motion=sum(r['kind']==k and r['witnessed_motion_proposal_exists'] for r in out),reviewed_extent=sum(r['kind']==k and r['boundary_extent_match_exists'] for r in out)) for k in ['full_normal','dynamic_complete_blocks']},
        scope='Reviewed source onset/offset uncertainty bands and direct moving witnesses. Not whole-sign validity, human independent gold, or a selector. Post-motion hold is not required to be covered.',
        target_sha256=hashlib.sha256((U/'reviews/sequence_target/MOTION_EXTENTS.json').read_bytes()).hexdigest(),training_use=False))
    print(read(dest)['counts'])

if __name__=='__main__':main()
