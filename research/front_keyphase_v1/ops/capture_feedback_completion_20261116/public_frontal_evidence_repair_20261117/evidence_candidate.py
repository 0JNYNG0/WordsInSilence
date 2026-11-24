"""Reuse frozen observers; repair evidence aggregation only (isolated candidate)."""
from pathlib import Path
import sys
D=Path(__file__).resolve().parent;G=D.parent;O=G.parent;N=G/'frontal_scope_recovery_20261117'
sys.path[:0]=[str(G/'public_visibility_roles_20261116'),str(G/'apple_v1')]
from apple_candidate import RoleEvidenceCandidate,adapter,shape_check,np
from visibility import hand_evidence
from frontal_evidence_logic import partial_match,overlap_bridges

class EvidenceLesson(RoleEvidenceCandidate):
    def analyze(self,raw):
        q=super().analyze(raw);ch=adapter.convert(raw['xy_px'],raw['score'],raw['timestamp_us'],raw['frame_id'],int(raw['source_width']),int(raw['source_height']))
        ts=ch['timestamp_us'];replacements={}
        mouth=next((e for e in q['events'] if e['region']==1),None)
        if mouth:
            a,b=mouth['start'],mouth['end'];sh=q['observations']['shape']['R']
            match=partial_match(sh['fingers'],sh['available'],[1,0,0,0])
            c=shape_check('index_shape',match,1,a,b,np.ones(len(ts),bool),ts)
            c['evidence_mode']='known_finger_contradiction_unknown_separate'
            replacements[c['id']]=c
        pair=next((e for e in q['events'] if e['region']==2),None)
        if pair:
            a,b=pair['context'];facts=hand_evidence(ch,'R');states=self.surface.sequence(ch['tcn280'],'R')
            c=next(c.copy() for c in q['criteria'] if c['id']=='wipe_shape')
            base=shape_check('wipe_shape',states,0,a,b,~facts['uncertain'],ts)
            bridges,rejected=overlap_bridges(states,facts,ts,a,b)
            remaining=np.array(facts['uncertain'],copy=True)
            for lo,hi in bridges:remaining[lo:hi+1]=False
            ids=np.flatnonzero(remaining[a:b+1])+a
            runs=np.split(ids,np.flatnonzero(np.diff(ids)>1)+1) if len(ids) else []
            unresolved=[[int(x[0]),int(x[-1])] for x in runs if len(x)>=4]
            if c['status']=='unknown' and base['status']=='supported' and not unresolved:
                c['status']='supported';c.pop('unobserved_spans',None)
            c['bridged_projected_overlap']=bridges;c['unresolved_overlap']=rejected
            c['hidden_finger_state_verified']=False
            c['evidence_mode']='visible_palm_evidence_before_after_short_overlap'
            replacements[c['id']]=c
        q['criteria']=[replacements.get(c['id'],c) for c in q['criteria']]
        q['evidence_repair_version']='public_front_evidence_20261117'
        return q
