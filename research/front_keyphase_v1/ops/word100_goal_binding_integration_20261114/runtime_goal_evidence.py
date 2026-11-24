"""Isolated, explicitly disabled-by-default goal observation extension.

The deployed server does not import this module. The caller must explicitly
provide unnormalized feedback280 observations, native times and 14 web phases.
"""
import copy
import numpy as np
from context_binding import ContextGoalObserver
from observable_goal import assess_observed_goal
from event_trace import combine_components

class ResearchGoalExtension:
    def __init__(self):self.observer=ContextGoalObserver()
    def attach(self,result,x,ts,fid,phase):
        x=np.asarray(x);phase=np.asarray(phase);ts=np.asarray(ts);fid=np.asarray(fid)
        if x.ndim!=2 or x.shape[1]!=280 or phase.shape!=(len(x),14):raise ValueError('feedback280 and WEB14 required; original ROLE11 is incompatible')
        if len(ts)!=len(x) or len(fid)!=len(x) or not np.all(np.diff(ts)>0):raise ValueError('Aligned native timestamps required')
        attached=self.observer.attach(result,x,ts,fid,phase=phase)
        byid={c['id']:c for c in result.get('cards',[])};components=[]
        for c in attached.pop('goal_components'):
            events=[];card=byid[c['card']];motion=card.get('motion_evidence')or {}
            for event in c['events']:
                e=assess_observed_goal(event,ts)
                # Preserve the original scored observation contract, while
                # standardizing the new gap notice to the established UI schema.
                if 'message'in e.get('hint',{}):
                    e['hint']['text_ko']=e['hint'].pop('message')
                    e['hint']['reason']='intervening_observations_insufficient'
                e['joint_evidence']=combine_components(e,motion.get('events',[]),card.get('shape_evidence',[]))
                if e['joint_evidence']['state']=='component_difference_observed':
                    e['hint']=dict(text_ko='도달 위치만으로 통과시키지 않아요. 같은 구간의 손모양·방향 안내를 함께 확인해 주세요.',
                                   reason='other_observed_component_conflict',display_enabled=False,
                                   semantic_completion=None,educational_error=None)
                events.append(e)
            c['events']=events;components.append(c)
        assert attached==result
        out=copy.deepcopy(result);out['goal_observation']=dict(version='isolated_goal_observation_v1',components=components,
            display_enabled=False,research_only=True,whole_action_complete=None,educational_error=None,
            does_not_select_or_delete_actions=True)
        return out
