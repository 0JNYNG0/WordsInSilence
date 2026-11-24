"""Behavioral contracts: real proposer integration and isolated conflict cases."""
import json
import numpy as np
from pathlib import Path
from candidate import HandAssigner,EvidenceTrack,PALM
U=Path(__file__).resolve().parent
def hand(x,y):
 h=np.array([[0,0],[3,-6],[6,-12],[9,-18],[12,-24],[-12,-15],[-12,-28],[-12,-40],[-12,-52],[0,-18],[0,-32],[0,-47],[0,-61],[12,-16],[12,-30],[12,-43],[12,-56],[24,-12],[24,-24],[24,-36],[24,-48]],float)
 return h+[x,y]
def body():
 p=np.zeros((33,2));p[11]=[700,150];p[12]=[300,150];p[15]=[600,400];p[16]=[400,400]
 u=np.zeros(33,bool);u[[11,12]]=True;return p,u
def call(a,hh,lab,t,p=None,u=None):
 if p is None:p,u=body()
 return a.assign(np.asarray(hh,float).reshape(-1,21,2),lab,[.99]*len(lab),p,u,t,1200,900)

def run():
 tests=[];a=HandAssigner();h=hand(600,400)
 for f in range(8):
  out,d=call(a,[h],[0],f*33333)
  assert out=={0:0} and d['identity_state']==['provisional'] and not a.confirmed
 assert d['origin'][0]['frame']==0 and d['parent_frame'][0]==6
 tests.append('repeated_label_seed_never_self_confirms')
 p,u=body();u[15]=True
 out,d=call(a,[h],[0],8*33333,p,u)
 assert d['identity_state']==['confirmed'] and d['origin'][0]['frame']==8
 tests.append('new_independent_pose_can_confirm')
 out,d=call(a,[h],[1],9*33333)
 assert out=={0:0} and d['identity_state']==['confirmed'] and d['origin'][0]['frame']==8
 tests.append('confirmed_history_survives_raw_label_flip')
 empty,d=call(a,[],[],2_000_000);assert not empty and not a.confirmed and not a.provisional
 tests.append('empty_and_expired_state_do_not_create_hands')
 call(a,[h],[0],3_000_000,p,u)
 out,d=call(a,[hand(400,400)],[1],0)
 assert out=={0:1} and d['identity_state']==['provisional'] and a.frame==0
 tests.append('nonmonotonic_time_clears_identity')
 a.reset();assert not a.confirmed and not a.provisional and a.frame==-1
 tests.append('clip_reset_has_no_identity_carryover')
 a=HandAssigner();p,u=body();p[15]=[350,400];p[16]=[650,400];u[[15,16]]=True
 hh=[hand(350,400),hand(650,400)]
 out,d=call(a,hh,[1,0],0,p,u);assert out=={0:0,1:1} and d['identity_state']==['confirmed','confirmed']
 p[[15,16]]=[500,400]
 out,d=call(a,hh,[1,0],33333,p,u);assert out=={0:0,1:1} and not d['guards']
 tests.append('crossing_with_flipped_labels_and_coincident_pose_preserved')
 # Stub isolates the new duplicate policy from the older proposer's own
 # duplicate suppression; integration is checked again over full public data.
 class Stub:
  def __init__(self,side,reasons):self.side=side;self.reasons=reasons
  def assign(self,*x):
   return dict(enumerate(self.side)),{'scale':400.,'pose_distance':[[None,None] for _ in self.side],'reasons':self.reasons,'assigned_sides':self.side}
 def duplicate(own=False,shift=4):
  a=HandAssigner();a.proposer=Stub([0,1],['label_without_close_pose','temporal'])
  h0=hand(400,400);h1=hand(400+shift,400)
  # The anchor's past observation is an exact match. Both hypotheses are
  # near one real track; unique matching favors the exact anchor.
  a.confirmed[1]=EvidenceTrack(h1[PALM],0,0,np.zeros(2),0,'current_unique_pose',0)
  if own:a.confirmed[0]=EvidenceTrack(h0[PALM],0,0,np.zeros(2),0,'current_unique_pose',0)
  a.last_time=0;a.frame=0
  return a,[h0,h1]
 a,hh=duplicate();copy=np.array(hh).copy();out,d=call(a,hh,[0,1],33333)
 assert out=={0:None,1:1} and d['guards'][0]['kind']=='overlapping_region_with_confirmed_opposite_anchor'
 assert np.array_equal(hh,copy) and not d['coordinates_generated']
 tests.append('duplicate_risk_withheld_raw_coordinates_preserved')
 a,hh=duplicate(shift=6);out,d=call(a,hh,[0,1],33333)
 assert out=={0:0,1:1} and not d['guards'] and d['region_overlap_min_area']<.80
 tests.append('below_geometric_conflict_threshold_preserved')
 a,hh=duplicate(True);out,d=call(a,hh,[0,1],33333)
 assert out=={0:0,1:1} and not d['guards']
 tests.append('two_reachable_confirmed_hands_protect_real_overlap')
 a,hh=duplicate();a.confirmed.clear();out,d=call(a,hh,[0,1],33333)
 assert out=={0:0,1:1} and not d['guards']
 tests.append('overlap_alone_does_not_reject_two_hands')
 a=HandAssigner();a.proposer=Stub([1],['label_without_close_pose']);a.confirmed[1]=EvidenceTrack(hand(400,400)[PALM],0,0,np.zeros(2),0,'current_unique_pose',0);a.last_time=0;a.frame=0
 out,d=call(a,[hand(900,400)],[1],33333)
 assert out=={0:None} and a.confirmed[1].frame==0
 tests.append('label_jump_cannot_overwrite_recent_confirmed_identity')
 # Reflection changes screen positions but not anatomical labels or outcome.
 a,hh=duplicate();p,u=body();hh=np.asarray(hh);hh[:,:,0]=1200-hh[:,:,0];p[:,0]=1200-p[:,0]
 for tr in a.confirmed.values():tr.palm[:,0]=1200-tr.palm[:,0]
 out,d=call(a,hh,[0,1],33333,p,u);assert out=={0:None,1:1}
 tests.append('horizontal_reflection_not_screen_half_labeling')
 return tests

if __name__=='__main__':
 tests=run()
 with (U/'CONTRACTS.json').open('x',encoding='utf-8') as f:json.dump({'status':'PASS','tests':tests},f,indent=2)
 print('PASS',len(tests))
