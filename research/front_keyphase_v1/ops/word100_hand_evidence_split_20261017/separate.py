"""Separate location, shape observation, and overlap evidence.

No aggregate quality score/state: uncertainty about unobserved finger shape must
not overwrite current arm-chain location evidence. No repair or training inputs.
"""
import sys
from copy import deepcopy
from pathlib import Path
U=Path(__file__).resolve().parent
sys.path.insert(0,str(U.parent/'word100_hand_region_uncertainty_20261017'))
from region import HandRegion,HandVisibility

LOCATION={
    'observed':'observed_points', 'partial_observation':'partial_points',
    'inside_missing':'wrist_interior', 'overlap_likely':'wrist_interior',
    'boundary_outside_likely':'wrist_boundary_or_outside',
    'assignment_uncertain':'assignment_unresolved', 'unknown':'unknown'}

def separate_evidence(baseline,region):
    if baseline['state'] not in LOCATION:
        raise ValueError('Unknown baseline observation state')
    if baseline['side']!=region['side'] or region['baseline_state']!=baseline['state']:
        raise ValueError('Baseline and region must refer to the same side and observation')
    old=baseline['state']
    observed=old=='observed'
    shape=('usable_observed_points' if observed else 'partial_observation' if old=='partial_observation' else 'unobserved')
    missing=old in ('inside_missing','overlap_likely','boundary_outside_likely')
    source=region['region_status']
    # 'overlap_likely' is area-supported only if the region component actually
    # evaluated a hull; its fallback can retain the old proximity-only label.
    if source=='estimated_envelope':
        rs=region['state']
        support=('area_supported_suspicion' if rs=='overlap_likely' else
                 'region_uncertain' if rs in ('region_uncertain','boundary_uncertain') else
                 'not_supported_across_envelope')
    else:
        support='region_unavailable'
    if not missing:support='not_assessed_for_missing_hand'
    return {
      'side':baseline['side'],
      'position':{'state':LOCATION[old], 'basis':'existing_current_observation_and_arm_chain',
                  'whole_hand_visibility_proven':False,
                  'evidence':deepcopy(baseline['evidence'])},
      'shape':{'state':shape, 'estimated_region_is_observed_shape':False},
      'overlap':{'proximity_hint':old=='overlap_likely', 'area_evidence':support,
                 'occlusion_cause_confirmed':False},
      'region':{'status':source,'state':region['state'],
                'age_seconds':region['evidence'].get('region_age_seconds'),
                'uncertainty_pixels':region['evidence'].get('uncertainty_pixels'),
                'not_calibrated_probability':True},
      'coordinate_repair':False, 'eligible_for_automatic_exclusion':False
    }

class SeparatedHandDiagnostics:
    """Current-view streaming adapter. Each stream owns its own causal history."""
    def __init__(self):self.reset()
    def reset(self):
        self.location=HandVisibility();self.region=HandRegion()
    def step(self,*args,**kwargs):
        location=self.location.step(*args,**kwargs)
        region=self.region.step(*args,**kwargs)
        return [separate_evidence(a,b) for a,b in zip(location,region)]
