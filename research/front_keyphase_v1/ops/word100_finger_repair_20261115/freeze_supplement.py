from common import *
assert not(U/'SUPPLEMENT_FROZEN.json').exists()
write(U/'SUPPLEMENT_FROZEN.json',dict(reason='Whole-hand and shared-finger candidates correctly distinguish added3pattern but have different old-source errors and one shared error. Never replace existing qualified coarse feedback with these alone.',
 observer_rule='All3expanded_whole_hand and all3shared_finger_chain must confidently agree at the observed frame. If all3old_v2 confidently agree on an opposing state, mark conflict unknown. No use of correct answer in this observer fusion.',
 presentation_rule='Individual correction only refines an existing attributed coarse handshape difference within the same actual span. It cannot create an error card, remove earlier direction/order cues or claim completion. Require>=3valid frames spanning>=0.10s with consistent finger difference; this protects against isolated transient jitters, without marking short动作 correct.',
 capability='Right/left long fingers only; no thumb. Verified combinations and partial coverage reported. Do not manufacture V/fist label.',
 scopes=['old895cells','additional72cells including6heldout ring/little-separated anchors','public122conditions with original auto-selected spans','normal speed,masking,order/direction preservation'],
 no_web_promotion_until_all_gates=True,no_threshold_search=True,source_exposure='Development reused for diagnosis. Not a new blind error accuracy.',pins={str(p):sha(p)for d in ['expanded_whole_hand','shared_finger_chain']for p in (U/d).glob('*.pt')}))
print('Frozen supplementary policy before replay.')
