"""Keep accepted movement/shape weights; separate hand observability roles."""
from pathlib import Path
import sys

D = Path(__file__).resolve().parent
APPLE = D.parent / 'apple_v1'
sys.path.insert(0, str(APPLE))
from required_role_candidate import RequiredRoleCandidate, adapter, shape_check, contiguous, np
from visibility import hand_evidence


class RoleEvidenceCandidate(RequiredRoleCandidate):
    def analyze(self, raw):
        q = super().analyze(raw)
        pair = next((e for e in q['events'] if e['region'] == 2), None)
        if pair is None:
            return q
        ch = adapter.convert(raw['xy_px'], raw['score'], raw['timestamp_us'], raw['frame_id'],
                             int(raw['source_width']), int(raw['source_height']))
        a, b = pair['context']
        replacements = {}
        # These are the existing lesson's roles; the evidence helper is bilateral.
        for hand, cid, expected, preparation in [('L', 'support_shape', 1, True),
                                                ('R', 'wipe_shape', 0, False)]:
            facts = hand_evidence(ch, hand)
            first = max(0, a - 6) if preparation else a
            if preparation:
                # Preparation is already checked before cover. Do not expose
                # unreliable post-cover guesses just because the other hand is
                # missing. The hidden support state's continuation is unverified.
                c = next(c.copy() for c in q['criteria'] if c['id'] == cid)
            else:
                states = self.surface.sequence(ch['tcn280'], hand)
                c = shape_check(cid, states, expected, first, b, ~facts['uncertain'], ch['timestamp_us'])
            c['semantic_target'] = 'observed_palm_surface_form'
            spans = lambda mask: [[int(x[0]), int(x[-1])] for x in
                                  contiguous(np.flatnonzero(mask[first:b + 1]) + first)]
            c['observability'] = {name: spans(facts[name]) for name in
                                  ['own_missing', 'opposite_missing', 'projected_overlap']}
            c['hidden_finger_state_verified'] = False
            c['evidence_mode'] = 'observed_preparation_or_visible_state' if preparation else 'observed_active_hand'
            if not preparation:
                bad = [rr for rr in contiguous(np.flatnonzero(facts['uncertain'][a:b + 1]) + a)
                       if len(rr) >= 4]
                if bad and c['status'] == 'supported':
                    c['status'] = 'unknown'
                    c['unobserved_spans'] = [[int(rr[0]), int(rr[-1])] for rr in bad]
            replacements[cid] = c
        q['criteria'] = [replacements.get(c['id'], c) for c in q['criteria']]
        q['visibility_version'] = 'independent_hand_evidence_v1'
        return q
