"""Research-only evidence presentation; never repair coordinates or approve a sign.

Keep saved detector-gated comparison separately from the joint RGB/pose model's
estimate. RGB features being present is NOT proof that a hand is visible or that
the estimate is right. No word ID, new threshold, imputation or reshoot rule.
"""
from copy import deepcopy
import math


def describe_node(node, rgb_features_present):
    result = deepcopy(node)
    channels = {}
    for target, probability in node['probabilities'].items():
        present = (probability is not None and math.isfinite(probability)
                   and 0 <= probability <= 1)
        supported = bool(node['support'][target])
        assert not supported or present
        available = bool(present and rgb_features_present)
        channels[target] = {
            'saved_comparison_value': node['values'][target],
            'saved_detector_gate': supported,
            'model_probability': probability,
            'model_estimate': int(probability >= .5) if available else None,
            'model_input': 'RGB_VIDEO_FEATURES_PLUS_POSE_AND_MASKS',
            'estimate_status': ('DETECTOR_SUPPORTED_MODEL_ESTIMATE' if supported and available
                                else 'MODEL_ESTIMATE_WITHOUT_DETECTOR_SUPPORT' if available
                                else 'NO_AVAILABLE_MULTIMODAL_ESTIMATE'),
            'rgb_features_present': bool(rgb_features_present),
            'hand_visibility_verified': None,
            'image_only_accuracy_verified': False,
            'same_action_stage': None,
            'educational_correction': None,
            # Per-frame absence cannot establish that another recording is needed.
            'request_new_recording': None,
        }
    result['evidence_channels'] = channels
    return result


def describe_comparison(saved, query, reference, target):
    q = query['evidence_channels'][target]
    r = reference['evidence_channels'][target]
    available = q['model_estimate'] is not None and r['model_estimate'] is not None
    return {
        'saved_comparison': deepcopy(saved),
        'query_evidence': deepcopy(q), 'reference_evidence': deepcopy(r),
        'advisory_model_difference': (q['model_estimate'] != r['model_estimate']) if available else None,
        'advisory_status': ('ESTIMATES_DIFFER' if q['model_estimate'] != r['model_estimate']
                            else 'ESTIMATES_AGREE') if available else 'MODEL_OUTPUT_UNAVAILABLE',
        'model_agreement_is_correct_action': False,
        'saved_support_promoted': False,
        'educational_correction': None,
        'request_new_recording': None,
    }
