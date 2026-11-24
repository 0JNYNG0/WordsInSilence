"""Independent synthetic contract checks for the common observed-role binder.

No camera, training data, user footage or neural inference is involved.
Run after the implementation exists:
  python -X utf8 test_binding_contract.py --module binding.py
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
import unittest

import numpy as np

sys.dont_write_bytecode = True
U = Path(__file__).resolve().parent
BINDER = None


def sample(labels, times=None, valid=None):
    p = np.full((len(labels), 11), .01, dtype=np.float64)
    for i, label in enumerate(labels):
        if label is not None:
            p[i, label] = .9
    t = np.arange(len(labels), dtype=np.int64) * 33333 if times is None else np.asarray(times, dtype=np.int64)
    v = np.ones(len(labels), bool) if valid is None else np.asarray(valid, bool)
    return p, v, t


def spans(occurrences):
    return sorted((int(o['family']), int(o['a']), int(o['b'])) for o in occurrences)


class BindingContract(unittest.TestCase):
    def bind(self, labels, families=(1, 2), times=None, valid=None):
        p, v, t = sample(labels, times, valid)
        return BINDER.bind_occurrences(p, v, t, list(families))

    def test_two_distinct_observed_actions(self):
        o = self.bind([None, 1, 1, None, 2, 2, None])
        self.assertEqual(spans(o), [(1, 1, 2), (2, 4, 5)])
        self.assertEqual(BINDER.order_relation(o, 1, 2), 'supported')

    def test_actual_reversed_order_is_not_repaired(self):
        o = self.bind([2, 2, None, 1, 1])
        self.assertEqual(spans(o), [(1, 3, 4), (2, 0, 1)])
        self.assertEqual(BINDER.order_relation(o, 1, 2), 'reversed')

    def test_family_enumeration_does_not_force_order(self):
        labels = [2, 2, None, 1, 1]
        a = self.bind(labels, (1, 2))
        b = self.bind(labels, (2, 1))
        self.assertEqual(spans(a), spans(b))
        self.assertEqual(BINDER.order_relation(a, 1, 2), 'reversed')
        self.assertEqual(BINDER.order_relation(b, 1, 2), 'reversed')

    def test_requested_order_does_not_change_observation(self):
        o = self.bind([1, 1, None, 2, 2])
        before = json.dumps(o, sort_keys=True)
        self.assertEqual(BINDER.order_relation(o, 1, 2), 'supported')
        self.assertEqual(BINDER.order_relation(o, 2, 1), 'reversed')
        self.assertEqual(json.dumps(o, sort_keys=True), before)

    def test_all_occurrences_retained(self):
        o = self.bind([1, 1, None, 2, 2, None, 1, 1])
        self.assertEqual(spans(o), [(1, 0, 1), (1, 6, 7), (2, 3, 4)])
        self.assertEqual(BINDER.order_relation(o, 1, 2), 'ambiguous')

    def test_concurrent_conflicting_family_is_ambiguous(self):
        p, v, t = sample([1, 1, 1])
        p[:, 2] = .9
        o = BINDER.bind_occurrences(p, v, t, [1, 2])
        self.assertEqual(spans(o), [(1, 0, 2), (2, 0, 2)])
        self.assertEqual(BINDER.order_relation(o, 1, 2), 'ambiguous')

    def test_absent_action_does_not_get_a_fabricated_span(self):
        o = self.bind([1, 1, None, None])
        self.assertEqual(spans(o), [(1, 0, 1)])
        self.assertEqual(BINDER.order_relation(o, 1, 2), 'association_unknown')

    def test_all_observation_missing(self):
        o = self.bind([1, 1, 2, 2], valid=[False] * 4)
        self.assertEqual(o, [])
        self.assertEqual(BINDER.order_relation(o, 1, 2), 'association_unknown')

    def test_one_missing_frame_splits_evidence(self):
        o = self.bind([1, 1, 1, 1, 1], valid=[True, True, False, True, True])
        self.assertEqual(spans(o), [(1, 0, 1), (1, 3, 4)])
        self.assertTrue(all(not (x['a'] <= 2 <= x['b']) for x in o))

    def test_short_13ms_evidence_is_retained(self):
        o = self.bind([None, 1, 1, None, 2, 2], times=[0, 1000, 14000, 15000, 16000, 29000])
        self.assertEqual(spans(o), [(1, 1, 2), (2, 4, 5)])
        self.assertEqual(BINDER.order_relation(o, 1, 2), 'supported')

    def test_single_frame_is_not_silently_deleted(self):
        o = self.bind([None, 1, None, 2, None], times=[0, 1000, 2000, 3000, 4000])
        self.assertEqual(spans(o), [(1, 1, 1), (2, 3, 3)])

    def test_uniform_time_scaling_preserves_order_and_span(self):
        p, v, t = sample([2, 2, None, 1, 1])
        reference = None
        for factor in (.25, .75, 1., 1.25, 4.):
            o = BINDER.bind_occurrences(p, v, (t * factor).astype(np.int64), [1, 2])
            if reference is None:
                reference = spans(o)
            self.assertEqual(spans(o), reference)
            self.assertEqual(BINDER.order_relation(o, 1, 2), 'reversed')

    def test_same_family_is_not_two_distinct_required_actions(self):
        o = self.bind([1, 1, None, 1, 1])
        self.assertEqual(BINDER.order_relation(o, 1, 1), 'not_applicable')

    def test_other_family_ids_use_the_same_code(self):
        o = self.bind([7, 7, None, 4, 4], families=(4, 7))
        self.assertEqual(spans(o), [(4, 3, 4), (7, 0, 1)])
        self.assertEqual(BINDER.order_relation(o, 4, 7), 'reversed')

    def test_no_expected_direction_or_word_inputs(self):
        import inspect
        names = set(inspect.signature(BINDER.bind_occurrences).parameters)
        self.assertFalse(names & {'word', 'word_id', 'desired_direction', 'expected_order', 'target_order', 'gold'})

    def test_input_arrays_not_modified(self):
        p, v, t = sample([2, 2, None, 1, 1])
        copies = [a.copy() for a in (p, v, t)]
        BINDER.bind_occurrences(p, v, t, [1, 2])
        for a, b in zip((p, v, t), copies):
            np.testing.assert_array_equal(a, b)


def main():
    global BINDER
    parser = argparse.ArgumentParser()
    parser.add_argument('--module', type=Path, default=U / 'binding.py')
    parser.add_argument('--output', type=Path, default=U / 'BINDING_CONTRACT_TESTS.json')
    args = parser.parse_args()
    path = args.module.resolve()
    if not path.is_file():
        raise SystemExit('Binder implementation missing: ' + str(path))
    spec = importlib.util.spec_from_file_location('_common_binder_contract_target', path)
    BINDER = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(BINDER)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(BindingContract)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    data = dict(passed=result.wasSuccessful(), tests=result.testsRun,
                failures=[dict(test=str(t), detail=e) for t, e in result.failures],
                errors=[dict(test=str(t), detail=e) for t, e in result.errors],
                module=str(path), module_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope='Synthetic interface/behavior contracts; not performance or human KSL truth',
                author='Separate Codex subagent', training=False, inference=False, camera=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
