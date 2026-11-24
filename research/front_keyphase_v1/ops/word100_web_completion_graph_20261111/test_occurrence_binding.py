import unittest
import numpy as np
from occurrence_binding import select_occurrence, union_indices


class Contract(unittest.TestCase):
    def select(self, p, r, t=None, valid=None, child=False):
        n = len(p)
        return select_occurrence(p, r, np.ones(n, bool) if valid is None else valid,
                                 np.arange(n) * 33333 if t is None else t,
                                 .65, 60., endpoint_child=child)

    def test_context_peak_does_not_trim_action(self):
        a = self.select([.9] * 8, [.1, .2, .9, .9, .3, .2, .1, .1])
        self.assertEqual(a['span'], [0, 7])
        self.assertEqual(a['anchors'], [2, 3])
        self.assertIsNone(a['semantic_completion'])

    def test_missing_middle_is_not_painted(self):
        a = self.select([.9] * 6, [.9] * 6, valid=[1, 1, 0, 0, 1, 1])
        self.assertEqual([r['span'] for r in a['candidates']], [[0, 1], [4, 5]])
        self.assertEqual(union_indices([[0, 1], [4, 5]]), [0, 1, 4, 5])

    def test_endpoint_speed_does_not_erase_observed_arrival(self):
        for speed in [.75, 1, 1.25]:
            a = self.select([.9, .9], [.9, .9], np.arange(2) * 33333 / speed, child=True)
            self.assertEqual(a['span'], [0, 1])
            self.assertIsNone(a['semantic_completion'])

    def test_no_context_support_is_unknown(self):
        self.assertIsNone(self.select([.9] * 6, [.1] * 6)['span'])

    def test_all_missing_is_unknown(self):
        self.assertIsNone(self.select([.9] * 6, [.9] * 6, valid=[0] * 6)['span'])

    def test_no_expected_order_argument(self):
        a = self.select([.9, .9, 0, .9, .9], [.8, .8, 0, .95, .95])
        self.assertEqual(a['span'], [3, 4])
        self.assertEqual(len(a['candidates']), 2)

    def test_unreliable_single_endpoint_is_not_completion(self):
        a = self.select([0, .99, 0], [0, .99, 0], child=True)
        self.assertIsNone(a['span'])

    def test_time_invalid_rejected(self):
        with self.assertRaises(AssertionError):
            self.select([.9, .9], [.9, .9], [1, 1])


if __name__ == '__main__':
    unittest.main()
