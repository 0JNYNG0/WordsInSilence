import unittest

import numpy as np

import paired_finetune as pf


V = pf.validated_subset_indices
BAD = (TypeError, ValueError)


class TestSubsetContract(unittest.TestCase):
    def test_selects_known_values(self):
        pred = np.array([10, 11, 12, 13])
        target = np.array([10, 99, 12, 7])
        sel = V(np.array([2, 0]), 4)
        n = int(sel.size)
        self.assertEqual(n, 2)
        self.assertEqual(sel.dtype, np.int64)
        np.testing.assert_array_equal(sel, [2, 0])
        np.testing.assert_array_equal(pred[sel], [12, 10])
        np.testing.assert_array_equal(target[sel], [12, 10])
        self.assertEqual(int((pred[sel] == target[sel]).sum()), 2)

    def test_flatnonzero_output_accepted(self):
        mask = np.array([False, True, False, True, True])
        sel = V(np.flatnonzero(mask), 5)
        np.testing.assert_array_equal(sel, [1, 3, 4])
        self.assertEqual(sel.dtype, np.int64)

    def test_order_preserved(self):
        np.testing.assert_array_equal(V(np.array([3, 1, 2]), 5), [3, 1, 2])
        np.testing.assert_array_equal(V([4, 0], 5), [4, 0])

    def test_other_integer_dtypes(self):
        for dt in (np.int8, np.int32, np.uint16, np.uint64):
            out = V(np.array([1, 0], dtype=dt), 3)
            self.assertEqual(out.dtype, np.int64)
            np.testing.assert_array_equal(out, [1, 0])

    def test_empty(self):
        out = V(np.array([], dtype=np.int64), 855)
        self.assertEqual(out.size, 0)
        self.assertEqual(out.dtype, np.int64)
        self.assertEqual(int(out.size), 0)
        self.assertEqual(V(np.flatnonzero(np.zeros(5, dtype=bool)), 5).size, 0)

    def test_returns_copy(self):
        src = np.array([0, 1], dtype=np.int64)
        out = V(src, 2)
        out[0] = 1
        self.assertEqual(int(src[0]), 0)

    def test_rejects_bool(self):
        with self.assertRaises(BAD):
            V(np.array([True, False, True]), 3)

    def test_rejects_float(self):
        with self.assertRaises(BAD):
            V(np.array([0.0, 1.0]), 3)
        with self.assertRaises(BAD):
            V(np.array([], dtype=np.float64), 3)

    def test_rejects_2d(self):
        with self.assertRaises(BAD):
            V(np.array([[0, 1]]), 3)
        with self.assertRaises(BAD):
            V(np.array(1), 3)

    def test_rejects_negative(self):
        with self.assertRaises(BAD):
            V(np.array([-1, 0]), 3)

    def test_rejects_out_of_bounds(self):
        with self.assertRaises(BAD):
            V(np.array([0, 3]), 3)
        with self.assertRaises(BAD):
            V(np.array([0]), 0)

    def test_rejects_duplicates(self):
        with self.assertRaises(BAD):
            V(np.array([1, 2, 1]), 3)

    def test_rejects_bad_n(self):
        with self.assertRaises(BAD):
            V(np.array([0]), 2.0)
        with self.assertRaises(BAD):
            V(np.array([0]), True)
        with self.assertRaises(BAD):
            V(np.array([], dtype=np.int64), -1)


if __name__ == '__main__':
    unittest.main()
