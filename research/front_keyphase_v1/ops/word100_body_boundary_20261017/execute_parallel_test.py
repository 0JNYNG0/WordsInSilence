"""Scheduling-only tests: no detector, source archive, preparation or training."""
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import unittest

from execute_parallel import bounded_dispatch


class ParallelDispatchTests(unittest.TestCase):
    def test_limit_and_order(self):
        started = []
        guard = threading.Lock()
        active = 0
        maximum = 0

        def work(row):
            nonlocal active, maximum
            with guard:
                started.append(row['clip_id'])
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.01)
            with guard:
                active -= 1
            return {'status': 'ROW_COMPLETE', 'clip_id': row['clip_id']}

        callbacks = []
        with ThreadPoolExecutor(max_workers=2) as pool:
            result, dispatched, failed = bounded_dispatch(pool,
                [{'clip_id': str(i)} for i in range(8)], work, (), 5, 30, callbacks.append)
        self.assertEqual(started, ['0', '1', '2', '3', '4'])
        self.assertEqual(dispatched, 5)
        self.assertEqual(len(result), 5)
        self.assertEqual(len(callbacks), 5)
        self.assertEqual(maximum, 2)
        self.assertFalse(failed)

    def test_first_failure_stops_dispatch_and_retains_inflight_success(self):
        started = []

        def work(row):
            started.append(row['clip_id'])
            if row['clip_id'] == '0':
                time.sleep(0.01)
                return {'status': 'ROW_FAILED', 'clip_id': '0'}
            time.sleep(0.04)
            return {'status': 'ROW_COMPLETE', 'clip_id': row['clip_id']}

        callbacks = []
        with ThreadPoolExecutor(max_workers=2) as pool:
            result, dispatched, failed = bounded_dispatch(pool,
                [{'clip_id': str(i)} for i in range(8)], work, (), 8, 30, callbacks.append)
        self.assertEqual(started, ['0', '1'])
        self.assertEqual(dispatched, 2)
        self.assertTrue(failed)
        self.assertEqual({r['status'] for r in result}, {'ROW_COMPLETE', 'ROW_FAILED'})
        self.assertEqual(len(callbacks), 2)

    def test_worker_crash_becomes_failure(self):
        def work(row):
            raise RuntimeError('synthetic worker crash')

        with ThreadPoolExecutor(max_workers=2) as pool:
            result, dispatched, failed = bounded_dispatch(pool,
                [{'clip_id': str(i)} for i in range(5)], work, (), 5, 30, lambda r: None)
        self.assertTrue(failed)
        self.assertLessEqual(dispatched, 2)
        self.assertTrue(all(r.get('worker_crash') for r in result))

    def test_dispatch_deadline_does_not_cancel_inflight(self):
        def work(row):
            time.sleep(0.03)
            return {'status': 'ROW_COMPLETE', 'clip_id': row['clip_id']}

        with ThreadPoolExecutor(max_workers=2) as pool:
            result, dispatched, failed = bounded_dispatch(pool,
                [{'clip_id': str(i)} for i in range(5)], work, (), 5, 0.02, lambda r: None)
        self.assertFalse(failed)
        self.assertEqual(dispatched, 2)
        self.assertEqual(len(result), 2)

    def test_empty(self):
        with ThreadPoolExecutor(max_workers=2) as pool:
            result, dispatched, failed = bounded_dispatch(pool, [], lambda row: None,
                                                           (), 4, 5, lambda r: None)
        self.assertEqual((result, dispatched, failed), ([], 0, False))


if __name__ == '__main__':
    unittest.main()
