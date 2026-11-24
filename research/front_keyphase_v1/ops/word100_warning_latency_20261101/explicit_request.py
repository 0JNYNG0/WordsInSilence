"""Dedicated synchronous request for one existing encoder consumer thread.

Create ONE wrapper per _encode invocation; reuse the compiled model, not this
wrapper, across videos. Construction/import does not create a model/request or
run inference. The first caller creates the request on its own thread. The
candidate encoder already owns each contiguous input until the call completes.

This is a narrow execution-lifetime experiment, not a diagnosed latency fix.
Official API and installed _ov_api.py agree that CompiledModel.__call__ caches
an InferRequest and defaults to share_inputs=True, share_outputs=False.
https://docs.openvino.ai/2025/api/ie_python_api/_autosummary/openvino.CompiledModel.html
https://docs.openvino.ai/2025/api/ie_python_api/_autosummary/openvino.InferRequest.html
"""
from pathlib import Path
import json
import threading
import time


class ExplicitRequest:
    def __init__(self, compiled, profile=None, jsonl_path=None):
        if profile is not None and not isinstance(profile, list):
            raise TypeError('profile must be a list or None')
        self.compiled = compiled
        self.profile = profile if profile is not None else []
        self.jsonl_path = Path(jsonl_path) if jsonl_path is not None else None
        # Reserve a new ledger before any work. Never overwrite prior attempts.
        if self.jsonl_path is not None:
            with self.jsonl_path.open('x', encoding='utf8'):
                pass
        self._request = None
        self._owner = None
        self._lock = threading.Lock()
        self._failed = False
        self._calls = 0

    def _event(self, record):
        if self.jsonl_path is not None:
            with self.jsonl_path.open('a', encoding='utf8') as handle:
                handle.write(json.dumps(record, separators=(',', ':'), allow_nan=False)+'\n')
                handle.flush()

    def __call__(self, items):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('ExplicitRequest concurrent/reentrant call forbidden')
        try:
            current = threading.current_thread()
            if self._owner is not None and self._owner is not current:
                raise RuntimeError('ExplicitRequest belongs to another consumer thread; create a new wrapper per video')
            if self._failed:
                raise RuntimeError('ExplicitRequest failed previously; no implicit inference retry')
            self._owner = current
            index = self._calls
            self._calls += 1
            began = time.perf_counter()
            row = dict(window_index=index, thread_id=threading.get_ident(),
                       thread_name=current.name, request_create_seconds=0.,
                       infer_seconds=None, share_inputs=True, share_outputs=False)
            phase = 'request_create' if self._request is None else 'infer'
            infer_started = None
            try:
                self._event(dict(event='CALL_BEGIN', phase=phase,
                                 monotonic_seconds=began, **row))
                if self._request is None:
                    t = time.perf_counter()
                    self._request = self.compiled.create_infer_request()
                    row['request_create_seconds'] = time.perf_counter()-t
                    self._event(dict(event='REQUEST_CREATED', window_index=index,
                                     thread_id=row['thread_id'], seconds=row['request_create_seconds']))
                phase = 'infer'
                infer_started = time.perf_counter()
                result = self._request.infer(items, share_inputs=True, share_outputs=False)
                row['infer_seconds'] = time.perf_counter()-infer_started
            except BaseException as exc:
                self._failed = True
                if infer_started is not None:
                    row['infer_seconds'] = time.perf_counter()-infer_started
                row.update(status='FAILED', failed_phase=phase, error_type=type(exc).__name__,
                           call_seconds_before_end_log=time.perf_counter()-began)
                self.profile.append(row)
                try:
                    self._event(dict(event='CALL_END', **row))
                except BaseException as log_error:
                    exc.add_note('Timing log failed: '+type(log_error).__name__)
                raise
            row.update(status='COMPLETE', call_seconds_before_end_log=time.perf_counter()-began)
            self.profile.append(row)
            try:
                self._event(dict(event='CALL_END', **row))
            except BaseException:
                self._failed = True
                raise
            # Preserve the original OVDict/int-key API without reshaping,
            # recasting, precision changes, output caching or extra inference.
            return result
        finally:
            self._lock.release()
