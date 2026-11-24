"""Post-benchmark transport fix. Frozen measured server.py remains unchanged."""
from server import make_server as original_server
from pipeline import U
import json, time

def expected_abort(error):
    return isinstance(error, ConnectionAbortedError) and getattr(error,'winerror',None)==10053

def make_server(engine):
    server=original_server(engine)
    parent=server.RequestHandlerClass
    class CancelAwareHandler(parent):
        def handle(self):
            try:super().handle()
            except ConnectionAbortedError as error:
                if not expected_abort(error):raise
                # A browser cancels an old video range request when src/seek changes.
                # Record the cancellation; do not treat it as a detector/job failure.
                with (U/'TRANSPORT_EVENTS.jsonl').open('a',encoding='utf8') as out:
                    out.write(json.dumps(dict(event='client_video_request_cancelled',winerror=10053,at_ns=time.perf_counter_ns()))+'\n')
    server.RequestHandlerClass=CancelAwareHandler
    return server

def selftest():
    e=ConnectionAbortedError('synthetic browser cancellation');e.winerror=10053
    assert expected_abort(e)
    other=ConnectionAbortedError('other transport issue');other.winerror=10054
    assert not expected_abort(other) and not expected_abort(RuntimeError('real job failure'))
    return dict(expected_10053_only=True,other_failures_not_swallowed=True)
