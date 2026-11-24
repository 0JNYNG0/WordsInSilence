"""Read-only local inference review. Never starts camera, inference, or training."""
from pathlib import Path
import datetime, json, os, time, urllib.request

HERE = Path(__file__).resolve().parent
OPS = HERE.parents[1]
URL = 'http://127.0.0.1:8821'
ACCEPTANCE = OPS / 'word100_live_g6_20261114/WEB_ACCEPTANCE.json'
TARGETS = {'WORD0099', 'WORD1290'}

def now():
    return datetime.datetime.now().astimezone().isoformat()

def write(name, value):
    target = HERE / name
    temp = target.with_suffix(target.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(target)

def get(path, token=None):
    request = urllib.request.Request(URL + path, headers={'X-Session-Token': token} if token else {})
    with urllib.request.urlopen(request, timeout=3) as response:
        return json.load(response)

def main():
    began = now()
    token = get('/api/meta')['token']
    initial = get('/api/status', token)
    excluded_attempt = initial.get('attempt_id') if initial['state'] == 'done' else None
    seen, saved = {}, {}
    deadline = time.monotonic() + 1200
    status = dict(started_at=began, pid=os.getpid(), status='WAITING_FOR_USER_CAPTURE',
                  training_use=False, tuning_use=False, candidate_selection_use=False,
                  camera_started_by_observer=False, runtime_changed=False, excluded_initial_attempt=excluded_attempt)
    write('OBSERVER.json', status)
    while time.monotonic() < deadline:
        try:
            s = get('/api/status', token)
            attempt = s.get('attempt_id')
            if attempt and attempt != excluded_attempt:
                timeline = seen.setdefault(attempt, [])
                if not timeline or timeline[-1]['state'] != s['state']:
                    timeline.append(dict(time=now(), state=s['state'], elapsed=s.get('elapsed'), error=s.get('error')))
                if s['state'] == 'error':
                    write(attempt + '_error.json', dict(attempt_id=attempt, status=s, timeline=timeline, training_use=False))
                if s['state'] == 'done' and s.get('has_result'):
                    data = get('/api/result?attempt=' + attempt, token)
                    result = data['result']
                    if result.get('source') == 'user_inference_only' and result.get('word') in TARGETS:
                        # Existing video/RTMW files remain in their original local temp folder.
                        # Preserve the produced decisions, not a new copy of video or coordinates.
                        result = {k: v for k, v in result.items() if k not in ('query_xy', 'timestamp_us', 'frame_id', 'video')}
                        acceptance = None
                        if ACCEPTANCE.exists():
                            a = json.loads(ACCEPTANCE.read_text('utf-8-sig'))
                            if a.get('attempt_id') == attempt:
                                acceptance = a
                        item = dict(saved_at=now(), training_use=False, tuning_use=False, candidate_selection_use=False,
                                    result=result, outcome=data.get('outcome'), timeline=timeline,
                                    browser_ack=acceptance, capture_status=s,
                                    review_scope='Observed runtime output only; not correctness ground truth')
                        if attempt not in saved or (acceptance and not saved[attempt].get('browser_ack')):
                            saved[attempt] = item
                            write(attempt + '_result.json', item)
                            print(json.dumps(dict(attempt=attempt, word=result['word'], stop_reason=result.get('stop_reason'),
                                                  browser_ack=bool(acceptance)), ensure_ascii=False), flush=True)
            words = {item['result']['word'] for item in saved.values() if item.get('browser_ack')}
            status.update(updated_at=now(), status='RECORDS_SAVED_AWAITING_REVIEW' if words == TARGETS else 'WAITING_FOR_USER_CAPTURE',
                          state=s['state'], attempts=list(saved), completed_words=sorted(words))
            write('OBSERVER.json', status)
            if words == TARGETS:
                return
        except Exception as error:
            status.update(last_poll_error=str(error), updated_at=now())
            write('OBSERVER.json', status)
        time.sleep(0.5)
    status.update(status='WATCH_WINDOW_ENDED', updated_at=now(), note='Review any saved attempts; user may resume later.')
    write('OBSERVER.json', status)

if __name__ == '__main__':
    main()
