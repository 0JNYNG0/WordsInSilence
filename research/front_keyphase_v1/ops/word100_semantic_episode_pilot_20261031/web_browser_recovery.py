"""Explicitly invoked four-case browser replay; no camera or old benchmark rerun."""
import base64
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from web_replay import U, RUNS, ALIASES, Engine, make_server, write_new, read, sha, first_failure


def main():
    # Root reviews protocol and invokes `run`; importing/checking cannot replay videos.
    for name in ['WEB_RESULT.json', 'WEB_RUNNING.json', 'WEB_RECOVERY_EXECUTION_COMPLETE.json']:
        assert not (U / name).exists(), f'{name} exists; preserve first attempt and inspect'
    assert (U / 'WEB_FROZEN.json').exists(), 'Run reviewed web_replay.py freeze first'
    assert not RUNS.exists(), 'Partial/completed web runs exist; no automatic retry'
    write_new(U / 'WEB_RUNNING.json', dict(pid=os.getpid(), status='STARTING', aliases=ALIASES))
    engine = server = server_thread = chrome = ws = None
    try:
        import websocket
        engine = Engine()
        server = make_server(engine)
        assert server.server_port != 8765
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        profile = U / 'web_chrome_recovery_profile'
        profile.mkdir(exist_ok=False)
        portfile = profile / 'DevToolsActivePort'
        command = ['C:/Program Files/Google/Chrome/Application/chrome.exe',
                   '--headless=new', '--remote-debugging-port=0',
                   '--remote-allow-origins=http://127.0.0.1',
                   '--disable-background-networking', '--disable-component-update',
                   '--no-first-run', '--user-data-dir=' + str(profile),
                   '--window-size=1440,1100', 'about:blank']
        chrome = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                  creationflags=subprocess.CREATE_NO_WINDOW)
        tabs = []
        for _ in range(150):
            assert chrome.poll() is None, 'Isolated Chrome exited before debugging became ready'
            try:
                port = int(portfile.read_text().splitlines()[0])
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/json/list', timeout=2) as response:
                    tabs = json.load(response)
                if any(t['type'] == 'page' for t in tabs):
                    break
            except (OSError, ValueError):
                pass
            time.sleep(.1)
        tab = next(t for t in tabs if t['type'] == 'page')
        ws = websocket.create_connection(tab['webSocketDebuggerUrl'], origin='http://127.0.0.1', timeout=600)
        sequence = 0
        errors = []

        def call(method, params=None):
            nonlocal sequence
            sequence += 1
            ws.send(json.dumps(dict(id=sequence, method=method, params=params or {})))
            while True:
                message = json.loads(ws.recv())
                if message.get('method') == 'Runtime.exceptionThrown':
                    errors.append(message)
                if message.get('id') == sequence:
                    if 'error' in message:
                        raise RuntimeError(message)
                    return message.get('result', {})

        def js(expression):
            value = call('Runtime.evaluate', dict(expression=expression, returnByValue=True, awaitPromise=True))
            if value.get('exceptionDetails'):
                raise RuntimeError(value)
            return value.get('result', {}).get('value')

        def screenshot(name):
            image = call('Page.captureScreenshot', dict(format='png'))['data']
            path = U / name
            with path.open('xb') as f:
                f.write(base64.b64decode(image))

        call('Page.enable')
        call('Runtime.enable')
        call('Page.navigate', dict(url=f'http://127.0.0.1:{server.server_port}/'))
        for _ in range(100):
            if js('window.uiReady===true'):
                break
            time.sleep(.1)
        assert js('window.uiReady===true')
        output, media_checks = [], []
        for alias in ALIASES:
            result = js(f'runOne({json.dumps(alias)})')
            assert result['alias'] == alias and result['repeat'] == 1
            assert result['mode'] == 'during_recording'
            output.append(result)
            assert js("document.querySelector('#refs').children.length===3")
            assert js("document.querySelector('#adoption').textContent.includes('채택 전')")
            # Loadeddata was already included in final ACK. Verify actual aligned seeking separately.
            media = js("""(async()=>{
              const slider=document.querySelector('#seek');
              if(slider.disabled)return {alignment_available:false,videos:[...document.querySelectorAll('video')].map(v=>({width:v.videoWidth,ready:v.readyState,time:v.currentTime}))};
              slider.value=Math.floor(Number(slider.max)/2);seekPair();
              const until=performance.now()+10000;
              while([...document.querySelectorAll('video')].some(v=>v.seeking||v.readyState<2)){
                if(performance.now()>until)throw Error('Aligned video seek timeout');
                await new Promise(r=>setTimeout(r,40));
              }
              return {alignment_available:true,videos:[...document.querySelectorAll('video')].map(v=>({width:v.videoWidth,ready:v.readyState,time:v.currentTime})),displayed:document.querySelector('#time').textContent};
            })()""")
            assert len(media['videos']) == 2 and all(v['width'] > 0 and v['ready'] >= 2 for v in media['videos'])
            if media['alignment_available']:
                assert all(v['time'] > 0 for v in media['videos'])
            media_checks.append(dict(alias=alias, **media))
            js("document.querySelector('#result').scrollIntoView()")
            screenshot(f'WEB_UI_{alias}.png')
            print(json.dumps({k: result[k] for k in ['alias', 'frames', 'backend_after_end_seconds',
                'backend_ready_to_2raf_ack_seconds', 'backend_ready_to_media_and_paint_ack_seconds']}, ensure_ascii=False), flush=True)
        call('Emulation.setDeviceMetricsOverride', dict(width=760, height=1100, deviceScaleFactor=1, mobile=False))
        js("document.querySelector('#result').scrollIntoView()")
        await_paint = "new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)))"
        js(await_paint)
        assert js('document.documentElement.scrollWidth<=window.innerWidth+2')
        screenshot('WEB_UI_NARROW.png')
        assert not errors, errors
        assert len(output) == 4 and len({r['alias'] for r in output}) == 4
        engine.check_pins()
        report = dict(status='FOUR_REPLAYS_COMPLETE_PENDING_INDEPENDENT_REVIEW',
            rows=output, media_seek_checks=media_checks, application_setup_seconds=engine.setup_seconds,
            fixed_seed=42, model_selection_by_ui=False, whole_sign_verdict=None,
            protected_hashes_unchanged=True, no_old12_replays=True,
            js_errors=errors, narrow_no_horizontal_page_overflow=True,
            server_port=server.server_port, production_port_untouched=True,
            browser=call('Browser.getVersion'), timing_definition=engine.frozen['timing'],
            timing_scope=engine.frozen['timing_scope'], protocol_sha256=sha(U / 'WEB_FROZEN.json'))
        write_new(U / 'WEB_RESULT.json', report)
        call('Browser.close')
    except BaseException as error:
        first_failure(U / 'WEB_RECOVERY_FIRST_FAILURE.json', error)
        raise
    finally:
        if ws:
            ws.close()
        if server:
            server.shutdown()
            server.server_close()
        if server_thread:
            server_thread.join(timeout=10)
        if engine and engine.worker and engine.worker.is_alive():
            # Do not silently start a replacement or delete a partial observation.
            engine.worker.join(timeout=120)
        if chrome:
            try:
                chrome.wait(timeout=10)
            except subprocess.TimeoutExpired:
                chrome.terminate()
                chrome.wait(timeout=10)
        write_new(U / 'WEB_RECOVERY_EXECUTION_COMPLETE.json', dict(pid=os.getpid(),
            status='COMPLETE' if (U / 'WEB_RESULT.json').exists() else 'FAILED',
            server_closed=True, chrome_closed=chrome is None or chrome.poll() is not None,
            worker_finished=engine is None or engine.worker is None or not engine.worker.is_alive()))
        lock = U / 'WEB_RUNNING.json'
        if lock.exists() and read(lock)['pid'] == os.getpid():
            lock.unlink()


if __name__ == '__main__':
    if sys.argv[1:] != ['run']:
        raise SystemExit('Actual reviewed execution only: python -B web_browser_test.py run')
    main()
