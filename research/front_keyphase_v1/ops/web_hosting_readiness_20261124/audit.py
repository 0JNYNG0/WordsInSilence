"""Read-only source/package audit. No camera, inference, training or upload."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json

D = Path(__file__).resolve().parent
ROOT = D.parents[3]
O = D.parent
G = O / 'capture_feedback_completion_20261116'
M = G / 'moa_coaching_room_20261117/motion_dock_20261117'

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def evidence(path, needles):
    p = O / path
    lines = p.read_text(encoding='utf-8-sig').splitlines()
    result = []
    for needle in needles:
        matches = [i + 1 for i, line in enumerate(lines) if needle in line]
        if not matches:
            raise AssertionError(f'Missing evidence: {p}: {needle}')
        result.append(dict(text=needle, lines=matches))
    return dict(path=str(p.relative_to(ROOT)), sha256=sha(p), matches=result)

sources = [
    evidence('word100_action_feedback_demo_20261108/capture.py', [
        'cv2.VideoCapture(0,cv2.CAP_DSHOW)', "tempfile.mkdtemp(prefix='wis_action_demo_')", "Recorder(self.directory/'recorded.mp4'", "stream.finish(self.directory/'rtmw')"]),
    evidence('word100_rtmw_stream_reuse_20261108/stream_runtime.py', [
        "backend='openvino',device=device", "np.savez_compressed(p/'observations.npz'", 'engine.session_lock.acquire(False)']),
    evidence('web_tutorial_live_integration_20261113/server.py', [
        'SERVICE=CaptureService();QUEST=', 'def valid_host(self):', 'save_attempt_result(SERVICE.directory']),
    evidence('web_tutorial_live_integration_20261113/attempt_records.py', [
        'temporary.write_text', 'tuning_use=False']),
    evidence('capture_feedback_completion_20261116/moa_coaching_room_20261117/motion_dock_20261117/serve.py', [
        "s.LocalServer(('127.0.0.1',s.PORT)"]),
    evidence('capture_feedback_completion_20261116/moa_coaching_room_20261117/motion_dock_20261117/site/live.js', [
        "'/preview.mjpeg?token='", "command('/api/open')"]),
    evidence('capture_feedback_completion_20261116/selective_rtmw_anchor_repair_20261117/retry_runtime.py', [
        "VideoFrames(s.directory/'recorded.mp4',raw)"]),
]
admission = read(M / 'web/ADMISSION.json')
pin_results = [dict(path=p, exists=Path(p).is_file(), matches=Path(p).is_file() and sha(Path(p)) == h)
               for p, h in admission['pins'].items()]
if not all(p['matches'] for p in pin_results):
    raise AssertionError('Existing admitted files changed; do not call baseline preserved')
models = []
for p in sorted((O / 'word100_rtmw_mediapipe_compare_20261108/models').rglob('*.onnx')):
    models.append(dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size, sha256=sha(p)))
portability = read(G / 'apple_v1/web/PORTABILITY_CHECK.json')
report = dict(
    task='web_hosting_readiness_20261124',
    status='SOURCE_AUDIT_COMPLETE_NOT_DEPLOYED',
    at=datetime.now(timezone(timedelta(hours=9))).isoformat(),
    author='Codex', verification='Same Codex: separate source assertions and file hashes; not a deployment or hardware benchmark',
    current=dict(camera='Python OpenCV on the host PC, not visitor getUserMedia',
                 rtmw='OpenVINO GPU backend', web_host='127.0.0.1 only',
                 session='global capture/quest state; not isolated for multiple visitors',
                 retention='Local temporary video, pose arrays and diagnosis results; not no-retention',
                 retry='Reads recorded.mp4; disk-free migration must preserve selective reobservation',
                 existing_audit=portability),
    constraints=dict(classroom='Different Windows desktop; GPU unknown',
                     requested='Internet URL, independent of development laptop and W HDD',
                     hosting_excluded='ChatGPT/OpenAI hosting',
                     visitor_training=False, visitor_tuning=False,
                     visitor_video_retention=False,
                     conservative_design='No external RGB upload by default. User permits keypoint use, objects to sensitive video retention.',
                     cloud_budget='Not approved; user asks why renting is needed'),
    detector_onnx_files=models,
    detector_onnx_total_bytes=sum(p['bytes'] for p in models),
    detector_onnx_size_scope='Only person detector + RTMW ONNX weights; excludes AI feedback, runtime, avatars and all other files',
    sources=sources,
    original_pin_count=len(pin_results), original_pins_all_match=True,
    actions=dict(runtime_code_changed=False, private_recordings_read=False,
                 camera_opened=False, training=False, inference=False,
                 external_upload=False, paid_resource_created=False, public_deployment=False),
    remaining=[
        'Keep presentation fallback separate from URL-only architecture: local Windows runtime packaging can avoid GPU rental but requires classroom hardware test.',
        'For URL-only with no video upload, verify browser-side RTMW/YOLOX compatibility, coordinate parity and sustainable capture rate before choosing hosting.',
        'Package explicit runtime/model/avatar/reference allowlist with relocatable paths; exclude recordings, data ZIPs, personal data and secrets. Check rights for any publicly served reference assets.',
        'Replace visitor capture and recording with bounded local memory. Preserve timestamps, selective reobservation and comparison playback.',
        'If sending only keypoints, isolate sessions and prohibit payload logs/retention; reuse fixed TCN/feedback models. GPU rental is not automatically required for keypoint-only processing.',
        'Compare frozen public frontal cases and outputs, test abort/disconnect cleanup, then classroom camera/autostop/end-to-end wait. No new learning from visitors.',
    ],
    official_sources=[
        'https://docs.netlify.com/build/functions/overview/',
        'https://docs.netlify.com/build/server/get-started/',
        'https://onnxruntime.ai/docs/get-started/with-javascript/web.html',
        'https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html',
        'https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia',
    ],
)
(D / 'AUDIT.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(status=report['status'], pins=len(pin_results), preserved=True,
                      detector_mib=round(report['detector_onnx_total_bytes'] / 2**20, 1),
                      source_files=len(sources)), ensure_ascii=False))
