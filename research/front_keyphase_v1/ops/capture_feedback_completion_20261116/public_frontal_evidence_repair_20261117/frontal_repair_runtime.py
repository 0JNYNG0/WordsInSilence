"""Isolated frontal evidence/recording-readiness repair runtime."""
from pathlib import Path
import sys,json,hashlib
D=Path(__file__).resolve().parent;G=D.parent;N=G/'frontal_scope_recovery_20261117'
sys.path.insert(0,str(N))
from frontal_web_adapter import FeedbackConsumer as FrontConsumer,CaptureService as FrontCapture,Worker,bind_dependency
sys.path.insert(0,str(D))
from ready_capture import READY_CAMERA_LOOP

class EvidenceWorker(Worker):
    __init__=bind_dependency(Worker.__init__,'U',D/'apple_runtime')

class FeedbackConsumer(FrontConsumer):
    _original_init=bind_dependency(FrontConsumer.__init__,'AppleRoleWorker',EvidenceWorker)
    def __init__(self):
        self._original_init();self.evidence_admitted=False
        p=D/'web/ADMISSION.json'
        if p.exists():
            a=json.loads(p.read_text(encoding='utf-8'))
            self.evidence_admitted=a['admitted'] is True and all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in a['pins'].items())
        self.apple_admitted=self.apple_admitted and self.evidence_admitted
    def raw(self,raw,word):
        q=super().raw(raw,word);q['evidence_repair_version']='public_front_evidence_20261117';return q

class CaptureService(FrontCapture):
    prepare=bind_dependency(FrontCapture.prepare,'FeedbackConsumer',FeedbackConsumer)
    camera_loop=bind_dependency(FrontCapture.camera_loop,'_paced_camera_loop',READY_CAMERA_LOOP)
    def start(self,word):
        if not self.diagnostics.get('auto_end_ready',False):
            raise ValueError('양팔을 편하게 내리고 잠시 기다려 주세요. 자동 종료 준비가 되면 시작할 수 있어요.')
        return super().start(word)
    def status(self):
        q=super().status();q['auto_end_ready']=self.diagnostics.get('auto_end_ready',False)
        q['preparation_message']=self.diagnostics.get('preparation_message','양팔을 편하게 내리고 잠시 기다려 주세요.')
        q['evidence_repair_active']=bool(getattr(getattr(self,'consumer',None),'evidence_admitted',False))
        return q
