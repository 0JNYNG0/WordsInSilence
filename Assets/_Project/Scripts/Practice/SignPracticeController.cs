using System;
using System.Collections.Generic;
using System.Threading;
using TMPro;
using UnityEngine;
using UnityEngine.UI;
using WordsInSilence.Calibration;
using WordsInSilence.Camera;
using WordsInSilence.Core;
using WordsInSilence.Personalization;
using WordsInSilence.Pose;
using WordsInSilence.Recognition;
using WordsInSilence.Segmentation;
using WordsInSilence.Settings;

namespace WordsInSilence.Practice
{
    /// <summary>
    /// 수어 연습 게임 씬의 메인 컨트롤러.
    /// HandPoseIntegrator로부터 프레임을 받아
    /// 보정 → 중립 감지 → 자동 캡처 → 인식 → 피드백 흐름을 관리한다.
    ///
    /// Unity Editor에서 씬 배선:
    /// 1. 빈 GameObject에 이 스크립트를 추가
    /// 2. _integrator에 HandPoseIntegrator 드래그
    /// 3. TMP_Text 4개를 UI Canvas에 만들어 연결
    /// 4. _profileId에 사용자 이름 입력 (또는 "default")
    /// </summary>
    public sealed class SignPracticeController : MonoBehaviour
    {
        [Header("Camera")]
        [SerializeField] RawImage _cameraDisplay;
        [SerializeField] bool _mirrorDisplay = true;

        [Header("Pipeline")]
        [SerializeField] HandPoseIntegrator _integrator;

        [Header("Settings")]
        [SerializeField] RecognitionSettings _settings;
        [SerializeField] string _profileId = "default";
        [SerializeField] string[] _targetSignIds = { "인사" };

        [Header("UI (optional)")]
        [SerializeField] TMP_Text _textState;
        [SerializeField] TMP_Text _textGuidance;
        [SerializeField] TMP_Text _textScore;
        [SerializeField] TMP_Text _textEnrollment;

        // ─── 내부 시스템 ────────────────────────────────────────────────

        enum Phase { Calibration, Practice }
        Phase _phase = Phase.Calibration;

        CalibrationController _calibration;
        NeutralPoseDetector _neutralDetector;
        AttemptStateMachine _stateMachine;
        SignRecognizer _recognizer;
        PersonalizationService _personalization;
        SignGameBridge _gameBridge;

        // 레퍼런스 로딩
        List<SignReference> _references = new List<SignReference>();

        // 카메라
        UnityWebCamFrameSource _cameraSource;
        CancellationTokenSource _cameraCts;

        // 최신 프레임
        HandPoseFrame _latestFrame;

        // 진단용 (문제 해결 후 제거)
        float _diagTimer;
        int _diagNullCount;
        int _diagFrameCount;

        // ─── MonoBehaviour ──────────────────────────────────────────────

        async void Start()
        {
            // 카메라 자동 시작
            await AutoStartCameraAsync();

            var settings = _settings != null ? _settings : RecognitionSettings.Default;

            // 레퍼런스 로드
            LoadReferences();

            // 보정 시스템
            _calibration = new CalibrationController(settings);
            _calibration.StateChanged += OnCalibrationStateChanged;
            _calibration.GuidanceChanged += OnGuidanceChanged;

            // 중립 감지기 (보정 완료 후 상태 머신과 공유)
            _neutralDetector = new NeutralPoseDetector(settings);

            // 인식기
            _recognizer = new SignRecognizer(settings);

            // 개인화
            _personalization = new PersonalizationService(_profileId, settings);

            // 게임 브릿지
            _gameBridge = new SignGameBridge();
            _gameBridge.SetAllowedSigns(_targetSignIds);
            _gameBridge.OnSignSuccess += OnSignSuccess;
            _gameBridge.OnSignRetry += OnSignRetry;
            _gameBridge.OnCameraRecovery += OnCameraRecovery;

            // 프레임 구독
            if (_integrator != null)
                _integrator.UnifiedFrameArrived += OnFrame;

            // 보정 시작
            var cameraInfo = new CameraFingerprint { DeviceName = "default" };
            _calibration.Begin(_profileId, cameraInfo);
            _phase = Phase.Calibration;

            SetState("보정 중...");
            SetGuidance("카메라 앞에서 상체가 보이도록 앉아 주세요.");
        }

        void OnDestroy()
        {
            if (_integrator != null)
                _integrator.UnifiedFrameArrived -= OnFrame;

            _cameraCts?.Cancel();
            _cameraCts?.Dispose();
            if (_cameraSource is IDisposable disposable)
                disposable.Dispose();
        }

        void OnFrame(HandPoseFrame frame)
        {
            _latestFrame = frame;
        }

        void Update()
        {
            if (_latestFrame == null && _phase == Phase.Practice) return;

            float dt = Time.deltaTime;

            switch (_phase)
            {
                case Phase.Calibration:
                    _calibration.Tick(_latestFrame, dt);

                    if (_calibration.State == CalibrationState.Complete)
                    {
                        StartPractice();
                    }
                    break;

                case Phase.Practice:
                    _stateMachine?.Tick(_latestFrame, dt);
                    UpdatePracticeUI();
                    // ─── 진단 로그 (문제 해결 후 제거) ───
                    _diagTimer += dt;
                    if (_latestFrame != null) _diagFrameCount++;
                    else _diagNullCount++;
                    if (_diagTimer >= 2f)
                    {
                        var f = _latestFrame;
                        string frameInfo = f == null ? "null" :
                            $"HasPose={f.HasPose} Shoulders={f.HasValidShoulders} Q={f.Quality}";
                        string neutralInfo = _neutralDetector == null ? "detector=null" :
                            $"baseline={_neutralDetector.HasBaseline} neutral={_neutralDetector.IsCurrentlyNeutral} hold={_neutralDetector.NeutralHoldTime:F2} why={_neutralDetector.LastReason ?? "ok"}";
                        string smInfo = _stateMachine == null ? "sm=null" : $"state={_stateMachine.State}";
                        Debug.Log($"[Diag] {smInfo} | frame: {frameInfo} | {neutralInfo} | frames={_diagFrameCount} nulls={_diagNullCount}");
                        _diagTimer = 0f;
                        _diagFrameCount = 0;
                        _diagNullCount = 0;
                    }
                    break;
            }

            // ESC로 취소
            if (Input.GetKeyDown(KeyCode.Escape))
            {
                if (_phase == Phase.Practice && _stateMachine != null)
                    _stateMachine.Cancel();
            }

            _latestFrame = null; // 소비 완료
        }

        // ─── 카메라 자동 시작 ──────────────────────────────────────────

        async System.Threading.Tasks.Task AutoStartCameraAsync()
        {
            if (_cameraDisplay == null)
            {
                Debug.LogWarning("[SignPractice] _cameraDisplay(RawImage)가 연결되지 않았습니다. 카메라를 자동 시작하지 않습니다.");
                return;
            }

            _cameraSource = new UnityWebCamFrameSource();
            _cameraCts = new CancellationTokenSource();

            var request = new CameraStartRequest
            {
                DeviceName = null, // 기본 카메라
                RequestedWidth = 640,
                RequestedHeight = 480,
                RequestedFps = 30,
                MirrorDisplay = _mirrorDisplay
            };

            var result = await _cameraSource.StartAsync(request, _cameraCts.Token);

            if (!result.IsSuccess)
            {
                Debug.LogError($"[SignPractice] 카메라 시작 실패: {result.ErrorDetail} [{result.ErrorCode}]");
                SetGuidance($"카메라 시작 실패: {result.ErrorDetail}");
                return;
            }

            _cameraDisplay.texture = _cameraSource.ActiveTexture;

            // 미러 표시
            if (_mirrorDisplay)
                _cameraDisplay.uvRect = new Rect(1f, 0f, -1f, 1f);
            else
                _cameraDisplay.uvRect = new Rect(0f, 0f, 1f, 1f);

            // 회전 보정
            int rotation = _cameraSource.ActiveTexture.videoRotationAngle;
            if (rotation != 0)
                _cameraDisplay.rectTransform.localRotation = Quaternion.Euler(0f, 0f, -rotation);

            Debug.Log($"[SignPractice] 카메라 자동 시작 완료: {result.DeviceName} ({result.ActualWidth}x{result.ActualHeight})");
        }

        // ─── 보정 → 연습 전환 ──────────────────────────────────────────

        void StartPractice()
        {
            _phase = Phase.Practice;

            // 보정 결과에서 중립 기준 가져오기
            if (_calibration.Profile?.Neutral != null)
                _neutralDetector.SetBaseline(_calibration.Profile.Neutral);

            // 상태 머신 시작
            var settings = _settings != null ? _settings : RecognitionSettings.Default;
            _stateMachine = new AttemptStateMachine(_neutralDetector, settings);
            _stateMachine.StateChanged += OnAttemptStateChanged;
            _stateMachine.AttemptCompleted += OnAttemptCompleted;
            _stateMachine.Start();

            SignPracticeEvents.EmitCalibrationCompleted(_profileId);
            SetState("연습 준비");
            SetGuidance("양손을 편안히 내리고 준비해 주세요.");
        }

        // ─── 보정 콜백 ─────────────────────────────────────────────────

        void OnCalibrationStateChanged(CalibrationState state)
        {
            SetState($"보정: {state}");

            // C4 확인 단계면 자동 수락 (간소화)
            if (state == CalibrationState.Confirmation)
                _calibration.ConfirmCalibration();
        }

        void OnGuidanceChanged(string msg)
        {
            if (!string.IsNullOrEmpty(msg))
                SetGuidance(msg);
        }

        // ─── 상태 머신 콜백 ────────────────────────────────────────────

        void OnAttemptStateChanged(AttemptState from, AttemptState to)
        {
            switch (to)
            {
                case AttemptState.WaitForNeutral:
                    SetState("대기");
                    SetGuidance("양손을 편안히 내리고 준비해 주세요.");
                    break;
                case AttemptState.Armed:
                    SetState("준비 완료");
                    SetGuidance("동작을 따라 한 뒤 다시 손을 내려 주세요.");
                    SignPracticeEvents.EmitAttemptArmed(_targetSignIds.Length > 0 ? _targetSignIds[0] : "");
                    break;
                case AttemptState.Capturing:
                    if (from == AttemptState.EndCandidate)
                    {
                        SetState("캡처 재개");
                        SignPracticeEvents.EmitAttemptResumed(_stateMachine.AttemptId);
                    }
                    else
                    {
                        SetState("캡처 중...");
                        SignPracticeEvents.EmitAttemptStarted(_stateMachine.AttemptId);
                    }
                    SetGuidance("");
                    break;
                case AttemptState.EndCandidate:
                    SetState("종료 대기...");
                    SetGuidance("재측정하려면 다시 손을 올려주세요!");
                    var settings = _settings != null ? _settings : RecognitionSettings.Default;
                    SignPracticeEvents.EmitEndCandidateStarted(settings.NeutralHoldForEnd);
                    break;
                case AttemptState.Cancelled:
                    SetState("취소됨");
                    SetGuidance("ESC로 취소했습니다. 다시 시작하려면 손을 내려 주세요.");
                    break;
                case AttemptState.InvalidAttempt:
                    SetState("무효");
                    SetGuidance("동작을 인식하지 못했습니다. 다시 시도해 주세요.");
                    break;
                case AttemptState.Feedback:
                    // OnAttemptCompleted에서 처리
                    break;
            }
        }

        void OnAttemptCompleted(AttemptCompletionResult completion)
        {
            // 인식
            var result = _recognizer.Recognize(completion.CapturedFrames, _references);

            // 개인 점수 (등록된 경우)
            if (result.UserFeatures != null && !string.IsNullOrEmpty(result.BestSignId))
            {
                float personalScore = _personalization.ComputePersonalScore(
                    result.BestSignId, result.UserFeatures, result.GlobalScore);
                result.PersonalScore = personalScore;
                result.CombinedScore = _personalization.ComputeCombinedScore(
                    result.BestSignId, result.GlobalScore, personalScore);
            }

            // 피드백 표시
            var messages = SignFeedbackMapper.BuildMessages(result);
            SetState(messages.StatusMessage);

            string scoreText = $"점수: {result.GlobalScore:P0}";
            if (!float.IsNaN(result.CombinedScore))
                scoreText += $" (혼합: {result.CombinedScore:P0})";
            if (!string.IsNullOrEmpty(result.BestSignId))
                scoreText += $"\n인식: {result.BestSignId}";
            SetScore(scoreText);

            // 피드백 메시지
            string guidance = "";

            // 추적 이탈 안내
            if (completion.TrackingGapCount > 0)
            {
                if (!string.IsNullOrEmpty(completion.InvalidReason) && completion.InvalidReason == "camera_lost")
                    guidance += "[카메라] 손이 카메라 밖으로 나갔습니다. 동작 중 손이 화면 안에 있도록 해 주세요.\n";
                else
                    guidance += $"[카메라] 추적이 {completion.TrackingGapCount}회 끊겼습니다 ({completion.TrackingGapTotalSeconds:F1}초). 손이 화면 안에 있도록 해 주세요.\n";
            }

            if (messages.HasCameraIssues)
                guidance += "[카메라] " + string.Join(", ", messages.CameraMessages) + "\n";
            if (messages.HasPerformanceIssues)
                guidance += string.Join("\n", messages.PerformanceMessages);
            if (string.IsNullOrEmpty(guidance))
                guidance = messages.StatusMessage;
            SetGuidance(guidance);

            // 게임 이벤트
            var evt = new SignAttemptEvent
            {
                AttemptId = completion.AttemptId,
                SignId = result.BestSignId,
                Status = result.Status,
                GlobalScore = result.GlobalScore,
                CombinedScore = float.IsNaN(result.CombinedScore) ? result.GlobalScore : result.CombinedScore,
                FeedbackCodes = result.FeedbackCodes,
            };
            SignPracticeEvents.EmitAttemptFinished(evt);
            _gameBridge.ProcessResult(evt, Time.realtimeSinceStartup);

            // 등록 시도 (passed일 때)
            if (result.Status == AttemptResultStatus.Passed && !string.IsNullOrEmpty(result.BestSignId))
            {
                var enrollResult = _personalization.TryEnroll(
                    result.BestSignId, result, completion.CapturedFrames,
                    _calibration.Session?.SessionId ?? "unknown");

                if (enrollResult.Accepted)
                {
                    SetEnrollment($"등록: {enrollResult.AcceptedTotal}/{enrollResult.RequiredTotal}");
                    SignPracticeEvents.EmitEnrollmentProgress(
                        result.BestSignId, enrollResult.AcceptedTotal, enrollResult.RequiredTotal);

                    if (enrollResult.EnrollmentReady)
                        SetEnrollment($"{result.BestSignId} 학습 완료!");
                }
            }

            // 피드백 확인 후 다음 시도로 (2초 후 자동)
            Invoke(nameof(AcknowledgeAndRestart), 2.5f);
        }

        void AcknowledgeAndRestart()
        {
            if (_stateMachine != null && _stateMachine.State == AttemptState.Feedback)
            {
                _stateMachine.AcknowledgeFeedback();
                SetScore("");
            }
        }

        // ─── 게임 브릿지 콜백 ──────────────────────────────────────────

        void OnSignSuccess(string signId)
        {
            Debug.Log($"[SignPractice] 성공: {signId}");
        }

        void OnSignRetry(string signId)
        {
            Debug.Log($"[SignPractice] 재시도: {signId}");
        }

        void OnCameraRecovery(string signId)
        {
            Debug.Log($"[SignPractice] 카메라 복구 필요: {signId}");
        }

        // ─── 레퍼런스 로딩 ─────────────────────────────────────────────

        void LoadReferences()
        {
            string[] motionIds = HandPoseRecordingRepository.ListAll();
            foreach (string id in motionIds)
            {
                var recording = HandPoseRecordingRepository.Load(id);
                if (recording == null) continue;

                // 대상 수어만 필터링 (설정된 경우)
                bool isTarget = _targetSignIds == null || _targetSignIds.Length == 0;
                if (!isTarget)
                {
                    foreach (var targetId in _targetSignIds)
                    {
                        string displayName = string.IsNullOrEmpty(recording.DisplayName)
                            ? recording.MotionId : recording.DisplayName;
                        if (displayName.Contains(targetId) || recording.MotionId.Contains(targetId))
                        {
                            isTarget = true;
                            break;
                        }
                    }
                }

                if (!isTarget) continue;

                // 특징 사전 추출
                SignFeatureSet features = null;
                if (recording.Frames != null && recording.Frames.Length >= 3)
                    features = SignFeatureExtractor.ExtractFromRecording(recording.Frames);

                _references.Add(new SignReference
                {
                    SignId = recording.MotionId,
                    DisplayName = string.IsNullOrEmpty(recording.DisplayName)
                        ? recording.MotionId : recording.DisplayName,
                    Recording = recording,
                    Features = features,
                });
            }

            Debug.Log($"[SignPractice] 레퍼런스 {_references.Count}개 로드");
        }

        // ─── UI 업데이트 ───────────────────────────────────────────────

        void UpdatePracticeUI()
        {
            if (_stateMachine == null) return;

            // 캡처 중일 때 경과 시간 표시
            if (_stateMachine.State == AttemptState.Capturing)
            {
                SetState($"캡처 중... {_stateMachine.CaptureElapsed:F1}초");
            }
        }

        void SetState(string text)
        {
            if (_textState != null) _textState.text = text;
        }

        void SetGuidance(string text)
        {
            if (_textGuidance != null) _textGuidance.text = text;
        }

        void SetScore(string text)
        {
            if (_textScore != null) _textScore.text = text;
        }

        void SetEnrollment(string text)
        {
            if (_textEnrollment != null) _textEnrollment.text = text;
        }
    }
}
