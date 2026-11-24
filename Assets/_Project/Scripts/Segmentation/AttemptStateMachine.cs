using System;
using System.Collections.Generic;
using UnityEngine;
using WordsInSilence.Calibration;
using WordsInSilence.Pose;
using WordsInSilence.Settings;

namespace WordsInSilence.Segmentation
{
    /// <summary>
    /// 중립 기반 자동 시작·종료·재개 상태 머신.
    /// WaitForNeutral → Armed → Capturing → EndCandidate → Evaluating
    /// 순수 C# — 외부에서 매 프레임 Tick()을 호출한다.
    /// </summary>
    public sealed class AttemptStateMachine
    {
        readonly RecognitionSettings _settings;
        readonly NeutralPoseDetector _neutralDetector;
        readonly MotionOnsetDetector _motionDetector;
        readonly CircularFrameBuffer _preBuffer;

        AttemptState _state = AttemptState.Inactive;
        string _attemptId;

        // 타이머
        float _armedElapsed;
        float _captureElapsed;
        float _endCandidateElapsed;
        float _trackingGapElapsed;
        float _captureGraceElapsed;

        // 추적 이탈 통계
        int _trackingGapCount;
        float _trackingGapTotalSeconds;

        // 캡처된 프레임
        readonly List<HandPoseFrame> _capturedFrames = new List<HandPoseFrame>();
        long _captureStartMs;

        // 이벤트
        public event Action<AttemptState, AttemptState> StateChanged;
        public event Action<AttemptCompletionResult> AttemptCompleted;

        // 읽기 전용 접근
        public AttemptState State => _state;
        public string AttemptId => _attemptId;
        public float CaptureElapsed => _captureElapsed;
        public int CapturedFrameCount => _capturedFrames.Count;

        public AttemptStateMachine(
            NeutralPoseDetector neutralDetector,
            RecognitionSettings settings = null)
        {
            _settings = settings ?? RecognitionSettings.Default;
            _neutralDetector = neutralDetector;
            _motionDetector = new MotionOnsetDetector(_settings);

            // 30fps 기준 0.35초 ≈ 11프레임
            int bufferSize = Mathf.Max(1, Mathf.CeilToInt(_settings.PreCaptureBufferSeconds * 30f));
            _preBuffer = new CircularFrameBuffer(bufferSize);
        }

        /// <summary>
        /// 상태 머신 시작. WaitForNeutral로 진입한다.
        /// </summary>
        public void Start()
        {
            ResetInternal();
            SetState(AttemptState.WaitForNeutral);
        }

        /// <summary>
        /// 상태 머신 정지.
        /// </summary>
        public void Stop()
        {
            SetState(AttemptState.Inactive);
            ResetInternal();
        }

        /// <summary>
        /// 사용자 취소 (Esc 등).
        /// </summary>
        public void Cancel()
        {
            if (_state == AttemptState.Inactive) return;

            var result = BuildResult(AttemptResultStatus.Invalid, "cancelled");
            SetState(AttemptState.Cancelled);
            AttemptCompleted?.Invoke(result);
        }

        /// <summary>
        /// Feedback 상태에서 다음 시도로 전환.
        /// </summary>
        public void AcknowledgeFeedback()
        {
            if (_state != AttemptState.Feedback) return;
            ResetInternal();
            SetState(AttemptState.WaitForNeutral);
        }

        /// <summary>
        /// 매 프레임 호출.
        /// </summary>
        public void Tick(HandPoseFrame frame, float deltaTime)
        {
            switch (_state)
            {
                case AttemptState.WaitForNeutral:
                    TickWaitForNeutral(frame, deltaTime);
                    break;
                case AttemptState.Armed:
                    TickArmed(frame, deltaTime);
                    break;
                case AttemptState.Capturing:
                    TickCapturing(frame, deltaTime);
                    break;
                case AttemptState.EndCandidate:
                    TickEndCandidate(frame, deltaTime);
                    break;
            }
        }

        // ─── WaitForNeutral ─────────────────────────────────────────────────

        void TickWaitForNeutral(HandPoseFrame frame, float deltaTime)
        {
            var neutralResult = _neutralDetector.CheckNeutral(frame, deltaTime);

            if (neutralResult.IsNeutral &&
                neutralResult.HoldTime >= _settings.NeutralHoldForArmed)
            {
                _neutralDetector.ResetHoldTimer();
                _motionDetector.Reset();
                _preBuffer.Clear();
                _armedElapsed = 0f;
                _attemptId = Guid.NewGuid().ToString("N").Substring(0, 12);
                SetState(AttemptState.Armed);
            }
        }

        // ─── Armed ──────────────────────────────────────────────────────────

        void TickArmed(HandPoseFrame frame, float deltaTime)
        {
            _armedElapsed += deltaTime;

            // 순환 버퍼에 프레임 추가
            if (frame != null)
                _preBuffer.Add(frame);

            // 시작 대기 타임아웃
            if (_armedElapsed >= _settings.ArmedTimeoutSeconds)
            {
                ResetInternal();
                SetState(AttemptState.WaitForNeutral);
                return;
            }

            // 캡처 시작 트리거: 움직임 감지 (body-relative 변위)
            bool moving = _motionDetector.Update(frame);

            // 보조 트리거: 보정 시 손이 안 보이던 사용자가 손을 올린 경우
            bool handAppeared = false;
            if (!moving && _neutralDetector.HasBaseline)
            {
                var baseline = _neutralDetector.Baseline;
                if (baseline != null && !baseline.HandsVisible)
                {
                    handAppeared = frame != null && (frame.HasLeftHand || frame.HasRightHand);
                }
            }

            if (moving || handAppeared)
            {
                BeginCapture(frame);
            }
        }

        // ─── Capturing ──────────────────────────────────────────────────────

        void TickCapturing(HandPoseFrame frame, float deltaTime)
        {
            _captureElapsed += deltaTime;
            _captureGraceElapsed += deltaTime;

            // 최대 수행 시간 초과 → 캡처 종료 후 평가 (invalid 아님)
            if (_captureElapsed >= _settings.MaxCaptureDurationSeconds)
            {
                FinishWithEvaluation();
                return;
            }

            // 추적 gap 검사
            if (frame == null || !frame.HasPose)
            {
                if (_trackingGapElapsed == 0f)
                    _trackingGapCount++;
                _trackingGapElapsed += deltaTime;
                _trackingGapTotalSeconds += deltaTime;
                if (_trackingGapElapsed > _settings.TrackingGapToleranceSeconds)
                {
                    // 추적 이탈이 길면 캡처된 프레임으로 평가 (invalid 대신)
                    FinishWithEvaluation("camera_lost");
                    return;
                }
                return;
            }

            _trackingGapElapsed = 0f;
            _capturedFrames.Add(frame);

            // 유예기간 동안 중립 체크 안 함
            if (_captureGraceElapsed < _settings.CaptureGracePeriodSeconds)
                return;

            // 중립 복귀 체크
            var neutralResult = _neutralDetector.CheckNeutral(frame, deltaTime);
            if (neutralResult.IsNeutral)
            {
                _endCandidateElapsed = neutralResult.HoldTime;
                SetState(AttemptState.EndCandidate);
            }
        }

        // ─── EndCandidate ───────────────────────────────────────────────────

        void TickEndCandidate(HandPoseFrame frame, float deltaTime)
        {
            _captureElapsed += deltaTime;

            // 최대 수행 시간 초과 → 캡처 종료 후 평가
            if (_captureElapsed >= _settings.MaxCaptureDurationSeconds)
            {
                FinishWithEvaluation();
                return;
            }

            if (frame != null)
                _capturedFrames.Add(frame);

            var neutralResult = _neutralDetector.CheckNeutral(frame, deltaTime);

            if (!neutralResult.IsNeutral)
            {
                // 중립 깨짐 → 새 캡처 시작 (타이머·프레임 초기화)
                _capturedFrames.Clear();
                _captureElapsed = 0f;
                _captureGraceElapsed = 0f;
                _trackingGapElapsed = 0f;
                _motionDetector.Reset();
                _preBuffer.Clear();
                _neutralDetector.ResetHoldTimer();
                _attemptId = Guid.NewGuid().ToString("N").Substring(0, 12);
                SetState(AttemptState.Capturing);
                return;
            }

            _endCandidateElapsed = neutralResult.HoldTime;

            // 3초 유지 → 평가
            if (_endCandidateElapsed >= _settings.NeutralHoldForEnd)
            {
                FinishWithEvaluation();
            }
        }

        // ─── 평가 완료 공통 ─────────────────────────────────────────────────

        void FinishWithEvaluation(string extraReason = null)
        {
            SetState(AttemptState.Evaluating);

            var trimmedFrames = TrimCapturedFrames();
            var result = BuildResult(AttemptResultStatus.Passed, extraReason, trimmedFrames);
            result.TrackingGapCount = _trackingGapCount;
            result.TrackingGapTotalSeconds = _trackingGapTotalSeconds;
            AttemptCompleted?.Invoke(result);

            SetState(AttemptState.Feedback);
        }

        // ─── 캡처 시작 공통 ─────────────────────────────────────────────────

        void BeginCapture(HandPoseFrame frame)
        {
            _capturedFrames.Clear();
            var preFrames = _preBuffer.ToArray();
            _capturedFrames.AddRange(preFrames);
            if (frame != null) _capturedFrames.Add(frame);

            _captureStartMs = frame?.TimestampMs ?? 0;
            _captureElapsed = 0f;
            _captureGraceElapsed = 0f;
            _trackingGapElapsed = 0f;
            SetState(AttemptState.Capturing);
        }

        // ─── 구간 정제 ──────────────────────────────────────────────────────

        List<HandPoseFrame> TrimCapturedFrames()
        {
            if (_capturedFrames.Count == 0) return _capturedFrames;

            float postMarginMs = _settings.PostCaptureMarginSeconds * 1000f;
            float endNeutralMs = _settings.NeutralHoldForEnd * 1000f;

            long lastMs = _capturedFrames[_capturedFrames.Count - 1].TimestampMs;
            long neutralStartMs = lastMs - (long)endNeutralMs;
            long cutoffMs = neutralStartMs + (long)postMarginMs;

            var trimmed = new List<HandPoseFrame>();
            foreach (var f in _capturedFrames)
            {
                if (f.TimestampMs <= cutoffMs)
                    trimmed.Add(f);
            }

            if (trimmed.Count < 5 && _capturedFrames.Count >= 5)
                return new List<HandPoseFrame>(_capturedFrames);

            return trimmed;
        }

        // ─── 유틸 ───────────────────────────────────────────────────────────

        AttemptCompletionResult BuildResult(
            AttemptResultStatus status,
            string invalidReason,
            List<HandPoseFrame> trimmedFrames = null)
        {
            return new AttemptCompletionResult
            {
                AttemptId = _attemptId,
                Status = status,
                InvalidReason = invalidReason,
                CapturedFrames = trimmedFrames ?? new List<HandPoseFrame>(_capturedFrames),
                TotalCapturedCount = _capturedFrames.Count,
                CaptureDurationSeconds = _captureElapsed,
            };
        }

        void SetState(AttemptState newState)
        {
            if (_state == newState) return;
            var old = _state;
            _state = newState;
            StateChanged?.Invoke(old, newState);
        }

        void ResetInternal()
        {
            _attemptId = null;
            _armedElapsed = 0f;
            _captureElapsed = 0f;
            _endCandidateElapsed = 0f;
            _trackingGapElapsed = 0f;
            _captureGraceElapsed = 0f;
            _capturedFrames.Clear();
            _preBuffer.Clear();
            _motionDetector.Reset();
            _neutralDetector.ResetHoldTimer();
            _trackingGapCount = 0;
            _trackingGapTotalSeconds = 0f;
        }
    }

    /// <summary>
    /// 상태 머신이 attempt 완료 시 발행하는 결과.
    /// </summary>
    public sealed class AttemptCompletionResult
    {
        public string AttemptId;
        public AttemptResultStatus Status;
        public string InvalidReason;
        public List<HandPoseFrame> CapturedFrames;
        public int TotalCapturedCount;
        public float CaptureDurationSeconds;
        /// <summary>캡처 중 추적이 끊긴 횟수.</summary>
        public int TrackingGapCount;
        /// <summary>추적 이탈 총 시간 (초).</summary>
        public float TrackingGapTotalSeconds;
    }
}
