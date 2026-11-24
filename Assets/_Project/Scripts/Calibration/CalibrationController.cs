using System;
using System.Collections.Generic;
using UnityEngine;
using WordsInSilence.Pose;
using WordsInSilence.Settings;

namespace WordsInSilence.Calibration
{
    /// <summary>
    /// 2단계 보정 흐름을 관리한다.
    ///   Step 1 (BodyMeasurement): 팔을 올려 어깨·팔꿈치·손이 보이는 상태 유지 → 신체 비율 측정
    ///   Step 2 (NeutralCalibration): 편안한 자세로 2~3초 유지 → 중립 자세 등록
    /// 각 단계에서 안정성을 검증하고, 불안정하면 재측정을 요청한다.
    /// MonoBehaviour가 아닌 순수 C# — 외부에서 매 프레임 Tick()을 호출한다.
    /// </summary>
    public sealed class CalibrationController
    {
        readonly RecognitionSettings _settings;
        readonly CameraReadinessEvaluator _framingEvaluator;
        readonly NeutralPoseDetector _neutralDetector;
        readonly ReachCheckEvaluator _reachEvaluator;

        CalibrationState _state = CalibrationState.Idle;
        float _framingGoodTime;
        const float FramingGoodRequired = 1.5f;

        // Body Measurement (Step 1)
        readonly List<HandPoseFrame> _bodyMeasurementFrames = new List<HandPoseFrame>();
        float _bodyMeasurementElapsed;
        int _bodyMeasurementRetryCount;

        // Neutral Calibration (Step 2)
        float _neutralTransitionElapsed;
        readonly List<HandPoseFrame> _neutralFrames = new List<HandPoseFrame>();
        float _neutralCollectionElapsed;
        int _neutralRetryCount;

        // 결과
        ActiveSessionCalibration _session;
        UserCalibrationProfile _profile;
        string _activeProfileId;

        public CalibrationState State => _state;
        public ActiveSessionCalibration Session => _session;
        public UserCalibrationProfile Profile => _profile;
        public string ActiveProfileId => _activeProfileId;
        public NeutralPoseDetector NeutralDetector => _neutralDetector;

        // 상태 변경 이벤트
        public event Action<CalibrationState> StateChanged;
        // UI 안내 메시지
        public event Action<string> GuidanceChanged;

        // 하위 평가기 결과 (UI 표시용)
        public FramingResult LastFramingResult { get; private set; }
        public NeutralCheckResult LastNeutralResult { get; private set; }
        public ReachCheckFrameResult LastReachResult { get; private set; }
        public NeutralCalibrationResult LastCalibrationResult { get; private set; }
        public BodyMeasurementResult LastBodyMeasurementResult { get; private set; }

        public CalibrationController(RecognitionSettings settings = null)
        {
            _settings = settings ?? RecognitionSettings.Default;
            _framingEvaluator = new CameraReadinessEvaluator(_settings);
            _neutralDetector = new NeutralPoseDetector(_settings);
            _reachEvaluator = new ReachCheckEvaluator(_settings);
        }

        /// <summary>
        /// C0: 프로필 선택 후 보정 시작.
        /// </summary>
        public void Begin(string profileId, CameraFingerprint cameraInfo)
        {
            _activeProfileId = profileId;

            // 프로필 로드 또는 신규 생성
            _profile = UserProfileRepository.Load(profileId);
            bool isNewProfile = _profile == null;

            if (isNewProfile)
            {
                _profile = new UserCalibrationProfile
                {
                    UserProfileId = profileId,
                    CreatedUtc = DateTime.UtcNow.ToString("o"),
                };
            }

            // 세션 생성
            _session = new ActiveSessionCalibration
            {
                SessionId = Guid.NewGuid().ToString("N").Substring(0, 12),
                UserProfileId = profileId,
                CreatedUtc = DateTime.UtcNow.ToString("o"),
                CameraInfo = cameraInfo ?? new CameraFingerprint(),
            };

            // 카메라 변경 확인
            bool cameraChanged = UserProfileRepository.HasCameraChanged(_profile, cameraInfo);
            _session.SafeFrame.FullReachCheckRequired = isNewProfile || cameraChanged;

            _framingGoodTime = 0f;
            _bodyMeasurementFrames.Clear();
            _bodyMeasurementRetryCount = 0;
            _neutralFrames.Clear();
            _neutralRetryCount = 0;
            _reachEvaluator.Reset();

            SetState(CalibrationState.FramingCheck);
        }

        /// <summary>
        /// 매 프레임 호출.
        /// </summary>
        public void Tick(HandPoseFrame frame, float deltaTime)
        {
            switch (_state)
            {
                case CalibrationState.FramingCheck:
                    TickFramingCheck(frame, deltaTime);
                    break;
                case CalibrationState.BodyMeasurement:
                    TickBodyMeasurement(frame, deltaTime);
                    break;
                case CalibrationState.NeutralCalibration:
                    TickNeutralCalibration(frame, deltaTime);
                    break;
                case CalibrationState.ReachCheck:
                    TickReachCheck(frame, deltaTime);
                    break;
                case CalibrationState.Confirmation:
                    break;
            }
        }

        // ─── Step 0: Framing Check ────────────────────────────────────────

        void TickFramingCheck(HandPoseFrame frame, float deltaTime)
        {
            LastFramingResult = _framingEvaluator.EvaluateFraming(frame);

            if (LastFramingResult.IsGood)
            {
                _framingGoodTime += deltaTime;
                if (_framingGoodTime >= FramingGoodRequired)
                {
                    // → Step 1: Body Measurement
                    _bodyMeasurementFrames.Clear();
                    _bodyMeasurementElapsed = 0f;
                    SetState(CalibrationState.BodyMeasurement);
                    EmitGuidance("신체 정보를 측정합니다. 어깨, 팔꿈치, 손이 모두 보이게 유지해 주세요.");
                }
                else
                {
                    EmitGuidance($"프레이밍 양호. {FramingGoodRequired - _framingGoodTime:F1}초 유지...");
                }
            }
            else
            {
                _framingGoodTime = 0f;
                EmitGuidance(LastFramingResult.Guidance);
            }
        }

        // ─── Step 1: Body Measurement ─────────────────────────────────────

        void TickBodyMeasurement(HandPoseFrame frame, float deltaTime)
        {
            _bodyMeasurementElapsed += deltaTime;

            // 프레임이 유효하면 수집
            if (frame != null && frame.HasPose && frame.HasValidShoulders)
            {
                _bodyMeasurementFrames.Add(frame);
            }

            float holdSeconds = _settings.BodyMeasurementHoldSeconds;
            float remaining = holdSeconds - _bodyMeasurementElapsed;

            if (remaining > 0f)
            {
                EmitGuidance($"신체 측정 중... 자세를 유지해 주세요. ({remaining:F0}초)");
                return;
            }

            // 수집 완료 → 안정성 검증
            LastBodyMeasurementResult = BodyScaleCalculator.ComputeWithValidation(_bodyMeasurementFrames);

            if (LastBodyMeasurementResult.Success)
            {
                // 세션·프로필에 반영
                _profile.BodyScale = LastBodyMeasurementResult.Scale;
                _session.CurrentScale.ShoulderWidthNorm = LastBodyMeasurementResult.Scale.ShoulderWidthNorm;

                // → Step 2: Neutral Calibration (전환 대기 포함)
                _neutralDetector.BeginCalibration();
                _neutralTransitionElapsed = 0f;
                _neutralFrames.Clear();
                _neutralCollectionElapsed = 0f;
                SetState(CalibrationState.NeutralCalibration);
                EmitGuidance("신체 측정 완료! 이제 편안한 자세로 돌아가 주세요...");
            }
            else
            {
                // 재시도
                _bodyMeasurementRetryCount++;
                if (_bodyMeasurementRetryCount >= _settings.BodyMeasurementMaxRetries)
                {
                    SetState(CalibrationState.Failed);
                    EmitGuidance("신체 측정에 실패했습니다. 보정을 다시 시작해 주세요.");
                }
                else
                {
                    _bodyMeasurementFrames.Clear();
                    _bodyMeasurementElapsed = 0f;
                    EmitGuidance($"{LastBodyMeasurementResult.FailureReason} (재시도 {_bodyMeasurementRetryCount}/{_settings.BodyMeasurementMaxRetries})");
                }
            }
        }

        // ─── Step 2: Neutral Calibration ──────────────────────────────────

        void TickNeutralCalibration(HandPoseFrame frame, float deltaTime)
        {
            float transitionWait = _settings.NeutralTransitionWaitSeconds;

            // 전환 대기: 사용자가 팔을 내릴 시간
            _neutralTransitionElapsed += deltaTime;
            if (_neutralTransitionElapsed < transitionWait)
            {
                float remaining = transitionWait - _neutralTransitionElapsed;
                EmitGuidance($"편안한 자세로 돌아가 주세요... ({remaining:F0}초)");
                return;
            }

            // 프레임 수집
            if (frame != null && frame.HasPose && frame.HasValidShoulders)
            {
                _neutralDetector.AddCalibrationFrame(frame);
                _neutralFrames.Add(frame);
            }

            _neutralCollectionElapsed += deltaTime;
            float holdSeconds = _settings.NeutralCalibrationHoldSeconds;
            float collectionRemaining = holdSeconds - _neutralCollectionElapsed;

            if (collectionRemaining > 0f)
            {
                EmitGuidance($"중립 자세 등록 중... 움직이지 말고 기다려 주세요. ({collectionRemaining:F0}초)");
                return;
            }

            // 수집 완료 → 중립 보정
            LastCalibrationResult = _neutralDetector.FinishCalibration();

            if (LastCalibrationResult.Success)
            {
                // 중립 기준의 안정성 추가 검증
                string stabilityError = ValidateNeutralStability(_neutralFrames);
                if (stabilityError != null)
                {
                    // 불안정 → 재시도
                    _neutralRetryCount++;
                    if (_neutralRetryCount >= _settings.BodyMeasurementMaxRetries)
                    {
                        SetState(CalibrationState.Failed);
                        EmitGuidance("중립 자세 등록에 실패했습니다. 보정을 다시 시작해 주세요.");
                        return;
                    }

                    _neutralDetector.BeginCalibration();
                    _neutralFrames.Clear();
                    _neutralCollectionElapsed = 0f;
                    EmitGuidance($"{stabilityError} (재시도 {_neutralRetryCount}/{_settings.BodyMeasurementMaxRetries})");
                    return;
                }

                // 세션에 반영
                _session.Neutral.BottomExitValidated = LastCalibrationResult.Baseline.BottomExitAllowed;
                _session.Neutral.MotionNoiseFloor = LastCalibrationResult.Baseline.MotionNoiseFloor;
                _session.Quality.ValidFrameRatio = LastCalibrationResult.ValidFrameRatio;

                // 프로필에 반영
                _profile.Neutral = LastCalibrationResult.Baseline;
                _profile.Camera = _session.CameraInfo;

                // ReachCheck 생략 (노트북 환경)
                _session.SafeFrame.QuickCheckPassed = true;
                FinalizeCalibration("good");
                SetState(CalibrationState.Confirmation);
                EmitGuidance("보정 완료! 확인해 주세요.");
            }
            else
            {
                // 보정 실패 → 재시도
                _neutralRetryCount++;
                if (_neutralRetryCount >= _settings.BodyMeasurementMaxRetries)
                {
                    SetState(CalibrationState.Failed);
                    EmitGuidance("중립 자세 등록에 실패했습니다. 보정을 다시 시작해 주세요.");
                }
                else
                {
                    _neutralDetector.BeginCalibration();
                    _neutralFrames.Clear();
                    _neutralCollectionElapsed = 0f;
                    EmitGuidance($"{LastCalibrationResult.FailureReason} (재시도 {_neutralRetryCount}/{_settings.BodyMeasurementMaxRetries})");
                }
            }
        }

        /// <summary>
        /// 중립 프레임의 body-relative 손목 위치 안정성을 검증한다.
        /// 안정하면 null, 불안정하면 오류 메시지를 반환한다.
        /// </summary>
        string ValidateNeutralStability(List<HandPoseFrame> frames)
        {
            if (frames.Count < 5) return null; // 샘플 부족 시 통과 (FinishCalibration에서 이미 검증)

            var leftXs = new List<float>();
            var leftYs = new List<float>();
            var rightXs = new List<float>();
            var rightYs = new List<float>();

            foreach (var f in frames)
            {
                if (!f.IsValidForBodyRelativeEvaluation) continue;
                if (float.IsNaN(f.LeftWristBodyRelative.x)) continue;

                leftXs.Add(f.LeftWristBodyRelative.x);
                leftYs.Add(f.LeftWristBodyRelative.y);
                rightXs.Add(f.RightWristBodyRelative.x);
                rightYs.Add(f.RightWristBodyRelative.y);
            }

            if (leftXs.Count < 3) return null; // body-relative 데이터 부족 → 통과

            // MAD로 안정성 측정 (어깨너비 단위)
            float maxSpread = Mathf.Max(
                ComputeMAD(leftXs), ComputeMAD(leftYs),
                ComputeMAD(rightXs), ComputeMAD(rightYs));

            // 어깨너비 단위 0.08 이상의 분산 → 불안정
            if (maxSpread > 0.08f)
            {
                return "측정 중 움직임이 감지되었습니다. 자세를 고정하고 다시 시도해 주세요.";
            }

            return null;
        }

        static float ComputeMAD(List<float> values)
        {
            if (values.Count == 0) return 0f;
            var sorted = new List<float>(values);
            sorted.Sort();
            float median = sorted[sorted.Count / 2];
            var deviations = new List<float>(values.Count);
            foreach (var v in values)
                deviations.Add(Mathf.Abs(v - median));
            deviations.Sort();
            return deviations[deviations.Count / 2];
        }

        // ─── Step 3: Reach Check (현재 생략) ──────────────────────────────

        void TickReachCheck(HandPoseFrame frame, float deltaTime)
        {
            LastReachResult = _reachEvaluator.Update(frame, deltaTime);

            if (_reachEvaluator.IsComplete)
            {
                _profile.SafeFrame = _reachEvaluator.Result;
                _session.SafeFrame.QuickCheckPassed = true;
                FinalizeCalibration("good");
                SetState(CalibrationState.Confirmation);
                EmitGuidance("보정 완료! 확인해 주세요.");
            }
            else
            {
                string phaseInstruction = _reachEvaluator.CurrentPhase switch
                {
                    ReachCheckPhase.LeftArm => "왼팔을 편안히 왼쪽으로 뻗어 주세요.",
                    ReachCheckPhase.RightArm => "오른팔을 편안히 오른쪽으로 뻗어 주세요.",
                    ReachCheckPhase.BothArmsUp => "양손을 머리 위쪽으로 올려 주세요.",
                    _ => ""
                };

                EmitGuidance(LastReachResult.Guidance ?? phaseInstruction);
            }
        }

        /// <summary>
        /// C5 확인: 사용자가 보정 결과를 수락.
        /// </summary>
        public void ConfirmCalibration()
        {
            if (_state != CalibrationState.Confirmation) return;

            // 프로필 저장
            _profile.Quality.Status = _session.Quality.Status;
            _profile.Quality.ValidFrameRatio = _session.Quality.ValidFrameRatio;
            _profile.UpdatedUtc = DateTime.UtcNow.ToString("o");
            UserProfileRepository.Save(_profile);

            // 고품질 세션이면 장기 기준에 반영
            if (_session.Quality.EligibleForStableBaselineUpdate)
            {
                _profile.BodyBaseline.AcceptedSessionCount++;
                _profile.BodyBaseline.LastAcceptedSessionId = _session.SessionId;
                UserProfileRepository.Save(_profile);
            }

            SetState(CalibrationState.Complete);
            EmitGuidance(null);
        }

        /// <summary>
        /// 보정 재시작.
        /// </summary>
        public void RestartCalibration()
        {
            _framingGoodTime = 0f;
            _bodyMeasurementFrames.Clear();
            _bodyMeasurementRetryCount = 0;
            _neutralFrames.Clear();
            _neutralRetryCount = 0;
            _neutralDetector.BeginCalibration();
            _reachEvaluator.Reset();
            SetState(CalibrationState.FramingCheck);
        }

        /// <summary>
        /// 보정 취소.
        /// </summary>
        public void Cancel()
        {
            SetState(CalibrationState.Failed);
        }

        void FinalizeCalibration(string qualityStatus)
        {
            _session.Quality.Status = qualityStatus;
            _session.Quality.EligibleForStableBaselineUpdate =
                qualityStatus == "good" && _session.Quality.ValidFrameRatio >= 0.8f;
        }

        void SetState(CalibrationState newState)
        {
            if (_state == newState) return;
            _state = newState;
            StateChanged?.Invoke(newState);
        }

        void EmitGuidance(string message)
        {
            GuidanceChanged?.Invoke(message);
        }
    }
}
