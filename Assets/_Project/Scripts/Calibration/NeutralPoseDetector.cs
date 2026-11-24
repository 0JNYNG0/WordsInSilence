using System.Collections.Generic;
using UnityEngine;
using WordsInSilence.Pose;
using WordsInSilence.Settings;

namespace WordsInSilence.Calibration
{
    /// <summary>
    /// 사용자의 중립 자세를 감지한다.
    /// 핵심 기준: Pose 스켈레톤의 팔 각도(팔꿈치 굽힘)와 팔 방향(수직 여부).
    /// Hand Landmarker 불필요 — Pose 랜드마크만으로 판정.
    /// </summary>
    public sealed class NeutralPoseDetector
    {
        readonly RecognitionSettings _settings;

        // MediaPipe Pose 인덱스
        const int LeftShoulder = 11;
        const int RightShoulder = 12;
        const int LeftElbow = 13;
        const int RightElbow = 14;
        const int LeftWrist = 15;
        const int RightWrist = 16;

        NeutralPoseData _baseline;
        bool _hasBaseline;
        int _diagSkip; // 진단용 (문제 해결 후 제거)

        float _neutralHoldTime;
        bool _isNeutral;

        readonly List<NeutralSample> _calibrationSamples = new List<NeutralSample>();

        public bool HasBaseline => _hasBaseline;
        public bool IsCurrentlyNeutral => _isNeutral;
        public float NeutralHoldTime => _neutralHoldTime;
        public NeutralPoseData Baseline => _baseline;
        public string LastReason { get; private set; } // 진단용

        public NeutralPoseDetector(RecognitionSettings settings = null)
        {
            _settings = settings ?? RecognitionSettings.Default;
        }

        public void SetBaseline(NeutralPoseData baseline)
        {
            _baseline = baseline;
            _hasBaseline = baseline != null;
        }

        public void BeginCalibration()
        {
            _calibrationSamples.Clear();
        }

        public void AddCalibrationFrame(HandPoseFrame frame)
        {
            if (frame == null || !frame.HasPose || !frame.HasValidShoulders)
                return;

            var pts = frame.PoseFrame?.AllPoints;
            if (pts == null || pts.Length <= RightWrist) return;

            var sample = new NeutralSample
            {
                HasLeftHand = frame.HasLeftHand,
                HasRightHand = frame.HasRightHand,
                LeftArmAngle = PoseBodyMath.ArmAngleDegrees(pts[LeftShoulder], pts[LeftElbow], pts[LeftWrist]),
                RightArmAngle = PoseBodyMath.ArmAngleDegrees(pts[RightShoulder], pts[RightElbow], pts[RightWrist]),
                LeftArmVertical = PoseBodyMath.ArmVerticalAngleDegrees(pts[LeftShoulder], pts[LeftWrist]),
                RightArmVertical = PoseBodyMath.ArmVerticalAngleDegrees(pts[RightShoulder], pts[RightWrist]),
                LeftWristBodyRelative = frame.IsValidForBodyRelativeEvaluation
                    ? frame.LeftWristBodyRelative : PoseBodyMath.BodyRelative(pts[LeftWrist], frame.ShoulderCenter, frame.ShoulderWidth),
                RightWristBodyRelative = frame.IsValidForBodyRelativeEvaluation
                    ? frame.RightWristBodyRelative : PoseBodyMath.BodyRelative(pts[RightWrist], frame.ShoulderCenter, frame.ShoulderWidth),
            };

            _calibrationSamples.Add(sample);
        }

        public NeutralCalibrationResult FinishCalibration()
        {
            var result = new NeutralCalibrationResult();

            if (_calibrationSamples.Count < 5)
            {
                result.Success = false;
                result.FailureReason = "보정 프레임이 부족합니다. 자세를 유지해 주세요.";
                return result;
            }

            var baseline = new NeutralPoseData();

            // 손 가시성 판단
            int handsVisibleCount = 0;
            foreach (var s in _calibrationSamples)
                if (s.HasLeftHand || s.HasRightHand) handsVisibleCount++;
            baseline.HandsVisible = (float)handsVisibleCount / _calibrationSamples.Count > 0.5f;
            baseline.BottomExitAllowed = !baseline.HandsVisible;

            // 팔꿈치 각도 중앙값
            var leftAngles = new List<float>();
            var rightAngles = new List<float>();
            foreach (var s in _calibrationSamples)
            {
                if (!float.IsNaN(s.LeftArmAngle)) leftAngles.Add(s.LeftArmAngle);
                if (!float.IsNaN(s.RightArmAngle)) rightAngles.Add(s.RightArmAngle);
            }
            if (leftAngles.Count >= 3 && rightAngles.Count >= 3)
            {
                baseline.HasArmAngleBaseline = true;
                leftAngles.Sort();
                rightAngles.Sort();
                baseline.LeftArmAngleNeutral = leftAngles[leftAngles.Count / 2];
                baseline.RightArmAngleNeutral = rightAngles[rightAngles.Count / 2];
            }

            // 팔 수직 각도 중앙값
            var leftVerticals = new List<float>();
            var rightVerticals = new List<float>();
            foreach (var s in _calibrationSamples)
            {
                if (!float.IsNaN(s.LeftArmVertical)) leftVerticals.Add(s.LeftArmVertical);
                if (!float.IsNaN(s.RightArmVertical)) rightVerticals.Add(s.RightArmVertical);
            }
            if (leftVerticals.Count >= 3 && rightVerticals.Count >= 3)
            {
                baseline.HasArmVerticalBaseline = true;
                leftVerticals.Sort();
                rightVerticals.Sort();
                baseline.LeftArmVerticalNeutral = leftVerticals[leftVerticals.Count / 2];
                baseline.RightArmVerticalNeutral = rightVerticals[rightVerticals.Count / 2];
            }

            // body-relative 손목 위치 중앙값 (참고용)
            baseline.HasWristBaseline = true;
            baseline.LeftWristNeutralBR = MedianVector2(_calibrationSamples, s => s.LeftWristBodyRelative);
            baseline.RightWristNeutralBR = MedianVector2(_calibrationSamples, s => s.RightWristBodyRelative);

            // 움직임 노이즈
            baseline.MotionNoiseFloor = ComputeMotionNoise(_calibrationSamples);

            _baseline = baseline;
            _hasBaseline = true;

            result.Success = true;
            result.Baseline = baseline;
            result.ValidFrameRatio = 1f; // 모든 프레임이 유효 (Pose 기반)
            result.TotalSamples = _calibrationSamples.Count;
            return result;
        }

        /// <summary>
        /// 매 프레임 호출: 현재 프레임이 중립 자세인지 판정한다.
        /// Pose 스켈레톤의 팔 각도 + 팔 방향으로 판정. Hand 검출 불필요.
        /// </summary>
        public NeutralCheckResult CheckNeutral(HandPoseFrame frame, float deltaTime)
        {
            var result = new NeutralCheckResult();

            if (!_hasBaseline)
            {
                result.IsNeutral = false;
                result.Reason = "NO_BASELINE";
                _isNeutral = false;
                _neutralHoldTime = 0f;
                LastReason = result.Reason;
                return result;
            }

            if (frame == null)
            {
                result.IsNeutral = false;
                result.Reason = "NULL_FRAME";
                _isNeutral = false;
                _neutralHoldTime = 0f;
                LastReason = result.Reason;
                return result;
            }

            if (!frame.HasPose)
            {
                result.IsNeutral = false;
                result.Reason = "NO_POSE";
                _isNeutral = false;
                _neutralHoldTime = 0f;
                LastReason = result.Reason;
                return result;
            }

            if (!frame.HasValidShoulders)
            {
                result.IsNeutral = false;
                result.Reason = "NO_SHOULDERS";
                _isNeutral = false;
                _neutralHoldTime = 0f;
                LastReason = result.Reason;
                return result;
            }

            // ─── 핵심 중립 판정: 팔 각도 + 팔 방향 ───
            bool isNeutral = CheckArmsNeutral(frame, out string armDiag);
            result.ArmsNeutral = isNeutral;
            if (!isNeutral) result.Reason = armDiag;

            if (isNeutral)
            {
                _neutralHoldTime += deltaTime;
            }
            else
            {
                _neutralHoldTime = 0f;
            }

            _isNeutral = isNeutral;
            result.IsNeutral = isNeutral;
            result.HoldTime = _neutralHoldTime;
            LastReason = result.Reason; // 진단용
            return result;
        }

        public void ResetHoldTimer()
        {
            _neutralHoldTime = 0f;
        }

        /// <summary>
        /// Pose 스켈레톤으로 중립 자세 판정:
        /// 1) 보정 시 손이 안 보였다면, 손이 보이는 동안은 중립이 아님
        /// 2) 상완(어깨→팔꿈치) 방향이 수직에서 NeutralArmVerticalMax 이내
        /// </summary>
        bool CheckArmsNeutral(HandPoseFrame frame, out string diag)
        {
            // 보정 시 손이 안 보였는데 양손 중 어느 하나라도 보이면 → 아직 손을 안 내린 것
            // 양손이 모두 사라져야 중립으로 판정 (한 손만 카메라 밖으로 나간 건 중립 아님)
            if (_baseline != null && !_baseline.HandsVisible)
            {
                if (frame.HasLeftHand || frame.HasRightHand)
                {
                    diag = frame.HasLeftHand && frame.HasRightHand ? "BOTH_HANDS" :
                           frame.HasLeftHand ? "LEFT_HAND" : "RIGHT_HAND";
                    return false;
                }
            }

            var pts = frame.PoseFrame?.AllPoints;
            if (pts == null || pts.Length <= RightWrist)
            {
                diag = $"NO_POINTS(PoseFrame null={frame.PoseFrame == null}, len={pts?.Length ?? -1})";
                return false;
            }

            // 상완 방향: 어깨→팔꿈치가 수직(아래)에 가까운지
            float leftVertical = PoseBodyMath.ArmVerticalAngleDegrees(pts[LeftShoulder], pts[LeftElbow]);
            float rightVertical = PoseBodyMath.ArmVerticalAngleDegrees(pts[RightShoulder], pts[RightElbow]);

            float maxVertical = _settings.NeutralArmVerticalMax; // 기본 45°

            diag = $"L={leftVertical:F1}° R={rightVertical:F1}° max={maxVertical}°";

            // NaN은 통과 (측정 불가한 경우)
            if (!float.IsNaN(leftVertical) && leftVertical > maxVertical) return false;
            if (!float.IsNaN(rightVertical) && rightVertical > maxVertical) return false;

            return true;
        }

        // ─── 유틸 ─────────────────────────────────────────────────────────

        static float ComputeMotionNoise(List<NeutralSample> samples)
        {
            if (samples.Count < 3) return 0.01f;

            var diffs = new List<float>();
            for (int i = 1; i < samples.Count; i++)
            {
                var a = samples[i].LeftWristBodyRelative;
                var b = samples[i - 1].LeftWristBodyRelative;
                if (float.IsNaN(a.x) || float.IsNaN(b.x)) continue;
                float dx = a.x - b.x;
                float dy = a.y - b.y;
                diffs.Add(Mathf.Sqrt(dx * dx + dy * dy));
            }

            if (diffs.Count == 0) return 0.01f;
            diffs.Sort();
            return diffs[diffs.Count / 2];
        }

        static Vector2 MedianVector2(List<NeutralSample> samples, System.Func<NeutralSample, Vector2> selector)
        {
            var xs = new List<float>();
            var ys = new List<float>();
            foreach (var s in samples)
            {
                var v = selector(s);
                if (float.IsNaN(v.x) || float.IsNaN(v.y)) continue;
                xs.Add(v.x);
                ys.Add(v.y);
            }
            if (xs.Count == 0) return new Vector2(float.NaN, float.NaN);
            xs.Sort();
            ys.Sort();
            return new Vector2(xs[xs.Count / 2], ys[ys.Count / 2]);
        }

        struct NeutralSample
        {
            public bool HasLeftHand;
            public bool HasRightHand;
            public float LeftArmAngle;
            public float RightArmAngle;
            public float LeftArmVertical;
            public float RightArmVertical;
            public Vector2 LeftWristBodyRelative;
            public Vector2 RightWristBodyRelative;
        }
    }

    public sealed class NeutralCalibrationResult
    {
        public bool Success;
        public string FailureReason;
        public NeutralPoseData Baseline;
        public float ValidFrameRatio;
        public int TotalSamples;
    }

    public sealed class NeutralCheckResult
    {
        public bool IsNeutral;
        public float HoldTime;
        public string Reason;
        public bool ArmsNeutral;
    }
}
