using System.Collections.Generic;
using UnityEngine;
using WordsInSilence.Pose;

namespace WordsInSilence.Calibration
{
    /// <summary>
    /// 프레임 시퀀스에서 어깨너비 기준 상대 비율을 계산한다.
    /// 실제 센티미터가 아닌 정규화된 비율값.
    /// </summary>
    public static class BodyScaleCalculator
    {
        // MediaPipe Pose indices
        const int LeftShoulder = 11;
        const int RightShoulder = 12;
        const int LeftElbow = 13;
        const int RightElbow = 14;
        const int LeftWrist = 15;
        const int RightWrist = 16;
        const int LeftHip = 23;
        const int RightHip = 24;

        /// <summary>
        /// 보정 프레임들로부터 상대 신체 비율을 계산한다.
        /// </summary>
        public static BodyScaleData ComputeFromFrames(List<HandPoseFrame> frames)
        {
            return ComputeWithValidation(frames).Scale;
        }

        /// <summary>
        /// 프레임에서 신체 비율을 계산하고 안정성을 검증한다.
        /// </summary>
        public static BodyMeasurementResult ComputeWithValidation(List<HandPoseFrame> frames)
        {
            var result = new BodyMeasurementResult();
            result.Scale = new BodyScaleData();

            if (frames == null || frames.Count < 3)
            {
                result.Success = false;
                result.FailureReason = "측정 프레임이 부족합니다. 자세를 유지해 주세요.";
                return result;
            }

            var shoulderWidths = new List<float>();
            var leftUpperArms = new List<float>();
            var rightUpperArms = new List<float>();
            var leftForearms = new List<float>();
            var rightForearms = new List<float>();
            var torsos = new List<float>();

            int elbowVisibleCount = 0;
            int wristVisibleCount = 0;

            foreach (var frame in frames)
            {
                if (!frame.HasPose || !frame.HasValidShoulders) continue;
                var pts = frame.PoseFrame?.AllPoints;
                if (pts == null || pts.Length <= 24) continue;

                float sw = Vector2.Distance(
                    new Vector2(pts[LeftShoulder].x, pts[LeftShoulder].y),
                    new Vector2(pts[RightShoulder].x, pts[RightShoulder].y));
                if (sw < 0.02f) continue;

                shoulderWidths.Add(sw);

                // 팔꿈치/손목 가시성 (화면 범위 내)
                bool elbowsInFrame = IsInFrame(pts[LeftElbow]) && IsInFrame(pts[RightElbow]);
                bool wristsInFrame = IsInFrame(pts[LeftWrist]) && IsInFrame(pts[RightWrist]);
                if (elbowsInFrame) elbowVisibleCount++;
                if (wristsInFrame) wristVisibleCount++;

                // Upper arm: shoulder → elbow
                float lua = Vector2.Distance(V2(pts[LeftShoulder]), V2(pts[LeftElbow]));
                float rua = Vector2.Distance(V2(pts[RightShoulder]), V2(pts[RightElbow]));
                leftUpperArms.Add(lua / sw);
                rightUpperArms.Add(rua / sw);

                // Forearm: elbow → wrist
                float lfa = Vector2.Distance(V2(pts[LeftElbow]), V2(pts[LeftWrist]));
                float rfa = Vector2.Distance(V2(pts[RightElbow]), V2(pts[RightWrist]));
                leftForearms.Add(lfa / sw);
                rightForearms.Add(rfa / sw);

                // Torso: shoulder center → hip center
                if (pts.Length > 24)
                {
                    Vector2 shoulderCenter = (V2(pts[LeftShoulder]) + V2(pts[RightShoulder])) * 0.5f;
                    Vector2 hipCenter = (V2(pts[LeftHip]) + V2(pts[RightHip])) * 0.5f;
                    float torso = Vector2.Distance(shoulderCenter, hipCenter);
                    torsos.Add(torso / sw);
                }
            }

            if (shoulderWidths.Count < 3)
            {
                result.Success = false;
                result.FailureReason = "어깨가 안정적으로 감지되지 않았습니다. 다시 시도해 주세요.";
                return result;
            }

            // 안정성 검증: 비율의 MAD(중앙절대편차)가 너무 크면 거부
            float swMedian = Median(new List<float>(shoulderWidths));
            float swMad = MAD(shoulderWidths, swMedian);
            float swCv = swMad / Mathf.Max(swMedian, 0.001f);
            result.ShoulderWidthStability = swCv;

            if (swCv > 0.15f)
            {
                result.Success = false;
                result.FailureReason = "측정 중 움직임이 너무 컸습니다. 자세를 유지한 채 다시 시도해 주세요.";
                return result;
            }

            result.Scale.ShoulderWidthNorm = swMedian;

            if (leftUpperArms.Count >= 3)
            {
                float luaMedian = Median(new List<float>(leftUpperArms));
                float ruaMedian = Median(new List<float>(rightUpperArms));
                float luaMad = MAD(leftUpperArms, luaMedian);
                float ruaMad = MAD(rightUpperArms, ruaMedian);
                float armCv = Mathf.Max(
                    luaMad / Mathf.Max(luaMedian, 0.001f),
                    ruaMad / Mathf.Max(ruaMedian, 0.001f));
                result.UpperArmStability = armCv;

                if (armCv <= 0.20f)
                {
                    result.Scale.HasUpperArmRatio = true;
                    result.Scale.LeftUpperArmOverShoulder = luaMedian;
                    result.Scale.RightUpperArmOverShoulder = ruaMedian;
                }
            }

            if (leftForearms.Count >= 3)
            {
                float lfaMedian = Median(new List<float>(leftForearms));
                float rfaMedian = Median(new List<float>(rightForearms));
                float lfaMad = MAD(leftForearms, lfaMedian);
                float rfaMad = MAD(rightForearms, rfaMedian);
                float faCv = Mathf.Max(
                    lfaMad / Mathf.Max(lfaMedian, 0.001f),
                    rfaMad / Mathf.Max(rfaMedian, 0.001f));
                result.ForearmStability = faCv;

                if (faCv <= 0.20f)
                {
                    result.Scale.HasForearmRatio = true;
                    result.Scale.LeftForearmOverShoulder = lfaMedian;
                    result.Scale.RightForearmOverShoulder = rfaMedian;
                }
            }

            if (torsos.Count >= 3)
            {
                result.Scale.HasTorsoRatio = true;
                result.Scale.TorsoOverShoulder = Median(torsos);
            }

            result.ValidFrameCount = shoulderWidths.Count;
            result.ElbowVisibleRatio = shoulderWidths.Count > 0
                ? (float)elbowVisibleCount / shoulderWidths.Count : 0f;
            result.WristVisibleRatio = shoulderWidths.Count > 0
                ? (float)wristVisibleCount / shoulderWidths.Count : 0f;
            result.Success = true;
            return result;
        }

        static bool IsInFrame(Vector3 pt)
        {
            return pt.x >= 0f && pt.x <= 1f && pt.y >= 0f && pt.y <= 1f;
        }

        static Vector2 V2(Vector3 v) => new Vector2(v.x, v.y);

        static float Median(List<float> values)
        {
            values.Sort();
            return values[values.Count / 2];
        }

        static float MAD(List<float> values, float median)
        {
            var deviations = new List<float>(values.Count);
            foreach (var v in values)
                deviations.Add(Mathf.Abs(v - median));
            return Median(deviations);
        }
    }

    /// <summary>
    /// 신체 측정 결과 (안정성 검증 포함).
    /// </summary>
    public sealed class BodyMeasurementResult
    {
        public bool Success;
        public string FailureReason;
        public BodyScaleData Scale;
        public int ValidFrameCount;
        public float ElbowVisibleRatio;
        public float WristVisibleRatio;
        public float ShoulderWidthStability;
        public float UpperArmStability;
        public float ForearmStability;
    }
}
