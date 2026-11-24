using System;
using UnityEngine;

namespace WordsInSilence.Calibration
{
    /// <summary>
    /// 장기 사용자 신체 기준 및 카메라 설정 프로필.
    /// 여러 고품질 세션에서 강건하게 갱신된다.
    /// </summary>
    [Serializable]
    public sealed class UserCalibrationProfile
    {
        public int SchemaVersion = 1;
        public string UserProfileId;
        public string CreatedUtc;
        public string UpdatedUtc;

        public StableBodyBaseline BodyBaseline = new StableBodyBaseline();
        public CameraFingerprint Camera = new CameraFingerprint();
        public BodyScaleData BodyScale = new BodyScaleData();
        public NeutralPoseData Neutral = new NeutralPoseData();
        public SafeFrameData SafeFrame = new SafeFrameData();
        public CalibrationQuality Quality = new CalibrationQuality();
    }

    /// <summary>
    /// 여러 고품질 세션에서 얻은 강건 통계 기반 신체 기준.
    /// </summary>
    [Serializable]
    public sealed class StableBodyBaseline
    {
        public int AcceptedSessionCount;
        public string Aggregation = "robust_median_mad";
        public string LastAcceptedSessionId;
    }

    /// <summary>
    /// 카메라 장치 정보. 장치/해상도 변경 감지에 사용.
    /// </summary>
    [Serializable]
    public sealed class CameraFingerprint
    {
        public string DeviceName;
        public int Width;
        public int Height;
        public float Fps;
        public int RotationDegrees;
        public bool PreviewMirrored = true;
    }

    /// <summary>
    /// 어깨너비 기준 상대적 신체 비율.
    /// 실제 센티미터 아님 — 정규화된 상대값.
    /// </summary>
    [Serializable]
    public sealed class BodyScaleData
    {
        public float ShoulderWidthNorm;

        public bool HasUpperArmRatio;
        public float LeftUpperArmOverShoulder;
        public float RightUpperArmOverShoulder;

        public bool HasForearmRatio;
        public float LeftForearmOverShoulder;
        public float RightForearmOverShoulder;

        public bool HasTorsoRatio;
        public float TorsoOverShoulder;
    }

    /// <summary>
    /// 중립 자세 기준값. 손이 화면 아래로 벗어난 경우도 조건부 허용.
    /// </summary>
    [Serializable]
    public sealed class NeutralPoseData
    {
        public bool HandsVisible;
        public bool BottomExitAllowed;

        /// <summary>보정된 중립 팔꿈치 각도가 있는지.</summary>
        public bool HasArmAngleBaseline;
        /// <summary>왼팔 중립 팔꿈치 각도 (도). 180=완전 펴짐.</summary>
        public float LeftArmAngleNeutral;
        /// <summary>오른팔 중립 팔꿈치 각도 (도).</summary>
        public float RightArmAngleNeutral;

        /// <summary>보정된 팔 수직 각도가 있는지.</summary>
        public bool HasArmVerticalBaseline;
        /// <summary>왼팔 수직 각도 (도). 0=수직, 90=수평.</summary>
        public float LeftArmVerticalNeutral;
        /// <summary>오른팔 수직 각도 (도).</summary>
        public float RightArmVerticalNeutral;

        /// <summary>보정된 중립 손목 위치가 있는지 (참고용).</summary>
        public bool HasWristBaseline;
        /// <summary>왼손목 중립 body-relative 좌표.</summary>
        public Vector2 LeftWristNeutralBR;
        /// <summary>오른손목 중립 body-relative 좌표.</summary>
        public Vector2 RightWristNeutralBR;

        public float MotionNoiseFloor;
    }

    /// <summary>
    /// 동작 안전 영역 (카메라 프레이밍).
    /// </summary>
    [Serializable]
    public sealed class SafeFrameData
    {
        public float LeftMargin;
        public float RightMargin;
        public float TopMargin;
        public bool ReachCheckPassed;
    }

    /// <summary>
    /// 보정 품질 상태.
    /// </summary>
    [Serializable]
    public sealed class CalibrationQuality
    {
        /// <summary>"unvalidated", "good", "marginal", "failed"</summary>
        public string Status = "unvalidated";
        public float ValidFrameRatio;
    }
}
