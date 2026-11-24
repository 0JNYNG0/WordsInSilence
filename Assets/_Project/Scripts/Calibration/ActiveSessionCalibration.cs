using System;
using UnityEngine;

namespace WordsInSilence.Calibration
{
    /// <summary>
    /// 현재 실행 세션의 활성 보정값.
    /// 매 실행마다 새로 생성·교체된다. 장기 프로필을 대체하지 않는다.
    /// </summary>
    [Serializable]
    public sealed class ActiveSessionCalibration
    {
        public int SchemaVersion = 1;
        public string SessionId;
        public string UserProfileId;
        public string CreatedUtc;
        public string AppVersion;

        public CameraFingerprint CameraInfo = new CameraFingerprint();

        public SessionScaleData CurrentScale = new SessionScaleData();
        public SessionNeutralData Neutral = new SessionNeutralData();
        public SessionSafeFrame SafeFrame = new SessionSafeFrame();
        public SessionQuality Quality = new SessionQuality();
    }

    /// <summary>
    /// 현재 세션의 어깨 스케일 및 몸 중심.
    /// </summary>
    [Serializable]
    public sealed class SessionScaleData
    {
        public float ShoulderWidthNorm;
        public bool HasBodyCenter;
        public Vector3 BodyCenter;
    }

    /// <summary>
    /// 현재 세션의 중립 자세 측정값.
    /// </summary>
    [Serializable]
    public sealed class SessionNeutralData
    {
        public float HoldSeconds = 2.0f;
        public bool BottomExitValidated;
        public float MotionNoiseFloor;
    }

    /// <summary>
    /// 현재 세션의 안전 프레임 상태.
    /// </summary>
    [Serializable]
    public sealed class SessionSafeFrame
    {
        public bool QuickCheckPassed;
        public bool FullReachCheckRequired;
    }

    /// <summary>
    /// 현재 세션의 보정 품질.
    /// </summary>
    [Serializable]
    public sealed class SessionQuality
    {
        /// <summary>"unvalidated", "good", "marginal", "failed"</summary>
        public string Status = "unvalidated";
        public float ValidFrameRatio;
        public bool EligibleForStableBaselineUpdate;
    }
}
