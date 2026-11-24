using System;
using UnityEngine;

namespace WordsInSilence.Segmentation
{
    /// <summary>
    /// 개별 수어 시도의 기록. 판정 결과, 캡처 정보, 점수를 포함한다.
    /// </summary>
    [Serializable]
    public sealed class SignAttemptRecord
    {
        public int SchemaVersion = 1;
        public string AttemptId;
        public string UserProfileId;
        public string SignId;
        public string SceneContext;
        public string CreatedUtc;

        public AttemptCaptureInfo Capture = new AttemptCaptureInfo();
        public AttemptCameraQuality CameraQuality = new AttemptCameraQuality();
        public AttemptScores Scores = new AttemptScores();
        public AttemptResult Result = new AttemptResult();
        public AttemptPersonalization Personalization = new AttemptPersonalization();
    }

    [Serializable]
    public sealed class AttemptCaptureInfo
    {
        public long StartTimestampMs;
        public long EndTimestampMs;
        public long DurationMs;
        public int TrackedFrameCount;
        public bool RawVideoSaved;
        public string SessionId;
        public string ViewLabel = "front";
        public string EstimatedTorsoYawBucket = "front";
    }

    [Serializable]
    public sealed class AttemptCameraQuality
    {
        public float ValidFrameRatio;
        public bool Occluded;
        public bool CriticalLandmarksObservable = true;
        public bool OutOfFrame;
        public long TrackingGapMs;
    }

    [Serializable]
    public sealed class AttemptScores
    {
        public float GlobalScore = float.NaN;
        public float PersonalScore = float.NaN;
        public float CombinedScore = float.NaN;
        public float Handshape = float.NaN;
        public float Position = float.NaN;
        public float Orientation = float.NaN;
        public float Trajectory = float.NaN;
        public float TwoHandRelation = float.NaN;

        public bool HasPersonalScore => !float.IsNaN(PersonalScore);
    }

    [Serializable]
    public sealed class AttemptResult
    {
        /// <summary>"passed", "incorrect", "uncertain", "invalid"</summary>
        public string Status = "invalid";
        public string[] FeedbackCodes = Array.Empty<string>();
    }

    [Serializable]
    public sealed class AttemptPersonalization
    {
        public bool EligibleForEnrollment;
        public bool AddedToPersonalProfile;
        public string RejectionReason;
    }
}
