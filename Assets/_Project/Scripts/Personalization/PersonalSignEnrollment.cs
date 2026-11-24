using System;

namespace WordsInSilence.Personalization
{
    /// <summary>
    /// 사용자별 수어 개인 등록 상태 및 품질.
    /// 승인된 반복 템플릿을 누적하고 개인 판정에 사용한다.
    /// </summary>
    [Serializable]
    public sealed class PersonalSignEnrollment
    {
        public int SchemaVersion = 1;
        public string UserProfileId;
        public string SignId;

        /// <summary>"not_started", "in_progress", "ready"</summary>
        public string EnrollmentStatus = "not_started";

        public int RequiredAcceptedRepetitions = 3;
        public int MaxAttemptsPerEnrollmentRound = 5;

        /// <summary>"front", "slight_left_preferred", etc.</summary>
        public string CaptureViewPolicy = "front";

        public string[] AcceptedAttemptIds = Array.Empty<string>();
        public string[] RejectedAttemptIds = Array.Empty<string>();

        public string FeatureSchema = "sign_features_v1";

        public PersonalRanges Ranges = new PersonalRanges();
        public EnrollmentQuality Quality = new EnrollmentQuality();

        public int AcceptedCount => AcceptedAttemptIds?.Length ?? 0;
        public bool IsReady => EnrollmentStatus == "ready" &&
                               AcceptedCount >= RequiredAcceptedRepetitions;
    }

    [Serializable]
    public sealed class PersonalRanges
    {
        public float DurationMedianMs;
        public float DurationMadMs;
        public float MotionScale;

        public bool HasPositionOffset;
        public float[] PositionOffset;
    }

    [Serializable]
    public sealed class EnrollmentQuality
    {
        public float WithinUserConsistency;
        public float GlobalAnchorScore;
        /// <summary>"unvalidated", "good", "marginal"</summary>
        public string Status = "unvalidated";
    }
}
