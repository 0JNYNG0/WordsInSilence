using System;
using System.Collections.Generic;
using WordsInSilence.Recognition;
using WordsInSilence.Segmentation;

namespace WordsInSilence.Core
{
    /// <summary>
    /// 수어 연습 시스템의 이벤트 계약.
    /// 게임 로직(대화, 퀘스트)은 이 이벤트를 구독한다.
    /// </summary>
    public static class SignPracticeEvents
    {
        // ─── 보정 ──────────────────────────────────────────────────────────
        public static event Action<string> CalibrationCompleted; // profileId

        // ─── 상태 머신 ────────────────────────────────────────────────────
        public static event Action<string> AttemptArmed;           // signId
        public static event Action<string> AttemptStarted;         // attemptId
        public static event Action<float> EndCandidateStarted;     // requiredHoldSeconds
        public static event Action<string> AttemptResumed;         // attemptId

        // ─── 판정 ──────────────────────────────────────────────────────────
        public static event Action<SignAttemptEvent> AttemptFinished;

        // ─── 등록 ──────────────────────────────────────────────────────────
        public static event Action<string, int, int> EnrollmentProgress; // signId, accepted, required
        public static event Action<string> SignPassed;           // signId
        public static event Action<string> SignUncertain;        // signId

        // ─── 발행 메서드 ──────────────────────────────────────────────────

        public static void EmitCalibrationCompleted(string profileId)
            => CalibrationCompleted?.Invoke(profileId);

        public static void EmitAttemptArmed(string signId)
            => AttemptArmed?.Invoke(signId);

        public static void EmitAttemptStarted(string attemptId)
            => AttemptStarted?.Invoke(attemptId);

        public static void EmitEndCandidateStarted(float requiredHoldSeconds)
            => EndCandidateStarted?.Invoke(requiredHoldSeconds);

        public static void EmitAttemptResumed(string attemptId)
            => AttemptResumed?.Invoke(attemptId);

        public static void EmitAttemptFinished(SignAttemptEvent evt)
            => AttemptFinished?.Invoke(evt);

        public static void EmitEnrollmentProgress(string signId, int accepted, int required)
            => EnrollmentProgress?.Invoke(signId, accepted, required);

        public static void EmitSignPassed(string signId)
            => SignPassed?.Invoke(signId);

        public static void EmitSignUncertain(string signId)
            => SignUncertain?.Invoke(signId);
    }

    /// <summary>
    /// 시도 완료 이벤트 데이터.
    /// </summary>
    public sealed class SignAttemptEvent
    {
        public string AttemptId;
        public string SignId;
        public AttemptResultStatus Status;
        public float GlobalScore;
        public float CombinedScore;
        public List<string> FeedbackCodes = new List<string>();
    }
}
