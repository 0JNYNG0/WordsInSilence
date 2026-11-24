using System;
using UnityEngine;
using WordsInSilence.Recognition.Common;
using WordsInSilence.Recognition.LiveInput;

namespace WordsInSilence.Practice.PetRobotVerticalSlice
{
    /// <summary>
    /// Step 10P v1.0c observability seam for the frozen common-2D auxiliary signal,
    /// repaired in v1.0c1.
    ///
    /// It READS one finalized <see cref="PracticeRecognitionAttempt"/> and records it at
    /// most once, through <see cref="CommonSignalDiagnosticGate"/>. It never classifies,
    /// never fuses, never stores and never decides: removing every call site restores the
    /// previous behaviour exactly, because nothing here is written back into the attempt,
    /// the outcome, the session, the repository or the profile.
    ///
    /// The record carries only the auxiliary classification fields, the fused evidence
    /// state and the existing game-consumed outcome. Free-text detail passes through the
    /// central sanitizer, so no filesystem location reaches the log or the screen, and it
    /// carries no landmark, no frame payload and no personal identifier. The capture
    /// session id is an opaque technical attempt id and is never joined with a path or a
    /// user name.
    /// </summary>
    public static class CommonSignalAttemptDiagnostics
    {
        static readonly CommonSignalDiagnosticGate SharedGate = new CommonSignalDiagnosticGate();

        /// <summary>
        /// Where a record is written. Replaceable so a test can observe the exact number
        /// of records, or inject a failing sink. Defaults to the Unity console.
        /// </summary>
        public static Action<string> Sink = DefaultSink;

        /// <summary>
        /// OPTIONAL read-only observation hook, null by default. When null this class behaves
        /// exactly as before. It is invoked only after a capture was recorded for the first
        /// time, so it inherits the existing dedup, and any fault it raises is swallowed at
        /// the call site: it can never change the record, the gate, the return value or the
        /// capture path. Nothing it does is written back into the attempt or the outcome.
        /// </summary>
        public static Action<PracticeRecognitionAttempt> Observer;

        public static CommonSignalDiagnosticGate Gate { get { return SharedGate; } }
        public static int RecordCount { get { return SharedGate.RecordedCount; } }
        public static string LastRecord { get { return SharedGate.LastRecord; } }

        static void DefaultSink(string line)
        {
            Debug.Log(line);
        }

        /// <summary>Restores the default sink and forgets recorded ids. Test helper.</summary>
        public static void ResetForTest()
        {
            Sink = DefaultSink;
            Observer = null;
            SharedGate.Reset();
        }

        /// <summary>Value written when a World outcome does not apply or is unknown.</summary>
        public const string Unavailable = "unavailable";

        /// <summary>
        /// Step 10W v1.0. Names how the guided outcome and the canonical World outcome
        /// relate on this attempt, so a reader never has to guess which layer won.
        /// </summary>
        internal static string DescribeAgreement(
            GuidedQuestOutcomeKind guided,
            TrustedHybridOutcomeKind? questAuthoritative,
            TrustedHybridOutcomeKind? legacyHybrid,
            bool questMode)
        {
            if (!questMode || !questAuthoritative.HasValue) return "not_quest_mode";
            if (!legacyHybrid.HasValue) return "legacy_hybrid_unavailable";
            return legacyHybrid.Value == questAuthoritative.Value
                ? "legacy_agrees_with_canonical"
                : "legacy_disagrees_with_canonical_and_was_ignored";
        }

        /// <summary>Builds the record text for one attempt. Pure; safe on null.</summary>
        public static string Build(
            PracticeRecognitionAttempt attempt,
            TrustedHybridOutcomeKind? questAuthoritativeOutcome = null,
            TrustedHybridOutcomeKind? legacyHybridOutcome = null,
            bool questMode = false)
        {
            if (attempt == null) return string.Empty;

            CommonSignClassification signal = attempt.CommonSignal;
            GuidedQuestOutcome outcome = attempt.Outcome;
            CommonEvidence evidence = outcome == null ? null : outcome.Common;

            return CommonSignalDiagnosticRecord.Compose(
                attempt.CaptureSessionId,
                attempt.ExpectedWord,
                signal != null,
                signal == null ? string.Empty : signal.ArtifactVersion,
                signal == null ? string.Empty : signal.ArtifactSha256,
                signal == null ? string.Empty : signal.RequestedTarget,
                signal != null && signal.Eligible,
                (signal == null ? CommonSignFailureReason.None : signal.FailureReason).ToString(),
                signal == null ? string.Empty : signal.FailureDetail,
                signal == null ? double.NaN : signal.GreetingProbability,
                signal == null ? double.NaN : signal.ThankYouProbability,
                signal == null ? string.Empty : signal.TopClass,
                signal == null ? double.NaN : signal.TopProbability,
                signal == null ? double.NaN : signal.ProbabilityMargin,
                signal != null && signal.ExpectedTargetSupport,
                signal == null ? double.NaN : signal.DecisionValue,
                signal == null ? 0 : signal.SourceFrameCount,
                (evidence == null ? CommonEvidenceState.Unavailable : evidence.State).ToString(),
                outcome == null ? string.Empty : outcome.Kind.ToString(),
                attempt.PredictedWord,
                attempt.CaptureState.ToString(),
                attempt.StrictAccepted,
                attempt.StrictExpectedWordClassAccept,
                attempt.StrictQuestSuccess,
                outcome != null && outcome.GuidedProgressAllowed,
                evidence != null && evidence.GrantsSuccess,
                evidence != null && evidence.GrantsLearnerError,
                evidence != null && evidence.GrantsVeto,
                questAuthoritativeOutcome.HasValue
                    ? questAuthoritativeOutcome.Value.ToString() : Unavailable,
                questMode ? PetRobotSignPracticeSession.WorldOutcomeAuthority
                          : Unavailable,
                legacyHybridOutcome.HasValue
                    ? legacyHybridOutcome.Value.ToString() : Unavailable,
                DescribeAgreement(
                    outcome == null ? GuidedQuestOutcomeKind.TechnicalFailure
                                    : outcome.Kind,
                    questAuthoritativeOutcome, legacyHybridOutcome, questMode));
        }

        /// <summary>
        /// Records one finalized attempt at most once per unique capture session id.
        /// Read-only side effect: a log line. Composer and sink failures are isolated by
        /// the gate and never reach the capture path.
        /// </summary>
        public static bool Emit(
            PracticeRecognitionAttempt attempt,
            TrustedHybridOutcomeKind? questAuthoritativeOutcome = null,
            TrustedHybridOutcomeKind? legacyHybridOutcome = null,
            bool questMode = false)
        {
            if (attempt == null) return false;
            PracticeRecognitionAttempt captured = attempt;
            TrustedHybridOutcomeKind? canonical = questAuthoritativeOutcome;
            TrustedHybridOutcomeKind? legacy = legacyHybridOutcome;
            bool quest = questMode;
            bool recorded = SharedGate.TryRecord(
                attempt.CaptureSessionId,
                () => Build(captured, canonical, legacy, quest),
                Sink ?? DefaultSink);
            // optional, opt-in, read-only, and only for a capture recorded for the FIRST
            // time, so the existing bounded dedup is reused rather than duplicated. Any
            // observer fault is contained here: the record, the gate and this return value
            // are already final and are never revisited.
            if (recorded)
            {
                Action<PracticeRecognitionAttempt> observer = Observer;
                if (observer != null)
                {
                    try { observer(captured); }
                    catch (Exception) { }
                }
            }
            return recorded;
        }
    }
}
