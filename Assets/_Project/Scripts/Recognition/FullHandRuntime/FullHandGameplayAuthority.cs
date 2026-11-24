// F27 - the explicit, off-by-default connection between the ALREADY COMPUTED full-hand
// (46-point: 21 joints per hand + 4 arm points) capture evidence and gameplay outcomes.
//
// This file adds NO geometry, NO classifier, NO threshold and NO fitted number. It reads a
// FullHandGameCaptureResult that the F21 bridge already produced for THIS capture and turns
// exactly one already-gated method decision into a typed gameplay decision. It never scores,
// never re-adapts, never re-normalizes, never touches a file, an environment path, a camera
// or a participant profile, and it never enrolls anything.
//
// The gameplay authority is the HYBRID method over the COMMON-ONLY bank (outcome.Before).
// The common+participant bank (outcome.After) is retained by F20 for comparison and is
// deliberately NOT read here: earlier measurements showed personal additions can raise false
// accepts, so nothing promotes it.
//
// "Accepted" means the frozen word-level bank gate accepted this capture for its declared
// target. It is NOT a certification that a correct Korean Sign Language sign was performed,
// and no probability of correctness is produced anywhere in this file.
using System;

namespace WordsInSilence.Recognition.FullHandRuntime
{
    /// <summary>Safe, non-secret gameplay-seam refusal codes. No paths, no identities.</summary>
    public static class FullHandGameplayCodes
    {
        public const string None = "";
        /// <summary>Gameplay was enabled but this capture carried no full-hand evidence.</summary>
        public const string EvidenceUnavailable = "GAMEPLAY_EVIDENCE_UNAVAILABLE";
        /// <summary>Gameplay was enabled while the pipeline itself was not configured.</summary>
        public const string PipelineNotConfigured = "GAMEPLAY_PIPELINE_NOT_CONFIGURED";
        public const string CaptureIdentityMismatch = "GAMEPLAY_CAPTURE_IDENTITY_MISMATCH";
        /// <summary>The owning capture did not finalize on the full-hand route.</summary>
        public const string CaptureNotFinalized = "GAMEPLAY_CAPTURE_NOT_FINALIZED";
        public const string TargetMismatch = "GAMEPLAY_TARGET_MISMATCH";
        public const string BankUnavailable = "GAMEPLAY_BANK_UNAVAILABLE";
        public const string MethodTechnicalFailure = "GAMEPLAY_METHOD_TECHNICAL_FAILURE";
        public const string CaptureSourceFailure = "GAMEPLAY_CAPTURE_SOURCE_FAILURE";
        public const string EvaluatorFault = "GAMEPLAY_EVALUATOR_FAULT";
    }

    /// <summary>
    /// The five distinguishable gameplay verdicts. Only <see cref="Accepted"/> may ever
    /// reach a quest success; every other value is a distinct, observable refusal of
    /// progress and none of them may be collapsed into another.
    /// </summary>
    public enum FullHandGameplayDecisionKind
    {
        /// <summary>Gate ACCEPTED and Top1 equals the supported declared target.</summary>
        Accepted,
        /// <summary>Gate ACCEPTED but Top1 is a different word. Never progress.</summary>
        AcceptedDifferentWord,
        /// <summary>A valid abstention: the model did not separate the classes.</summary>
        NotDistinguished,
        /// <summary>The declared target is outside the model vocabulary.</summary>
        UnsupportedWord,
        /// <summary>A technical refusal: no learner performance was measured.</summary>
        TechnicalRefusal,
    }

    /// <summary>
    /// Immutable, fully attributed gameplay decision for ONE capture. Every field is copied
    /// from evidence that already existed; nothing here is recomputed or invented.
    /// </summary>
    public sealed class FullHandGameplayDecision
    {
        /// <summary>Fixed literal naming the authority that decided this attempt.</summary>
        public const string SourceId = "wis_full_hand_gameplay_common_only_before_v1";

        public FullHandGameplayDecisionKind Kind { get; private set; }
        /// <summary>Always <see cref="FullHandMethods.Hybrid"/> for this route.</summary>
        public string Method { get; private set; }
        /// <summary>Always <see cref="FullHandPipelineBanks.CommonOnly"/> for this route.</summary>
        public string BankId { get; private set; }
        public string ModelId { get; private set; }
        public string GateConfigId { get; private set; }
        public string GateId { get; private set; }
        public string ParticipantId { get; private set; }
        public string CaptureSessionId { get; private set; }
        public string QuerySampleId { get; private set; }
        public string QueryFeatureDigest { get; private set; }
        public string ExpectedWord { get; private set; }
        public string DeclaredTarget { get; private set; }
        /// <summary>The ranked winner, or empty when nothing was ranked.</summary>
        public string Top1 { get; private set; }
        public string MethodStatus { get; private set; }
        public string MethodDecision { get; private set; }
        public double Score { get; private set; }
        public double Margin { get; private set; }
        public double Quality { get; private set; }
        public int FailedPairs { get; private set; }
        public int FrameCount { get; private set; }
        public int EffectiveReferenceCount { get; private set; }
        public string FailureCode { get; private set; }
        public string FailureDetail { get; private set; }
        public string Source { get { return SourceId; } }

        /// <summary>The ONLY property that may grant quest progress.</summary>
        public bool GrantsQuestSuccess
        {
            get { return Kind == FullHandGameplayDecisionKind.Accepted; }
        }

        FullHandGameplayDecision() { }

        /// <summary>One line of provenance for a log or a technical panel. No secrets.</summary>
        public string ProvenanceSummary()
        {
            return "source=" + SourceId +
                   " kind=" + Kind +
                   " method=" + Method +
                   " bank=" + BankId +
                   " model=" + Short(ModelId) +
                   " gate_config=" + Short(GateConfigId) +
                   " gate=" + Short(GateId) +
                   " target=" + Or(DeclaredTarget) +
                   " top1=" + Or(Top1) +
                   " refs=" + EffectiveReferenceCount +
                   (string.IsNullOrEmpty(FailureCode) ? string.Empty : " failure=" + FailureCode);
        }

        static string Or(string value)
        {
            return string.IsNullOrEmpty(value) ? "<none>" : value;
        }

        static string Short(string value)
        {
            if (string.IsNullOrEmpty(value)) return "<none>";
            return value.Length <= 12 ? value : value.Substring(0, 12);
        }

        internal static FullHandGameplayDecision ForMethod(
            FullHandGameplayDecisionKind kind, string expectedWord, int frameCount,
            FullHandRecognitionOutcome outcome, FullHandBankOutcome bank,
            FullHandMethodDecision method, string failureCode, string failureDetail)
        {
            return new FullHandGameplayDecision
            {
                Kind = kind,
                Method = method.Method,
                BankId = bank.BankId,
                ModelId = outcome.CommonModelId ?? string.Empty,
                GateConfigId = outcome.GateConfigId ?? string.Empty,
                GateId = method.GateId ?? string.Empty,
                ParticipantId = outcome.ParticipantId ?? string.Empty,
                CaptureSessionId = outcome.QuerySampleId ?? string.Empty,
                QuerySampleId = outcome.QuerySampleId ?? string.Empty,
                QueryFeatureDigest = outcome.QueryFeatureDigest ?? string.Empty,
                ExpectedWord = expectedWord ?? string.Empty,
                DeclaredTarget = outcome.DeclaredTarget ?? string.Empty,
                Top1 = method.Top1 ?? string.Empty,
                MethodStatus = method.Status ?? string.Empty,
                MethodDecision = method.Decision ?? string.Empty,
                Score = method.Score,
                Margin = method.Margin,
                Quality = method.Quality,
                FailedPairs = method.FailedPairs,
                FrameCount = frameCount,
                EffectiveReferenceCount = bank.EffectiveReferenceCount,
                FailureCode = failureCode ?? FullHandGameplayCodes.None,
                FailureDetail = failureDetail ?? string.Empty,
            };
        }

        /// <summary>
        /// A technical refusal. It never gains an invented score, an invented Top1 or an
        /// invented model identity, and it can never grant progress.
        /// </summary>
        public static FullHandGameplayDecision Refusal(string captureSessionId, string expectedWord,
                                                       int frameCount, string failureCode,
                                                       string failureDetail)
        {
            return new FullHandGameplayDecision
            {
                Kind = FullHandGameplayDecisionKind.TechnicalRefusal,
                Method = FullHandMethods.Hybrid,
                BankId = FullHandPipelineBanks.CommonOnly,
                ModelId = string.Empty,
                GateConfigId = string.Empty,
                GateId = string.Empty,
                ParticipantId = string.Empty,
                CaptureSessionId = captureSessionId ?? string.Empty,
                QuerySampleId = string.Empty,
                QueryFeatureDigest = string.Empty,
                ExpectedWord = expectedWord ?? string.Empty,
                DeclaredTarget = string.Empty,
                Top1 = string.Empty,
                MethodStatus = string.Empty,
                MethodDecision = string.Empty,
                Score = double.NaN,
                Margin = double.NaN,
                Quality = double.NaN,
                FailedPairs = 0,
                FrameCount = frameCount,
                EffectiveReferenceCount = 0,
                FailureCode = string.IsNullOrEmpty(failureCode)
                    ? FullHandGameplayCodes.EvidenceUnavailable : failureCode,
                FailureDetail = failureDetail ?? string.Empty,
            };
        }
    }

    /// <summary>
    /// The explicit gameplay opt-in. Reads one environment variable and nothing else: no
    /// file, no participant, no profile, no model. Owning components read it ONCE, at
    /// construction, and hold the resulting mode for their whole lifetime.
    /// </summary>
    public static class FullHandGameplayConfiguration
    {
        public const string EnableVariable = "WIS_FULL_HAND_GAMEPLAY";
        public const string EnabledValue = "1";

        /// <summary>True only for the exact opt-in value.</summary>
        public static bool IsEnabled()
        {
            try
            {
                return string.Equals(Environment.GetEnvironmentVariable(EnableVariable),
                                     EnabledValue, StringComparison.Ordinal);
            }
            catch (Exception) { return false; }
        }
    }

    /// <summary>
    /// Turns ONE already produced <see cref="FullHandGameCaptureResult"/> into ONE typed
    /// gameplay decision. It never calls the pipeline, never compares two attempts and never
    /// reads a second capture: every input belongs to the single attempt being decided.
    /// </summary>
    public static class FullHandGameplayEvaluator
    {
        /// <summary>The gameplay authority method. Fixed, not configurable at runtime.</summary>
        public const string AuthorityMethod = FullHandMethods.Hybrid;

        /// <summary>
        /// The entry the owning capture route uses. <paramref name="captureFinalized"/> is
        /// the controller's own record that THIS capture finalized on the full-hand route:
        /// a cancelled, unfinalized or invalidly re-finalized capture can therefore never
        /// consume a still-cached earlier result as a new success.
        /// </summary>
        public static FullHandGameplayDecision DecideForCapture(FullHandGameCaptureResult evidence,
                                                                string expectedWord,
                                                                string captureSessionId,
                                                                bool pipelineConfigured,
                                                                bool captureFinalized)
        {
            if (!captureFinalized)
                return FullHandGameplayDecision.Refusal(captureSessionId, expectedWord,
                    evidence == null ? 0 : evidence.FrameCount,
                    FullHandGameplayCodes.CaptureNotFinalized,
                    "this capture did not finalize on the full-hand route");
            return Decide(evidence, expectedWord, captureSessionId, pipelineConfigured);
        }

        /// <summary>
        /// <paramref name="evidence"/> must be the result the controller produced for
        /// <paramref name="captureSessionId"/>; a result whose identity does not match is
        /// refused rather than scored against this attempt.
        /// </summary>
        public static FullHandGameplayDecision Decide(FullHandGameCaptureResult evidence,
                                                      string expectedWord,
                                                      string captureSessionId,
                                                      bool pipelineConfigured)
        {
            int frames = evidence == null ? 0 : evidence.FrameCount;
            try
            {
                if (string.IsNullOrEmpty(captureSessionId))
                    return FullHandGameplayDecision.Refusal(captureSessionId, expectedWord, frames,
                        FullHandGameplayCodes.CaptureIdentityMismatch,
                        "the attempt carried no capture identity");

                if (evidence == null)
                    return FullHandGameplayDecision.Refusal(captureSessionId, expectedWord, 0,
                        pipelineConfigured
                            ? FullHandGameplayCodes.EvidenceUnavailable
                            : FullHandGameplayCodes.PipelineNotConfigured,
                        pipelineConfigured
                            ? "full-hand gameplay is enabled but this capture produced no full-hand evidence"
                            : "full-hand gameplay is enabled while the full-hand pipeline is not configured");

                if (!string.Equals(evidence.CaptureSessionId, captureSessionId,
                                   StringComparison.Ordinal))
                    return FullHandGameplayDecision.Refusal(captureSessionId, expectedWord, frames,
                        FullHandGameplayCodes.CaptureIdentityMismatch,
                        "the full-hand evidence belongs to another capture");

                if (!evidence.Succeeded || evidence.Outcome == null)
                    return FullHandGameplayDecision.Refusal(captureSessionId, expectedWord, frames,
                        string.IsNullOrEmpty(evidence.FailureCode)
                            ? FullHandGameplayCodes.EvidenceUnavailable : evidence.FailureCode,
                        evidence.FailureDetail);

                FullHandRecognitionOutcome outcome = evidence.Outcome;
                if (!string.Equals(outcome.QuerySampleId, captureSessionId, StringComparison.Ordinal))
                    return FullHandGameplayDecision.Refusal(captureSessionId, expectedWord, frames,
                        FullHandGameplayCodes.CaptureIdentityMismatch,
                        "the scored sample identity is not this attempt");

                if (string.IsNullOrEmpty(expectedWord) ||
                    !string.Equals(outcome.DeclaredTarget, expectedWord, StringComparison.Ordinal))
                    return FullHandGameplayDecision.Refusal(captureSessionId, expectedWord, frames,
                        FullHandGameplayCodes.TargetMismatch,
                        "the scored declared target is not the prompted word");

                // the gameplay bank is the COMMON-ONLY "Before" bank; "After" is retained by
                // F20 for comparison only and is deliberately not consulted here
                FullHandBankOutcome bank = outcome.Before;
                if (bank == null)
                    return FullHandGameplayDecision.Refusal(captureSessionId, expectedWord, frames,
                        FullHandGameplayCodes.BankUnavailable,
                        "the common-only bank outcome is absent");

                FullHandMethodDecision method = bank.Decision(AuthorityMethod);
                if (method == null)
                    return FullHandGameplayDecision.Refusal(captureSessionId, expectedWord, frames,
                        FullHandGameplayCodes.BankUnavailable,
                        "the authority method decision is absent");

                if (string.Equals(method.Decision, FullHandDecisions.UnsupportedTarget,
                                  StringComparison.Ordinal) || !method.TargetSupported)
                    return FullHandGameplayDecision.ForMethod(
                        FullHandGameplayDecisionKind.UnsupportedWord, expectedWord, frames,
                        outcome, bank, method, FullHandGameplayCodes.None, string.Empty);

                if (string.Equals(method.Decision, FullHandDecisions.TechnicalFailure,
                                  StringComparison.Ordinal))
                    return FullHandGameplayDecision.ForMethod(
                        FullHandGameplayDecisionKind.TechnicalRefusal, expectedWord, frames,
                        outcome, bank, method, FullHandGameplayCodes.MethodTechnicalFailure,
                        method.Reason ?? string.Empty);

                if (string.Equals(method.Decision, FullHandDecisions.Accepted,
                                  StringComparison.Ordinal))
                    return FullHandGameplayDecision.ForMethod(
                        method.TargetMatches
                            ? FullHandGameplayDecisionKind.Accepted
                            : FullHandGameplayDecisionKind.AcceptedDifferentWord,
                        expectedWord, frames, outcome, bank, method,
                        FullHandGameplayCodes.None, string.Empty);

                // the remaining gated verdict is a valid abstention
                return FullHandGameplayDecision.ForMethod(
                    FullHandGameplayDecisionKind.NotDistinguished, expectedWord, frames,
                    outcome, bank, method, FullHandGameplayCodes.None, string.Empty);
            }
            catch (Exception exception)
            {
                // a fault in this seam is a refusal, never a silent legacy fallback
                return FullHandGameplayDecision.Refusal(captureSessionId, expectedWord, frames,
                    FullHandGameplayCodes.EvaluatorFault, exception.GetType().Name);
            }
        }

        /// <summary>
        /// The refusal used when the capture itself failed at its source. It exists so a
        /// source failure keeps refusing even if a bridge result for the same capture had
        /// already been produced.
        /// </summary>
        public static FullHandGameplayDecision RefuseForSourceFailure(string captureSessionId,
                                                                      string expectedWord,
                                                                      string detail)
        {
            return FullHandGameplayDecision.Refusal(captureSessionId, expectedWord, 0,
                FullHandGameplayCodes.CaptureSourceFailure,
                string.IsNullOrEmpty(detail) ? "the capture source failed" : detail);
        }
    }

    /// <summary>
    /// The learner-facing Korean text for each verdict. It states what the model did and
    /// never claims certified sign correctness, a probability, or a camera fault for an
    /// ordinary abstention.
    ///
    /// Learner text carries NO technical identity: no hash, no source id, no enum name and
    /// no raw word tag. Provenance stays typed on the decision and on the technical panel.
    /// </summary>
    public static class FullHandGameplayFeedback
    {
        public const string NotDistinguished =
            "동작을 확실히 구별하지 못했어요. 다시 시도해 주세요";
        public const string DifferentWord =
            "다른 단어의 동작으로 인식했어요. 다시 시도해 주세요";
        public const string Unsupported =
            "이 단어는 지금 인식 모델이 배우지 않은 단어예요. 지금은 '인사'와 '고마워'만 확인할 수 있어요.";
        public const string TechnicalRefusal =
            "인식을 끝내지 못했어요. 기술적인 문제라서 연습 결과로 기록하지 않아요.";
        /// <summary>
        /// Generic and deliberately worded as recognition, not as certified sign
        /// correctness. It names no word tag and promises no saved data.
        /// </summary>
        public const string Accepted =
            "잘했어요! 이번 동작을 지금 배우는 표현으로 알아봤어요.";

        /// <summary>
        /// The note shown wherever personalization would normally be reported. Full-hand
        /// enrollment is NOT implemented, so this route never reports "개인화 완료".
        /// </summary>
        public const string EnrollmentPending =
            "전체 손 인식 결과는 아직 개인화 학습 데이터로 저장하지 않아요. " +
            "(전체 손 개인화 등록 기능은 준비 중이에요.)";

        public static string Korean(FullHandGameplayDecision decision)
        {
            if (decision == null) return TechnicalRefusal;
            switch (decision.Kind)
            {
                case FullHandGameplayDecisionKind.Accepted:
                    return Accepted;
                case FullHandGameplayDecisionKind.AcceptedDifferentWord:
                    return DifferentWord;
                case FullHandGameplayDecisionKind.NotDistinguished:
                    return NotDistinguished;
                case FullHandGameplayDecisionKind.UnsupportedWord:
                    return Unsupported;
                default:
                    return TechnicalRefusal;
            }
        }
    }
}
