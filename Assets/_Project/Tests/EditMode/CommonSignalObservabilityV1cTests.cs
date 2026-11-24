using System;
using NUnit.Framework;
using WordsInSilence.Practice.PetRobotVerticalSlice;
using WordsInSilence.Recognition.Common;
using WordsInSilence.Recognition.LiveInput;

namespace WordsInSilence.Tests.EditMode
{
    /// <summary>
    /// Step 10P v1.0c: the frozen common-2D auxiliary signal must be RECOVERABLE from a
    /// finalized practice attempt as one structured record, and the record must stay
    /// diagnostic - complete, single-line, privacy-clean and authority-free.
    /// </summary>
    public sealed class CommonSignalObservabilityV1cTests
    {
        static readonly string[] RequiredKeys =
        {
            "schema", "captureSessionId", "expectedWord", "commonSignalPresent",
            "artifactVersion", "artifactSha256", "requestedTarget", "eligible",
            "failureReason", "failureDetail", "failureDetailRedacted", "greetingProbability",
            "thankYouProbability", "topClass", "topProbability", "probabilityMargin",
            "expectedTargetSupport", "decisionValue", "sourceFrameCount",
            "commonEvidenceState", "outcomeKind", "predictedWord", "captureState",
            "strictAccepted", "strictExpectedWordClassAccept", "strictQuestSuccess",
            "guidedProgressAllowed", "evidenceGrantsSuccess",
            "evidenceGrantsLearnerError", "evidenceGrantsVeto", "nonFinite",
        };

        [SetUp]
        public void ResetDiagnostics()
        {
            CommonSignalAttemptDiagnostics.ResetForTest();
        }

        static CommonSignClassification Eligible(string target, bool support)
        {
            bool greetingTop = (target == "GREETING") == support;
            double g = greetingTop ? 0.9 : 0.1;
            return CommonSignClassification.Success(
                "common_2d_v1_6", "0123456789abcdef0123456789abcdef", target,
                g, 1.0 - g, greetingTop ? 2.2 : -2.2, 32);
        }

        static GuidedQuestOutcome Route(
            string expected, string predicted, bool success, CommonSignClassification common,
            PersonalLiveCaptureState state = PersonalLiveCaptureState.Completed)
        {
            return GuidedQuestOutcomeEvaluator.EvaluateForRoute(
                PersonalLiveProfileMode.MultiPrototypeCandidate, state, true,
                expected, predicted, success, success, success, false, false, common);
        }

        static PracticeRecognitionAttempt Attempt(
            string expected, string predicted, bool success, CommonSignClassification common,
            PersonalLiveCaptureState state = PersonalLiveCaptureState.Completed)
        {
            return new PracticeRecognitionAttempt(
                "session-0001", expected, predicted, state,
                Route(expected, predicted, success, common, state),
                null, null, PersonalLiveProfileMode.MultiPrototypeCandidate.ToString(),
                "profile-a", 0.42d, null, null,
                new PracticeRecognitionMetrics
                {
                    StrictAccepted = success,
                    StrictExpectedWordClassAccept = success,
                    StrictQuestSuccess = success,
                    WinningPrototypeId = string.Empty,
                },
                common);
        }

        static string Value(string record, string key)
        {
            int at = record.IndexOf("\"" + key + "\":", StringComparison.Ordinal);
            Assert.Greater(at, -1, "record is missing key " + key);
            int start = at + key.Length + 3;
            if (record[start] == '"')
            {
                int end = start + 1;
                while (end < record.Length && !(record[end] == '"' && record[end - 1] != '\\')) end++;
                return record.Substring(start + 1, end - start - 1);
            }
            if (record[start] == '[')
            {
                int close = record.IndexOf(']', start);
                Assert.Greater(close, start, "unterminated array for " + key);
                return record.Substring(start, close - start + 1);
            }
            int stop = start;
            while (stop < record.Length && record[stop] != ',' && record[stop] != '}') stop++;
            return record.Substring(start, stop - start);
        }

        [Test]
        public void EligibleSupportedAttemptExposesEveryContractField()
        {
            string record = CommonSignalAttemptDiagnostics.Build(
                Attempt("GREETING", "GREETING", true, Eligible("GREETING", true)));

            StringAssert.StartsWith(CommonSignalDiagnosticRecord.Prefix, record);
            foreach (string key in RequiredKeys)
                StringAssert.Contains("\"" + key + "\":", record);

            Assert.AreEqual(CommonSignalDiagnosticRecord.Schema, Value(record, "schema"));
            Assert.AreEqual("common_2d_v1_6", Value(record, "artifactVersion"));
            Assert.AreEqual("0123456789abcdef0123456789abcdef", Value(record, "artifactSha256"));
            Assert.AreEqual("GREETING", Value(record, "requestedTarget"));
            Assert.AreEqual("true", Value(record, "eligible"));
            Assert.AreEqual("None", Value(record, "failureReason"));
            Assert.AreEqual("GREETING", Value(record, "topClass"));
            Assert.AreEqual("true", Value(record, "expectedTargetSupport"));
            Assert.AreEqual("32", Value(record, "sourceFrameCount"));
            Assert.AreEqual(CommonEvidenceState.Support.ToString(),
                Value(record, "commonEvidenceState"));
            Assert.AreEqual(GuidedQuestOutcomeKind.StrictSuccess.ToString(),
                Value(record, "outcomeKind"));
            Assert.AreEqual(0.9d, double.Parse(Value(record, "greetingProbability"),
                System.Globalization.CultureInfo.InvariantCulture), 1e-12);
            Assert.AreEqual(0.1d, double.Parse(Value(record, "thankYouProbability"),
                System.Globalization.CultureInfo.InvariantCulture), 1e-12);
            Assert.AreEqual("[]", Value(record, "nonFinite"),
                "an eligible classification has no non-finite value");
        }

        [Test]
        public void NonSupportAndTechnicalEvidenceAreDistinguishable()
        {
            string nonSupport = CommonSignalAttemptDiagnostics.Build(
                Attempt("GREETING", "GREETING", true, Eligible("GREETING", false)));
            Assert.AreEqual(CommonEvidenceState.NonSupport.ToString(),
                Value(nonSupport, "commonEvidenceState"));
            Assert.AreEqual("THANK_YOU", Value(nonSupport, "topClass"));
            Assert.AreEqual("false", Value(nonSupport, "expectedTargetSupport"));

            string failed = CommonSignalAttemptDiagnostics.Build(
                Attempt("THANK_YOU", "THANK_YOU", true, CommonSignClassification.Failed(
                    "common_2d_v1_6", "0123456789abcdef0123456789abcdef", "THANK_YOU",
                    CommonSignFailureReason.SequenceMissing,
                    "no active common-2D capture was produced for this attempt")));
            Assert.AreEqual(CommonEvidenceState.TechnicalEvidence.ToString(),
                Value(failed, "commonEvidenceState"));
            Assert.AreEqual("false", Value(failed, "eligible"));
            Assert.AreEqual(CommonSignFailureReason.SequenceMissing.ToString(),
                Value(failed, "failureReason"));
            StringAssert.Contains("no active common-2D capture", Value(failed, "failureDetail"));
        }

        [Test]
        public void NonFiniteValuesBecomeNullAndAreNamed()
        {
            string record = CommonSignalAttemptDiagnostics.Build(
                Attempt("GREETING", "GREETING", true, CommonSignClassification.Failed(
                    "common_2d_v1_6", "sha", "GREETING",
                    CommonSignFailureReason.ModelUnavailable, "model unavailable")));

            foreach (string key in new[] { "greetingProbability", "thankYouProbability",
                                           "topProbability", "probabilityMargin",
                                           "decisionValue" })
            {
                Assert.AreEqual("null", Value(record, key), key + " must be JSON null");
                StringAssert.Contains("\"" + key + "\"", Value(record, "nonFinite"));
            }
        }

        [Test]
        public void RecordIsExactlyOneLineAndCarriesNoPayloadOrPath()
        {
            string record = CommonSignalAttemptDiagnostics.Build(
                Attempt("THANK_YOU", "THANK_YOU", true, CommonSignClassification.Failed(
                    "common_2d_v1_6", "sha", "THANK_YOU",
                    CommonSignFailureReason.InvalidShape,
                    "line one\r\nline two\twith \"quotes\" and \\ backslash")));

            Assert.AreEqual(-1, record.IndexOf('\n'));
            Assert.AreEqual(-1, record.IndexOf('\r'));
            StringAssert.Contains("line one", record);
            Assert.IsFalse(record.Contains("C:\\"), "no host path may appear");
            Assert.IsFalse(record.Contains("landmark"), "no landmark payload may appear");
            Assert.IsFalse(record.Contains("Coordinates"), "no frame payload may appear");
        }

        [Test]
        public void LongDetailIsTruncatedSoOneAttemptStaysOneRecord()
        {
            string record = CommonSignalAttemptDiagnostics.Build(
                Attempt("GREETING", "GREETING", true, CommonSignClassification.Failed(
                    "common_2d_v1_6", "sha", "GREETING",
                    CommonSignFailureReason.NonFiniteInput, new string('x', 2000))));
            string detail = Value(record, "failureDetail");
            Assert.LessOrEqual(detail.Length,
                CommonSignalDiagnosticRecord.MaxDetailLength + 3);
            StringAssert.EndsWith("...", detail);
        }

        [Test]
        public void OutOfContractTargetIsNotApplicableAndNotAFailure()
        {
            string record = CommonSignalAttemptDiagnostics.Build(
                Attempt("LOVE", "LOVE", true, null));
            Assert.AreEqual(CommonEvidenceState.NotApplicable.ToString(),
                Value(record, "commonEvidenceState"));
            Assert.AreEqual("false", Value(record, "commonSignalPresent"));
            Assert.AreEqual(CommonSignFailureReason.None.ToString(),
                Value(record, "failureReason"));
            Assert.AreEqual(GuidedQuestOutcomeKind.StrictSuccess.ToString(),
                Value(record, "outcomeKind"));
        }

        [Test]
        public void DiagnosticsNeverCarryAcceptanceAuthorityAndNeverMutateTheAttempt()
        {
            PracticeRecognitionAttempt attempt =
                Attempt("GREETING", "GREETING", true, Eligible("GREETING", true));
            GuidedQuestOutcomeKind kindBefore = attempt.Outcome.Kind;
            bool acceptedBefore = attempt.StrictAccepted;
            bool questBefore = attempt.StrictQuestSuccess;
            CommonEvidenceState stateBefore = attempt.Outcome.Common.State;

            string first = CommonSignalAttemptDiagnostics.Build(attempt);
            string second = CommonSignalAttemptDiagnostics.Build(attempt);

            Assert.AreEqual(first, second, "the record builder must be pure");
            Assert.AreEqual(kindBefore, attempt.Outcome.Kind);
            Assert.AreEqual(acceptedBefore, attempt.StrictAccepted);
            Assert.AreEqual(questBefore, attempt.StrictQuestSuccess);
            Assert.AreEqual(stateBefore, attempt.Outcome.Common.State);
            Assert.AreEqual("false", Value(first, "evidenceGrantsSuccess"));
            Assert.AreEqual("false", Value(first, "evidenceGrantsLearnerError"));
            Assert.AreEqual("false", Value(first, "evidenceGrantsVeto"));
        }

        [Test]
        public void NullAttemptProducesNoRecord()
        {
            Assert.AreEqual(string.Empty, CommonSignalAttemptDiagnostics.Build(null));
            Assert.DoesNotThrow(() => CommonSignalAttemptDiagnostics.Emit(null));
        }
    }
}
