using System;
using System.Collections.Generic;
using NUnit.Framework;
using WordsInSilence.Practice.PetRobotVerticalSlice;
using WordsInSilence.Recognition.Common;
using WordsInSilence.Recognition.LiveInput;

namespace WordsInSilence.Tests.EditMode
{
    /// <summary>
    /// Step 10P v1.0c1: the common-2D diagnostic must be recorded exactly once per unique
    /// finalized attempt, must never publish a filesystem location, and must never let a
    /// diagnostic failure reach the capture path.
    ///
    /// These are runtime-behaviour tests: they drive the real
    /// <see cref="PetRobotSignPracticeSession"/> through test doubles and count the
    /// records that actually reach the sink. They are not substring or call-site counts.
    /// </summary>
    public sealed class CommonSignalExactOncePrivacyV1c1Tests
    {
        readonly List<string> emitted = new List<string>();

        [SetUp]
        public void SetUp()
        {
            emitted.Clear();
            CommonSignalAttemptDiagnostics.ResetForTest();
            CommonSignalAttemptDiagnostics.Sink = line => emitted.Add(line);
        }

        [TearDown]
        public void TearDown()
        {
            CommonSignalAttemptDiagnostics.ResetForTest();
        }

        // ---- doubles ------------------------------------------------------

        sealed class FakeRepository : IPersonalizationCandidateRepository
        {
            public int SaveCallCount { get; private set; }
            public string RootDirectory { get { return "<test>"; } }
            public string LearnerId { get { return "test_learner"; } }
            public CandidateCountSnapshot Counts { get { return CandidateCountSnapshot.Empty; } }
            public CandidateCountSnapshot ReloadAndVerify() { return CandidateCountSnapshot.Empty; }

            public CandidateSaveResult Save(PersonalizationCandidateSaveRequest request)
            {
                SaveCallCount++;
                return new CandidateSaveResult(CandidateStorageStatus.Saved, "<test>",
                    "saved", CandidateCountSnapshot.Empty);
            }
        }

        sealed class FakeAdapter : IPersonalLivePracticeRecognitionAdapter
        {
            public string NextSessionId = "session-1";
            public string ExpectedWord = "GREETING";
            public string FailureDetail = string.Empty;
            public bool Eligible = true;

            /// <summary>
            /// Step 10P v1.0c2b: opt-in only. The default stays StrictSuccess so the
            /// other tests are unaffected. A test that needs a RETRYABLE attempt sets
            /// this, and the produced attempt is what the approved evaluator classifies
            /// as a technical failure: a Failed capture state, no strict result and no
            /// predicted word. A technical failure leaves the word incomplete, which is
            /// exactly why the approved rule then allows Retry.
            /// </summary>
            public bool NextIsTechnicalFailure;

            public PersonalLiveCaptureState CaptureState
            {
                get
                {
                    return NextIsTechnicalFailure
                        ? PersonalLiveCaptureState.Failed
                        : PersonalLiveCaptureState.Completed;
                }
            }
            public string CaptureSessionId { get { return NextSessionId; } }
            public string ActiveProfileId { get { return "profile-test"; } }
            public bool CandidateRouteActive { get { return true; } }
            public bool SetExpectedWord(string word) { ExpectedWord = word; return true; }
            public bool BeginCapture() { return true; }
            public bool ResetCapture() { return true; }

            public PracticeRecognitionAttempt FinalizeCapture()
            {
                CommonSignClassification common = Eligible
                    ? CommonSignClassification.Success(
                        "common_2d_v1_6", "0123456789abcdef", ExpectedWord, 0.8, 0.2, 1.4, 32)
                    : CommonSignClassification.Failed(
                        "common_2d_v1_6", "0123456789abcdef", ExpectedWord,
                        CommonSignFailureReason.ModelUnavailable, FailureDetail);
                bool technical = NextIsTechnicalFailure;
                bool strict = !technical;
                string predicted = technical ? string.Empty : ExpectedWord;
                PersonalLiveCaptureState state = technical
                    ? PersonalLiveCaptureState.Failed
                    : PersonalLiveCaptureState.Completed;

                // The outcome is produced by the approved evaluator, not asserted here:
                // a Failed capture state with no strict result and no predicted word is
                // what it classifies as GuidedQuestOutcomeKind.TechnicalFailure.
                GuidedQuestOutcome outcome = GuidedQuestOutcomeEvaluator.EvaluateForRoute(
                    PersonalLiveProfileMode.MultiPrototypeCandidate,
                    state, strict, ExpectedWord, predicted,
                    strict, strict, strict, false, false, common);
                return new PracticeRecognitionAttempt(
                    NextSessionId, ExpectedWord, predicted,
                    state, outcome, null, null,
                    PersonalLiveProfileMode.MultiPrototypeCandidate.ToString(),
                    "profile-test", 0.4d, null, null,
                    new PracticeRecognitionMetrics
                    {
                        StrictAccepted = strict,
                        StrictExpectedWordClassAccept = strict,
                        StrictQuestSuccess = strict,
                        WinningPrototypeId = string.Empty,
                    },
                    common);
            }
        }

        static PetRobotSignPracticeSession NewSession(FakeAdapter adapter, FakeRepository repo)
        {
            return new PetRobotSignPracticeSession(adapter, repo);
        }

        // ---- F1: exact once ------------------------------------------------

        [Test]
        public void FirstUniqueFinalizedAttemptRecordsExactlyOnce()
        {
            var adapter = new FakeAdapter();
            var session = NewSession(adapter, new FakeRepository());

            Assert.IsTrue(session.StartExpression());
            Assert.IsTrue(session.FinishExpression());

            Assert.AreEqual(1, emitted.Count, "one finalized attempt must record once");
            StringAssert.StartsWith(CommonSignalDiagnosticRecord.Prefix, emitted[0]);
            StringAssert.Contains("\"captureSessionId\":\"session-1\"", emitted[0]);
        }

        /// <summary>
        /// Reaches a SECOND finalized capture the way the approved rules actually allow:
        /// the first attempt is a technical failure, which leaves the word incomplete and
        /// is precisely the outcome the session permits a retry after. A StrictSuccess
        /// masters the word and correctly forbids a retry, which is why the earlier
        /// version of this test failed - the production rule was right, the premise was
        /// not. Nothing about success, mastery or retry policy is changed here.
        /// </summary>
        [Test]
        public void DuplicateCallbackForTheSameSessionAddsNoRecord()
        {
            var adapter = new FakeAdapter { NextIsTechnicalFailure = true };
            var repo = new FakeRepository();
            var session = NewSession(adapter, repo);

            Assert.IsTrue(session.StartExpression());
            Assert.IsTrue(session.FinishExpression());
            int afterFirst = emitted.Count;

            // the retryable precondition is asserted, never assumed
            Assert.AreEqual(GuidedQuestOutcomeKind.TechnicalFailure,
                session.LastAttempt.Outcome.Kind);
            Assert.AreEqual(PetRobotPracticeState.Feedback, session.State);
            Assert.IsFalse(session.IsCurrentWordComplete,
                "a technical failure must not master the word");
            Assert.IsTrue(session.CanRetry, "the approved rule allows a retry here");
            Assert.AreEqual(0, repo.SaveCallCount,
                "a technical failure is not learner performance and is not stored");

            Assert.IsTrue(session.Retry());

            // the adapter replays the SAME capture session id on the second cycle
            Assert.IsTrue(session.StartExpression());
            Assert.IsTrue(session.FinishExpression());

            Assert.AreEqual(1, afterFirst);
            Assert.AreEqual(1, emitted.Count, "a duplicate callback must not record again");
            Assert.AreEqual(CandidateStorageStatus.Duplicate, session.LastStorageResult.Status,
                "the existing duplicate meaning must be preserved");
            Assert.AreEqual(0, repo.SaveCallCount,
                "a duplicate callback must not store a candidate");
        }

        /// <summary>
        /// Same real precondition as the duplicate test, but the second cycle carries a
        /// NEW capture session id, so it must record a second time - once per distinct
        /// finalized attempt, in order.
        /// </summary>
        [Test]
        public void RetryWithANewSessionIdRecordsAgainExactlyOnce()
        {
            var adapter = new FakeAdapter { NextIsTechnicalFailure = true };
            var repo = new FakeRepository();
            var session = NewSession(adapter, repo);

            Assert.IsTrue(session.StartExpression());
            Assert.IsTrue(session.FinishExpression());
            Assert.AreEqual(1, emitted.Count);

            Assert.AreEqual(GuidedQuestOutcomeKind.TechnicalFailure,
                session.LastAttempt.Outcome.Kind);
            Assert.AreEqual(PetRobotPracticeState.Feedback, session.State);
            Assert.IsFalse(session.IsCurrentWordComplete);
            Assert.IsTrue(session.CanRetry);
            Assert.AreEqual(0, repo.SaveCallCount);

            adapter.NextSessionId = "session-2";
            Assert.IsTrue(session.Retry());
            Assert.IsTrue(session.StartExpression());
            Assert.IsTrue(session.FinishExpression());

            Assert.AreEqual(2, emitted.Count);
            StringAssert.Contains("\"captureSessionId\":\"session-1\"", emitted[0]);
            StringAssert.Contains("\"captureSessionId\":\"session-2\"", emitted[1]);
            Assert.AreEqual(0, repo.SaveCallCount,
                "neither technical-failure cycle stores a candidate");
        }

        [Test]
        public void BlankCaptureSessionIdRecordsNoAcceptedAttempt()
        {
            var adapter = new FakeAdapter { NextSessionId = "   " };
            var session = NewSession(adapter, new FakeRepository());

            Assert.IsTrue(session.StartExpression());
            Assert.IsTrue(session.FinishExpression());

            Assert.AreEqual(0, emitted.Count,
                "an attempt with no capture session id is not an accepted attempt");
        }

        [Test]
        public void GateItselfEnforcesExactOnceIndependently()
        {
            var gate = new CommonSignalDiagnosticGate();
            var sink = new List<string>();

            Assert.IsTrue(gate.TryRecord("a", () => "r1", sink.Add));
            Assert.IsFalse(gate.TryRecord("a", () => "r2", sink.Add));
            Assert.IsFalse(gate.TryRecord(string.Empty, () => "r3", sink.Add));
            Assert.IsFalse(gate.TryRecord(null, () => "r4", sink.Add));
            Assert.IsTrue(gate.TryRecord("b", () => "r5", sink.Add));

            Assert.AreEqual(2, sink.Count);
            Assert.AreEqual(2, gate.RecordedCount);
            Assert.AreEqual(3, gate.SuppressedCount);
        }

        // ---- F3: fail non-interference --------------------------------------

        [Test]
        public void SinkFailureNeverReachesTheCapturePath()
        {
            CommonSignalAttemptDiagnostics.Sink =
                line => { throw new InvalidOperationException("sink exploded"); };
            var adapter = new FakeAdapter();
            var repo = new FakeRepository();
            var session = NewSession(adapter, repo);

            Assert.IsTrue(session.StartExpression());
            Assert.DoesNotThrow(() => session.FinishExpression());
            Assert.AreEqual(PetRobotPracticeState.Feedback, session.State);
            Assert.IsNotNull(session.LastAttempt);
            Assert.AreEqual(GuidedQuestOutcomeKind.StrictSuccess, session.LastAttempt.Outcome.Kind);
            Assert.AreEqual(1, repo.SaveCallCount,
                "the approved candidate flow must complete despite a diagnostic failure");
        }

        [Test]
        public void ComposerFailureIsIsolatedAndLeavesABoundedMarker()
        {
            var gate = new CommonSignalDiagnosticGate();
            var sink = new List<string>();

            Assert.DoesNotThrow(() => gate.TryRecord(
                "a", () => { throw new InvalidOperationException("composer exploded"); },
                sink.Add));

            Assert.AreEqual(1, sink.Count);
            StringAssert.StartsWith(CommonSignalDiagnosticRecord.Prefix, sink[0]);
            StringAssert.Contains("compose_failed", sink[0]);
            Assert.IsFalse(sink[0].Contains("exploded"),
                "an exception message may carry a path and is never published");
            Assert.AreEqual(-1, sink[0].IndexOf('\n'));
            Assert.AreEqual(1, gate.IsolatedFailureCount);
        }

        // ---- F2: failure-detail privacy --------------------------------------

        static readonly string[][] PathFixtures =
        {
            new[] { "model directory not found: C:\\Users\\local-user\\AppData\\LocalLow\\model",
                    "C:\\Users\\local-user" },
            new[] { "model directory not found: C:/Users/local-user/AppData/LocalLow/model",
                    "C:/Users/local-user" },
            new[] { "model directory not found: \\\\BUILDSRV\\share\\models\\v1_6", "BUILDSRV" },
            new[] { "manifest unreadable: file:///C:/Users/local-user/models/frozen.json", "file://" },
            new[] { "model directory not found: /home/local-user/models/frozen/v1_6", "/home/local-user" },
            new[] { "model directory not found: ~/models/frozen/v1_6", "~/models" },
        };

        [Test]
        public void EveryPathShapeIsRedactedInTheRecordAndInTheUiText()
        {
            foreach (string[] fixture in PathFixtures)
            {
                string raw = fixture[0];
                string secret = fixture[1];

                bool redacted;
                string uiText = CommonSignalDiagnosticRecord.SanitizeDetail(raw, out redacted);
                Assert.IsTrue(redacted, "redaction flag must be set for " + secret);
                Assert.IsFalse(uiText.Contains(secret), "UI text leaked " + secret);
                StringAssert.Contains(CommonSignalDiagnosticRecord.RedactedPathMarker, uiText);

                var adapter = new FakeAdapter { Eligible = false, FailureDetail = raw };
                var session = NewSession(adapter, new FakeRepository());
                emitted.Clear();
                CommonSignalAttemptDiagnostics.ResetForTest();
                CommonSignalAttemptDiagnostics.Sink = line => emitted.Add(line);
                Assert.IsTrue(session.StartExpression());
                Assert.IsTrue(session.FinishExpression());

                Assert.AreEqual(1, emitted.Count);
                Assert.IsFalse(emitted[0].Contains(secret), "record leaked " + secret);
                StringAssert.Contains("\"failureDetailRedacted\":true", emitted[0]);
                StringAssert.Contains("\"failureReason\":\"ModelUnavailable\"", emitted[0],
                    "the typed reason must survive redaction");
            }
        }

        [Test]
        public void CleanDetailIsNotFlaggedAndControlCharactersStaySingleLine()
        {
            bool redacted;
            string clean = CommonSignalDiagnosticRecord.SanitizeDetail(
                "input sequence had 0 valid landmarks", out redacted);
            Assert.IsFalse(redacted);
            Assert.AreEqual("input sequence had 0 valid landmarks", clean);

            bool controlRedacted;
            string flattened = CommonSignalDiagnosticRecord.SanitizeDetail(
                "line one\r\nline two\twith \"quotes\"", out controlRedacted);
            Assert.IsTrue(controlRedacted);
            Assert.AreEqual(-1, flattened.IndexOf('\n'));
            Assert.AreEqual(-1, flattened.IndexOf('\r'));
            Assert.AreEqual(-1, flattened.IndexOf('\t'));
            StringAssert.Contains("line one", flattened);
        }

        [Test]
        public void OverlongDetailStaysWithinFourHundredCharactersIncludingTheMarker()
        {
            bool redacted;
            string bounded = CommonSignalDiagnosticRecord.SanitizeDetail(
                new string('x', 4000), out redacted);
            Assert.IsTrue(redacted);
            Assert.LessOrEqual(bounded.Length, CommonSignalDiagnosticRecord.MaxDetailLength,
                "the marker counts towards the 400 character bound");
            StringAssert.EndsWith(CommonSignalDiagnosticRecord.TruncationMarker, bounded);
            Assert.AreEqual(-1, bounded.IndexOf('\n'));
        }

        // ---- Step 10P v1.0c2 additions -----------------------------------------

        [Test]
        public void AGateNeverForgetsAnAcceptedSessionIdDuringItsLifetime()
        {
            var gate = new CommonSignalDiagnosticGate();
            var lines = new List<string>();

            Assert.IsTrue(gate.TryRecord("first-id", () => "r", lines.Add));
            for (int index = 0; index < 600; index++)
                Assert.IsTrue(gate.TryRecord("bulk-" + index, () => "r", lines.Add));

            int before = lines.Count;
            Assert.AreEqual(601, before);
            Assert.IsFalse(gate.ShouldRecord("first-id"),
                "an id accepted 600 ids ago must still be remembered");
            Assert.IsFalse(gate.TryRecord("first-id", () => "r", lines.Add));
            Assert.AreEqual(before, lines.Count, "no additional sink line for a replayed id");
            Assert.AreEqual(601, gate.RememberedSessionCount);
        }

        static readonly string[][] EdgePathFixtures =
        {
            new[] { @"model directory not found: C:\Users\local-user Smith\AppData\LocalLow\model\frozen.json",
                    "Alice", "Smith", "AppData", "frozen.json" },
            new[] { "model directory not found: C:/Users/local-user Smith/AppData/LocalLow/model/frozen.json",
                    "Alice", "Smith", "AppData", "frozen.json" },
            new[] { @"share unreachable: \\BUILD SERVER\private share\models\frozen.json",
                    "BUILD SERVER", "private share", "models", "frozen.json" },
            new[] { "manifest unreadable: file:///C:/Users/local-user Smith/models/frozen.json",
                    "Alice", "Smith", "file:", "frozen.json" },
            new[] { "model directory not found: /home/local-user smith/models/frozen.json",
                    "home", "alice", "smith", "frozen.json" },
            new[] { "model directory not found: /secret", "secret" },
            new[] { "model directory not found: ~/secret", "secret", "~/" },
        };

        [Test]
        public void WhitespaceAndRootPathsLeaveNoUsernameOrSuffixFragment()
        {
            foreach (string[] fixture in EdgePathFixtures)
            {
                string raw = fixture[0];
                bool redacted;
                string uiText = CommonSignalDiagnosticRecord.SanitizeDetail(raw, out redacted);

                Assert.IsTrue(redacted, raw);
                StringAssert.Contains(CommonSignalDiagnosticRecord.RedactedPathMarker, uiText);
                for (int index = 1; index < fixture.Length; index++)
                    Assert.IsFalse(uiText.Contains(fixture[index]),
                        "UI text leaked " + fixture[index] + " from " + raw);

                var adapter = new FakeAdapter { Eligible = false, FailureDetail = raw };
                var session = NewSession(adapter, new FakeRepository());
                emitted.Clear();
                CommonSignalAttemptDiagnostics.ResetForTest();
                CommonSignalAttemptDiagnostics.Sink = line => emitted.Add(line);
                Assert.IsTrue(session.StartExpression());
                Assert.IsTrue(session.FinishExpression());

                Assert.AreEqual(1, emitted.Count);
                for (int index = 1; index < fixture.Length; index++)
                    Assert.IsFalse(emitted[0].Contains(fixture[index]),
                        "record leaked " + fixture[index] + " from " + raw);
                StringAssert.Contains("\"failureDetailRedacted\":true", emitted[0]);
                StringAssert.Contains("\"failureReason\":\"ModelUnavailable\"", emitted[0]);
            }
        }

        [Test]
        public void OrdinaryTextWithSlashesIsNotRedacted()
        {
            bool redacted;
            string text = CommonSignalDiagnosticRecord.SanitizeDetail(
                "input sequence had 0/32 valid landmarks and/or 0 frames", out redacted);
            Assert.IsFalse(redacted, "a slash inside a token is not a path start");
            Assert.AreEqual("input sequence had 0/32 valid landmarks and/or 0 frames", text);
        }
    }
}
