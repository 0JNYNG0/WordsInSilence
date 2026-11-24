// M3_H3_OBSERVER_R5 - tests for the optional H3 observation seam.
//
// R5 corrects four concrete defects the R4 review found:
//   * path validation now refuses drive-relative, root-relative, UNC, device and URI forms,
//     and the synthetic good paths are Windows-absolute with existence and loading injected,
//     so no real file system is touched;
//   * the "no slash" assertions were guaranteed to fail because the fixed and approved
//     thresholds field legitimately contains '/', so they now assert the exact slash count
//     and the absence of path shapes instead of banning a valid character;
//   * the valid-startup test now runs the real Initialize with a synthetic VALID bundle and
//     verifies delivery through the DEFAULT sink Initialize installs, rather than manually
//     installing the hook and replacing the sink it was supposed to test;
//   * the bound assertions now check the exact declared maximum, marker included.
//
// EditMode is the right mode: nothing under test is a MonoBehaviour. The synthetic snapshot
// and the synthetic bundle use the real constructors, reachable because
// Assets/_Project/Scripts/AssemblyInfo.cs declares
// InternalsVisibleTo("WordsInSilence.Tests.EditMode"). No learner repository, default
// identity, camera, scene or bundle FILE is touched.
//
// These prove WIRING, ISOLATION, EXTRACTION, REDACTION and BOUNDS. They do not claim to
// prove the numerical fidelity of the H3 port; R2's parity run already covers that.
using System;
using System.Collections.Generic;
using System.Text.RegularExpressions;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;
using WordsInSilence.Practice.PetRobotVerticalSlice;
using WordsInSilence.Practice.PetRobotVerticalSlice.H3Observation;
using WordsInSilence.Recognition.Core;
using WordsInSilence.Recognition.H3Runtime;
using WordsInSilence.Recognition.LiveInput;

namespace WordsInSilence.Tests.EditMode
{
    [TestFixture]
    public sealed class H3AttemptObserverTests
    {
        const string GoodRoot = @"C:\synthetic\wis_h3_bundle";
        static readonly string FakeSha256 = new string('a', 64);

        [SetUp]
        public void SetUp()
        {
            H3ObservationBundle.ResetForTest();
            H3AttemptObserver.ResetForTest();
            H3ObserverStartup.ResetForTest();
            CommonSignalAttemptDiagnostics.ResetForTest();
        }

        [TearDown]
        public void TearDown()
        {
            H3AttemptObserver.Uninstall();
            H3ObservationBundle.ResetForTest();
            H3AttemptObserver.ResetForTest();
            H3ObserverStartup.ResetForTest();
            CommonSignalAttemptDiagnostics.ResetForTest();
        }

        // ---- synthetic fixtures, no personal data, no file system -------------

        static RecognitionSequence CreateSequence()
        {
            int length = 10;
            int landmarks = RecognitionSequence.RequiredLandmarkCount;
            var coordinates = new double[length * landmarks * 3];
            var mask = new bool[length * landmarks];
            for (int index = 0; index < mask.Length; index++) mask[index] = true;
            return new RecognitionSequence(
                "test_sample", "GREETING", "normal", "train",
                FakeSha256, FakeSha256, length, landmarks, coordinates, mask);
        }

        static CommonTwoDimensionalCapture CreateCapture(int frames, int landmarks)
        {
            var coordinates = new double[frames * landmarks * 3];
            var mask = new bool[frames * landmarks];
            for (int index = 0; index < mask.Length; index++) mask[index] = true;
            return new CommonTwoDimensionalCapture(
                frames, landmarks, coordinates, mask, mask.Length, 0, 640, 480,
                "test_convention");
        }

        static PersonalLiveRecognitionSnapshot CreateSnapshot(
            CommonTwoDimensionalCapture capture, string failure)
        {
            var scores = new Dictionary<string, double>
            {
                { "GREETING", 0.21 }, { "LOVE", 0.94 }, { "THANK_YOU", 0.66 }
            };
            var accept = new Dictionary<string, bool>
            {
                { "GREETING", true }, { "LOVE", false }, { "THANK_YOU", false }
            };
            var routing = new PersonalLiveRoutingResult(
                "GREETING", true, true, true, scores, accept, 0.21, "prototype-1",
                new string[0]);
            var preprocessing = new ApprovedPreprocessingResult(
                24, 300, 2,
                new WordsInSilence.Recognition.Runtime.NormalizedRecognitionFrame[0]);
            return new PersonalLiveRecognitionSnapshot(
                "snapshot", "GREETING", preprocessing, CreateSequence(), routing,
                "test_profile", true, capture, failure);
        }

        static PracticeRecognitionAttempt CreateAttempt(
            string captureId, PersonalLiveRecognitionSnapshot snapshot)
        {
            return new PracticeRecognitionAttempt(
                captureId, "GREETING", "GREETING", PersonalLiveCaptureState.Completed,
                null, snapshot, CreateSequence(), "route", "test_profile", 0.5, null, null);
        }

        static PracticeRecognitionAttempt ObservableAttempt(string captureId)
        {
            return CreateAttempt(captureId,
                CreateSnapshot(CreateCapture(24, H3Constants.LandmarkCount), null));
        }

        /// <summary>A structurally valid bundle, built from the real R2 types.</summary>
        static H3RuntimeBundle SyntheticBundle()
        {
            int width = H3Constants.PairFeatureCount;
            var mean = new double[width];
            var scale = new double[width];
            var coefficients = new double[width];
            for (int index = 0; index < width; index++)
            {
                scale[index] = 1.0;
                coefficients[index] = 0.01;
            }
            var versions = new Dictionary<string, string>
            {
                { "coord_contract", "common2d_v1_6_input" },
                { "alignment", "maskaware_band010_path_h1v2" },
                { "pair_features", "h3_alignedpair36_v1" },
                { "normalization", "identity_common_v1" },
                { "model", "h3_candidate_v1" },
            };
            int frames = H3Constants.ResampleFrames;
            int landmarks = H3Constants.LandmarkCount;
            var bank = new Dictionary<string, H3ReferenceSequence[]>();
            string[] vocabulary = { "GREETING", "THANK_YOU" };
            for (int index = 0; index < vocabulary.Length; index++)
            {
                var coordinates = new double[frames * landmarks * H3Constants.CoordDims];
                var mask = new bool[frames * landmarks];
                for (int slot = 0; slot < mask.Length; slot++) mask[slot] = true;
                bank[vocabulary[index]] =
                    new[] { new H3ReferenceSequence(coordinates, mask, frames) };
            }
            return new H3RuntimeBundle(
                "m3_general_hybrid_h3_artifact_v1", "h3_pair_v1",
                new string('b', 64), new string('c', 64), new string('d', 64),
                versions, vocabulary, new[] { "different_word", "same_word" },
                H3PairGeometry.FeatureNames(), mean, scale, coefficients, 0.0, bank);
        }

        sealed class CountingScorer : IH3ObservationScorer
        {
            public bool Ready = true;
            public Exception Throw;
            public H3DecisionResult Result = AcceptedResult();
            public string ProvenanceValue = "model=h3_candidate_v1 payload=abc123def456";
            public int ScoreCalls;
            public double[] LastCoordinates;
            public bool[] LastMask;
            public int LastFrameCount;

            public bool IsReady { get { return Ready; } }
            public string NotReadyReason { get { return Ready ? string.Empty : "test_off"; } }
            public string Provenance { get { return ProvenanceValue; } }

            public H3DecisionResult Score(double[] coordinates, bool[] mask, int frameCount)
            {
                ScoreCalls++;
                LastCoordinates = coordinates;
                LastMask = mask;
                LastFrameCount = frameCount;
                if (Throw != null) throw Throw;
                return Result;
            }
        }

        static H3DecisionResult AcceptedResult()
        {
            return new H3DecisionResult
            {
                Status = H3Status.Accept,
                QualityCode = H3FailureCodes.Ok,
                Top1 = "GREETING",
                Predicted = "GREETING",
                Classes = new[] { "GREETING", "THANK_YOU" },
                RawCompatibility = new[] { 0.81, 0.12 },
                RelativeShareNotAPosterior = new[] { 0.87, 0.13 },
                BestRawCompatibility = 0.81,
                RunnerUpRawCompatibility = 0.12,
                RelativeGatePassed = true,
                AbsoluteGatePassed = true,
            };
        }

        static CountingScorer ReadyScorer()
        {
            var scorer = new CountingScorer();
            H3AttemptObserver.Scorer = scorer;
            return scorer;
        }

        static void UseValidBundle(Action<int> onLoad)
        {
            H3ObservationBundle.CommandLineSourceForTest =
                () => new[] { "game.exe", H3ObservationBundle.FlagName, GoodRoot };
            H3ObservationBundle.DirectoryExistsForTest = path => true;
            int loads = 0;
            H3ObservationBundle.LoaderForTest = path =>
            {
                loads++;
                if (onLoad != null) onLoad(loads);
                return SyntheticBundle();
            };
        }

        static int Count(string text, char character)
        {
            int total = 0;
            for (int index = 0; index < text.Length; index++)
                if (text[index] == character) total++;
            return total;
        }

        // ==== 1. fully qualified local path validation =========================

        [TestCase(@"C:\bundle")]
        [TestCase(@"C:/bundle")]
        [TestCase(@"d:\a\b\c")]
        [TestCase(@"C:\")]
        public void FullyQualifiedLocalPaths_AreAccepted(string value)
        {
            Assert.IsTrue(H3ObservationBundle.IsFullyQualifiedLocalDirectory(value));
        }

        [TestCase(@"C:bundle", TestName = "drive relative")]
        [TestCase(@"\bundle", TestName = "root relative backslash")]
        [TestCase("/bundle", TestName = "root relative slash")]
        [TestCase(@"\\server\share", TestName = "UNC network")]
        [TestCase(@"\\?\C:\bundle", TestName = "extended length device")]
        [TestCase(@"\\.\PhysicalDrive0", TestName = "device namespace")]
        [TestCase("file://C:/bundle", TestName = "file URI")]
        [TestCase("http://host/bundle", TestName = "http URI")]
        [TestCase("bundle", TestName = "bare relative")]
        [TestCase("C:", TestName = "drive only")]
        [TestCase("C:\u0001\\x", TestName = "control character")]
        public void NonLocalOrRelativePaths_AreRefused(string value)
        {
            Assert.IsFalse(H3ObservationBundle.IsFullyQualifiedLocalDirectory(value));
        }

        [TestCase(@"C:bundle")]
        [TestCase(@"\\server\share")]
        [TestCase("relative/bundle")]
        [TestCase("file://C:/bundle")]
        public void NonLocalFlagValue_IsRefusedBeforeAnyExistenceOrLoaderCall(string value)
        {
            int existenceChecks = 0;
            int loads = 0;
            H3ObservationBundle.CommandLineSourceForTest =
                () => new[] { "game.exe", H3ObservationBundle.FlagName, value };
            H3ObservationBundle.DirectoryExistsForTest = path => { existenceChecks++; return true; };
            H3ObservationBundle.LoaderForTest = path => { loads++; return SyntheticBundle(); };

            Assert.AreEqual(H3BundleState.BundleRootNotFullyQualifiedLocal,
                            H3ObservationBundle.State);
            Assert.AreEqual(0, existenceChecks, "no existence check may run for a refused path");
            Assert.AreEqual(0, loads, "no loader call may run for a refused path");
            Assert.AreEqual(0, H3ObservationBundle.LoadAttempts);
            Assert.IsFalse(H3ObservationBundle.ReasonCode.Contains(value),
                "the flag value must never appear in the reason code");
        }

        [Test]
        public void MissingRoot_IsRefusedUsingAnInjectedExistenceCheck()
        {
            H3ObservationBundle.CommandLineSourceForTest =
                () => new[] { "game.exe", H3ObservationBundle.FlagName, GoodRoot };
            H3ObservationBundle.DirectoryExistsForTest = path => false;

            Assert.AreEqual(H3BundleState.BundleRootMissing, H3ObservationBundle.State);
            Assert.AreEqual(0, H3ObservationBundle.LoadAttempts);
        }

        [Test]
        public void FlagAbsent_IsInertAndNeverLoads()
        {
            H3ObservationBundle.CommandLineSourceForTest =
                () => new[] { "game.exe", "-other", "value" };
            Assert.AreEqual(H3BundleState.FlagAbsent, H3ObservationBundle.State);
            Assert.AreEqual(0, H3ObservationBundle.LoadAttempts);
        }

        [Test]
        public void FlagDuplicated_IsRefused()
        {
            H3ObservationBundle.CommandLineSourceForTest = () => new[]
            {
                "game.exe", H3ObservationBundle.FlagName, GoodRoot,
                H3ObservationBundle.FlagName, @"C:\other"
            };
            Assert.AreEqual(H3BundleState.FlagDuplicated, H3ObservationBundle.State);
            Assert.AreEqual(0, H3ObservationBundle.LoadAttempts);
        }

        [Test]
        public void ArgumentListLongerThanTheScanLimit_IsRefused()
        {
            var arguments = new string[H3ObservationBundle.MaxScannedArguments + 2];
            for (int index = 0; index < arguments.Length; index++) arguments[index] = "x";
            arguments[0] = H3ObservationBundle.FlagName;
            arguments[1] = GoodRoot;
            H3ObservationBundle.CommandLineSourceForTest = () => arguments;

            Assert.AreEqual(H3BundleState.ArgumentListTooLong, H3ObservationBundle.State);
            Assert.AreEqual(0, H3ObservationBundle.LoadAttempts);
        }

        [TestCase("")]
        [TestCase("   ")]
        [TestCase("\t")]
        public void WhitespaceOrEmptyValue_IsRefused(string value)
        {
            H3ObservationBundle.CommandLineSourceForTest =
                () => new[] { "game.exe", H3ObservationBundle.FlagName, value };
            Assert.AreEqual(H3BundleState.FlagValueMissing, H3ObservationBundle.State);
        }

        [Test]
        public void LoaderReturningNull_IsATypedFailure()
        {
            H3ObservationBundle.CommandLineSourceForTest =
                () => new[] { "game.exe", H3ObservationBundle.FlagName, GoodRoot };
            H3ObservationBundle.DirectoryExistsForTest = path => true;
            H3ObservationBundle.LoaderForTest = path => null;

            Assert.AreEqual(H3BundleState.BundleLoadFailed, H3ObservationBundle.State);
            Assert.AreEqual("bundle_null", H3ObservationBundle.ReasonCode);
        }

        [Test]
        public void InvalidBundle_ReportsTheTypedCodeOnly()
        {
            H3ObservationBundle.CommandLineSourceForTest =
                () => new[] { "game.exe", H3ObservationBundle.FlagName, GoodRoot };
            H3ObservationBundle.DirectoryExistsForTest = path => true;
            H3ObservationBundle.LoaderForTest = path =>
            {
                throw new H3FormatException(H3FormatCodes.DigestMismatch, @"C:\secret\payload");
            };

            Assert.AreEqual(H3BundleState.BundleInvalid, H3ObservationBundle.State);
            StringAssert.Contains(H3FormatCodes.DigestMismatch, H3ObservationBundle.ReasonCode);
            Assert.IsFalse(H3ObservationBundle.ReasonCode.Contains("secret"));
            Assert.IsFalse(H3ObservationBundle.ReasonCode.Contains("\\"));
        }

        [Test]
        public void RepeatedReadinessChecks_LoadAtMostOnce()
        {
            int loaderCalls = 0;
            UseValidBundle(count => loaderCalls = count);

            for (int index = 0; index < 5; index++)
            {
                bool ignoredReady = H3ObservationBundle.IsReady;
                H3BundleState ignoredState = H3ObservationBundle.State;
                string ignoredReason = H3ObservationBundle.ReasonCode;
                object ignoredModel = H3ObservationBundle.Model;
                object ignoredBundle = H3ObservationBundle.Bundle;
            }

            Assert.AreEqual(1, loaderCalls, "the bundle resolves at most once per process");
            Assert.AreEqual(1, H3ObservationBundle.LoadAttempts);
            Assert.IsTrue(H3ObservationBundle.IsReady);
        }

        // ==== 2. the record's fixed syntax versus variable metadata ============

        [Test]
        public void HostileProvenance_IsSanitizedWithoutBreakingTheFixedThresholds()
        {
            CountingScorer scorer = ReadyScorer();
            scorer.ProvenanceValue =
                "C:\\Users\\local-user\\secret\\bundle.npz\nline2\ttab /unix/path/here "
                + new string('x', 400);
            var observations = new List<string>();
            H3AttemptObserver.Sink = observations.Add;
            H3AttemptObserver.Install();

            CommonSignalAttemptDiagnostics.Emit(ObservableAttempt("cap-hostile"));
            string line = observations[0];

            Assert.IsFalse(line.Contains("\\"), "no backslash may appear anywhere");
            Assert.IsFalse(line.Contains("://"));
            Assert.IsFalse(line.Contains("C:"), "no drive-qualified shape survives");
            Assert.IsFalse(line.Contains("\n"));
            Assert.IsFalse(line.Contains("\t"));
            // the ONLY '/' characters are the two in the fixed, approved thresholds field
            Assert.AreEqual(2, Count(line, '/'),
                "sanitized metadata contributes no slash; the two are thresholds=a/b/c");
            StringAssert.Contains("thresholds=0.6/0.15/0.5", line);
            Assert.LessOrEqual(line.Length, H3RecordText.MaxRecordLength);
        }

        [Test]
        public void HostileClassLabels_AreSanitizedAndCountsStayVisible()
        {
            CountingScorer scorer = ReadyScorer();
            int classCount = 12;
            var classes = new string[classCount];
            var raw = new double[classCount];
            var shares = new double[classCount];
            for (int index = 0; index < classCount; index++)
            {
                classes[index] = "C:/word/" + index + "\n";
                raw[index] = 0.5;
                shares[index] = 1.0 / classCount;
            }
            scorer.Result = new H3DecisionResult
            {
                Status = H3Status.Abstain,
                QualityCode = H3FailureCodes.Ok,
                Top1 = classes[0],
                Classes = classes,
                RawCompatibility = raw,
                RelativeShareNotAPosterior = shares,
                BestRawCompatibility = 0.5,
                RunnerUpRawCompatibility = 0.5,
                RelativeGatePassed = false,
                AbsoluteGatePassed = true,
            };
            var observations = new List<string>();
            H3AttemptObserver.Sink = observations.Add;
            H3AttemptObserver.Install();

            CommonSignalAttemptDiagnostics.Emit(ObservableAttempt("cap-classes"));
            string line = observations[0];

            Assert.AreEqual(2, Count(line, '/'), "only the fixed thresholds field has slashes");
            Assert.IsFalse(line.Contains("\n"));
            StringAssert.Contains("scored_classes=12", line);
            StringAssert.Contains("reported_classes=8", line);
            StringAssert.Contains("omitted_classes=4", line);
            StringAssert.Contains("NO_PROMOTION", line);
            Assert.LessOrEqual(line.Length, H3RecordText.MaxRecordLength);
        }

        [Test]
        public void HostileCorrelation_IsSanitized()
        {
            ReadyScorer();
            var observations = new List<string>();
            H3AttemptObserver.Sink = observations.Add;
            H3AttemptObserver.Install();

            CommonSignalAttemptDiagnostics.Emit(
                ObservableAttempt("D:\\dir\\capture 1\nsecond"));
            string line = observations[0];

            Assert.IsFalse(line.Contains("\\"));
            Assert.IsFalse(line.Contains("D:"));
            StringAssert.Contains("attempt=D__dir_capture_1_second", line);
            Assert.AreEqual(1, line.Split('\n').Length);
        }

        // ==== 3. genuine startup =============================================

        [Test]
        public void Startup_WithNoFlag_IsEntirelyInert()
        {
            var lines = new List<string>();
            H3ObserverStartup.SetupSinkForTest = lines.Add;
            H3ObservationBundle.CommandLineSourceForTest = () => new[] { "game.exe" };

            H3ObserverStartup.Initialize();

            Assert.AreEqual(0, lines.Count, "a normal build must see nothing at all");
            Assert.IsFalse(H3ObserverStartup.Activated);
            Assert.IsFalse(H3AttemptObserver.Installed);
            Assert.IsNull(H3AttemptObserver.Sink);
        }

        [Test]
        public void Startup_WithInvalidFlag_EmitsOneTypedLineAndInstallsNothing()
        {
            var lines = new List<string>();
            H3ObserverStartup.SetupSinkForTest = lines.Add;
            H3ObservationBundle.CommandLineSourceForTest =
                () => new[] { "game.exe", H3ObservationBundle.FlagName, @"\\server\share" };

            H3ObserverStartup.Initialize();

            Assert.AreEqual(1, lines.Count, "exactly one short setup line");
            StringAssert.Contains("setup=UNAVAILABLE", lines[0]);
            StringAssert.Contains("BundleRootNotFullyQualifiedLocal", lines[0]);
            Assert.IsFalse(H3ObserverStartup.Activated);
            Assert.IsFalse(H3AttemptObserver.Installed);
            Assert.IsNull(H3AttemptObserver.Sink);
            Assert.LessOrEqual(lines[0].Length, H3RecordText.MaxRecordLength);
        }

        [Test]
        public void Startup_WithValidBundle_InstallsAndDeliversThroughTheDefaultSink()
        {
            var setupLines = new List<string>();
            H3ObserverStartup.SetupSinkForTest = setupLines.Add;
            int loaderCalls = 0;
            UseValidBundle(count => loaderCalls = count);
            var diagnostics = new List<string>();
            CommonSignalAttemptDiagnostics.Sink = diagnostics.Add;

            H3ObserverStartup.Initialize();

            Assert.IsTrue(H3ObserverStartup.Activated);
            Assert.AreEqual(1, setupLines.Count);
            StringAssert.Contains("setup=READY", setupLines[0]);
            StringAssert.Contains("model=h3_candidate_v1", setupLines[0]);
            StringAssert.Contains("NO_PROMOTION", setupLines[0]);
            Assert.IsTrue(H3AttemptObserver.Installed, "startup installed the hook itself");
            Assert.IsNotNull(H3AttemptObserver.Sink,
                "startup installed its own default sink; the test does not replace it");
            Assert.AreEqual(1, loaderCalls);

            // delivery through the DEFAULT sink Initialize installed, observed by LogAssert
            LogAssert.Expect(LogType.Log, new Regex(@"^\[H3Observe\] attempt=cap-startup"));
            Assert.IsTrue(CommonSignalAttemptDiagnostics.Emit(ObservableAttempt("cap-startup")));

            Assert.AreEqual(1, H3AttemptObserver.ObservationCount);
            string record = H3AttemptObserver.LastRecord;
            StringAssert.Contains("scored_classes=2", record);
            StringAssert.Contains("relative_share_NOT_a_posterior", record);
            StringAssert.Contains("thresholds=0.6/0.15/0.5", record);
            StringAssert.Contains("NO_PROMOTION", record);
            Assert.AreEqual(1, diagnostics.Count,
                "the existing diagnostic record is unaffected");
            Assert.LessOrEqual(record.Length, H3RecordText.MaxRecordLength);
        }

        [Test]
        public void ScorerException_AfterGenuineStartup_LeavesTheCanonicalRecordIntact()
        {
            H3ObserverStartup.SetupSinkForTest = line => { };
            UseValidBundle(null);
            var diagnostics = new List<string>();
            CommonSignalAttemptDiagnostics.Sink = diagnostics.Add;
            H3ObserverStartup.Initialize();
            Assert.IsTrue(H3ObserverStartup.Activated);

            // isolate scoring only; the hook and the default sink stay as startup left them
            var scorer = new CountingScorer { Throw = new InvalidOperationException("boom") };
            H3AttemptObserver.Scorer = scorer;

            LogAssert.Expect(LogType.Log, new Regex(@"status=TECHNICAL_FAILURE"));
            Assert.IsTrue(CommonSignalAttemptDiagnostics.Emit(ObservableAttempt("cap-throw")),
                "a scorer exception must not change the canonical return value");

            Assert.AreEqual(1, scorer.ScoreCalls);
            Assert.AreEqual(1, diagnostics.Count);
            Assert.AreEqual(1, CommonSignalAttemptDiagnostics.RecordCount);
            StringAssert.Contains("InvalidOperationException", H3AttemptObserver.LastRecord);
            Assert.IsFalse(H3AttemptObserver.LastRecord.Contains("boom"),
                "an exception message must never reach the record");
            Assert.IsFalse(CommonSignalAttemptDiagnostics.Emit(ObservableAttempt("cap-throw")),
                "the existing dedup still holds after an observer failure");
        }

        // ==== 4. exact declared bounds ========================================

        [Test]
        public void Token_NeverExceedsItsDeclaredBoundIncludingTheMarker()
        {
            string truncated = H3RecordText.Token(new string('a', 40), 8);
            Assert.AreEqual(8, truncated.Length);
            Assert.AreEqual(H3RecordText.TruncationMark, truncated[truncated.Length - 1]);

            Assert.AreEqual("a_b_c", H3RecordText.Token("a/b\\c", 16));
            Assert.AreEqual(H3RecordText.Empty, H3RecordText.Token(null, 16));
            Assert.AreEqual(2, H3RecordText.Token(null, 2).Length,
                "even the placeholder respects a tiny bound");
            Assert.AreEqual(1, H3RecordText.Token("abcdef", 1).Length);
            Assert.AreEqual(4, H3RecordText.Token("abcd", 4).Length,
                "an exactly fitting value is not marked as truncated");
            Assert.AreEqual("abcd", H3RecordText.Token("abcd", 4));
        }

        [Test]
        public void MetadataToken_KeepsStructureButRemovesPathShapes()
        {
            Assert.AreEqual("model=h3_candidate_v1 payload=abc",
                H3RecordText.MetadataToken("model=h3_candidate_v1 payload=abc", 64),
                "key=value structure must stay parseable");
            Assert.AreEqual("C__Users_x", H3RecordText.MetadataToken(@"C:\Users\local-user", 64),
                "a drive colon and separators are removed");
            Assert.AreEqual("_unix_path", H3RecordText.MetadataToken("/unix/path", 64));
            string truncated = H3RecordText.MetadataToken(new string('a', 40), 8);
            Assert.AreEqual(8, truncated.Length);
            Assert.AreEqual(H3RecordText.TruncationMark, truncated[truncated.Length - 1]);
        }

        [Test]
        public void BoundLine_NeverExceedsTheMaximumAndKeepsTheRequiredSuffix()
        {
            string suffix = H3RecordText.ClaimLimits;
            string bounded = H3RecordText.BoundLine(new string('a', 5000), suffix);

            Assert.AreEqual(H3RecordText.MaxRecordLength, bounded.Length,
                "a truncated line fills the declared bound exactly, marker included");
            StringAssert.EndsWith(suffix, bounded);
            StringAssert.Contains(H3RecordText.TruncationMark.ToString(), bounded);
            StringAssert.Contains("NO_PROMOTION", bounded);

            string shortLine = "[H3Observe] short";
            Assert.AreEqual(shortLine, H3RecordText.BoundLine(shortLine, suffix),
                "a line within the bound is returned unchanged");
            Assert.IsFalse(H3RecordText.BoundLine("a\nb", suffix).Contains("\n"));
        }

        [Test]
        public void TokenList_ReportsAnExplicitOverflowCount()
        {
            StringAssert.Contains("+2_more",
                H3RecordText.TokenList(new[] { "a", "b", "c", "d" }, 2, 8));
            Assert.AreEqual(H3RecordText.Empty, H3RecordText.TokenList(null, 4, 8));
        }

        // ==== unchanged seam behaviour ========================================

        [Test]
        public void Emit_WithObserverInstalled_LeavesTheDiagnosticRecordIdentical()
        {
            var records = new List<string>();
            CommonSignalAttemptDiagnostics.Sink = records.Add;
            Assert.IsTrue(CommonSignalAttemptDiagnostics.Emit(ObservableAttempt("cap-x")));
            string withoutObserver = records[0].Replace("cap-x", "CAP");

            CommonSignalAttemptDiagnostics.ResetForTest();
            records.Clear();
            CommonSignalAttemptDiagnostics.Sink = records.Add;
            ReadyScorer();
            H3AttemptObserver.Sink = line => { };
            H3AttemptObserver.Install();

            Assert.IsTrue(CommonSignalAttemptDiagnostics.Emit(ObservableAttempt("cap-y")));
            string withObserver = records[0].Replace("cap-y", "CAP");

            Assert.AreEqual(withoutObserver, withObserver);
            Assert.AreEqual(1, H3AttemptObserver.ObservationCount);
        }

        [Test]
        public void Emit_Twice_RecordsOnceAndObservesOnce()
        {
            ReadyScorer();
            var observations = new List<string>();
            H3AttemptObserver.Sink = observations.Add;
            H3AttemptObserver.Install();

            Assert.IsTrue(CommonSignalAttemptDiagnostics.Emit(ObservableAttempt("cap-dup")));
            Assert.IsFalse(CommonSignalAttemptDiagnostics.Emit(ObservableAttempt("cap-dup")));
            Assert.AreEqual(1, CommonSignalAttemptDiagnostics.RecordCount);
            Assert.AreEqual(1, observations.Count);
        }

        [Test]
        public void Emit_WithScorerNotReady_StillRecordsAndObservesNothing()
        {
            CountingScorer scorer = ReadyScorer();
            scorer.Ready = false;
            var observations = new List<string>();
            H3AttemptObserver.Sink = observations.Add;
            H3AttemptObserver.Install();

            Assert.IsTrue(CommonSignalAttemptDiagnostics.Emit(ObservableAttempt("cap-off")));
            Assert.AreEqual(1, CommonSignalAttemptDiagnostics.RecordCount);
            Assert.AreEqual(0, observations.Count);
            Assert.AreEqual(0, scorer.ScoreCalls);
        }

        [Test]
        public void Emit_WithThrowingHook_KeepsReturnRecordAndDedup()
        {
            var records = new List<string>();
            CommonSignalAttemptDiagnostics.Sink = records.Add;
            CommonSignalAttemptDiagnostics.Observer =
                attempt => { throw new InvalidOperationException("hook"); };

            Assert.IsTrue(CommonSignalAttemptDiagnostics.Emit(ObservableAttempt("cap-hook")));
            Assert.AreEqual(1, records.Count);
            Assert.IsFalse(CommonSignalAttemptDiagnostics.Emit(ObservableAttempt("cap-hook")));
        }

        [Test]
        public void Emit_WithThrowingObserverSink_KeepsReturnAndRecord()
        {
            ReadyScorer();
            H3AttemptObserver.Sink = line => { throw new InvalidOperationException("sink"); };
            H3AttemptObserver.Install();

            Assert.IsTrue(CommonSignalAttemptDiagnostics.Emit(ObservableAttempt("cap-sink")));
            Assert.AreEqual(1, CommonSignalAttemptDiagnostics.RecordCount);
            Assert.AreEqual(1, H3AttemptObserver.ObservationCount);
        }

        [Test]
        public void Emit_DoesNotMutateTheAttemptOrItsOutcome()
        {
            ReadyScorer();
            H3AttemptObserver.Sink = line => { };
            H3AttemptObserver.Install();

            PracticeRecognitionAttempt attempt = ObservableAttempt("cap-immutable");
            string capture = attempt.CaptureSessionId;
            string predicted = attempt.PredictedWord;
            bool accepted = attempt.StrictAccepted;
            object outcome = attempt.Outcome;
            object recognition = attempt.Recognition;

            CommonSignalAttemptDiagnostics.Emit(attempt);

            Assert.AreEqual(capture, attempt.CaptureSessionId);
            Assert.AreEqual(predicted, attempt.PredictedWord);
            Assert.AreEqual(accepted, attempt.StrictAccepted);
            Assert.AreSame(outcome, attempt.Outcome);
            Assert.AreSame(recognition, attempt.Recognition);
        }

        [Test]
        public void Extract_NoSnapshot_IsNotObservedAndNeverScored()
        {
            CountingScorer scorer = ReadyScorer();
            var observations = new List<string>();
            H3AttemptObserver.Sink = observations.Add;
            H3AttemptObserver.Install();

            CommonSignalAttemptDiagnostics.Emit(CreateAttempt("cap-nosnap", null));

            StringAssert.Contains("status=NOT_OBSERVED", observations[0]);
            StringAssert.Contains("no_recognition_snapshot", observations[0]);
            Assert.AreEqual(0, scorer.ScoreCalls);
        }

        [Test]
        public void Extract_MissingCommon2D_IsNotObservedAndNeverScored()
        {
            CountingScorer scorer = ReadyScorer();
            var observations = new List<string>();
            H3AttemptObserver.Sink = observations.Add;
            H3AttemptObserver.Install();

            CommonSignalAttemptDiagnostics.Emit(CreateAttempt(
                "cap-nocommon", CreateSnapshot(null, "target is out of contract")));

            StringAssert.Contains("status=NOT_OBSERVED", observations[0]);
            StringAssert.Contains("common2d_unavailable", observations[0]);
            Assert.AreEqual(0, scorer.ScoreCalls);
        }

        [Test]
        public void Extract_UnsupportedLandmarkCount_IsNotObservedAndNeverScored()
        {
            CountingScorer scorer = ReadyScorer();
            var observations = new List<string>();
            H3AttemptObserver.Sink = observations.Add;
            H3AttemptObserver.Install();

            CommonSignalAttemptDiagnostics.Emit(CreateAttempt(
                "cap-badlandmarks", CreateSnapshot(CreateCapture(24, 21), null)));

            StringAssert.Contains("landmark_count_unsupported", observations[0]);
            StringAssert.Contains("landmarks=21", observations[0]);
            Assert.AreEqual(0, scorer.ScoreCalls);
        }

        [Test]
        public void Extract_PassesPrivateCopiesOfTheCaptureToTheScorer()
        {
            CountingScorer scorer = ReadyScorer();
            H3AttemptObserver.Sink = line => { };
            H3AttemptObserver.Install();

            CommonTwoDimensionalCapture capture =
                CreateCapture(24, H3Constants.LandmarkCount);
            CommonSignalAttemptDiagnostics.Emit(
                CreateAttempt("cap-copy", CreateSnapshot(capture, null)));

            Assert.AreEqual(1, scorer.ScoreCalls);
            Assert.AreEqual(24, scorer.LastFrameCount);
            Assert.AreNotSame(capture.CopyCoordinates(), scorer.LastCoordinates);

            scorer.LastCoordinates[0] = 12345.0;
            scorer.LastMask[0] = false;
            Assert.AreEqual(0.0, capture.CopyCoordinates()[0]);
            Assert.IsTrue(capture.CopyValidMask()[0]);
        }

        [Test]
        public void InstallAndUninstall_OnlyToggleTheOptionalHook()
        {
            Assert.IsNull(CommonSignalAttemptDiagnostics.Observer);
            H3AttemptObserver.Install();
            Assert.IsNotNull(CommonSignalAttemptDiagnostics.Observer);
            H3AttemptObserver.Uninstall();
            Assert.IsNull(CommonSignalAttemptDiagnostics.Observer);
        }

        [Test]
        public void DuplicateAndBlankCaptureIds_AreAdmittedCorrectly()
        {
            Assert.IsTrue(H3AttemptObserver.TryBeginObservation("capture-A"));
            Assert.IsFalse(H3AttemptObserver.TryBeginObservation("capture-A"));
            Assert.IsTrue(H3AttemptObserver.TryBeginObservation("capture-B"));
            Assert.IsFalse(H3AttemptObserver.TryBeginObservation(null));
            Assert.IsFalse(H3AttemptObserver.TryBeginObservation(string.Empty));
        }
    }
}
