using System;
using System.IO;
using System.Linq;
using System.Reflection;
using NUnit.Framework;
using WordsInSilence.Practice.PetRobotVerticalSlice;
using WordsInSilence.Recognition.Common;
using WordsInSilence.Recognition.LiveInput;

namespace WordsInSilence.Tests.EditMode
{
    /// <summary>
    /// Step 10P v1.0 focused EditMode tests for the frozen v1.6 common-2D auxiliary
    /// signal. EditMode only: no PlayMode, no webcam, no Scene or Prefab authoring.
    ///
    /// Numeric parity against the 30 canonical v1.6 inputs is proven separately by the
    /// offline C# harness in the Delivery, because the training tensors are deliberately
    /// NOT installed as Unity build assets.
    /// </summary>
    public sealed class FrozenCommonSignIntegrationV1Tests
    {
        static string ModelDirectory => CommonSignClassifierProvider.DefaultModelDirectory;

        static double[] ValidCoordinates(int frames)
        {
            var coordinates = new double[frames * FrozenCommonSignModel.LandmarkCount * 3];
            for (int i = 0; i < coordinates.Length; i++) coordinates[i] = 0.01 * (i % 17);
            return coordinates;
        }

        static bool[] ValidMask(int frames, bool value = true)
        {
            var mask = new bool[frames * FrozenCommonSignModel.LandmarkCount];
            for (int i = 0; i < mask.Length; i++) mask[i] = value;
            return mask;
        }

        // ---- artifact identity --------------------------------------------

        [Test]
        public void InstalledArtifactLoadsAndMatchesPinnedDigest()
        {
            Assert.IsTrue(Directory.Exists(ModelDirectory), "model directory missing: " + ModelDirectory);
            FrozenCommonSignClassifier classifier = CommonSignClassifierProvider.Shared;
            Assert.IsTrue(classifier.ModelAvailable,
                classifier.LoadFailure + ": " + classifier.LoadDetail);
            Assert.AreEqual(FrozenCommonSignModel.ExpectedCombinedSha256, classifier.ArtifactSha256);
            Assert.AreEqual("common_2d_v1_6", classifier.ArtifactVersion);
        }

        [Test]
        public void EveryInstalledArrayMatchesItsPinnedShaAndByteLength()
        {
            var expected = new (string File, string Sha, int Elements)[]
            {
                (FrozenCommonSignModel.ScalerMeanFile, FrozenCommonSignModel.ScalerMeanSha256, 1536),
                (FrozenCommonSignModel.ScalerScaleFile, FrozenCommonSignModel.ScalerScaleSha256, 1536),
                (FrozenCommonSignModel.PcaMeanFile, FrozenCommonSignModel.PcaMeanSha256, 1536),
                (FrozenCommonSignModel.PcaComponentsFile, FrozenCommonSignModel.PcaComponentsSha256, 4 * 1536),
                (FrozenCommonSignModel.LrCoefFile, FrozenCommonSignModel.LrCoefSha256, 4),
                (FrozenCommonSignModel.LrInterceptFile, FrozenCommonSignModel.LrInterceptSha256, 1),
            };
            foreach (var entry in expected)
            {
                string path = Path.Combine(ModelDirectory, entry.File);
                Assert.IsTrue(File.Exists(path), "missing " + entry.File);
                byte[] bytes = File.ReadAllBytes(path);
                Assert.AreEqual(entry.Elements * sizeof(double), bytes.Length, entry.File + " byte length");
                using (var sha = System.Security.Cryptography.SHA256.Create())
                {
                    string hex = string.Concat(sha.ComputeHash(bytes).Select(b => b.ToString("x2")));
                    Assert.AreEqual(entry.Sha, hex, entry.File + " SHA-256");
                }
            }
        }

        [Test]
        public void TamperedArtifactFailsClosedAndCannotClassify()
        {
            string temp = Path.Combine(Path.GetTempPath(), "v10p_tamper_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(temp);
            try
            {
                foreach (string file in Directory.GetFiles(ModelDirectory))
                    File.Copy(file, Path.Combine(temp, Path.GetFileName(file)));
                string victim = Path.Combine(temp, FrozenCommonSignModel.LrCoefFile);
                byte[] bytes = File.ReadAllBytes(victim);
                bytes[0] ^= 0x01;
                File.WriteAllBytes(victim, bytes);

                FrozenCommonSignClassifier tampered = FrozenCommonSignClassifier.Load(temp);
                Assert.IsFalse(tampered.ModelAvailable);
                Assert.AreEqual(CommonSignFailureReason.ModelIntegrityMismatch, tampered.LoadFailure);

                CommonSignClassification result =
                    tampered.Classify(ValidCoordinates(8), ValidMask(8), 8, "GREETING");
                Assert.IsFalse(result.Eligible);
                Assert.IsFalse(result.ExpectedTargetSupport);
            }
            finally
            {
                try { Directory.Delete(temp, true); } catch (IOException) { }
            }
        }

        [Test]
        public void MissingModelDirectoryFailsClosed()
        {
            FrozenCommonSignClassifier absent =
                FrozenCommonSignClassifier.Load(Path.Combine(ModelDirectory, "__absent__"));
            Assert.IsFalse(absent.ModelAvailable);
            CommonSignClassification result =
                absent.Classify(ValidCoordinates(8), ValidMask(8), 8, "GREETING");
            Assert.IsFalse(result.Eligible);
            Assert.IsFalse(result.ExpectedTargetSupport);
        }

        // ---- fail-closed input handling ------------------------------------

        [Test]
        public void AllMissingLandmarksIsTechnicalFailureNotSuccess()
        {
            CommonSignClassification result = CommonSignClassifierProvider.Shared
                .Classify(ValidCoordinates(8), ValidMask(8, false), 8, "GREETING");
            Assert.IsFalse(result.Eligible);
            Assert.AreEqual(CommonSignFailureReason.AllLandmarksMissing, result.FailureReason);
            Assert.IsFalse(result.ExpectedTargetSupport);
        }

        [Test]
        public void InvalidShapeIsTechnicalFailure()
        {
            CommonSignClassification result = CommonSignClassifierProvider.Shared
                .Classify(new double[7], ValidMask(8), 8, "GREETING");
            Assert.IsFalse(result.Eligible);
            Assert.AreEqual(CommonSignFailureReason.InvalidShape, result.FailureReason);
            Assert.IsFalse(result.ExpectedTargetSupport);
        }

        [Test]
        public void NonFiniteInputIsTechnicalFailure()
        {
            double[] coordinates = ValidCoordinates(8);
            coordinates[15] = double.NaN;
            CommonSignClassification nan = CommonSignClassifierProvider.Shared
                .Classify(coordinates, ValidMask(8), 8, "GREETING");
            Assert.IsFalse(nan.Eligible);
            Assert.AreEqual(CommonSignFailureReason.NonFiniteInput, nan.FailureReason);

            coordinates[15] = double.PositiveInfinity;
            CommonSignClassification infinite = CommonSignClassifierProvider.Shared
                .Classify(coordinates, ValidMask(8), 8, "GREETING");
            Assert.IsFalse(infinite.Eligible);
            Assert.AreEqual(CommonSignFailureReason.NonFiniteInput, infinite.FailureReason);
        }

        [Test]
        public void TargetOutsideTheFrozenTwoClassContractIsTechnicalFailure()
        {
            foreach (string target in new[] { "LOVE", "", null, "greeting" })
            {
                CommonSignClassification result = CommonSignClassifierProvider.Shared
                    .Classify(ValidCoordinates(8), ValidMask(8), 8, target);
                Assert.IsFalse(result.Eligible, "target=" + (target ?? "<null>"));
                Assert.AreEqual(CommonSignFailureReason.UnsupportedTarget, result.FailureReason);
                Assert.IsFalse(result.ExpectedTargetSupport);
            }
        }

        // ---- semantics ------------------------------------------------------

        [Test]
        public void OpposingTopOneNeverCreatesTargetSupport()
        {
            CommonSignClassification greetingTop = CommonSignClassification.Success(
                "common_2d_v1_6", "sha", "THANK_YOU", 0.99, 0.01, 4.0, 32);
            Assert.AreEqual("GREETING", greetingTop.TopClass);
            Assert.IsFalse(greetingTop.ExpectedTargetSupport);

            CommonSignClassification thankYouTop = CommonSignClassification.Success(
                "common_2d_v1_6", "sha", "GREETING", 0.02, 0.98, -3.0, 32);
            Assert.AreEqual("THANK_YOU", thankYouTop.TopClass);
            Assert.IsFalse(thankYouTop.ExpectedTargetSupport);
        }

        [Test]
        public void ProbabilityMappingAndTieBreakMatchTheFrozenConvention()
        {
            FrozenCommonSignModel.ToProbabilities(0.0, out double greeting, out double thankYou);
            Assert.AreEqual(0.5, greeting, 1e-15);
            Assert.AreEqual(0.5, thankYou, 1e-15);

            // positive decision favours class_order[1] = THANK_YOU
            FrozenCommonSignModel.ToProbabilities(5.0, out greeting, out thankYou);
            Assert.Greater(thankYou, greeting);

            // exact tie resolves to the first frozen class
            CommonSignClassification tie = CommonSignClassification.Success(
                "common_2d_v1_6", "sha", "GREETING", 0.5, 0.5, 0.0, 32);
            Assert.AreEqual("GREETING", tie.TopClass);
            Assert.AreEqual(0.0, tie.ProbabilityMargin, 1e-15);
        }

        [Test]
        public void CommonResultCarriesNoAcceptanceOrThresholdAuthority()
        {
            string[] forbidden =
            {
                "StrictSuccess", "StrictAccepted", "QuestSuccess", "QuestComplete",
                "ProfileActivated", "CandidateApproved", "Accepted",
                "GuidedProgressAllowed", "Threshold",
            };
            string[] members = typeof(CommonSignClassification)
                .GetMembers(BindingFlags.Public | BindingFlags.Instance)
                .Select(m => m.Name).ToArray();
            foreach (string bad in forbidden)
                Assert.IsFalse(members.Any(m => m.IndexOf(bad, StringComparison.Ordinal) >= 0),
                    "common result must not expose " + bad);
        }

        [Test]
        public void GuidedQuestOutcomeIgnoresTheCommonSignalEntirely()
        {
            // The auxiliary signal is not an input to the outcome seam at all, so the
            // frozen mapping cannot be moved by the common model under any value.
            ParameterInfo[] parameters = typeof(GuidedQuestOutcomeEvaluator)
                .GetMethod("EvaluateForRoute", BindingFlags.Public | BindingFlags.Static)
                .GetParameters();
            Assert.IsFalse(parameters.Any(p =>
                p.ParameterType == typeof(CommonSignClassification) ||
                p.Name.IndexOf("common", StringComparison.OrdinalIgnoreCase) >= 0));

            GuidedQuestOutcome wrongWord = GuidedQuestOutcomeEvaluator.EvaluateForRoute(
                PersonalLiveProfileMode.MultiPrototypeCandidate,
                PersonalLiveCaptureState.Completed, true, "GREETING", "THANK_YOU",
                false, false, false, false, false);
            Assert.AreEqual(GuidedQuestOutcomeKind.WrongWord, wrongWord.Kind);
            Assert.IsFalse(wrongWord.GuidedProgressAllowed);

            GuidedQuestOutcome strict = GuidedQuestOutcomeEvaluator.EvaluateForRoute(
                PersonalLiveProfileMode.MultiPrototypeCandidate,
                PersonalLiveCaptureState.Completed, true, "GREETING", "GREETING",
                true, true, true, false, false);
            Assert.AreEqual(GuidedQuestOutcomeKind.StrictSuccess, strict.Kind);
        }

        // ---- shared seam ----------------------------------------------------

        [Test]
        public void PracticeAndWorldShareOneClassifierInstanceAndOneArtifact()
        {
            FrozenCommonSignClassifier first = CommonSignClassifierProvider.Shared;
            FrozenCommonSignClassifier second = CommonSignClassifierProvider.Shared;
            Assert.AreSame(first, second,
                "both routes must resolve the same classifier instance when nothing is injected");
            Assert.AreEqual(first.ArtifactSha256, second.ArtifactSha256);
        }

        [Test]
        public void AdapterExposesTheCommonClassifierSeamForBothRoutes()
        {
            PropertyInfo seam = typeof(PersonalLivePracticeRecognitionAdapter)
                .GetProperty("CommonClassifier", BindingFlags.Public | BindingFlags.Instance);
            Assert.IsNotNull(seam, "the shared adapter must expose its common classifier");
            Assert.AreEqual(typeof(ICommonSignClassifier), seam.PropertyType);

            ParameterInfo[] parameters = typeof(PersonalLivePracticeRecognitionAdapter)
                .GetConstructors().Single().GetParameters();
            ParameterInfo injected = parameters.Single(p => p.Name == "commonClassifier");
            Assert.AreEqual(typeof(ICommonSignClassifier), injected.ParameterType);
            Assert.IsTrue(injected.IsOptional,
                "injection must stay optional so existing call sites keep compiling");
        }

        [Test]
        public void AttemptCommonSignalIsOptionalAndDefaultsToNull()
        {
            var attempt = new PracticeRecognitionAttempt(
                "session", "GREETING", "GREETING", PersonalLiveCaptureState.Completed,
                null, null, null, "route", "profile", double.NaN, null, null);
            Assert.IsNull(attempt.CommonSignal,
                "the auxiliary signal must be absent unless a classifier supplied one");

            CommonSignClassification signal = CommonSignClassification.Success(
                "common_2d_v1_6", "sha", "GREETING", 0.9, 0.1, 2.0, 32);
            var carried = new PracticeRecognitionAttempt(
                "session", "GREETING", "GREETING", PersonalLiveCaptureState.Completed,
                null, null, null, "route", "profile", double.NaN, null, null, null, signal);
            Assert.AreSame(signal, carried.CommonSignal);
            Assert.IsTrue(carried.CommonSignal.ExpectedTargetSupport);
            // carrying the signal changes none of the strict values
            Assert.IsFalse(carried.StrictAccepted);
            Assert.IsFalse(carried.StrictQuestSuccess);
        }
    }
}
