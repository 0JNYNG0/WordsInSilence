using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Security.Cryptography;
using NUnit.Framework;
using UnityEngine;
using WordsInSilence.Recognition.Core;
using WordsInSilence.Recognition.LiveInput;
using WordsInSilence.Recognition.Runtime;

namespace WordsInSilence.Tests.EditMode
{
    public sealed class PersonalLiveRecognitionIntegrationV1Tests
    {
        const string Step9IProfileHash = "468613729c0299d11a4c72343c17df0c4fc7ed5b36367c8ce86ea9d81b6b62cf";
        const string Step9ITestHash = "0db5cd7afbacadf54fb2cdb53af0211dfb3ef48758d41a0d9de9df05ca7cc959";
        const string Step9IScoresHash = "c3f98a8c49a52acd018e3366d9fbe88453e529256c51a77d92e1c3e8069a97aa";

        static string ProjectRoot => Directory.GetParent(Application.dataPath).FullName;
        static string ProfileDirectory => Path.Combine(Application.streamingAssetsPath,
            "RecognitionCoreV1", "PersonalProfiles", "step9h_current_user_v1_0");
        static string GlobalTemplateDirectory => Path.Combine(Application.streamingAssetsPath,
            "RecognitionCoreV1", "Templates");
        static string GlobalThresholdPath => Path.Combine(Application.streamingAssetsPath,
            "RecognitionCoreV1", "RuntimeGate", "frozen_threshold_config.json");
        static string FixtureDirectory => Path.Combine(Application.dataPath,
            "_Project", "Tests", "EditMode", "Fixtures", "PersonalRuntimeOptInIntegrationV1");

        [Serializable]
        sealed class GlobalAssetSnapshot
        {
            public GlobalAssetRecord[] assets;
        }

        [Serializable]
        sealed class GlobalAssetRecord
        {
            public string relative_path;
            public long size_bytes;
            public string sha256;
        }

        sealed class FakeObservationSource : IPersonalLiveObservationSource
        {
            Action<RawMediaPipeObservation> observations;
            Action<Exception> failures;

            public bool IsBound { get; set; } = true;
            public int SubscriberCount =>
                (observations == null ? 0 : observations.GetInvocationList().Length) +
                (failures == null ? 0 : failures.GetInvocationList().Length);

            public event Action<RawMediaPipeObservation> ObservationArrived
            {
                add { observations += value; }
                remove { observations -= value; }
            }

            public event Action<Exception> SourceFailed
            {
                add { failures += value; }
                remove { failures -= value; }
            }

            public void Emit(RawMediaPipeObservation observation) => observations?.Invoke(observation);
            public void Fail(Exception exception) => failures?.Invoke(exception);
        }

        sealed class FakeGlobalRuntime : IPersonalLiveGlobalRuntime
        {
        }

        sealed class FakeRuntime : IPersonalLiveRecognitionRuntime
        {
            readonly PersonalLiveRoutingResult result;
            public bool IsPersonalActive { get; set; } = true;
            public string ActiveProfileId { get; set; } = PersonalRuntimeProfileLoader.ProfileId;
            public int RecognitionCount { get; private set; }
            public string LastExpectedWord { get; private set; }

            public FakeRuntime(PersonalLiveRoutingResult result)
            {
                this.result = result;
            }

            public PersonalLiveRoutingResult Recognize(
                string sessionId,
                RecognitionSequence sequence,
                string expectedWord)
            {
                RecognitionCount++;
                LastExpectedWord = expectedWord;
                return result;
            }
        }

        sealed class FakeRuntimeFactory : IPersonalLiveRuntimeFactory
        {
            public bool ThrowDuringGlobalLoad { get; set; }
            public bool ThrowDuringActivation { get; set; }
            public int GlobalLoadCount { get; private set; }
            public int ActivationCount { get; private set; }
            public FakeRuntime Runtime { get; }

            public FakeRuntimeFactory(FakeRuntime runtime)
            {
                Runtime = runtime;
            }

            public IPersonalLiveGlobalRuntime LoadGlobal(PersonalLiveRuntimePaths paths)
            {
                GlobalLoadCount++;
                if (ThrowDuringGlobalLoad) throw new InvalidDataException("injected global failure");
                return new FakeGlobalRuntime();
            }

            public IPersonalLiveRecognitionRuntime CreateAndActivate(
                IPersonalLiveGlobalRuntime globalRuntime,
                PersonalLiveRuntimePaths paths)
            {
                ActivationCount++;
                if (ThrowDuringActivation) throw new InvalidDataException("injected personal activation failure");
                return Runtime;
            }
        }

        [Test]
        public void InitialState_RequiresBoundBridgeThenLoadsGlobalAndExplicitlyActivatesPersonalProfile()
        {
            var missing = new PersonalLiveRecognitionController(null, Factory(SuccessResult("GREETING")));
            Assert.IsFalse(missing.Initialize(Paths()));
            Assert.AreEqual(PersonalLiveFailureCode.BridgeMissing, missing.Failure.Code);

            var unboundSource = new FakeObservationSource { IsBound = false };
            var unbound = new PersonalLiveRecognitionController(unboundSource, Factory(SuccessResult("GREETING")));
            Assert.IsFalse(unbound.Initialize(Paths()));
            Assert.AreEqual(PersonalLiveFailureCode.BridgeNotBound, unbound.Failure.Code);

            var source = new FakeObservationSource();
            var runtime = new FakeRuntime(SuccessResult("GREETING"));
            var factory = new FakeRuntimeFactory(runtime);
            var controller = new PersonalLiveRecognitionController(source, factory);
            Assert.AreEqual(PersonalLiveCaptureState.Idle, controller.State);
            Assert.IsTrue(controller.Initialize(Paths()));
            Assert.AreEqual(PersonalLiveCaptureState.Ready, controller.State);
            Assert.AreEqual(1, factory.GlobalLoadCount);
            Assert.AreEqual(1, factory.ActivationCount);
            Assert.IsTrue(controller.IsPersonalModeActive);
            Assert.AreEqual(PersonalRuntimeProfileLoader.ProfileId, controller.ActivePersonalProfileId);
        }

        [Test]
        public void RuntimeAndPersonalActivationFailuresRemainDistinctTypedFailures()
        {
            var source = new FakeObservationSource();
            var globalFailureFactory = Factory(SuccessResult("GREETING"));
            globalFailureFactory.ThrowDuringGlobalLoad = true;
            var globalFailure = new PersonalLiveRecognitionController(source, globalFailureFactory);
            Assert.IsFalse(globalFailure.Initialize(Paths()));
            Assert.AreEqual(PersonalLiveFailureCode.RuntimeInitializationFailure, globalFailure.Failure.Code);

            var activationFailureFactory = Factory(SuccessResult("GREETING"));
            activationFailureFactory.ThrowDuringActivation = true;
            var activationFailure = new PersonalLiveRecognitionController(new FakeObservationSource(), activationFailureFactory);
            Assert.IsFalse(activationFailure.Initialize(Paths()));
            Assert.AreEqual(PersonalLiveFailureCode.PersonalProfileActivationFailure, activationFailure.Failure.Code);
        }

        [Test]
        public void SupportedWordsAreExactAndInvalidExpectedWordIsRejected()
        {
            CollectionAssert.AreEqual(
                new[] { "GREETING", "LOVE", "THANK_YOU" },
                PersonalLiveRecognitionController.SupportedWords);
            foreach (string word in PersonalLiveRecognitionController.SupportedWords)
            {
                var controller = ReadyController(new FakeObservationSource(), SuccessResult(word));
                Assert.IsTrue(controller.SetExpectedWord(word));
                Assert.AreEqual(word, controller.ExpectedWord);
            }

            var invalid = ReadyController(new FakeObservationSource(), SuccessResult("GREETING"));
            Assert.IsFalse(invalid.SetExpectedWord("UNKNOWN"));
            Assert.AreEqual(PersonalLiveFailureCode.InvalidExpectedWord, invalid.Failure.Code);
            Assert.IsNull(invalid.Result);
        }

        [Test]
        public void ObservationBeforeManualStartIsIgnored()
        {
            var source = new FakeObservationSource();
            var controller = ReadyController(source, SuccessResult("GREETING"));
            source.Emit(MakeValidObservation(0.0));
            Assert.AreEqual(PersonalLiveCaptureState.Ready, controller.State);
            Assert.AreEqual(0, controller.BufferedRawFrameCount);
        }

        [Test]
        public void BeginAddFinalizeResetStateRouteBuildsImmutableCompletedResult()
        {
            var source = new FakeObservationSource();
            var route = SuccessResult("GREETING");
            var runtime = new FakeRuntime(route);
            var controller = ReadyController(source, runtime);
            Assert.IsTrue(controller.BeginCapture("step9j-manual-take"));
            Assert.AreEqual(PersonalLiveCaptureState.Capturing, controller.State);
            source.Emit(MakeValidObservation(0.0));
            source.Emit(MakeValidObservation(0.03));
            Assert.AreEqual(2, controller.BufferedRawFrameCount);
            Assert.IsTrue(controller.FinalizeCapture());
            Assert.AreEqual(PersonalLiveCaptureState.Completed, controller.State);
            Assert.AreEqual(1, runtime.RecognitionCount);
            Assert.AreEqual("GREETING", runtime.LastExpectedWord);
            Assert.AreEqual(0, source.SubscriberCount);
            Assert.AreEqual(2, controller.Result.RawFrameCount);
            Assert.AreEqual(2, controller.Result.NormalizedFrameCount);
            Assert.AreEqual("step9j-manual-take", controller.Result.CaptureSessionId);
            Assert.IsTrue(controller.Reset());
            Assert.AreEqual(PersonalLiveCaptureState.Ready, controller.State);
            Assert.IsNull(controller.Result);
        }

        [Test]
        public void EmptyAndAllInvalidCapturesHaveSeparateTypedFailures()
        {
            var empty = ReadyController(new FakeObservationSource(), SuccessResult("GREETING"));
            Assert.IsTrue(empty.BeginCapture("empty"));
            Assert.IsFalse(empty.FinalizeCapture());
            Assert.AreEqual(PersonalLiveFailureCode.EmptyCapture, empty.Failure.Code);

            var source = new FakeObservationSource();
            var invalid = ReadyController(source, SuccessResult("GREETING"));
            Assert.IsTrue(invalid.BeginCapture("all-invalid"));
            source.Emit(MakeAbsentObservation(0.0));
            Assert.IsFalse(invalid.FinalizeCapture());
            Assert.AreEqual(PersonalLiveFailureCode.NoValidNormalizedLandmarks, invalid.Failure.Code);
            Assert.IsNull(invalid.Result);
        }

        [Test]
        public void TimestampAndConventionMismatchAreRejectedBeforePreprocessing()
        {
            var timestampSource = new FakeObservationSource();
            var timestamp = ReadyController(timestampSource, SuccessResult("GREETING"));
            timestamp.BeginCapture("timestamp");
            timestampSource.Emit(MakeValidObservation(1.0));
            timestampSource.Emit(MakeValidObservation(1.0));
            Assert.AreEqual(PersonalLiveCaptureState.Failed, timestamp.State);
            Assert.AreEqual(PersonalLiveFailureCode.ObservationRejected, timestamp.Failure.Code);

            var conventionSource = new FakeObservationSource();
            var convention = ReadyController(conventionSource, SuccessResult("GREETING"));
            convention.BeginCapture("convention");
            conventionSource.Emit(MakeValidObservation(0.0, new RawObservationConvention(
                "not-approved-live", RawObservationCoordinateSpace.MediaPipeNormalizedImage,
                "axes", "hands", "mirror")));
            Assert.AreEqual(PersonalLiveCaptureState.Failed, convention.State);
            Assert.AreEqual(PersonalLiveFailureCode.ObservationRejected, convention.Failure.Code);
            Assert.AreEqual(0, conventionSource.SubscriberCount);
        }

        [Test]
        public void BridgeFailurePinsFailureAndRemovesSubscriptions()
        {
            var source = new FakeObservationSource();
            var controller = ReadyController(source, SuccessResult("GREETING"));
            controller.BeginCapture("bridge-failure");
            source.Emit(MakeValidObservation(0.0));
            source.Fail(new InvalidOperationException("provider lifetime ended"));
            Assert.AreEqual(PersonalLiveCaptureState.Failed, controller.State);
            Assert.AreEqual(PersonalLiveFailureCode.BridgeFailure, controller.Failure.Code);
            Assert.AreEqual(0, source.SubscriberCount);
            Assert.AreEqual(0, controller.BufferedRawFrameCount);
            Assert.IsNull(controller.Result);
        }

        [Test]
        public void CompletedSnapshotIsImmutableAndCannotBeChangedByLaterLiveObservations()
        {
            var source = new FakeObservationSource();
            var controller = ReadyController(source, SuccessResult("GREETING"));
            controller.BeginCapture("immutable");
            source.Emit(MakeValidObservation(0.0));
            Assert.IsTrue(controller.FinalizeCapture());
            PersonalLiveRecognitionSnapshot snapshot = controller.Result;
            source.Emit(MakeValidObservation(0.03));
            Assert.AreSame(snapshot, controller.Result);
            Assert.AreEqual(1, snapshot.RawFrameCount);
            foreach (PropertyInfo property in typeof(PersonalLiveRecognitionSnapshot)
                .GetProperties(BindingFlags.Instance | BindingFlags.Public))
                Assert.IsFalse(property.CanWrite, property.Name);
            Assert.AreEqual(0, typeof(PersonalLiveRecognitionSnapshot)
                .GetFields(BindingFlags.Instance | BindingFlags.Public).Length);
        }

        [Test]
        public void ResultMapsPredictedAcceptanceExpectedClassAndStep9IQuestGateWithoutWeakening()
        {
            var source = new FakeObservationSource();
            var route = SuccessResult("GREETING");
            var controller = ReadyController(source, route);
            controller.BeginCapture("mapping");
            source.Emit(MakeValidObservation(0.0));
            Assert.IsTrue(controller.FinalizeCapture());
            PersonalLiveRecognitionSnapshot result = controller.Result;
            Assert.AreEqual("GREETING", result.PredictedWord);
            Assert.IsTrue(result.Accepted);
            Assert.IsTrue(result.ExpectedWordClassAccept);
            Assert.IsTrue(result.QuestSuccess);
            Assert.AreEqual(0.12, result.GreetingDistance, 0.0);
            Assert.AreEqual(0.34, result.LoveDistance, 0.0);
            Assert.AreEqual(0.56, result.ThankYouDistance, 0.0);
            Assert.AreEqual("prototype-GREETING", result.WinningPrototypeId);
            Assert.IsTrue(result.GreetingMarginNarrow);
        }

        [Test]
        public void ExistingStep9IWisrqQueriesRouteThroughApprovedPersonalSeamForAllThreeWords()
        {
            var factory = new ApprovedPersonalLiveRuntimeFactory();
            PersonalLiveRuntimePaths paths = Paths();
            IPersonalLiveGlobalRuntime global = factory.LoadGlobal(paths);
            IPersonalLiveRecognitionRuntime runtime = factory.CreateAndActivate(global, paths);
            Assert.IsTrue(runtime.IsPersonalActive);

            string queryDirectory = Path.Combine(FixtureDirectory, "Queries");
            string[] queryPaths = Directory.GetFiles(queryDirectory, "*.wisrq", SearchOption.TopDirectoryOnly);
            Assert.AreEqual(9, queryPaths.Length);
            var seen = new HashSet<string>(StringComparer.Ordinal);
            foreach (string path in queryPaths)
            {
                string file = Path.GetFileName(path);
                string expected = file.StartsWith("query_THANK_YOU_", StringComparison.Ordinal)
                    ? "THANK_YOU"
                    : file.StartsWith("query_GREETING_", StringComparison.Ordinal) ? "GREETING" : "LOVE";
                RecognitionSequence query = RecognitionSequenceBinary.LoadFile(path);
                PersonalLiveRoutingResult result = runtime.Recognize(query.SampleId, query, expected);
                Assert.AreEqual(expected, result.PredictedWord, query.SampleId);
                Assert.IsTrue(result.Accepted, query.SampleId);
                Assert.IsTrue(result.ExpectedWordClassAccept, query.SampleId);
                Assert.IsTrue(result.QuestSuccess, query.SampleId);
                seen.Add(expected);
            }
            CollectionAssert.AreEquivalent(
                new[] { "GREETING", "LOVE", "THANK_YOU" }, seen);
        }

        [Test]
        public void DifferentClosestWordNeverBecomesQuestSuccess()
        {
            var source = new FakeObservationSource();
            var route = Result(
                "LOVE", "GREETING", true, true, false,
                new Dictionary<string, double>(StringComparer.Ordinal)
                {
                    ["GREETING"] = 0.31,
                    ["LOVE"] = 0.20,
                    ["THANK_YOU"] = 0.50,
                });
            var controller = ReadyController(source, route);
            controller.SetExpectedWord("GREETING");
            controller.BeginCapture("wrong-word");
            source.Emit(MakeValidObservation(0.0));
            Assert.IsTrue(controller.FinalizeCapture());
            Assert.AreEqual("LOVE", controller.Result.PredictedWord);
            Assert.IsTrue(controller.Result.Accepted);
            Assert.IsTrue(controller.Result.ExpectedWordClassAccept);
            Assert.IsFalse(controller.Result.QuestSuccess);
        }

        [Test]
        public void ApprovedGlobalAndPersonalAssetsRemainByteIdentical()
        {
            AssertAsset(Path.Combine(ProfileDirectory, "personal_template_GREETING.wisrt"), 53564,
                "d8d5713334b22d9427b9b589bccf820cdbb4849357bae3dce1c368eaf0049681");
            AssertAsset(Path.Combine(ProfileDirectory, "personal_template_LOVE.wisrt"), 56262,
                "013bf5515eb8af33f300400893da64c853668b2fa0442e19203da62b2777ad4b");
            AssertAsset(Path.Combine(ProfileDirectory, "personal_template_THANK_YOU.wisrt"), 49319,
                "9bb9a7681d55a292a21d8b7f458b4b150c10555bc0ec163da692a780ed5280e5");
            AssertAsset(Path.Combine(ProfileDirectory, "personal_class_threshold_config.json"), 4811,
                "e84edd1f598c2d94f8412767f3125031e0be330886d3a315ed429798fe49726e");

            GlobalAssetSnapshot snapshot = JsonUtility.FromJson<GlobalAssetSnapshot>(File.ReadAllText(
                Path.Combine(FixtureDirectory, "global_runtime_asset_snapshot.json")));
            Assert.AreEqual(11, snapshot.assets.Length);
            foreach (GlobalAssetRecord asset in snapshot.assets)
                AssertAsset(ProjectPath(asset.relative_path), asset.size_bytes, asset.sha256);
        }

        [Test]
        public void Step9IApprovedFixtureAndTestSourceRemainUnchanged()
        {
            Assert.AreEqual(Step9IProfileHash, Sha256File(ProjectPath(
                "Assets/_Project/Scripts/Recognition/Runtime/PersonalRuntimeOptInProfile.cs")));
            Assert.AreEqual(Step9ITestHash, Sha256File(ProjectPath(
                "Assets/_Project/Tests/EditMode/PersonalRuntimeOptInIntegrationV1Tests.cs")));
            Assert.AreEqual(Step9IScoresHash, Sha256File(Path.Combine(
                FixtureDirectory, "personal_expected_scores.csv")));
            Assert.AreEqual(9, Directory.GetFiles(
                Path.Combine(FixtureDirectory, "Queries"), "*.wisrq", SearchOption.TopDirectoryOnly).Length);
        }

        [Test]
        public void CancelDisableAndDisposeLeaveZeroObservationSubscribers()
        {
            var cancelSource = new FakeObservationSource();
            var cancel = ReadyController(cancelSource, SuccessResult("GREETING"));
            cancel.BeginCapture("cancel");
            Assert.IsTrue(cancel.CancelCapture());
            Assert.AreEqual(PersonalLiveCaptureState.Cancelled, cancel.State);
            Assert.AreEqual(0, cancelSource.SubscriberCount);

            var disableSource = new FakeObservationSource();
            var disable = ReadyController(disableSource, SuccessResult("GREETING"));
            Assert.AreEqual(2, disableSource.SubscriberCount);
            disable.Disable();
            Assert.AreEqual(0, disableSource.SubscriberCount);

            var disposeSource = new FakeObservationSource();
            var dispose = ReadyController(disposeSource, SuccessResult("GREETING"));
            dispose.Dispose();
            Assert.AreEqual(0, disposeSource.SubscriberCount);
        }

        [Test]
        public void HarnessOnDisable_RemovesAllObservationSourceSubscribers()
        {
            GameObject owner = null;
            try
            {
                owner = new GameObject("Step9J-Harness-OnDisable-Test");
                owner.SetActive(false);
                PersonalLiveRecognitionHarness harness = owner.AddComponent<PersonalLiveRecognitionHarness>();
                var source = new FakeObservationSource();
                PersonalLiveRecognitionController controller = ReadyController(source, SuccessResult("GREETING"));
                SetPrivateHarnessField(harness, "controller", controller);

                Assert.AreEqual(2, source.SubscriberCount);
                InvokePrivateHarnessLifecycle(harness, "OnDisable");
                Assert.AreEqual(0, source.SubscriberCount);
                Assert.AreEqual(PersonalLiveCaptureState.Cancelled, controller.State);
            }
            finally
            {
                if (owner != null) UnityEngine.Object.DestroyImmediate(owner);
            }
        }

        [Test]
        public void HarnessOnDestroy_RemovesAllObservationSourceSubscribers()
        {
            GameObject owner = null;
            try
            {
                owner = new GameObject("Step9J-Harness-OnDestroy-Test");
                owner.SetActive(false);
                PersonalLiveRecognitionHarness harness = owner.AddComponent<PersonalLiveRecognitionHarness>();
                var source = new FakeObservationSource();
                PersonalLiveRecognitionController controller = ReadyController(source, SuccessResult("GREETING"));
                SetPrivateHarnessField(harness, "controller", controller);

                Assert.AreEqual(2, source.SubscriberCount);
                InvokePrivateHarnessLifecycle(harness, "OnDestroy");
                Assert.AreEqual(0, source.SubscriberCount);
                Assert.IsNull(GetPrivateHarnessField(harness, "controller"));
            }
            finally
            {
                if (owner != null) UnityEngine.Object.DestroyImmediate(owner);
            }
        }

        [Test]
        public void CaptureBridgeFailurePinsImmutablePrimaryFailureSnapshot()
        {
            var source = new FakeObservationSource();
            var controller = ReadyController(source, SuccessResult("GREETING"));
            Assert.IsTrue(controller.BeginCapture("primary-bridge-failure"));
            source.Emit(MakeValidObservation(0.0));
            source.Emit(MakeValidObservation(0.03));
            var bridgeException = new InvalidOperationException("provider lifetime ended");
            source.Fail(bridgeException);

            PersonalLivePrimaryFailureSnapshot primary = controller.PrimaryFailure;
            Assert.IsNotNull(primary);
            Assert.AreEqual(PersonalLiveFailureCode.BridgeFailure, primary.Code);
            Assert.AreEqual("Step 9C observation bridge failed: provider lifetime ended", primary.Message);
            Assert.AreEqual(typeof(InvalidOperationException).FullName, primary.ExceptionType);
            Assert.AreEqual(PersonalLiveCaptureState.Capturing, primary.StateBeforeFailure);
            Assert.AreEqual("primary-bridge-failure", primary.CaptureSessionId);
            Assert.AreEqual(2, primary.BufferedRawFrameCount);
            foreach (PropertyInfo property in typeof(PersonalLivePrimaryFailureSnapshot)
                .GetProperties(BindingFlags.Instance | BindingFlags.Public))
                Assert.IsFalse(property.CanWrite, property.Name);
            Assert.AreEqual(0, typeof(PersonalLivePrimaryFailureSnapshot)
                .GetFields(BindingFlags.Instance | BindingFlags.Public).Length);
        }

        [Test]
        public void PrimaryFailurePreservesBufferedRawFrameCountAfterFailureCleanup()
        {
            var source = new FakeObservationSource();
            var controller = ReadyController(source, SuccessResult("GREETING"));
            Assert.IsTrue(controller.BeginCapture("primary-buffer-count"));
            source.Emit(MakeValidObservation(0.0));
            source.Emit(MakeValidObservation(0.03));
            source.Emit(MakeValidObservation(0.06));
            Assert.AreEqual(3, controller.BufferedRawFrameCount);

            source.Fail(new InvalidDataException("capture bridge stopped"));

            Assert.AreEqual(0, controller.BufferedRawFrameCount);
            Assert.AreEqual(3, controller.PrimaryFailure.BufferedRawFrameCount);
            Assert.AreEqual("primary-buffer-count", controller.PrimaryFailure.CaptureSessionId);
        }

        [Test]
        public void FailedStateFollowupCommandsCannotOverwritePrimaryFailureSnapshot()
        {
            var source = new FakeObservationSource();
            var controller = ReadyController(source, SuccessResult("GREETING"));
            Assert.IsTrue(controller.BeginCapture("primary-followup"));
            source.Emit(MakeValidObservation(0.0));
            source.Fail(new InvalidOperationException("first asynchronous failure"));
            PersonalLivePrimaryFailureSnapshot primary = controller.PrimaryFailure;

            Assert.IsFalse(controller.FinalizeCapture());
            Assert.AreEqual(PersonalLiveFailureCode.InvalidStateTransition, controller.Failure.Code);
            Assert.AreSame(primary, controller.PrimaryFailure);
            Assert.IsFalse(controller.BeginCapture("later-begin"));
            Assert.AreSame(primary, controller.PrimaryFailure);
            Assert.IsFalse(controller.CancelCapture());
            Assert.AreSame(primary, controller.PrimaryFailure);
            source.IsBound = false;
            Assert.IsFalse(controller.Reset());
            Assert.AreEqual(PersonalLiveFailureCode.BridgeNotBound, controller.Failure.Code);
            Assert.AreSame(primary, controller.PrimaryFailure);
            Assert.AreEqual(PersonalLiveFailureCode.BridgeFailure, primary.Code);
            Assert.AreEqual("primary-followup", primary.CaptureSessionId);
            Assert.AreEqual(1, primary.BufferedRawFrameCount);
        }

        [Test]
        public void ObservationRejectionPinsExactPrimaryFailureCodeAndPrecleanupContext()
        {
            var source = new FakeObservationSource();
            var controller = ReadyController(source, SuccessResult("GREETING"));
            Assert.IsTrue(controller.BeginCapture("primary-observation-rejection"));
            source.Emit(MakeValidObservation(1.0));
            source.Emit(MakeValidObservation(1.0));

            PersonalLivePrimaryFailureSnapshot primary = controller.PrimaryFailure;
            Assert.AreEqual(PersonalLiveFailureCode.ObservationRejected, primary.Code);
            Assert.AreEqual("observation timestamps must be strictly increasing", primary.Message);
            Assert.AreEqual(string.Empty, primary.ExceptionType);
            Assert.AreEqual(PersonalLiveCaptureState.Capturing, primary.StateBeforeFailure);
            Assert.AreEqual("primary-observation-rejection", primary.CaptureSessionId);
            Assert.AreEqual(1, primary.BufferedRawFrameCount);
            Assert.AreEqual(0, controller.BufferedRawFrameCount);
        }

        [Test]
        public void SuccessfulResetClearsPrimaryFailureSnapshot()
        {
            var source = new FakeObservationSource();
            var controller = ReadyController(source, SuccessResult("GREETING"));
            Assert.IsTrue(controller.BeginCapture("primary-reset"));
            source.Emit(MakeValidObservation(0.0));
            source.Fail(new InvalidOperationException("resettable failure"));
            Assert.IsNotNull(controller.PrimaryFailure);

            Assert.IsTrue(controller.Reset());

            Assert.AreEqual(PersonalLiveCaptureState.Ready, controller.State);
            Assert.IsNull(controller.PrimaryFailure);
            Assert.IsNull(controller.Failure);
            Assert.AreEqual(string.Empty, controller.CaptureSessionId);
            Assert.AreEqual(0, controller.BufferedRawFrameCount);
            Assert.AreEqual(2, source.SubscriberCount);
        }

        static void SetPrivateHarnessField(
            PersonalLiveRecognitionHarness harness,
            string fieldName,
            object value)
        {
            FieldInfo field = typeof(PersonalLiveRecognitionHarness).GetField(
                fieldName,
                BindingFlags.Instance | BindingFlags.NonPublic);
            Assert.IsNotNull(field, fieldName);
            field.SetValue(harness, value);
        }

        static object GetPrivateHarnessField(
            PersonalLiveRecognitionHarness harness,
            string fieldName)
        {
            FieldInfo field = typeof(PersonalLiveRecognitionHarness).GetField(
                fieldName,
                BindingFlags.Instance | BindingFlags.NonPublic);
            Assert.IsNotNull(field, fieldName);
            return field.GetValue(harness);
        }

        static void InvokePrivateHarnessLifecycle(
            PersonalLiveRecognitionHarness harness,
            string methodName)
        {
            MethodInfo method = typeof(PersonalLiveRecognitionHarness).GetMethod(
                methodName,
                BindingFlags.Instance | BindingFlags.NonPublic);
            Assert.IsNotNull(method, methodName);
            method.Invoke(harness, null);
        }

        static PersonalLiveRecognitionController ReadyController(
            FakeObservationSource source,
            PersonalLiveRoutingResult result)
        {
            return ReadyController(source, new FakeRuntime(result));
        }

        static PersonalLiveRecognitionController ReadyController(
            FakeObservationSource source,
            FakeRuntime runtime)
        {
            var controller = new PersonalLiveRecognitionController(source, new FakeRuntimeFactory(runtime));
            Assert.IsTrue(controller.Initialize(Paths()));
            return controller;
        }

        static FakeRuntimeFactory Factory(PersonalLiveRoutingResult result) =>
            new FakeRuntimeFactory(new FakeRuntime(result));

        static PersonalLiveRuntimePaths Paths() => new PersonalLiveRuntimePaths(
            GlobalTemplateDirectory,
            GlobalThresholdPath,
            ProfileDirectory);

        static PersonalLiveRoutingResult SuccessResult(string predicted) => Result(
            predicted,
            predicted,
            true,
            true,
            true,
            new Dictionary<string, double>(StringComparer.Ordinal)
            {
                ["GREETING"] = predicted == "GREETING" ? 0.12 : 0.52,
                ["LOVE"] = predicted == "LOVE" ? 0.12 : 0.34,
                ["THANK_YOU"] = predicted == "THANK_YOU" ? 0.12 : 0.56,
            });

        static PersonalLiveRoutingResult Result(
            string predicted,
            string expectedWord,
            bool accepted,
            bool expectedAccept,
            bool questSuccess,
            Dictionary<string, double> scores)
        {
            var accepts = new Dictionary<string, bool>(StringComparer.Ordinal)
            {
                ["GREETING"] = false,
                ["LOVE"] = false,
                ["THANK_YOU"] = false,
            };
            accepts[predicted] = accepted;
            accepts[expectedWord] = expectedAccept;
            return new PersonalLiveRoutingResult(
                predicted,
                accepted,
                expectedAccept,
                questSuccess,
                scores,
                accepts,
                scores[predicted],
                "prototype-" + predicted,
                new[] { PersonalRuntimeProfileLoader.GreetingMarginWarning });
        }

        static RawMediaPipeObservation MakeValidObservation(
            double timestamp,
            RawObservationConvention convention = null)
        {
            var poseCoordinates = NaNCoordinates(33);
            var poseValid = new bool[33];
            WriteValid(poseCoordinates, poseValid, 11, -1.0, 0.0, 0.0);
            WriteValid(poseCoordinates, poseValid, 12, 1.0, 0.0, 0.0);
            WriteValid(poseCoordinates, poseValid, 23, -1.0, 2.0, 0.0);
            WriteValid(poseCoordinates, poseValid, 24, 1.0, 2.0, 0.0);
            WriteValid(poseCoordinates, poseValid, 13, -1.2, -0.5, 0.1);
            WriteValid(poseCoordinates, poseValid, 14, 1.2, -0.5, 0.1);
            WriteValid(poseCoordinates, poseValid, 15, -1.4, -1.0, 0.2);
            WriteValid(poseCoordinates, poseValid, 16, 1.4, -1.0, 0.2);
            var handCoordinates = NaNCoordinates(21);
            var handValid = new bool[21];
            foreach (int landmark in new[] { 0, 4, 8, 12, 16, 20 })
                WriteValid(handCoordinates, handValid, landmark,
                    -1.0 + landmark * 0.02, -0.8, landmark * 0.005);
            return new RawMediaPipeObservation(
                timestamp,
                convention ?? RawObservationConvention.LiveMediaPipe,
                new RawLandmarkObservationSet(33, poseCoordinates, poseValid, new bool[33], new bool[33]),
                new RawLandmarkObservationSet(21, handCoordinates, handValid, new bool[21], new bool[21]),
                RawLandmarkObservationSet.Absent(21));
        }

        static RawMediaPipeObservation MakeAbsentObservation(double timestamp) =>
            new RawMediaPipeObservation(
                timestamp,
                RawObservationConvention.LiveMediaPipe,
                RawLandmarkObservationSet.Absent(33),
                RawLandmarkObservationSet.Absent(21),
                RawLandmarkObservationSet.Absent(21));

        static double[] NaNCoordinates(int landmarkCount)
        {
            var result = new double[landmarkCount * 3];
            for (int index = 0; index < result.Length; index++) result[index] = double.NaN;
            return result;
        }

        static void WriteValid(
            double[] coordinates,
            bool[] valid,
            int landmark,
            double x,
            double y,
            double z)
        {
            int offset = landmark * 3;
            coordinates[offset] = x;
            coordinates[offset + 1] = y;
            coordinates[offset + 2] = z;
            valid[landmark] = true;
        }

        static string ProjectPath(string relativePath) =>
            Path.Combine(ProjectRoot, relativePath.Replace('/', Path.DirectorySeparatorChar));

        static void AssertAsset(string path, long expectedSize, string expectedSha)
        {
            Assert.IsTrue(File.Exists(path), path);
            Assert.AreEqual(expectedSize, new FileInfo(path).Length, path);
            Assert.AreEqual(expectedSha, Sha256File(path), path);
        }

        static string Sha256File(string path)
        {
            using (var sha = SHA256.Create())
            using (var stream = File.OpenRead(path))
                return string.Concat(sha.ComputeHash(stream).Select(value => value.ToString("x2")));
        }
    }
}
