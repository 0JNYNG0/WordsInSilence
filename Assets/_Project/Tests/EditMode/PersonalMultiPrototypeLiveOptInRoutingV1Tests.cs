using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using NUnit.Framework;
using UnityEditor;
using UnityEngine;
using WordsInSilence.Recognition.Core;
using WordsInSilence.Recognition.LiveInput;
using WordsInSilence.Recognition.Runtime;

namespace WordsInSilence.Tests.EditMode
{
    public sealed class PersonalMultiPrototypeLiveOptInRoutingV1Tests
    {
        const double Tolerance = 1e-12;
        const string CandidateProfileId = "step9l_current_user_multi_prototype_v1_0";
        const string LegacyProfileId = "step9h_current_user_v1_0";
        const string CandidatePrefabPath =
            "Assets/_Project/Resources/Diagnostics/PersonalMultiPrototypeLiveRecognitionTechnical.prefab";
        const string LegacyPrefabPath =
            "Assets/_Project/Resources/Diagnostics/PersonalLiveRecognitionTechnical.prefab";

        static string ProjectRoot => Directory.GetParent(Application.dataPath).FullName;
        static string GlobalTemplateDirectory => Path.Combine(Application.streamingAssetsPath,
            "RecognitionCoreV1", "Templates");
        static string GlobalThresholdPath => Path.Combine(Application.streamingAssetsPath,
            "RecognitionCoreV1", "RuntimeGate", "frozen_threshold_config.json");
        static string LegacyProfileDirectory => Path.Combine(Application.streamingAssetsPath,
            "RecognitionCoreV1", "PersonalProfiles", LegacyProfileId);
        static string CandidateProfileDirectory => Path.Combine(Application.streamingAssetsPath,
            "RecognitionCoreV1", "PersonalProfiles", CandidateProfileId);
        static string FixtureDirectory => Path.Combine(Application.dataPath,
            "_Project", "Tests", "EditMode", "Fixtures", "PersonalMultiPrototypeRuntimeIntegrationV1");

        [Serializable]
        sealed class QueryManifest { public QueryRecord[] queries; }

        [Serializable]
        sealed class QueryRecord
        {
            public string query_id;
            public string expected_word;
            public string relative_path;
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
        }

        IPersonalLiveGlobalRuntime globalHandle;
        RuntimeRecognitionService globalService;
        PersonalMultiPrototypeRuntimeProfile candidateProfile;
        QueryManifest queryManifest;
        Dictionary<string, RecognitionSequence> queries;

        [OneTimeSetUp]
        public void OneTimeSetUp()
        {
            var legacyFactory = new ApprovedPersonalLiveRuntimeFactory();
            globalHandle = legacyFactory.LoadGlobal(Paths(LegacyProfileDirectory));
            globalService = RuntimeRecognitionService.Load(GlobalTemplateDirectory, GlobalThresholdPath);
            candidateProfile = PersonalMultiPrototypeRuntimeProfileLoader.Load(CandidateProfileDirectory);
            queryManifest = JsonUtility.FromJson<QueryManifest>(
                File.ReadAllText(Path.Combine(FixtureDirectory, "query_manifest.json")));
            queries = queryManifest.queries.ToDictionary(
                item => item.query_id,
                item => RecognitionSequenceBinary.LoadFile(Path.Combine(
                    ProjectRoot,
                    item.relative_path.Replace('/', Path.DirectorySeparatorChar))),
                StringComparer.Ordinal);
        }

        [Test]
        public void LegacyHarnessDefaultAndFactoryRemainSingleMedoid()
        {
            GameObject owner = null;
            try
            {
                owner = new GameObject("Step9M-Legacy-Default");
                owner.SetActive(false);
                PersonalLiveRecognitionHarness harness = owner.AddComponent<PersonalLiveRecognitionHarness>();
                Assert.AreEqual(PersonalLiveProfileMode.LegacySingleMedoid, harness.ProfileMode);

                var factory = new ApprovedPersonalLiveRuntimeFactory();
                IPersonalLiveRecognitionRuntime runtime = factory.CreateAndActivate(
                    globalHandle,
                    Paths(LegacyProfileDirectory));
                Assert.IsTrue(runtime.IsPersonalActive);
                Assert.AreEqual(LegacyProfileId, runtime.ActiveProfileId);
                Assert.AreEqual(3, PersonalRuntimeProfileLoader.Load(LegacyProfileDirectory).Templates.Count);
            }
            finally
            {
                if (owner != null) UnityEngine.Object.DestroyImmediate(owner);
            }
        }

        [Test]
        public void ExplicitCandidateLiveFactoryActivatesFixedProfileAndNinePrototypes()
        {
            var factory = new ApprovedPersonalMultiPrototypeLiveRuntimeFactory();
            IPersonalLiveRecognitionRuntime runtime = factory.CreateAndActivate(
                globalHandle,
                Paths(CandidateProfileDirectory));

            Assert.IsTrue(runtime.IsPersonalActive);
            Assert.AreEqual(CandidateProfileId, runtime.ActiveProfileId);
            Assert.AreEqual(9, candidateProfile.Prototypes.Sum(item => item.Value.Count));
            foreach (string word in PersonalLiveRecognitionController.SupportedWords)
                Assert.AreEqual(3, candidateProfile.Prototypes[word].Count, word);
        }

        [Test]
        public void CandidateLiveRouteUsesPerWordMinimumDistanceForAllTwentySevenCells()
        {
            var factory = new ApprovedPersonalMultiPrototypeLiveRuntimeFactory();
            IPersonalLiveRecognitionRuntime runtime = factory.CreateAndActivate(
                globalHandle,
                Paths(CandidateProfileDirectory));
            int scoreCells = 0;
            int questSuccess = 0;

            foreach (QueryRecord record in queryManifest.queries)
            {
                RecognitionSequence query = queries[record.query_id];
                PersonalLiveRoutingResult routed = runtime.Recognize(
                    record.query_id,
                    query,
                    record.expected_word);
                foreach (string word in PersonalLiveRecognitionController.SupportedWords)
                {
                    var ranked = candidateProfile.Prototypes[word]
                        .Select(prototype => new
                        {
                            Prototype = prototype,
                            Distance = MaskAwareDtw.Compute(
                                query,
                                prototype,
                                RecognitionCore.RelativeBand).Distance,
                        })
                        .OrderBy(item => item.Distance)
                        .ThenBy(item => item.Prototype.SampleId, StringComparer.Ordinal)
                        .ToArray();
                    Assert.AreEqual(ranked[0].Distance, routed.Scores[word], Tolerance,
                        record.query_id + " -> " + word);
                    if (word == routed.PredictedWord)
                        Assert.AreEqual(ranked[0].Prototype.SampleId, routed.WinningPrototypeId);
                    scoreCells++;
                }
                Assert.AreEqual(record.expected_word, routed.PredictedWord);
                Assert.IsTrue(routed.QuestSuccess);
                questSuccess++;
            }

            Assert.AreEqual(27, scoreCells);
            Assert.AreEqual(9, questSuccess);
        }

        [Test]
        public void CandidateActivationFailureIsAtomicAndLeavesControllerInactive()
        {
            string copy = CopyDirectory(CandidateProfileDirectory);
            try
            {
                File.Delete(Path.Combine(copy, PersonalMultiPrototypeRuntimeProfileLoader.ConfigFileName));
                var source = new FakeObservationSource();
                var controller = new PersonalLiveRecognitionController(
                    source,
                    new ApprovedPersonalMultiPrototypeLiveRuntimeFactory());

                Assert.IsFalse(controller.Initialize(Paths(copy)));
                Assert.AreEqual(PersonalLiveCaptureState.Failed, controller.State);
                Assert.AreEqual(PersonalLiveFailureCode.PersonalProfileActivationFailure, controller.Failure.Code);
                Assert.IsFalse(controller.IsPersonalModeActive);
                Assert.AreEqual(string.Empty, controller.ActivePersonalProfileId);
                Assert.AreEqual(0, source.SubscriberCount);
            }
            finally
            {
                Directory.Delete(copy, true);
            }
        }

        [Test]
        public void CandidateDeactivationReturnsToUnchangedGlobalMode()
        {
            var session = new PersonalRuntimeOptInSession(globalService);
            session.ActivatePersonalMultiPrototypeProfile(CandidateProfileDirectory);
            Assert.AreEqual(CandidateProfileId, session.ActiveProfileId);

            session.DeactivatePersonalProfile();

            Assert.AreEqual(RuntimeOptInMode.Global, session.Mode);
            Assert.IsFalse(session.IsPersonalActive);
            Assert.AreEqual(string.Empty, session.ActiveProfileId);
        }

        [Test]
        public void CandidateHarnessOnDisableRemovesAllObservationSourceSubscribers()
        {
            GameObject owner = null;
            try
            {
                var source = new FakeObservationSource();
                PersonalLiveRecognitionController controller = ReadyCandidateController(source);
                owner = new GameObject("Step9M-Candidate-OnDisable");
                owner.SetActive(false);
                PersonalLiveRecognitionHarness harness = owner.AddComponent<PersonalLiveRecognitionHarness>();
                SetPrivateField(harness, "controller", controller);

                Assert.AreEqual(2, source.SubscriberCount);
                InvokeLifecycle(harness, "OnDisable");
                Assert.AreEqual(0, source.SubscriberCount);
                Assert.AreEqual(PersonalLiveCaptureState.Cancelled, controller.State);
            }
            finally
            {
                if (owner != null) UnityEngine.Object.DestroyImmediate(owner);
            }
        }

        [Test]
        public void CandidateHarnessOnDestroyRemovesAllObservationSourceSubscribers()
        {
            GameObject owner = null;
            try
            {
                var source = new FakeObservationSource();
                PersonalLiveRecognitionController controller = ReadyCandidateController(source);
                owner = new GameObject("Step9M-Candidate-OnDestroy");
                owner.SetActive(false);
                PersonalLiveRecognitionHarness harness = owner.AddComponent<PersonalLiveRecognitionHarness>();
                SetPrivateField(harness, "controller", controller);

                Assert.AreEqual(2, source.SubscriberCount);
                InvokeLifecycle(harness, "OnDestroy");
                Assert.AreEqual(0, source.SubscriberCount);
                Assert.IsNull(GetPrivateField(harness, "controller"));
            }
            finally
            {
                if (owner != null) UnityEngine.Object.DestroyImmediate(owner);
            }
        }

        [Test]
        public void CandidateTechnicalPrefabFixesCandidateModeAndPreservesLegacyPrefabDefault()
        {
            GameObject candidate = AssetDatabase.LoadAssetAtPath<GameObject>(CandidatePrefabPath);
            GameObject legacy = AssetDatabase.LoadAssetAtPath<GameObject>(LegacyPrefabPath);
            Assert.IsNotNull(candidate);
            Assert.IsNotNull(legacy);
            Assert.AreEqual("PersonalMultiPrototypeLiveRecognitionTechnical", candidate.name);

            PersonalLiveRecognitionHarness candidateHarness =
                candidate.GetComponent<PersonalLiveRecognitionHarness>();
            PersonalLiveRecognitionHarness legacyHarness =
                legacy.GetComponent<PersonalLiveRecognitionHarness>();
            Assert.IsNotNull(candidateHarness);
            Assert.IsNotNull(legacyHarness);
            Assert.AreEqual(PersonalLiveProfileMode.MultiPrototypeCandidate, candidateHarness.ProfileMode);
            Assert.AreEqual(PersonalLiveProfileMode.LegacySingleMedoid, legacyHarness.ProfileMode);
            Assert.AreSame(
                candidate.GetComponent<WordsInSilence.Recognition.PhysicalCompatibility.Step9CHandIdentityObservationBridge>(),
                candidateHarness.ObservationBridge);
        }

        PersonalLiveRecognitionController ReadyCandidateController(FakeObservationSource source)
        {
            var controller = new PersonalLiveRecognitionController(
                source,
                new ApprovedPersonalMultiPrototypeLiveRuntimeFactory());
            Assert.IsTrue(controller.Initialize(Paths(CandidateProfileDirectory)));
            Assert.AreEqual(CandidateProfileId, controller.ActivePersonalProfileId);
            return controller;
        }

        static PersonalLiveRuntimePaths Paths(string profileDirectory)
        {
            return new PersonalLiveRuntimePaths(
                GlobalTemplateDirectory,
                GlobalThresholdPath,
                profileDirectory);
        }

        static void SetPrivateField(object target, string fieldName, object value)
        {
            FieldInfo field = target.GetType().GetField(
                fieldName,
                BindingFlags.Instance | BindingFlags.NonPublic);
            Assert.IsNotNull(field, fieldName);
            field.SetValue(target, value);
        }

        static object GetPrivateField(object target, string fieldName)
        {
            FieldInfo field = target.GetType().GetField(
                fieldName,
                BindingFlags.Instance | BindingFlags.NonPublic);
            Assert.IsNotNull(field, fieldName);
            return field.GetValue(target);
        }

        static void InvokeLifecycle(PersonalLiveRecognitionHarness harness, string methodName)
        {
            MethodInfo method = typeof(PersonalLiveRecognitionHarness).GetMethod(
                methodName,
                BindingFlags.Instance | BindingFlags.NonPublic);
            Assert.IsNotNull(method, methodName);
            method.Invoke(harness, null);
        }

        static string CopyDirectory(string source)
        {
            string destination = Path.Combine(
                Path.GetTempPath(),
                "WordsInSilence-Step9M-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(destination);
            foreach (string file in Directory.GetFiles(source))
                File.Copy(file, Path.Combine(destination, Path.GetFileName(file)));
            return destination;
        }
    }
}
