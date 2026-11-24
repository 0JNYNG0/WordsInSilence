using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Security.Cryptography;
using System.Threading;
using System.Threading.Tasks;
using NUnit.Framework;
using UnityEngine;
using WordsInSilence.Camera;
using WordsInSilence.Debugging;
using WordsInSilence.Practice.PetRobotVerticalSlice;
using WordsInSilence.Recognition.LiveInput;
using WordsInSilence.Recognition.PhysicalCompatibility;

namespace WordsInSilence.Tests.EditMode
{
    public sealed class PetRobotCandidateCameraStartupRouteV1Tests
    {
        [Test]
        public void Scene_HasExactlyOneActiveCandidateCameraOwnerAndLegacyControllerRemainsInactive()
        {
            string scene = ReadProject("Assets/_Project/Scenes/VerticalSlice/PetRobotSignPracticeVerticalSlice.unity");
            Assert.AreEqual(1, Count(scene, "guid: e33729c5ad6f4e44ad02aa19b2710d40"));
            StringAssert.Contains("candidateCameraRoute: {fileID: 1669740324}", scene);
            StringAssert.Contains("m_Name: PracticeController\n  m_TagString: Untagged", scene);
            StringAssert.Contains("m_Name: PracticeController\n  m_TagString: Untagged\n  m_Icon: {fileID: 0}\n  m_NavMeshLayer: 0\n  m_StaticEditorFlags: 0\n  m_IsActive: 0", scene);
            string entry = ReadProject("Assets/_Project/Scripts/Practice/PetRobotVerticalSlice/PetRobotPracticeSceneEntry.cs");
            Assert.IsFalse(entry.Contains("SignPracticeController"));
            Assert.IsFalse(entry.Contains("SetActive(true)"));
        }

        [Test]
        public void CandidateStartup_StartsTheOwnedCameraExactlyOnce()
        {
            GameObject gameObject = new GameObject("candidate-camera-start-once");
            try
            {
                PetRobotCandidateCameraStartupRoute route =
                    gameObject.AddComponent<PetRobotCandidateCameraStartupRoute>();
                var source = new FakeCameraFrameSource(CameraStartResult.Success("fake", 640, 480, 30));
                ReplaceCameraSource(route, source);
                InvokeTask(route, "StartCameraOnceAsync");
                InvokeTask(route, "StartCameraOnceAsync");
                Assert.AreEqual(1, source.StartCalls);
                Assert.AreEqual(1, route.StartInvocationCount);
                Assert.IsTrue(route.CameraStarted);
                Assert.AreEqual(CandidateCameraStartupState.Starting, route.CameraState);
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(gameObject);
            }
        }

        [Test]
        public void PreviewAndHandPoseProviders_UseTheSameSerializedFrameSource()
        {
            string scene = ReadProject("Assets/_Project/Scenes/VerticalSlice/PetRobotSignPracticeVerticalSlice.unity");
            Assert.AreEqual(
                1,
                scene.Split('\n').Count(line =>
                    string.Equals(
                        line.TrimEnd('\r'),
                        "  cameraDisplay: {fileID: 447812208}",
                        StringComparison.Ordinal)));
            Assert.AreEqual(2, Count(scene, "_rawImage: {fileID: 447812208}"));
            StringAssert.Contains("captureConfig: {fileID: 11400000, guid: c7817e503402bd8498b8358b680830a9, type: 2}", scene);
            string ui = ReadProject("Assets/_Project/Scripts/Practice/PetRobotVerticalSlice/PetRobotPracticeUiPresenter.cs");
            StringAssert.Contains("cameraRoute.PreviewTexture", ui);
        }

        [Test]
        public void StartExpression_IsBlockedUntilCameraFrameAndProviderObservationAreReady()
        {
            var readiness = new FakeCameraReadiness();
            var recognition = new FakeRecognitionAdapter();
            var repository = new FakeRepository();
            using (var session = new PetRobotSignPracticeSession(recognition, repository, null, readiness))
            {
                Assert.IsFalse(session.StartExpression());
                Assert.AreEqual(0, recognition.BeginCalls);
                readiness.ReceiveFrame();
                Assert.IsFalse(session.StartExpression());
                Assert.AreEqual(0, recognition.BeginCalls);
                readiness.ReceiveObservation();
                Assert.IsTrue(session.CameraReady);
                Assert.IsTrue(session.StartExpression());
                Assert.AreEqual(1, recognition.BeginCalls);
            }
        }

        [Test]
        public void StartupFailure_IsNonScoringTechnicalFailureWithNoProgressOrStorage()
        {
            GameObject gameObject = new GameObject("candidate-camera-start-failure");
            try
            {
                PetRobotCandidateCameraStartupRoute route =
                    gameObject.AddComponent<PetRobotCandidateCameraStartupRoute>();
                ReplaceCameraSource(route,
                    new FakeCameraFrameSource(CameraStartResult.Failure("CAMERA_TIMEOUT", "fake timeout")));
                var recognition = new FakeRecognitionAdapter();
                var repository = new FakeRepository();
                using (var session = new PetRobotSignPracticeSession(recognition, repository, null, route))
                {
                    InvokeTask(route, "StartCameraOnceAsync");
                    Assert.AreEqual(CandidateCameraStartupState.Failed, route.CameraState);
                    Assert.AreEqual(GuidedQuestOutcomeKind.TechnicalFailure, session.LastLearningOutcome.Kind);
                    Assert.IsFalse(session.LastLearningOutcome.GuidedProgressAllowed);
                    Assert.IsFalse(session.IsCurrentWordComplete);
                    Assert.AreEqual(0, recognition.BeginCalls);
                    Assert.AreEqual(0, repository.SaveCalls);
                    Assert.AreEqual(CandidateStorageStatus.NotRequested, session.LastStorageResult.Status);
                    Assert.IsFalse(session.StartExpression());
                }
            }
            finally { UnityEngine.Object.DestroyImmediate(gameObject); }
        }

        [Test]
        public void DisableThenDestroy_StopDisposeAndObservationSubscriptionExactlyOnce()
        {
            GameObject gameObject = new GameObject("candidate-camera-cleanup-once");
            try
            {
                PetRobotCandidateCameraStartupRoute route =
                    gameObject.AddComponent<PetRobotCandidateCameraStartupRoute>();
                var source = new FakeCameraFrameSource(CameraStartResult.Success("fake", 640, 480, 30));
                ReplaceCameraSource(route, source);
                var bridge = gameObject.AddComponent<Step9CHandIdentityObservationBridge>();
                route.BindObservationBridge(bridge);
                Assert.IsTrue(route.ObservationSubscriptionActive);
                InvokeTask(route, "StartCameraOnceAsync");
                Assert.AreEqual(1, source.StartCalls);
                Invoke(route, "OnDisable");
                Assert.IsFalse(route.ObservationSubscriptionActive);
                Invoke(route, "OnDestroy");
                Assert.AreEqual(1, source.StopCalls);
                Assert.AreEqual(1, source.DisposeCalls);
                Assert.AreEqual(1, route.StopInvocationCount);
                Assert.AreEqual(1, route.DisposeInvocationCount);
                Assert.IsFalse(route.ObservationSubscriptionActive);
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(gameObject);
            }
        }

        [Test]
        public void ApprovedRecognitionCalculationAndCandidateProfileRemainByteIdentical()
        {
            AssertHash("Assets/_Project/Scripts/Recognition/Core/RecognitionCore.cs",
                "34ac176049a0340ef6983df29f433d0d5f4bb83720b7b8f7da9747fdfabef498");
            AssertHash("Assets/_Project/Scripts/Recognition/Core/MaskAwareDtw.cs",
                "69c76f869e1c2b123b06eb3e81668dc2dce7c555ce967d769e7bf6abb6977e1b");
            AssertHash("Assets/_Project/Scripts/Recognition/LiveInput/GuidedQuestOutcomeEvaluator.cs",
                "05a827d097c06012f0cec30c0c812e2b3aab8faa2e92865a163f07c08418fd1e");

            string root = "Assets/StreamingAssets/RecognitionCoreV1/PersonalProfiles/step9l_current_user_multi_prototype_v1_0/";
            AssertHash(root + "personal_multi_prototype_threshold_config.json",
                "42720034661a1fb9f6c673918f55fb0da3bf0507555463d4a13170718e81bfe0");
            AssertHash(root + "personal_runtime_profile_manifest.json",
                "f95508424f6e23dc8d046992be91f26c725aaf7d4fd91779832c2ea88653237d");
            var expected = new Dictionary<string, string>(StringComparer.Ordinal)
            {
                { "personal_prototype_GREETING_26f436475e2f4e8c832dbce00cbfe883.wisrt", "d5b56a6c714f22646b47e819a7a5d124433ca62b03f5968bfc8c9cf48c8d4a84" },
                { "personal_prototype_GREETING_87e5b54432ac49a197a596d5ff7c3845.wisrt", "d8d5713334b22d9427b9b589bccf820cdbb4849357bae3dce1c368eaf0049681" },
                { "personal_prototype_GREETING_step9c-take-01-greeting-f7875c16-ad18-425c-8ce5-39e9a6cc13f7.wisrt", "3ac4ec3e6468787c4f203503c9481c5f45bf245b1c4d4dc3d1e0841e6ff848fa" },
                { "personal_prototype_LOVE_cd65ad6b277c48b78fa50785070fdddc.wisrt", "5c17605bbc922f27c1ac54a97017e72c5812e89ae692aa4b81cf2139f7ccc2d8" },
                { "personal_prototype_LOVE_d958cbc2a1c44e03bb29bf64a6fae195.wisrt", "013bf5515eb8af33f300400893da64c853668b2fa0442e19203da62b2777ad4b" },
                { "personal_prototype_LOVE_step9c-take-02-love-6d68ddd7-2d7e-4afd-bd0a-b1989f8e52f2.wisrt", "6117c8204a2193b60639a0629a4006f6200c79862bca5fe372e08457b345a731" },
                { "personal_prototype_THANK_YOU_2b323671e3f64049a74556c71ff70154.wisrt", "9bb9a7681d55a292a21d8b7f458b4b150c10555bc0ec163da692a780ed5280e5" },
                { "personal_prototype_THANK_YOU_9926cef4a46649bea3139fbace2a0ff3.wisrt", "d6919f37afe725da6068654e446c1424ef132e451610c537c82f10bb4efb7fed" },
                { "personal_prototype_THANK_YOU_step9c-take-03-thank-you-53870472-c644-489c-a94d-95e9667ded42.wisrt", "cd2fc174c50c7f6c3e4adc089ac6ad12b2be60d90b8c003dc277c0b8597fdbe0" },
            };
            foreach (KeyValuePair<string, string> item in expected)
                AssertHash(root + item.Key, item.Value);
        }

        [Test]
        public void ApprovedStep9KAndLegacyScenesAndPrefabsRemainByteIdentical()
        {
            AssertHash("Assets/_Project/Resources/Diagnostics/PersonalLiveRecognitionTechnical.prefab",
                "2b2d1835d358bab3a59ee5e2fcd23570a6a40c86f6aed83f968364692fed0658");
            AssertHash("Assets/_Project/Resources/Diagnostics/PersonalMultiPrototypeLiveRecognitionTechnical.prefab",
                "c7f8ba196ef1d4df4dbe9ca6f028ae4f4ff1896743927c8d5163262ea9af3bee");
            AssertHash("Assets/_Project/Scenes/Game/03_SignPractice.unity",
                "dac9d8d51a561825314cd8fb4021e4e803590d9275d3d65b55af96bfa47338e8");
            AssertHash("Assets/_Project/Scenes/Technical/02_PoseIntegration.unity",
                "1fb83d0f304277317a4571319e432c8d85394c7f84abce9036e23a6b3de16a0b");
        }

        static string ReadProject(string relativePath) =>
            File.ReadAllText(Path.Combine(Directory.GetParent(Application.dataPath).FullName, relativePath))
                .Replace("\r\n", "\n");

        static void AssertHash(string relativePath, string expected)
        {
            string path = Path.Combine(Directory.GetParent(Application.dataPath).FullName, relativePath);
            using (SHA256 algorithm = SHA256.Create())
                Assert.AreEqual(expected,
                    string.Concat(algorithm.ComputeHash(File.ReadAllBytes(path)).Select(value => value.ToString("x2"))),
                    relativePath);
        }

        static int Count(string text, string value)
        {
            int count = 0;
            int offset = 0;
            while ((offset = text.IndexOf(value, offset, StringComparison.Ordinal)) >= 0)
            {
                count++;
                offset += value.Length;
            }
            return count;
        }

        static void ReplaceCameraSource(PetRobotCandidateCameraStartupRoute route, ICameraFrameSource source)
        {
            FieldInfo field = typeof(PetRobotCandidateCameraStartupRoute).GetField(
                "cameraSource", BindingFlags.Instance | BindingFlags.NonPublic);
            var previous = field.GetValue(route) as IDisposable;
            previous?.Dispose();
            field.SetValue(route, source);
        }

        static void InvokeTask(object target, string name)
        {
            var task = (Task)target.GetType().GetMethod(name, BindingFlags.Instance | BindingFlags.NonPublic)
                .Invoke(target, null);
            task.GetAwaiter().GetResult();
        }

        static void Invoke(object target, string name)
        {
            target.GetType().GetMethod(name, BindingFlags.Instance | BindingFlags.NonPublic)
                .Invoke(target, null);
        }

        sealed class FakeCameraFrameSource : ICameraFrameSource, IDisposable
        {
            readonly CameraStartResult result;
            public int StartCalls;
            public int StopCalls;
            public int DisposeCalls;
            public bool IsRunning { get; private set; }
            public string ActiveDeviceName => "fake";
            public WebCamTexture ActiveTexture => null;
            public event Action<CaptureDiagnostics> DiagnosticsUpdated { add { } remove { } }

            public FakeCameraFrameSource(CameraStartResult result) => this.result = result;
            public CameraDeviceInfo[] GetAvailableDevices() => new CameraDeviceInfo[0];
            public Task<CameraStartResult> StartAsync(CameraStartRequest request, CancellationToken cancellationToken)
            {
                StartCalls++;
                IsRunning = result.IsSuccess;
                return Task.FromResult(result);
            }
            public Task StopAsync(CancellationToken cancellationToken)
            {
                StopCalls++;
                IsRunning = false;
                return Task.CompletedTask;
            }
            public void Dispose() => DisposeCalls++;
        }

        sealed class FakeCameraReadiness : IPetRobotCandidateCameraReadiness
        {
            public event Action Changed;
            public CandidateCameraStartupState CameraState { get; private set; } = CandidateCameraStartupState.Starting;
            public bool IsReady => CameraState == CandidateCameraStartupState.Ready;
            public bool CameraStarted { get; private set; } = true;
            public bool HasReceivedFrame { get; private set; }
            public bool ProviderObservationReady { get; private set; }
            public int CameraFrameCount { get; private set; }
            public int ProviderObservationCount { get; private set; }
            public string LastErrorCode { get; private set; } = string.Empty;
            public string LastErrorMessage { get; private set; } = string.Empty;

            public void ReceiveFrame()
            {
                HasReceivedFrame = true;
                CameraFrameCount = 1;
                Evaluate();
            }
            public void ReceiveObservation()
            {
                ProviderObservationReady = true;
                ProviderObservationCount = 1;
                Evaluate();
            }
            public void Fail(string code, string message)
            {
                CameraStarted = false;
                LastErrorCode = code;
                LastErrorMessage = message;
                CameraState = CandidateCameraStartupState.Failed;
                Changed?.Invoke();
            }
            void Evaluate()
            {
                if (HasReceivedFrame && ProviderObservationReady)
                    CameraState = CandidateCameraStartupState.Ready;
                Changed?.Invoke();
            }
        }

        sealed class FakeRecognitionAdapter : IPersonalLivePracticeRecognitionAdapter
        {
            PersonalLiveCaptureState state = PersonalLiveCaptureState.Ready;
            public int BeginCalls;
            public PersonalLiveCaptureState CaptureState => state;
            public string CaptureSessionId => string.Empty;
            public string ActiveProfileId => "step9l_current_user_multi_prototype_v1_0";
            public bool CandidateRouteActive => true;
            public bool SetExpectedWord(string word) =>
                state == PersonalLiveCaptureState.Ready && PersonalLiveRecognitionController.IsSupportedWord(word);
            public bool BeginCapture()
            {
                if (state != PersonalLiveCaptureState.Ready) return false;
                BeginCalls++;
                state = PersonalLiveCaptureState.Capturing;
                return true;
            }
            public PracticeRecognitionAttempt FinalizeCapture() => null;
            public bool ResetCapture() => false;
        }

        sealed class FakeRepository : IPersonalizationCandidateRepository
        {
            public int SaveCalls;
            public string RootDirectory => "fake";
            public string LearnerId => "fake";
            public CandidateCountSnapshot Counts => CandidateCountSnapshot.Empty;
            public CandidateCountSnapshot ReloadAndVerify() => Counts;
            public CandidateSaveResult Save(PersonalizationCandidateSaveRequest request)
            {
                SaveCalls++;
                return new CandidateSaveResult(CandidateStorageStatus.Saved, "fake", "saved", Counts);
            }
        }
    }
}
