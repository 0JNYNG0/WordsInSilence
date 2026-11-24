// F7R4 EditMode lifecycle tests for the repeat host and its terminal-state rule.
//
// The host now does its dependency work at a deterministic LATER boundary (Start, then the
// next Update), not in OnEnable, so these tests invoke the real private lifecycle methods in
// that order by reflection. The host is built on an INACTIVE GameObject so this runner
// cannot invoke a Unity callback implicitly. Production needs no ExecuteAlways.
//
// Cursor expectations are always the state actually read back after a request, because this
// runner does not honour a locked cursor.
//
// These cover configuration, boundary and teardown behaviour ONLY; scene ordering with a
// real camera route, prefab loading and GUI layout need the reviewer's runtime. No real
// camera, private file, model or network is used. Nothing here has been executed.
using System;
using System.Collections.Generic;
using System.Reflection;
using NUnit.Framework;
using UnityEngine;
using WordsInSilence.Pose;
using WordsInSilence.Practice.PetRobotVerticalSlice;
using WordsInSilence.Recognition.FullHandRepeat;
using WordsInSilence.Recognition.FullHandRuntime;
using WordsInSilence.Recognition.LiveInput;

namespace WordsInSilence.Tests.EditMode
{
    public sealed class FullHandRepeatHostLifecycleTests
    {
        readonly List<GameObject> created = new List<GameObject>();
        bool cursorVisibleBefore;
        CursorLockMode cursorLockBefore;

        [SetUp]
        public void SetUp()
        {
            cursorVisibleBefore = Cursor.visible;
            cursorLockBefore = Cursor.lockState;
            FullHandRepeatDiagnosticHost.EnvironmentForTest = null;
        }

        [TearDown]
        public void TearDown()
        {
            foreach (GameObject entry in created)
                if (entry != null) UnityEngine.Object.DestroyImmediate(entry);
            created.Clear();
            FullHandRepeatDiagnosticHost.EnvironmentForTest = null;
            Cursor.visible = cursorVisibleBefore;
            Cursor.lockState = cursorLockBefore;
        }

        /// <summary>A stand-in for a world input behaviour the tool must suppress.</summary>
        sealed class FakeWorldInputBehaviour : MonoBehaviour { }

        GameObject NewInactive(string name)
        {
            var entry = new GameObject(name);
            entry.SetActive(false);
            created.Add(entry);
            return entry;
        }

        /// <summary>
        /// Builds the host on an INACTIVE object, wires it, and returns it. Nothing has run
        /// yet: every lifecycle step below is an explicit, verified call.
        /// </summary>
        FullHandRepeatDiagnosticHost CreateHost(bool wireRoute,
                                                out FakeWorldInputBehaviour worldInput)
        {
            GameObject worldObject = new GameObject("FakeWorldInput");
            created.Add(worldObject);
            worldInput = worldObject.AddComponent<FakeWorldInputBehaviour>();
            worldInput.enabled = true;

            PetRobotCandidateCameraStartupRoute route = null;
            HandPoseIntegrator integrator = null;
            if (wireRoute)
            {
                // both live on INACTIVE objects, so neither runs its own Awake here
                route = NewInactive("FakeRoute")
                    .AddComponent<PetRobotCandidateCameraStartupRoute>();
                integrator = NewInactive("FakeIntegrator").AddComponent<HandPoseIntegrator>();
            }

            var host = NewInactive("RepeatHostUnderTest")
                .AddComponent<FullHandRepeatDiagnosticHost>();
            host.WireForBuild(route, integrator, new Behaviour[] { worldInput });
            return host;
        }

        static void InvokeLifecycle(FullHandRepeatDiagnosticHost host, string method)
        {
            MethodInfo info = typeof(FullHandRepeatDiagnosticHost).GetMethod(
                method, BindingFlags.Instance | BindingFlags.NonPublic);
            Assert.IsNotNull(info, "the host must still declare a private " + method);
            info.Invoke(host, null);
        }

        /// <summary>
        /// Requests a cursor state and returns what the editor actually accepted, so a
        /// restoration assertion never depends on a mode this runner refuses.
        /// </summary>
        static void RequestCursor(bool visible, CursorLockMode lockState,
                                  out bool actualVisible, out CursorLockMode actualLock)
        {
            Cursor.visible = visible;
            Cursor.lockState = lockState;
            actualVisible = Cursor.visible;
            actualLock = Cursor.lockState;
        }

        // ---- 1. OFF does nothing at all, at any boundary -------------------------

        [Test]
        public void WhenTheRepeatFlagIsOffNothingIsLoadedBoundOrChanged()
        {
            var requested = new List<string>();
            FullHandRepeatDiagnosticHost.EnvironmentForTest =
                name => { requested.Add(name); return null; };
            bool visibleBefore;
            CursorLockMode lockBefore;
            RequestCursor(false, CursorLockMode.Locked, out visibleBefore, out lockBefore);

            FakeWorldInputBehaviour worldInput;
            FullHandRepeatDiagnosticHost host = CreateHost(true, out worldInput);
            InvokeLifecycle(host, "Awake");
            InvokeLifecycle(host, "OnEnable");
            InvokeLifecycle(host, "Start");
            InvokeLifecycle(host, "Update");

            Assert.IsNotEmpty(requested, "the repeat flag is genuinely consulted");
            CollectionAssert.AreEquivalent(
                new[] { FullHandRepeatDiagnosticHost.RepeatVariable },
                new List<string>(new HashSet<string>(requested)),
                "an OFF tool reads the repeat flag and nothing else, at any boundary");
            Assert.IsFalse(host.Armed);
            Assert.IsEmpty(host.SetupFault, "an OFF tool reports no fault at all");
            Assert.AreEqual(FullHandRepeatDiagnosticHost.PhaseOff, host.SetupPhase);
            Assert.IsFalse(host.InitializationPendingForTest);
            Assert.IsFalse(host.HarnessInstantiatedForTest, "no prefab is loaded");
            Assert.IsFalse(host.BridgeBoundForTest, "no bridge is bound");
            Assert.IsFalse(host.InputSuppressedForTest, "no world behaviour is disabled");
            Assert.IsFalse(host.CursorCapturedForTest);
            Assert.IsTrue(worldInput.enabled, "the world behaviour keeps its own state");
            Assert.AreEqual(visibleBefore, Cursor.visible, "the cursor is untouched");
            Assert.AreEqual(lockBefore, Cursor.lockState);
        }

        // ---- 2. pending before the boundary, then the typed configuration fault ---

        [Test]
        public void ABadManifestStaysPendingUntilTheLaterBoundaryThenFaultsAndRestores()
        {
            FullHandRepeatDiagnosticHost.EnvironmentForTest = name =>
            {
                if (name == FullHandRepeatDiagnosticHost.RepeatVariable) return "1";
                if (name == FullHandAttemptObserver.EnableVariable) return "1";
                if (name == FullHandAttemptObserver.ManifestVariable) return "relative.json";
                return null;
            };
            bool visibleBefore;
            CursorLockMode lockBefore;
            RequestCursor(false, CursorLockMode.Locked, out visibleBefore, out lockBefore);

            FakeWorldInputBehaviour worldInput;
            FullHandRepeatDiagnosticHost host = CreateHost(true, out worldInput);

            // Awake really acquires suppression and the cursor before any validation runs
            InvokeLifecycle(host, "Awake");
            Assert.IsTrue(host.InputSuppressedForTest, "Awake suppresses world input");
            Assert.IsFalse(worldInput.enabled, "the world behaviour is actually disabled");
            Assert.IsTrue(host.CursorCapturedForTest, "Awake takes the cursor");
            Assert.IsTrue(Cursor.visible, "the tool asks for a visible cursor");

            // OnEnable must NOT resolve anything: the scene has not finished enabling yet
            InvokeLifecycle(host, "OnEnable");
            Assert.IsTrue(host.InitializationPendingForTest,
                          "initialization waits for the later boundary");
            Assert.AreEqual(FullHandRepeatDiagnosticHost.PhasePendingBoundary,
                            host.SetupPhase);
            Assert.IsEmpty(host.SetupFault, "no fault may be decided during OnEnable");
            Assert.IsFalse(host.Armed);

            InvokeLifecycle(host, "Start");

            Assert.IsFalse(host.Armed, "a bad manifest must never arm the tool");
            Assert.AreEqual("BUNDLE_MANIFEST_NOT_ABSOLUTE", host.SetupFault);
            Assert.AreEqual(FullHandRepeatDiagnosticHost.PhaseFaulted, host.SetupPhase);
            Assert.IsFalse(host.InitializationPendingForTest);
            Assert.IsFalse(host.HarnessInstantiatedForTest, "nothing is loaded after the fault");
            Assert.IsFalse(host.BridgeBoundForTest);
            Assert.IsFalse(host.InputSuppressedForTest, "the failed setup hands input back");
            Assert.IsFalse(host.CursorCapturedForTest, "and hands the cursor back");
            Assert.IsTrue(worldInput.enabled, "the world behaviour is restored");
            Assert.AreEqual(visibleBefore, Cursor.visible, "the cursor is restored");
            Assert.AreEqual(lockBefore, Cursor.lockState);
        }

        // ---- 3. a serialized null reference is immediate, not a pending wait -------

        [Test]
        public void AnUnassignedRouteIsAnImmediateFaultRatherThanAPendingWait()
        {
            var requested = new List<string>();
            FullHandRepeatDiagnosticHost.EnvironmentForTest = name =>
            {
                requested.Add(name);
                return name == FullHandRepeatDiagnosticHost.RepeatVariable ? "1" : null;
            };

            FakeWorldInputBehaviour worldInput;
            FullHandRepeatDiagnosticHost host = CreateHost(false, out worldInput);
            InvokeLifecycle(host, "Awake");
            InvokeLifecycle(host, "OnEnable");
            Assert.IsEmpty(host.SetupFault, "OnEnable decides nothing");

            InvokeLifecycle(host, "Start");

            Assert.AreEqual("CAMERA_ROUTE_NOT_ASSIGNED", host.SetupFault,
                            "a null serialized reference is distinct from a not-yet-active one");
            Assert.AreEqual(FullHandRepeatDiagnosticHost.PhaseFaulted, host.SetupPhase);
            Assert.IsFalse(host.InitializationPendingForTest, "it does not keep waiting");
            CollectionAssert.DoesNotContain(requested,
                FullHandAttemptObserver.ManifestVariable,
                "a null reference is decided before any model or file work");
            Assert.IsTrue(worldInput.enabled, "ownership is handed back");
            Assert.IsFalse(host.CursorCapturedForTest);
        }

        // ---- 4. teardown is idempotent -------------------------------------------

        [Test]
        public void TeardownIsIdempotentAndRestoresOnlyOnce()
        {
            FullHandRepeatDiagnosticHost.EnvironmentForTest = name =>
                name == FullHandRepeatDiagnosticHost.RepeatVariable ? "1" : null;
            bool visibleBefore;
            CursorLockMode lockBefore;
            RequestCursor(false, CursorLockMode.Locked, out visibleBefore, out lockBefore);

            FakeWorldInputBehaviour worldInput;
            FullHandRepeatDiagnosticHost host = CreateHost(true, out worldInput);
            InvokeLifecycle(host, "Awake");
            Assert.IsTrue(host.InputSuppressedForTest);
            Assert.IsFalse(worldInput.enabled);

            InvokeLifecycle(host, "OnEnable");
            InvokeLifecycle(host, "Start");
            Assert.AreEqual("OBSERVER_NOT_ENABLED", host.SetupFault);
            Assert.IsFalse(host.InputSuppressedForTest, "the first cleanup already restored");
            Assert.IsTrue(worldInput.enabled);
            Assert.AreEqual(visibleBefore, Cursor.visible);
            Assert.AreEqual(lockBefore, Cursor.lockState);

            // someone changes the state AFTER the first cleanup; a second teardown must not
            // re-apply the stale saved values over it
            worldInput.enabled = false;
            bool visibleAfterChange;
            CursorLockMode lockAfterChange;
            RequestCursor(true, CursorLockMode.None, out visibleAfterChange,
                          out lockAfterChange);

            InvokeLifecycle(host, "OnDisable");
            InvokeLifecycle(host, "OnDestroy");

            Assert.IsFalse(worldInput.enabled,
                           "a second teardown must not re-apply an old saved state");
            Assert.AreEqual(visibleAfterChange, Cursor.visible);
            Assert.AreEqual(lockAfterChange, Cursor.lockState);
            Assert.IsFalse(host.Armed);
        }

        // ---- 5. the terminal-state rule closes exactly once ---------------------
        // unchanged from F7R2/F7R3: this test passed on every run.

        sealed class TerminalFakePort : IFullHandRepeatHarnessPort
        {
            public PersonalLiveCaptureState CurrentState = PersonalLiveCaptureState.Ready;
            public string Session = string.Empty;
            public int Buffered = 31;
            public int FinalizeCalls, CancelCalls, Rebuilds;
            public string Failure = string.Empty;

            public bool LiveInputBound { get { return true; } }
            public string EffectiveTarget { get; private set; }
            public PersonalLiveCaptureState State { get { return CurrentState; } }
            public string CaptureSessionId { get { return Session; } }
            public int BufferedRawFrameCount { get { return Buffered; } }

            public bool EnsureControllerReady()
            {
                if (CurrentState != PersonalLiveCaptureState.Ready)
                {
                    Rebuilds++;
                    Session = string.Empty;
                    Failure = string.Empty;
                    CurrentState = PersonalLiveCaptureState.Ready;
                }
                return true;
            }

            public bool TrySelectTarget(string word) { EffectiveTarget = word; return true; }

            public bool StartCapture()
            {
                Session = "capture_terminal_1";
                CurrentState = PersonalLiveCaptureState.Capturing;
                return true;
            }

            public bool FinalizeCapture() { FinalizeCalls++; return true; }
            public bool CancelCapture() { CancelCalls++; return true; }

            public bool ResetCapture()
            {
                Session = string.Empty;
                CurrentState = PersonalLiveCaptureState.Ready;
                return true;
            }

            public void FailFromSource()
            {
                CurrentState = PersonalLiveCaptureState.Failed;
                Failure = "BridgeFailure";
                Buffered = 0;
            }
        }

        sealed class RecordingSink : IFullHandObservationSink
        {
            public readonly Dictionary<string, string> Text =
                new Dictionary<string, string>(StringComparer.Ordinal);

            public void WriteBytes(string name, byte[] payload) { }

            public void WriteText(string name, string text)
            {
                if (Text.ContainsKey(name))
                    throw new System.IO.IOException("CreateNew refuses to overwrite " + name);
                Text[name] = text;
            }
        }

        [Test]
        public void ASourceFailureDuringCaptureIsClosedExactlyOnceAndPreservesTheOriginal()
        {
            var port = new TerminalFakePort();
            var sink = new RecordingSink();
            var coordinator = new FullHandRepeatCoordinator(port, sink, line => { }, null);
            string reason;
            Assert.IsTrue(coordinator.TryBegin(
                PersonalLiveRecognitionController.SupportedWords[0], "positive",
                "development", out reason), reason);

            Assert.IsFalse(FullHandRepeatTerminalWatch.ShouldCloseNow(true, port.State),
                           "a live capture is not closed");
            port.FailFromSource();
            Assert.IsTrue(FullHandRepeatTerminalWatch.ShouldCloseNow(true, port.State));

            // the host's Update reaction, invoked directly
            Assert.IsFalse(coordinator.TryFinish(false, out reason));
            Assert.AreEqual(0, port.FinalizeCalls,
                            "an already failed capture is never finalized over");
            Assert.AreEqual(0, port.CancelCalls);
            Assert.AreEqual("BridgeFailure", port.Failure, "the original failure is intact");
            Assert.AreEqual(1, sink.Text.Count);

            // a second reaction on the next tick must not write another record
            Assert.IsFalse(FullHandRepeatTerminalWatch.ShouldCloseNow(
                coordinator.CaptureActive, port.State));
            Assert.IsFalse(coordinator.TryFinish(false, out reason));
            Assert.AreEqual("NO_ACTIVE_CAPTURE", reason);
            Assert.AreEqual(1, sink.Text.Count, "the metadata is preserved exactly once");

            string record = sink.Text["repeat_attempt_0001.json"];
            StringAssert.Contains("\"outcome\": \"already_failed_before_close\"", record);
            StringAssert.Contains("\"private_capture_session_id\": \"capture_terminal_1\"",
                                  record);
            StringAssert.Contains("\"buffered_raw_frame_count\": null", record);
            Assert.AreEqual(0, port.Rebuilds, "no controller was recreated by the reaction");
        }
    }
}
