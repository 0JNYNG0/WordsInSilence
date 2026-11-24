using System;
using System.Collections.Generic;
using NUnit.Framework;
using WordsInSilence.Practice.PetRobotVerticalSlice;
using WordsInSilence.Recognition.LiveInput;

namespace WordsInSilence.Tests.EditMode
{
    /// <summary>
    /// Step 10P v1.0c2: the developer two-target validation must commit a target only
    /// after the adapter really started the capture, must call FinalizeCapture at most
    /// once per target, and must accept only a complete, unique, terminal attempt.
    ///
    /// These drive the SAME <see cref="CommonSignalTwoTargetValidationCoordinator"/> the
    /// Unity harness drives, through a spy adapter port, and assert adapter call counts,
    /// call order and terminal state. A plan-only test could not see any of this.
    /// </summary>
    public sealed class CommonSignalTwoTargetCoordinatorV1c2Tests
    {
        /// <summary>
        /// Step 10P v1.0d1: the double now models the real capture lifecycle instead of
        /// answering every call statelessly. The v1.0d live run failed precisely because
        /// the old stateless double could not express it:
        ///
        ///   Ready --SetExpectedWord--> Ready --BeginCapture--> Capturing
        ///   Capturing --FinalizeCapture--> Completed
        ///   Completed --ResetCapture--> Ready
        ///
        /// SetExpectedWord succeeds only from Ready, exactly like
        /// PersonalLivePracticeRecognitionAdapter, so removing the reset makes the happy
        /// path fail at the second Set - the same escape the live take hit.
        /// </summary>
        sealed class SpyPort : ICommonSignalValidationAdapterPort
        {
            public enum CaptureLifecycle { Ready, Capturing, Completed, Cancelled, Failed }

            public readonly List<string> Calls = new List<string>();
            public CaptureLifecycle State = CaptureLifecycle.Ready;
            public bool SetResult = true;
            public bool BeginResult = true;
            public bool ResetResult = true;
            public bool SetThrows, BeginThrows, FinalizeThrows, ResetThrows;
            public Func<int, ValidationAttemptInfo> FinalizeFactory =
                index => new ValidationAttemptInfo("live-" + index, "Completed",
                                                   "record-" + index);
            public int SetCalls, BeginCalls, FinalizeCalls, ResetCalls;

            public bool SetExpectedWord(string target)
            {
                SetCalls++;
                Calls.Add("Set:" + target);
                if (SetThrows) throw new InvalidOperationException("set exploded");
                // the controller only accepts a new expected word while it is Ready
                if (State != CaptureLifecycle.Ready) return false;
                return SetResult;
            }

            public bool BeginCapture()
            {
                BeginCalls++;
                Calls.Add("Begin");
                if (BeginThrows) throw new InvalidOperationException("begin exploded");
                if (State != CaptureLifecycle.Ready || !BeginResult) return false;
                State = CaptureLifecycle.Capturing;
                return true;
            }

            public ValidationAttemptInfo FinalizeCapture()
            {
                FinalizeCalls++;
                Calls.Add("Finalize");
                if (FinalizeThrows) throw new InvalidOperationException("finalize exploded");
                State = CaptureLifecycle.Completed;
                return FinalizeFactory == null ? null : FinalizeFactory(FinalizeCalls);
            }

            public bool ResetCapture()
            {
                ResetCalls++;
                Calls.Add("Reset");
                if (ResetThrows) throw new InvalidOperationException("reset exploded");
                if (!ResetResult) return false;
                if (State == CaptureLifecycle.Completed ||
                    State == CaptureLifecycle.Cancelled ||
                    State == CaptureLifecycle.Failed)
                {
                    State = CaptureLifecycle.Ready;
                    return true;
                }
                return false;
            }
        }

        List<string> sink;

        CommonSignalTwoTargetValidationCoordinator New(SpyPort port)
        {
            sink = new List<string>();
            return new CommonSignalTwoTargetValidationCoordinator(port, line => sink.Add(line));
        }

        // ---- C04 transactional begin -----------------------------------------

        [Test]
        public void SetExpectedWordFailureStopsWithoutCommittingOrCapturing()
        {
            foreach (bool throws in new[] { false, true })
            {
                var port = new SpyPort();
                if (throws) port.SetThrows = true;
                else port.SetResult = false;
                var coordinator = New(port);

                Assert.IsFalse(coordinator.Begin());
                Assert.AreEqual(TwoTargetValidationState.Stopped, coordinator.State);
                Assert.IsFalse(coordinator.Plan.CanFinalize,
                    "the plan must not have been committed");
                Assert.IsFalse(coordinator.CanFinalize);
                Assert.IsFalse(coordinator.CanBegin);
                Assert.AreEqual(0, port.BeginCalls, "capture must not be started");
                Assert.IsFalse(coordinator.Finalize());
                Assert.AreEqual(0, port.FinalizeCalls);
                Assert.AreEqual(0, sink.Count);
                Assert.IsNotEmpty(coordinator.StopReason);
            }
        }

        [Test]
        public void BeginCaptureFailureStopsWithNoFinalizeAndNoRetry()
        {
            foreach (bool throws in new[] { false, true })
            {
                var port = new SpyPort();
                if (throws) port.BeginThrows = true;
                else port.BeginResult = false;
                var coordinator = New(port);

                Assert.IsFalse(coordinator.Begin());
                Assert.AreEqual(TwoTargetValidationState.Stopped, coordinator.State);
                Assert.IsFalse(coordinator.Plan.CanFinalize);
                Assert.IsFalse(coordinator.Begin(), "no retry after a stop");
                Assert.AreEqual(1, port.SetCalls);
                Assert.AreEqual(1, port.BeginCalls);
                Assert.IsFalse(coordinator.Finalize());
                Assert.AreEqual(0, port.FinalizeCalls);
                Assert.AreEqual(0, coordinator.FinalizedCount);
            }
        }

        // ---- C05 single-invocation finalize ----------------------------------

        [Test]
        public void EveryFailingFinalizeBranchCallsTheAdapterExactlyOnce()
        {
            var branches = new Dictionary<string, Action<SpyPort>>
            {
                { "null", p => p.FinalizeFactory = index => null },
                { "throws", p => p.FinalizeThrows = true },
                { "blank", p => p.FinalizeFactory =
                    index => new ValidationAttemptInfo("   ", "Completed", "r") },
                { "nonterminal", p => p.FinalizeFactory =
                    index => new ValidationAttemptInfo("live-1", "Capturing", "r") },
            };

            foreach (KeyValuePair<string, Action<SpyPort>> branch in branches)
            {
                var port = new SpyPort();
                branch.Value(port);
                var coordinator = New(port);

                Assert.IsTrue(coordinator.Begin(), branch.Key);
                Assert.IsFalse(coordinator.Finalize(), branch.Key);
                Assert.IsFalse(coordinator.Finalize(), branch.Key + ": second finalize");
                Assert.IsFalse(coordinator.Finalize(), branch.Key + ": third finalize");

                Assert.AreEqual(1, port.FinalizeCalls,
                    branch.Key + ": FinalizeCapture must be invoked at most once");
                Assert.AreEqual(TwoTargetValidationState.Stopped, coordinator.State, branch.Key);
                Assert.AreEqual(0, coordinator.FinalizedCount, branch.Key);
                Assert.AreEqual(0, sink.Count, branch.Key);
                Assert.IsFalse(coordinator.CanBegin, branch.Key);
                Assert.IsFalse(coordinator.CanFinalize, branch.Key);
            }
        }

        [Test]
        public void AReusedCaptureSessionIdStopsTheSecondTarget()
        {
            var port = new SpyPort
            {
                FinalizeFactory = index => new ValidationAttemptInfo(
                    "shared-id", "Completed", "record")
            };
            var coordinator = New(port);

            Assert.IsTrue(coordinator.Begin());
            Assert.IsTrue(coordinator.Finalize());
            Assert.AreEqual(1, sink.Count);

            Assert.IsTrue(coordinator.Begin());
            Assert.IsFalse(coordinator.Finalize(), "a reused id is not a new take");
            Assert.AreEqual(TwoTargetValidationState.Stopped, coordinator.State);
            Assert.AreEqual(2, port.FinalizeCalls);
            Assert.AreEqual(1, coordinator.FinalizedCount);
            Assert.AreEqual(1, sink.Count);
        }

        // ---- C06 terminal attempt validation ---------------------------------

        [Test]
        public void OnlyTheAcceptedTerminalCaptureStateCounts()
        {
            foreach (string state in new[] { "Ready", "Capturing", "Failed", "Cancelled" })
            {
                var port = new SpyPort
                {
                    FinalizeFactory = index =>
                        new ValidationAttemptInfo("live-" + state, state, "record")
                };
                var coordinator = New(port);
                Assert.IsTrue(coordinator.Begin());
                Assert.IsFalse(coordinator.Finalize(), state + " must not count");
                Assert.AreEqual(0, coordinator.FinalizedCount, state);
                Assert.AreEqual(0, sink.Count, state);
                Assert.AreEqual(TwoTargetValidationState.Stopped, coordinator.State, state);
                StringAssert.Contains("non-terminal", coordinator.StopReason);
            }

            var good = new SpyPort();
            var ok = New(good);
            Assert.IsTrue(ok.Begin());
            Assert.IsTrue(ok.Finalize());
            Assert.AreEqual(1, ok.FinalizedCount);
            Assert.AreEqual(1, sink.Count);
            Assert.AreEqual(CommonSignalTwoTargetValidationCoordinator
                .AcceptedTerminalCaptureState, "Completed");
        }

        // ---- C07 happy path ---------------------------------------------------

        [Test]
        public void HappyPathCallsEachAdapterOperationExactlyTwiceInOrder()
        {
            var port = new SpyPort();
            var coordinator = New(port);

            Assert.IsTrue(coordinator.Begin());
            Assert.IsTrue(coordinator.Finalize());
            Assert.IsTrue(coordinator.Begin());
            Assert.IsTrue(coordinator.Finalize());

            Assert.AreEqual(TwoTargetValidationState.Completed, coordinator.State);
            Assert.AreEqual(2, port.SetCalls);
            Assert.AreEqual(2, port.BeginCalls);
            Assert.AreEqual(2, port.FinalizeCalls);
            Assert.AreEqual(1, port.ResetCalls, "exactly one lifecycle reset");
            Assert.AreEqual(1, coordinator.AdapterResetCaptureCalls);
            CollectionAssert.AreEqual(
                new[] { "Set:GREETING", "Begin", "Finalize", "Reset",
                        "Set:THANK_YOU", "Begin", "Finalize" },
                port.Calls);
            Assert.AreEqual(2, sink.Count);
            Assert.AreEqual(2, new HashSet<string>(sink, StringComparer.Ordinal).Count);
            Assert.AreEqual(2, coordinator.FinalizedCount);

            Assert.IsFalse(coordinator.Begin(), "nothing after completion");
            Assert.IsFalse(coordinator.Finalize());
            Assert.AreEqual(2, port.FinalizeCalls);
            Assert.AreEqual(1, port.ResetCalls, "no reset after final completion");
            foreach (string call in port.Calls)
                Assert.IsFalse(call.Contains("LOVE"), "LOVE must never be requested");
        }

        // ---- v1.0d1 reset lifecycle ------------------------------------------

        [Test]
        public void TheFirstTargetIsNeverPrecededByAReset()
        {
            var port = new SpyPort();
            var coordinator = New(port);

            Assert.IsTrue(coordinator.Begin());
            Assert.AreEqual(0, port.ResetCalls, "no reset before the first target");
            Assert.AreEqual(0, coordinator.AdapterResetCaptureCalls);
            CollectionAssert.AreEqual(new[] { "Set:GREETING", "Begin" }, port.Calls);

            Assert.IsTrue(coordinator.Finalize());
            Assert.AreEqual(0, port.ResetCalls,
                "the first finalize must not reset eagerly; the accepted record stays "
                + "committed even if a later reset fails");
        }

        [Test]
        public void AFailedResetIsTerminalAndKeepsTheAcceptedFirstTake()
        {
            foreach (bool throws in new[] { false, true })
            {
                var port = new SpyPort();
                var coordinator = New(port);

                Assert.IsTrue(coordinator.Begin());
                Assert.IsTrue(coordinator.Finalize());
                Assert.AreEqual(1, coordinator.FinalizedCount);
                Assert.AreEqual(1, sink.Count);

                if (throws) port.ResetThrows = true;
                else port.ResetResult = false;

                Assert.IsFalse(coordinator.Begin(), "the second target must not start");
                Assert.AreEqual(TwoTargetValidationState.Stopped, coordinator.State);
                StringAssert.Contains("ResetCapture", coordinator.StopReason);

                // the accepted GREETING take survives untouched
                Assert.AreEqual(1, coordinator.FinalizedCount);
                Assert.AreEqual(1, sink.Count);
                Assert.AreEqual(1, port.ResetCalls);
                // and nothing was attempted for the second target
                Assert.AreEqual(1, port.SetCalls);
                Assert.AreEqual(1, port.BeginCalls);
                Assert.AreEqual(1, port.FinalizeCalls);

                // no retry is possible afterwards
                Assert.IsFalse(coordinator.Begin());
                Assert.IsFalse(coordinator.Finalize());
                Assert.AreEqual(1, port.ResetCalls);
                Assert.AreEqual(1, port.SetCalls);
                Assert.AreEqual(1, port.BeginCalls);
                Assert.AreEqual(1, port.FinalizeCalls);
                Assert.AreEqual(1, sink.Count);
            }
        }

        [Test]
        public void WithoutTheResetTheSecondTargetCouldNotStart()
        {
            // Reproduces the v1.0d live escape directly against the lifecycle double:
            // once the controller is Completed, SetExpectedWord refuses, which is why a
            // reset is required rather than optional.
            var port = new SpyPort();
            Assert.IsTrue(port.SetExpectedWord("GREETING"));
            Assert.IsTrue(port.BeginCapture());
            Assert.IsNotNull(port.FinalizeCapture());
            Assert.AreEqual(SpyPort.CaptureLifecycle.Completed, port.State);

            Assert.IsFalse(port.SetExpectedWord("THANK_YOU"),
                "this is exactly the false the live run observed");

            Assert.IsTrue(port.ResetCapture());
            Assert.AreEqual(SpyPort.CaptureLifecycle.Ready, port.State);
            Assert.IsTrue(port.SetExpectedWord("THANK_YOU"));
        }

        // ---- no-write by shape -------------------------------------------------

        [Test]
        public void TheAdapterPortCannotExpressAnyPersistence()
        {
            foreach (var member in typeof(ICommonSignalValidationAdapterPort).GetMembers())
            {
                string text = member.ToString();
                Assert.IsFalse(text.Contains("Repository"), "port exposes " + text);
                Assert.IsFalse(text.Contains("Candidate"), "port exposes " + text);
                Assert.IsFalse(text.Contains("Profile"), "port exposes " + text);
                Assert.IsFalse(text.Contains("Quest"), "port exposes " + text);
                Assert.IsFalse(text.Contains("Session") && !text.Contains("SessionId"),
                    "port exposes " + text);
            }
        }
    }
}
