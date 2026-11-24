using System;
using WordsInSilence.Practice.PetRobotVerticalSlice;

namespace WordsInSilence.Recognition.LiveInput
{
    /// <summary>
    /// What one finalized attempt looks like to the coordinator. A plain value so the
    /// coordinator stays free of Unity and of the recognition types.
    /// </summary>
    public sealed class ValidationAttemptInfo
    {
        public string CaptureSessionId { get; }

        /// <summary>Name of the capture state the attempt ended in, verbatim.</summary>
        public string CaptureStateName { get; }

        /// <summary>The already-composed diagnostic line for this attempt.</summary>
        public string DiagnosticRecord { get; }

        public ValidationAttemptInfo(string captureSessionId, string captureStateName,
                                     string diagnosticRecord)
        {
            CaptureSessionId = captureSessionId ?? string.Empty;
            CaptureStateName = captureStateName ?? string.Empty;
            DiagnosticRecord = diagnosticRecord ?? string.Empty;
        }
    }

    /// <summary>
    /// The only three adapter operations this validation is allowed to perform. It
    /// deliberately cannot express a repository, a profile, a quest or any persistence,
    /// so a coordinator built on it has nothing to write with.
    /// </summary>
    public interface ICommonSignalValidationAdapterPort
    {
        bool SetExpectedWord(string target);
        bool BeginCapture();
        ValidationAttemptInfo FinalizeCapture();

        /// <summary>
        /// Step 10P v1.0d1: the one lifecycle operation the two-target validation needs.
        /// The live run showed why: after a finalize the capture controller sits in
        /// Completed, and SetExpectedWord only succeeds from Ready, so a second target
        /// can never start without returning the controller to Ready first. This
        /// forwards to the already-existing adapter reset; it adds no new capability and
        /// still cannot express a repository, candidate, profile, quest, session,
        /// PlayerPrefs or any persistence.
        /// </summary>
        bool ResetCapture();
    }

    public enum TwoTargetValidationState
    {
        Ready,
        CaptureStarted,
        Completed,
        Stopped,
    }

    /// <summary>
    /// Step 10P v1.0c2 orchestration for the developer-only two-target validation.
    ///
    /// It owns BOTH the target state and every decision to call the adapter, so the
    /// Unity harness cannot begin or finalize behind its back. Two properties matter:
    ///
    ///  TRANSACTIONAL BEGIN. A target is committed as begun only after SetExpectedWord
    ///  and BeginCapture have BOTH returned true. A false return or an exception leaves
    ///  the plan untouched and moves to a terminal stopped state, so the UI can never
    ///  offer Finalize for a capture that never started. State is never committed first
    ///  and rolled back, because a rollback window is itself observable ambiguity.
    ///
    ///  SINGLE-INVOCATION FINALIZE. The finalize latch is set BEFORE the adapter is
    ///  called, so FinalizeCapture can be invoked at most once per target no matter what
    ///  comes back - null, an exception, a blank or reused id, or a non-terminal capture
    ///  state. Any of those stops the run: no record, no advance, no retry, no second
    ///  call. Only a non-null attempt with a unique non-blank id whose capture state is
    ///  an accepted terminal state advances the plan and records one diagnostic line.
    ///
    /// The first failure is preserved verbatim in <see cref="StopReason"/> for review;
    /// nothing is retried and no threshold, target or model is adjusted afterwards.
    ///
    /// Pure: no UnityEngine dependency, so the real behaviour is compiled and executed
    /// offline as reproducible evidence.
    /// </summary>
    public sealed class CommonSignalTwoTargetValidationCoordinator
    {
        /// <summary>The only capture state a finalized take may end in.</summary>
        public const string AcceptedTerminalCaptureState = "Completed";

        readonly CommonSignalTwoTargetValidationPlan plan;
        readonly CommonSignalDiagnosticGate gate;
        readonly ICommonSignalValidationAdapterPort port;
        readonly Action<string> sink;

        bool finalizeInvoked;

        public CommonSignalTwoTargetValidationCoordinator(
            ICommonSignalValidationAdapterPort port,
            Action<string> sink,
            CommonSignalTwoTargetValidationPlan plan = null,
            CommonSignalDiagnosticGate gate = null)
        {
            this.port = port ?? throw new ArgumentNullException(nameof(port));
            this.sink = sink;
            this.plan = plan ?? new CommonSignalTwoTargetValidationPlan();
            this.gate = gate ?? new CommonSignalDiagnosticGate();
        }

        public TwoTargetValidationState State { get; private set; } =
            TwoTargetValidationState.Ready;
        public string StopReason { get; private set; } = string.Empty;
        public string LastRejection { get; private set; } = string.Empty;
        public int AdapterSetExpectedWordCalls { get; private set; }
        public int AdapterBeginCaptureCalls { get; private set; }
        public int AdapterFinalizeCaptureCalls { get; private set; }

        /// <summary>
        /// How many times the capture lifecycle was reset. Exactly one reset happens, at
        /// the start of the second target's Begin transaction; never before the first
        /// target and never after completion.
        /// </summary>
        public int AdapterResetCaptureCalls { get; private set; }
        public int RecordedLines { get { return gate.RecordedCount; } }
        public string LastRecord { get { return gate.LastRecord; } }
        public CommonSignalTwoTargetValidationPlan Plan { get { return plan; } }
        public string CurrentTarget { get { return plan.CurrentTarget; } }
        public int FinalizedCount { get { return plan.FinalizedCount; } }

        public bool CanBegin
        {
            get { return State == TwoTargetValidationState.Ready && plan.CanBegin; }
        }

        public bool CanFinalize
        {
            get
            {
                return State == TwoTargetValidationState.CaptureStarted
                    && !finalizeInvoked && plan.CanFinalize;
            }
        }

        void Stop(string reason)
        {
            State = TwoTargetValidationState.Stopped;
            StopReason = reason ?? string.Empty;
        }

        /// <summary>
        /// Validates the target, then calls the adapter, and only then commits the plan.
        /// Returns false without committing anything if the run cannot proceed.
        /// </summary>
        public bool Begin()
        {
            LastRejection = string.Empty;
            if (State != TwoTargetValidationState.Ready)
            {
                LastRejection = "begin is not available in state " + State;
                return false;
            }
            string target = plan.CurrentTarget;
            if (!plan.CanBegin || CommonSignalTwoTargetValidationPlan.IsForbiddenTarget(target))
            {
                LastRejection = "the plan does not allow beginning " + target;
                return false;
            }

            // --- lifecycle reset, only for a target that follows a finalized take.
            // It belongs to THIS Begin transaction, immediately before SetExpectedWord:
            // resetting eagerly inside the previous Finalize would put an already
            // accepted record at risk, and resetting after SetExpectedWord would undo
            // the target that was just selected. The plan is still untouched here, so a
            // reset failure cannot retract the GREETING take that is already committed.
            if (plan.FinalizedCount > 0)
            {
                try
                {
                    AdapterResetCaptureCalls++;
                    if (!port.ResetCapture())
                    {
                        Stop("ResetCapture returned false before " + target);
                        return false;
                    }
                }
                catch (Exception exception)
                {
                    Stop("ResetCapture threw " + exception.GetType().Name +
                         " before " + target);
                    return false;
                }
            }

            // --- adapter side effects first; the plan is untouched until both succeed
            try
            {
                AdapterSetExpectedWordCalls++;
                if (!port.SetExpectedWord(target))
                {
                    Stop("SetExpectedWord returned false for " + target);
                    return false;
                }
            }
            catch (Exception exception)
            {
                Stop("SetExpectedWord threw " + exception.GetType().Name + " for " + target);
                return false;
            }

            try
            {
                AdapterBeginCaptureCalls++;
                if (!port.BeginCapture())
                {
                    Stop("BeginCapture returned false for " + target);
                    return false;
                }
            }
            catch (Exception exception)
            {
                Stop("BeginCapture threw " + exception.GetType().Name + " for " + target);
                return false;
            }

            // --- commit
            if (!plan.TryBegin(target))
            {
                Stop("the plan refused a begin the adapter had already started: " +
                     plan.RejectionReason);
                return false;
            }
            State = TwoTargetValidationState.CaptureStarted;
            return true;
        }

        /// <summary>
        /// Latches the single allowed finalize for this target, calls the adapter once,
        /// and accepts the result only if it is a complete, unique, terminal attempt.
        /// </summary>
        public bool Finalize()
        {
            LastRejection = string.Empty;
            if (!CanFinalize)
            {
                LastRejection = "finalize is not available in state " + State +
                    (finalizeInvoked ? " (already invoked for this target)" : string.Empty);
                return false;
            }

            // The latch is set BEFORE the adapter call, so no result branch below can
            // leave a second invocation available.
            finalizeInvoked = true;
            string target = plan.CurrentTarget;

            ValidationAttemptInfo attempt;
            try
            {
                AdapterFinalizeCaptureCalls++;
                attempt = port.FinalizeCapture();
            }
            catch (Exception exception)
            {
                Stop("FinalizeCapture threw " + exception.GetType().Name + " for " + target);
                return false;
            }

            if (attempt == null)
            {
                Stop("FinalizeCapture produced no attempt for " + target);
                return false;
            }
            if (!string.Equals(attempt.CaptureStateName, AcceptedTerminalCaptureState,
                               StringComparison.Ordinal))
            {
                Stop("attempt for " + target + " ended in non-terminal capture state " +
                     (string.IsNullOrEmpty(attempt.CaptureStateName)
                         ? "<empty>" : attempt.CaptureStateName));
                return false;
            }
            if (!plan.TryFinalize(target, attempt.CaptureSessionId))
            {
                Stop("attempt for " + target + " was refused: " + plan.RejectionReason);
                return false;
            }

            gate.TryRecord(attempt.CaptureSessionId, () => attempt.DiagnosticRecord, sink);

            if (plan.Completed)
            {
                State = TwoTargetValidationState.Completed;
            }
            else
            {
                finalizeInvoked = false;
                State = TwoTargetValidationState.Ready;
            }
            return true;
        }
    }
}
