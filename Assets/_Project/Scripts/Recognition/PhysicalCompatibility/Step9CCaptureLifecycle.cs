using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;

namespace WordsInSilence.Recognition.PhysicalCompatibility
{
    public sealed class Step9CProviderBoundary
    {
        public int ProviderFrameCount { get; }
        public int BridgeObservationCount { get; }
        public long LifetimeProviderFrameCount { get; }
        public long LastLifetimeProviderFrameIndex { get; }
        public double FirstProviderTimestampSeconds { get; }
        public double LastProviderTimestampSeconds { get; }
        public double LastLifetimeProviderTimestampSeconds { get; }

        public Step9CProviderBoundary(
            int providerFrameCount,
            int bridgeObservationCount,
            long lifetimeProviderFrameCount,
            long lastLifetimeProviderFrameIndex,
            double firstProviderTimestampSeconds,
            double lastProviderTimestampSeconds,
            double lastLifetimeProviderTimestampSeconds)
        {
            if (providerFrameCount < 0) throw new ArgumentOutOfRangeException(nameof(providerFrameCount));
            if (bridgeObservationCount < 0 || bridgeObservationCount > providerFrameCount) throw new ArgumentOutOfRangeException(nameof(bridgeObservationCount));
            if (lifetimeProviderFrameCount < 0) throw new ArgumentOutOfRangeException(nameof(lifetimeProviderFrameCount));
            ProviderFrameCount = providerFrameCount;
            BridgeObservationCount = bridgeObservationCount;
            LifetimeProviderFrameCount = lifetimeProviderFrameCount;
            LastLifetimeProviderFrameIndex = lastLifetimeProviderFrameIndex;
            FirstProviderTimestampSeconds = firstProviderTimestampSeconds;
            LastProviderTimestampSeconds = lastProviderTimestampSeconds;
            LastLifetimeProviderTimestampSeconds = lastLifetimeProviderTimestampSeconds;
        }

        public static Step9CProviderBoundary Empty => new Step9CProviderBoundary(0, 0, 0, -1, double.NaN, double.NaN, double.NaN);
    }

    public sealed class PhysicalCaptureWindowSnapshot
    {
        public long PrepareAfterLifetimeProviderFrameIndex { get; }
        public double PrepareAfterProviderTimestampSeconds { get; }
        public long StartNextLifetimeProviderFrameIndex { get; }
        public double StartPreviousProviderTimestampSeconds { get; }
        public int FirstCaptureProviderFrameIndex { get; }
        public long FirstCaptureLifetimeProviderFrameIndex { get; }
        public double FirstCaptureProviderTimestampSeconds { get; }
        public int FinalCaptureProviderFrameIndex { get; }
        public long FinalCaptureLifetimeProviderFrameIndex { get; }
        public double FinalCaptureProviderTimestampSeconds { get; }
        public int ProviderFrameCount { get; }
        public int BridgeObservationCount { get; }

        internal PhysicalCaptureWindowSnapshot(Step9CProviderBoundary prepared, Step9CProviderBoundary started, Step9CProviderBoundary finalized)
        {
            PrepareAfterLifetimeProviderFrameIndex = prepared.LastLifetimeProviderFrameIndex;
            PrepareAfterProviderTimestampSeconds = prepared.LastLifetimeProviderTimestampSeconds;
            StartNextLifetimeProviderFrameIndex = started.LifetimeProviderFrameCount;
            StartPreviousProviderTimestampSeconds = started.LastLifetimeProviderTimestampSeconds;
            ProviderFrameCount = finalized.ProviderFrameCount;
            BridgeObservationCount = finalized.BridgeObservationCount;
            FirstCaptureProviderFrameIndex = finalized.ProviderFrameCount == 0 ? -1 : 0;
            FirstCaptureLifetimeProviderFrameIndex = finalized.ProviderFrameCount == 0 ? -1 : started.LifetimeProviderFrameCount;
            FirstCaptureProviderTimestampSeconds = finalized.FirstProviderTimestampSeconds;
            FinalCaptureProviderFrameIndex = finalized.ProviderFrameCount - 1;
            FinalCaptureLifetimeProviderFrameIndex = finalized.ProviderFrameCount == 0 ? -1 : finalized.LastLifetimeProviderFrameIndex;
            FinalCaptureProviderTimestampSeconds = finalized.ProviderFrameCount == 0 ? double.NaN : finalized.LastProviderTimestampSeconds;
        }
    }

    public sealed class Step9CCaptureDiagnosticSnapshot
    {
        readonly ReadOnlyCollection<Step9CHandIdentityDiagnostic> diagnostics;

        public IReadOnlyList<Step9CHandIdentityDiagnostic> Diagnostics => diagnostics;
        public string FailureType { get; }
        public string FailureMessage { get; }
        public PhysicalCaptureWindowSnapshot Window { get; }

        internal Step9CCaptureDiagnosticSnapshot(
            IList<Step9CHandIdentityDiagnostic> diagnostics,
            string failureType,
            string failureMessage,
            PhysicalCaptureWindowSnapshot window)
        {
            this.diagnostics = new ReadOnlyCollection<Step9CHandIdentityDiagnostic>(new List<Step9CHandIdentityDiagnostic>(diagnostics).ToArray());
            FailureType = failureType ?? string.Empty;
            FailureMessage = failureMessage ?? string.Empty;
            Window = window ?? throw new ArgumentNullException(nameof(window));
        }
    }

    /// <summary>Capture-only ledger; live preview events outside Start..Finalize are deliberately ignored.</summary>
    public sealed class Step9CCaptureDiagnosticLedger
    {
        enum LedgerState { Idle, Prepared, Capturing, Finalized }

        readonly List<Step9CHandIdentityDiagnostic> diagnostics = new List<Step9CHandIdentityDiagnostic>();
        LedgerState state;
        Step9CProviderBoundary prepared = Step9CProviderBoundary.Empty;
        Step9CProviderBoundary started = Step9CProviderBoundary.Empty;
        string failureType = string.Empty;
        string failureMessage = string.Empty;

        public bool IsCapturing => state == LedgerState.Capturing;
        public int DiagnosticCount => diagnostics.Count;
        public IReadOnlyList<Step9CHandIdentityDiagnostic> CurrentDiagnostics => diagnostics.AsReadOnly();
        public string FailureType => failureType;
        public string FailureMessage => failureMessage;

        public void Prepare(Step9CProviderBoundary boundary)
        {
            if (state == LedgerState.Capturing) throw new InvalidOperationException("cannot prepare while capture diagnostics are active");
            Clear();
            prepared = boundary ?? throw new ArgumentNullException(nameof(boundary));
            state = LedgerState.Prepared;
        }

        public void Begin(Step9CProviderBoundary boundary)
        {
            if (state != LedgerState.Prepared) throw new InvalidOperationException("capture diagnostics must be prepared before Start");
            Clear();
            started = boundary ?? throw new ArgumentNullException(nameof(boundary));
            state = LedgerState.Capturing;
        }

        public bool TryRecordDiagnostic(Step9CHandIdentityDiagnostic diagnostic)
        {
            if (state != LedgerState.Capturing) return false;
            diagnostics.Add(diagnostic ?? throw new ArgumentNullException(nameof(diagnostic)));
            return true;
        }

        public bool TryRecordFailure(Exception exception)
        {
            if (exception == null) throw new ArgumentNullException(nameof(exception));
            return TryRecordFailure(exception.GetType().FullName, exception.Message);
        }

        public bool TryRecordFailure(string type, string message)
        {
            if (state != LedgerState.Capturing) return false;
            if (string.IsNullOrEmpty(failureType))
            {
                failureType = type ?? string.Empty;
                failureMessage = message ?? string.Empty;
            }
            return true;
        }

        public Step9CCaptureDiagnosticSnapshot Finalize(Step9CProviderBoundary boundary)
        {
            if (state != LedgerState.Capturing) throw new InvalidOperationException("capture diagnostics are not active");
            var snapshot = new Step9CCaptureDiagnosticSnapshot(
                diagnostics,
                failureType,
                failureMessage,
                new PhysicalCaptureWindowSnapshot(prepared, started, boundary ?? throw new ArgumentNullException(nameof(boundary))));
            state = LedgerState.Finalized;
            return snapshot;
        }

        public void Reset()
        {
            Clear();
            prepared = Step9CProviderBoundary.Empty;
            started = Step9CProviderBoundary.Empty;
            state = LedgerState.Idle;
        }

        void Clear()
        {
            diagnostics.Clear();
            failureType = string.Empty;
            failureMessage = string.Empty;
        }
    }
}
