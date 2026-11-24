using System;
using System.Collections.Generic;

namespace WordsInSilence.Practice.PetRobotVerticalSlice
{
    /// <summary>
    /// Step 10P v1.0c1 exact-once gate and isolating sink for the common-2D diagnostic.
    ///
    /// Two guarantees, both enforced here rather than assumed from a call-site count:
    ///
    /// 1. EXACT ONCE. A finalized attempt is recorded once per unique, non-empty capture
    ///    session id. A repeated callback for the same id records nothing more, a blank
    ///    id records nothing, and a retry - which carries a new id - records once again.
    ///    The practice session additionally emits only after its own blank-or-duplicate
    ///    gate has accepted the attempt, so the two guards agree.
    ///
    /// 2. NON-INTERFERENCE. Composing the record and writing it are both isolated. A
    ///    failure in either becomes a bounded diagnostic marker or is dropped; it never
    ///    propagates into capture, finalization, candidate or outcome flow, and it never
    ///    changes recognition state or retries anything.
    ///
    /// This type is pure: no UnityEngine dependency, so the real production behaviour can
    /// be compiled and executed offline as reproducible evidence.
    /// </summary>
    public sealed class CommonSignalDiagnosticGate
    {
        // Step 10P v1.0c2: the previous build cleared the id set once it held 512
        // entries, so a much later repeat of an already recorded id could be logged a
        // second time - contradicting the exact-once contract. The set is now never
        // cleared, evicted or bounded for the lifetime of a gate instance: correctness
        // comes first, and a run finalizes a handful of attempts, so the memory held is
        // a few short strings.
        readonly HashSet<string> recordedSessions = new HashSet<string>(StringComparer.Ordinal);

        public int RecordedCount { get; private set; }
        public int SuppressedCount { get; private set; }
        public int IsolatedFailureCount { get; private set; }
        public string LastRecord { get; private set; } = string.Empty;

        /// <summary>True when this capture session has not been recorded yet.</summary>
        public bool ShouldRecord(string captureSessionId)
        {
            if (string.IsNullOrWhiteSpace(captureSessionId)) return false;
            return !recordedSessions.Contains(captureSessionId);
        }

        /// <summary>
        /// Records one attempt if, and only if, its capture session id is non-empty and
        /// has not been recorded before. Returns true when a record was handed to the
        /// sink. Never throws.
        /// </summary>
        public bool TryRecord(string captureSessionId, Func<string> compose, Action<string> sink)
        {
            if (string.IsNullOrWhiteSpace(captureSessionId))
            {
                SuppressedCount++;
                return false;
            }
            if (!recordedSessions.Add(captureSessionId))
            {
                SuppressedCount++;
                return false;
            }

            string line;
            try
            {
                line = compose == null ? string.Empty : compose();
            }
            catch (Exception exception)
            {
                IsolatedFailureCount++;
                line = CommonSignalDiagnosticRecord.ComposeDiagnosticFailure(
                    "compose_failed", exception.GetType().Name);
            }

            LastRecord = line ?? string.Empty;
            RecordedCount++;
            try
            {
                if (sink != null) sink(LastRecord);
            }
            catch (Exception)
            {
                // A diagnostic sink failure is isolated on purpose: the attempt has
                // already been finalized by the approved seams and must not be disturbed.
                IsolatedFailureCount++;
            }
            return true;
        }

        /// <summary>Number of ids this instance is remembering. Never shrinks.</summary>
        public int RememberedSessionCount { get { return recordedSessions.Count; } }

        /// <summary>
        /// Forgets recorded ids. TEST SETUP ONLY: nothing in the capture, finalize,
        /// outcome or validation flow calls this, so a production gate instance never
        /// forgets an id while it lives.
        /// </summary>
        public void Reset()
        {
            recordedSessions.Clear();
            RecordedCount = 0;
            SuppressedCount = 0;
            IsolatedFailureCount = 0;
            LastRecord = string.Empty;
        }
    }
}
