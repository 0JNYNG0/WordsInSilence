using System;
using System.Collections.Generic;

namespace WordsInSilence.Recognition.LiveInput
{
    /// <summary>
    /// Step 10P v1.0c1 fixed plan for the developer-only common-2D live validation:
    /// GREETING once, then THANK_YOU once. Nothing else.
    ///
    /// The plan owns no repository, no session, no profile and no quest state; it holds
    /// only the target order and which targets have been finalized. It cannot express a
    /// LOVE target, a repeat, a retry or a reorder, so those cannot happen by mistake
    /// in the live run:
    ///
    ///  - <see cref="Targets"/> is fixed at construction and contains exactly
    ///    GREETING then THANK_YOU.
    ///  - <see cref="TryBegin"/> only succeeds for the current target, only when the
    ///    previous target is finalized, and only once per target.
    ///  - <see cref="TryFinalize"/> only succeeds for a begun target, only once, and
    ///    only with a non-empty capture session id that has not been seen.
    ///
    /// This type is pure: no UnityEngine dependency, so the real behaviour is compiled
    /// and executed offline as reproducible evidence.
    /// </summary>
    public sealed class CommonSignalTwoTargetValidationPlan
    {
        public const string TargetGreeting = "GREETING";
        public const string TargetThankYou = "THANK_YOU";
        public const string ForbiddenTarget = "LOVE";

        static readonly string[] FixedTargets = { TargetGreeting, TargetThankYou };

        readonly bool[] begun = new bool[FixedTargets.Length];
        readonly bool[] finalized = new bool[FixedTargets.Length];
        readonly List<string> finalizedSessionIds = new List<string>();
        int index;

        public IList<string> Targets
        {
            get { return Array.AsReadOnly((string[])FixedTargets.Clone()); }
        }

        public int TargetCount { get { return FixedTargets.Length; } }
        public int FinalizedCount { get { return finalizedSessionIds.Count; } }
        public bool Completed { get { return index >= FixedTargets.Length; } }
        public string RejectionReason { get; private set; } = string.Empty;

        public IList<string> FinalizedSessionIds
        {
            get { return finalizedSessionIds.AsReadOnly(); }
        }

        /// <summary>The target awaiting a take, or empty when the plan is complete.</summary>
        public string CurrentTarget
        {
            get { return Completed ? string.Empty : FixedTargets[index]; }
        }

        public bool CanBegin
        {
            get { return !Completed && !begun[index]; }
        }

        public bool CanFinalize
        {
            get { return !Completed && begun[index] && !finalized[index]; }
        }

        /// <summary>A target this plan will never run. Present so the rule is explicit.</summary>
        public static bool IsForbiddenTarget(string target)
        {
            return string.Equals(target, ForbiddenTarget, StringComparison.Ordinal);
        }

        /// <summary>
        /// Approves beginning a capture for <paramref name="target"/>. Fails for any
        /// target other than the current one, for a second begin, and for LOVE.
        /// </summary>
        public bool TryBegin(string target)
        {
            RejectionReason = string.Empty;
            if (IsForbiddenTarget(target))
            {
                RejectionReason = "LOVE is not part of this validation and is never begun";
                return false;
            }
            if (Completed)
            {
                RejectionReason = "both targets are already finalized";
                return false;
            }
            if (!string.Equals(target, FixedTargets[index], StringComparison.Ordinal))
            {
                RejectionReason = "out of order: the current target is " + FixedTargets[index];
                return false;
            }
            if (begun[index])
            {
                RejectionReason = "this target has already been begun once; no retry";
                return false;
            }
            begun[index] = true;
            return true;
        }

        /// <summary>
        /// Approves finalizing the current target once, with a unique non-empty capture
        /// session id, and then advances to the next target.
        /// </summary>
        public bool TryFinalize(string target, string captureSessionId)
        {
            RejectionReason = string.Empty;
            if (IsForbiddenTarget(target))
            {
                RejectionReason = "LOVE is not part of this validation and is never finalized";
                return false;
            }
            if (Completed)
            {
                RejectionReason = "both targets are already finalized";
                return false;
            }
            if (!string.Equals(target, FixedTargets[index], StringComparison.Ordinal))
            {
                RejectionReason = "out of order: the current target is " + FixedTargets[index];
                return false;
            }
            if (!begun[index])
            {
                RejectionReason = "this target was never begun";
                return false;
            }
            if (finalized[index])
            {
                RejectionReason = "this target has already been finalized once; no repeat";
                return false;
            }
            if (string.IsNullOrWhiteSpace(captureSessionId))
            {
                RejectionReason = "a finalized take needs a capture session id";
                return false;
            }
            if (finalizedSessionIds.Contains(captureSessionId))
            {
                RejectionReason = "this capture session id was already finalized";
                return false;
            }

            finalized[index] = true;
            finalizedSessionIds.Add(captureSessionId);
            index++;
            return true;
        }
    }
}
