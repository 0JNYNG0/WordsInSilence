// M3_ROUTE_SOURCE_PREP_A1 (review fix) - draft, NOT installed.
//
// Player-side opt-in candidate root. Absent flag = today's behaviour, unchanged. A flag
// that is present but unusable FAILS CLOSED: no default fallback, no recognition, no
// camera, no candidate write. Validation runs once and is cached, so a blocked run never
// loops an exception per frame.
//
// The parsing and validation core takes an explicit string[] and a repository probe, so
// automated tests exercise the SAME code path the player uses. The Unity entry point only
// supplies Environment.GetCommandLineArgs().
//
// Public failures are a typed code plus a Korean sentence; no path, learner id or root
// content ever appears in them.
using System;
using System.IO;
using UnityEngine;

namespace WordsInSilence.Practice.PetRobotVerticalSlice
{
    public static class PetRobotPlayerCandidateRootOverride
    {
        public const string Flag = "-wisCandidateRoot";

        public enum RootFailure
        {
            None,
            FlagValueMissing,
            FlagDuplicated,
            PathNotFullyQualified,
            PathUnparseable,
            PathNotFound,
            RepositoryUnreadable,
            EnrollmentGateNotReady,
            LearnerIdentityMismatch
        }

        /// <summary>Validates a root against the real gate. Injected so tests use the
        /// same decision code without a Unity player.</summary>
        public delegate bool RootGateProbe(string learnerId, string root);

        public sealed class Resolution
        {
            public bool Requested;
            public string Root;
            public string LearnerId;
            public RootFailure Failure = RootFailure.None;
            public bool Allowed { get { return !Requested || Failure == RootFailure.None; } }
        }

        static Resolution cached;
        static bool reported;

        public static bool Requested { get { return cached != null && cached.Requested; } }
        public static bool Blocked
        {
            get { return cached != null && cached.Requested && cached.Failure != RootFailure.None; }
        }
        public static RootFailure Failure
        {
            get { return cached == null ? RootFailure.None : cached.Failure; }
        }

        public static string PublicFailureMessage { get { return MessageFor(Failure); } }

        public static string MessageFor(RootFailure failure)
        {
            switch (failure)
            {
                case RootFailure.None: return string.Empty;
                case RootFailure.FlagValueMissing:
                    return "데모 데이터 경로 인자가 비어 있습니다. 실행을 중단합니다.";
                case RootFailure.FlagDuplicated:
                    return "데모 데이터 경로 인자가 중복되었습니다. 실행을 중단합니다.";
                case RootFailure.PathNotFullyQualified:
                    return "데모 데이터 경로는 드라이브를 포함한 절대 경로여야 합니다. 실행을 중단합니다.";
                case RootFailure.PathUnparseable:
                    return "데모 데이터 경로 형식이 올바르지 않습니다. 실행을 중단합니다.";
                case RootFailure.PathNotFound:
                    return "지정한 데모 데이터 폴더를 찾을 수 없습니다. 실행을 중단합니다.";
                case RootFailure.RepositoryUnreadable:
                    return "데모 데이터를 읽거나 검증할 수 없습니다. 실행을 중단합니다.";
                case RootFailure.EnrollmentGateNotReady:
                    return "데모 데이터가 등록 기준을 통과하지 못했습니다. 실행을 중단합니다.";
                case RootFailure.LearnerIdentityMismatch:
                    return "학습자 식별자가 일치하지 않습니다. 실행을 중단합니다.";
                default:
                    return "데모 데이터 검증에 실패했습니다. 실행을 중단합니다.";
            }
        }

        // ---- the decision core: no Unity, no environment, fully testable
        public static Resolution Resolve(string[] arguments, string learnerProfileId,
            RootGateProbe probe)
        {
            var result = new Resolution { LearnerId = learnerProfileId };
            string value = null;
            int seen = 0;
            if (arguments != null)
            {
                for (int index = 0; index < arguments.Length; index++)
                {
                    if (!string.Equals(arguments[index], Flag, StringComparison.Ordinal)) continue;
                    seen++;
                    value = index + 1 < arguments.Length ? arguments[index + 1] : null;
                }
            }
            if (seen == 0) return result;                 // absent: unchanged default
            result.Requested = true;
            if (seen > 1) { result.Failure = RootFailure.FlagDuplicated; return result; }
            if (string.IsNullOrWhiteSpace(value)
                || value.StartsWith("-", StringComparison.Ordinal))
            { result.Failure = RootFailure.FlagValueMissing; return result; }

            // rejects "C:relative" and "\\relative" as well as plain relatives
            bool fullyQualified;
            try { fullyQualified = Path.IsPathFullyQualified(value); }
            catch (Exception) { result.Failure = RootFailure.PathUnparseable; return result; }
            if (!fullyQualified)
            { result.Failure = RootFailure.PathNotFullyQualified; return result; }

            string full;
            try { full = Path.GetFullPath(value); }
            catch (Exception) { result.Failure = RootFailure.PathUnparseable; return result; }

            bool exists;
            try { exists = Directory.Exists(full); }
            catch (Exception) { result.Failure = RootFailure.PathUnparseable; return result; }
            if (!exists) { result.Failure = RootFailure.PathNotFound; return result; }

            try
            {
                if (!probe(learnerProfileId, full))
                { result.Failure = RootFailure.EnrollmentGateNotReady; return result; }
            }
            catch (Exception)
            { result.Failure = RootFailure.RepositoryUnreadable; return result; }

            result.Root = full;
            return result;
        }

        /// <summary>The real gate: construct, reload, verify. Nothing is weakened.</summary>
        public static bool ProductionGateProbe(string learnerId, string root)
        {
            var repository = new PersonalizationCandidateRepository(learnerId, root);
            ITrustedPersonalizationCandidateRepository trusted = repository;
            TrustedCandidateInventory inventory = trusted.ReloadTrustedAndVerify();
            return inventory.OverallEnrollmentGateReady;
        }

        /// <summary>
        /// Player entry point. Returns false only in the fail-closed state; the caller
        /// must then stop before any gate, runtime, camera or save work.
        /// </summary>
        public static bool TryResolveForPlayer(string learnerProfileId, out string root)
        {
            if (DemoParticipantSession.IsActive)
            {
                // intentional empty-start route: a fresh participant has no enrollment yet, so the
                // pre-existing-enrollment requirement does not apply here. The quest's own readiness
                // gate downstream is unchanged. Mismatch or unavailable session fails closed.
                DemoParticipantIdentity demo = DemoParticipantSession.Current;
                if (!string.Equals(demo.LearnerId, learnerProfileId, StringComparison.Ordinal))
                    throw new InvalidOperationException(
                        "demo participant learner identity mismatch");
                root = demo.CandidateRoot;
                return true;
            }

            if (cached == null)
                cached = Resolve(Environment.GetCommandLineArgs(), learnerProfileId,
                    ProductionGateProbe);
            root = null;
            if (!cached.Requested) return true;
            if (cached.Failure != RootFailure.None) { ReportOnce(); return false; }
            if (!string.Equals(cached.LearnerId, learnerProfileId, StringComparison.Ordinal))
            {
                // a later consumer disagreeing is itself fail-closed, and it stays failed
                cached.Failure = RootFailure.LearnerIdentityMismatch;
                ReportOnce();
                return false;
            }
            root = cached.Root;
            return true;
        }

        static void ReportOnce()
        {
            if (reported) return;
            reported = true;
            Debug.LogError("[M3Route] " + Failure + " - " + PublicFailureMessage);
        }

#if UNITY_INCLUDE_TESTS
        public static void ResetForTests() { cached = null; reported = false; }
        public static void SeedForTests(Resolution resolution) { cached = resolution; reported = false; }
#endif
    }
}
