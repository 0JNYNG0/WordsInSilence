// F28 - the small production entry points that open the full-hand record store for the
// CURRENT participant and answer the one question the world gate asks: has this participant
// collected the required number of model-gate-accepted recordings for every word the CURRENT
// model actually knows?
//
// The ROUTE is owned by the gameplay opt-in alone. With WIS_FULL_HAND_GAMEPLAY=1 the full-hand
// route is selected; if the pipeline or its files are missing, this REFUSES that route and the
// legacy world/quest code is deliberately NOT selected as a fallback.
//
// There is no fake "true" here and no legacy 3/3/3 shortcut. Nothing in this file loads a
// stored record into the recognition pipeline or promotes it to a personalized reference.
using System;
using WordsInSilence.Practice.PetRobotVerticalSlice;
using WordsInSilence.Practice.PetRobotWorldIntegration;
using WordsInSilence.Recognition.FullHandRuntime;

namespace WordsInSilence.Practice.FullHandEnrollment
{
    public static class FullHandEnrollmentRuntime
    {
        // ---- narrow EditMode seam ------------------------------------------------
        //
        // EDITOR ONLY. The fields, the entry points and every branch that reads them are
        // compiled out of a player build, so a shipped build contains no test-mode override at
        // all and there is no environment variable or other runtime bypass. It exists so an
        // EditMode test can drive the REAL production decision path
        // (ProductionQuestGateVerifier -> quest gate -> store) with a real on-disk store
        // instead of a fake verifier.
#if UNITY_EDITOR
        static bool testOverrideActive;
        static bool testModeActive;
        static FullHandEnrollmentBinding testBinding;
        static FullHandParticipantCandidateStore testStore;

        public static void OverrideForTests(bool modeActive, FullHandEnrollmentBinding binding,
                                            FullHandParticipantCandidateStore store)
        {
            testOverrideActive = true;
            testModeActive = modeActive;
            testBinding = binding;
            testStore = store;
        }

        public static void ClearTestOverride()
        {
            testOverrideActive = false;
            testModeActive = false;
            testBinding = null;
            testStore = null;
        }
#endif

        // ---- route selection -----------------------------------------------------

        /// <summary>
        /// The route selector: the GAMEPLAY opt-in owns it. A missing pipeline does not fall
        /// back to the legacy route - it makes the full-hand route refuse.
        /// </summary>
        public static bool ModeActive
        {
            get
            {
#if UNITY_EDITOR
                if (testOverrideActive) return testModeActive;
#endif
                return FullHandEnrollmentBindingFactory.IsGameplayRouteRequested();
            }
        }

        /// <summary>
        /// The shared candidate root of the ACTIVE participant session. The root itself is
        /// shared across participants; the learner id segment inside it is the separation.
        /// </summary>
        public static string CandidateRootOrNull(PetRobotLearnerRootIdentity identity)
        {
            try
            {
                if (DemoParticipantSession.IsActive)
                {
                    DemoParticipantIdentity current = DemoParticipantSession.Current;
                    return current == null ? null : current.CandidateRoot;
                }
            }
            catch (Exception) { return null; }
            return identity != null && identity.HasExplicitRoot ? identity.ExplicitRoot : null;
        }

        /// <summary>
        /// Opens the store for the currently active participant, building a FRESH pinning from
        /// the operator configuration (no cache). <paramref name="identity"/> may be null; when
        /// it is supplied its learner id must be the ACTIVE participant, so a handoff from a
        /// previous participant can never open that participant's records.
        /// </summary>
        public static bool TryOpenForCurrentParticipant(
            PetRobotLearnerRootIdentity identity,
            out FullHandEnrollmentBinding binding,
            out FullHandParticipantCandidateStore store,
            out string failureCode, out string failureDetail)
        {
#if UNITY_EDITOR
            if (testOverrideActive)
            {
                binding = testBinding;
                store = testStore;
                bool ok = testModeActive && binding != null && store != null;
                failureCode = ok ? FullHandEnrollmentCodes.None
                                 : FullHandEnrollmentCodes.StoreUnavailable;
                failureDetail = ok ? string.Empty : "no test store is bound";
                if (ok && identity != null &&
                    !string.Equals(identity.LearnerId, binding.ParticipantId,
                                   StringComparison.Ordinal))
                {
                    failureCode = FullHandEnrollmentCodes.ParticipantChanged;
                    failureDetail = "the supplied identity is not the active participant";
                    binding = null;
                    store = null;
                    return false;
                }
                if (!ok) { binding = null; store = null; }
                return ok;
            }
#endif

            store = null;
            if (!FullHandEnrollmentBindingFactory.TryCreateForCurrentParticipant(
                    out binding, out failureCode, out failureDetail))
                return false;

            if (identity != null &&
                !string.Equals(identity.LearnerId, binding.ParticipantId, StringComparison.Ordinal))
            {
                failureCode = FullHandEnrollmentCodes.ParticipantChanged;
                failureDetail = "the supplied identity is not the active participant";
                binding = null;
                return false;
            }

            string root = CandidateRootOrNull(identity);
            if (string.IsNullOrWhiteSpace(root))
            {
                failureCode = FullHandEnrollmentCodes.RootRejected;
                failureDetail = "no local candidate root is available for this participant";
                binding = null;
                return false;
            }

            try
            {
                store = FullHandParticipantCandidateStore.Open(root, binding.ParticipantId, binding);
            }
            catch (FullHandFormatException exception)
            {
                failureCode = exception.Code;
                failureDetail = "the local record store was refused";
                binding = null;
                store = null;
                return false;
            }
            catch (Exception exception)
            {
                failureCode = FullHandEnrollmentCodes.StoreUnavailable;
                failureDetail = exception.GetType().Name;
                binding = null;
                store = null;
                return false;
            }

            failureCode = FullHandEnrollmentCodes.None;
            failureDetail = string.Empty;
            return true;
        }

        /// <summary>One line for a log or a technical panel. No learner-facing copy.</summary>
        public static string TechnicalSummary(FullHandParticipantCandidateStore store)
        {
            if (store == null) return "full_hand_records=<none>";
            return "full_hand_records " + store.CountSummary() +
                   " · refused=" + store.RefusedRecordCount +
                   " · quality=" + FullHandParticipantCandidateStore.QualityStatusModelGateAccepted;
        }
    }

    /// <summary>
    /// The full-hand replacement for the legacy 3/3/3 world gate. It asks the REAL store for
    /// the REAL counts of the CURRENT model vocabulary. A partially collected store, an
    /// incompletely scanned directory, a stale owner, a changed model or gate, an unreadable
    /// configuration and a missing participant all refuse.
    /// </summary>
    public static class FullHandEnrollmentQuestGate
    {
        /// <summary>
        /// The explicit form: an already opened store. It reloads first, so the counts are read
        /// from disk at the moment of the decision and never from a stale cache.
        /// </summary>
        public static bool Verify(FullHandParticipantCandidateStore store, out string failure)
        {
            if (store == null)
            {
                failure = FullHandEnrollmentCodes.StoreUnavailable;
                return false;
            }
            try { store.Reload(); }
            catch (Exception exception)
            {
                failure = FullHandEnrollmentCodes.StoreUnavailable + ": " +
                          exception.GetType().Name;
                return false;
            }
            if (!store.InventoryComplete)
            {
                failure = FullHandEnrollmentCodes.InventoryIncomplete + ": " +
                          store.LastFailureCode;
                return false;
            }
            if (!store.AllWordsCollected)
            {
                failure = "full-hand practice records incomplete: " + store.CountSummary();
                return false;
            }
            failure = string.Empty;
            return true;
        }

        /// <summary>The production form used by the world gate verifier.</summary>
        public static bool VerifyForIdentity(PetRobotLearnerRootIdentity identity,
                                             out string failure)
        {
            FullHandEnrollmentBinding binding;
            FullHandParticipantCandidateStore store;
            string code, detail;
            if (!FullHandEnrollmentRuntime.TryOpenForCurrentParticipant(
                    identity, out binding, out store, out code, out detail))
            {
                failure = code + (string.IsNullOrEmpty(detail) ? string.Empty : ": " + detail);
                return false;
            }
            return Verify(store, out failure);
        }
    }
}
