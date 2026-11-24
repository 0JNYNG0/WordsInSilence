#if UNITY_EDITOR
using System;
using System.IO;
using UnityEditor;
using UnityEngine;

namespace WordsInSilence.Practice.PetRobotVerticalSlice
{
    public enum DeveloperCandidateRootArmMode
    {
        None,
        EmptyClean,
        GateReadyExisting,
    }

    /// <summary>
    /// Editor-only, one-shot candidate-root override used for physical enrollment validation.
    /// It cannot be armed or resolved in a player build, and it never owns the neutral profile.
    /// Supports two modes: EmptyClean (Step 10K) and GateReadyExisting (Step 10M).
    /// </summary>
    [InitializeOnLoad]
    public static class PetRobotDeveloperCleanCandidateRootOverride
    {
        public const string LogPrefix = "[STEP10K_V2_2_CLEAN_ROOT]";

        const string SessionKeyPrefix = "WordsInSilence.Step10K.V2_2.CleanCandidateRoot.";
        const string ArmedRootKey = SessionKeyPrefix + "Armed";
        const string ActiveRootKey = SessionKeyPrefix + "Active";
        const string LastCompletedRootKey = SessionKeyPrefix + "LastCompleted";
        const string ArmedModeKey = SessionKeyPrefix + "ArmedMode";
        const string ArmMenu =
            "Tools/WordsInSilence/Developer Validation/Step 10K v2.2/Arm New Clean Candidate Root";
        const string StatusMenu =
            "Tools/WordsInSilence/Developer Validation/Step 10K v2.2/Log Candidate Root State";
        const string GateReadyArmMenu =
            "Tools/WordsInSilence/Developer Validation/Step 10M/Arm Existing Gate-Ready Candidate Root...";

        static PetRobotDeveloperCleanCandidateRootOverride()
        {
            EditorApplication.playModeStateChanged -= OnPlayModeStateChanged;
            EditorApplication.playModeStateChanged += OnPlayModeStateChanged;

            if (!EditorApplication.isPlayingOrWillChangePlaymode &&
                !string.IsNullOrEmpty(SessionState.GetString(ActiveRootKey, string.Empty)))
                DisarmAfterPlayMode("editor_domain_reload");
        }

        public static bool IsArmed =>
            !string.IsNullOrEmpty(SessionState.GetString(ArmedRootKey, string.Empty));

        public static bool HasActivePlayModeRoot =>
            !string.IsNullOrEmpty(SessionState.GetString(ActiveRootKey, string.Empty));

        public static string LastCompletedRoot =>
            SessionState.GetString(LastCompletedRootKey, string.Empty);

        public static string CurrentArmedMode =>
            SessionState.GetString(ArmedModeKey, DeveloperCandidateRootArmMode.None.ToString());

        public static string ResolveForBootstrap()
        {
            string activeRoot = SessionState.GetString(ActiveRootKey, string.Empty);
            if (string.IsNullOrEmpty(activeRoot))
            {
                if (EditorApplication.isPlaying && IsArmed)
                    throw new InvalidOperationException(
                        "developer candidate root was armed but not activated for this PlayMode session");
                return null;
            }

            string canonicalRoot;
            string failure;
            if (!DeveloperCleanCandidateRootPathPolicy.TryValidate(
                    activeRoot,
                    PersonalizationCandidateRepository.DefaultRootDirectory,
                    Application.dataPath,
                    false,
                    out canonicalRoot,
                    out failure))
                throw new InvalidOperationException(
                    "active developer candidate root failed safety validation: " + failure);

            string armedModeStr = SessionState.GetString(ArmedModeKey, string.Empty);
            if (armedModeStr == DeveloperCandidateRootArmMode.GateReadyExisting.ToString())
            {
                string gateFailure = ValidateGateReadyRoot(canonicalRoot);
                if (!string.IsNullOrEmpty(gateFailure))
                    throw new InvalidOperationException(
                        "gate-ready bootstrap re-validation failed: " + gateFailure);
            }
            else if (armedModeStr != DeveloperCandidateRootArmMode.EmptyClean.ToString())
            {
                throw new InvalidOperationException(
                    "active developer candidate root has unknown or missing mode: " + armedModeStr);
            }

            return canonicalRoot;
        }

        public static bool TryArmExistingEmptyRoot(string candidateRoot, out string failure)
        {
            failure = string.Empty;
            if (EditorApplication.isPlayingOrWillChangePlaymode)
            {
                failure = "candidate root can only be armed while the Editor is idle in Edit Mode";
                return false;
            }

            string canonicalRoot;
            if (!DeveloperCleanCandidateRootPathPolicy.TryValidate(
                    candidateRoot,
                    PersonalizationCandidateRepository.DefaultRootDirectory,
                    Application.dataPath,
                    true,
                    out canonicalRoot,
                    out failure))
                return false;

            SessionState.SetString(ArmedRootKey, canonicalRoot);
            SessionState.EraseString(ActiveRootKey);
            SessionState.SetString(ArmedModeKey, DeveloperCandidateRootArmMode.EmptyClean.ToString());
            Debug.Log(LogPrefix + " armed root=" + canonicalRoot +
                " mode=EmptyClean trusted_start=0/0/0");
            return true;
        }

        public static bool TryArmExistingGateReadyRoot(string candidateRoot, out string failure)
        {
            return TryArmExistingGateReadyRoot(candidateRoot, out failure, out _);
        }

        internal static bool TryArmExistingGateReadyRoot(
            string candidateRoot, out string failure, out TrustedCandidateInventory outInventory)
        {
            failure = string.Empty;
            outInventory = null;
            if (EditorApplication.isPlayingOrWillChangePlaymode)
            {
                failure = "candidate root can only be armed while the Editor is idle in Edit Mode";
                return false;
            }

            string canonicalRoot;
            if (!DeveloperCleanCandidateRootPathPolicy.TryValidate(
                    candidateRoot,
                    PersonalizationCandidateRepository.DefaultRootDirectory,
                    Application.dataPath,
                    false,
                    out canonicalRoot,
                    out failure))
            {
                SessionState.EraseString(ArmedRootKey);
                SessionState.EraseString(ActiveRootKey);
                SessionState.EraseString(ArmedModeKey);
                return false;
            }

            TrustedCandidateInventory inventory;
            string gateFailure = ValidateGateReadyRoot(canonicalRoot, out inventory);
            if (!string.IsNullOrEmpty(gateFailure))
            {
                failure = "gate-ready validation failed: " + gateFailure;
                SessionState.EraseString(ArmedRootKey);
                SessionState.EraseString(ActiveRootKey);
                SessionState.EraseString(ArmedModeKey);
                return false;
            }

            outInventory = inventory;
            SessionState.SetString(ArmedRootKey, canonicalRoot);
            SessionState.EraseString(ActiveRootKey);
            SessionState.SetString(ArmedModeKey,
                DeveloperCandidateRootArmMode.GateReadyExisting.ToString());
            Debug.Log(LogPrefix + " armed root=" + canonicalRoot +
                " mode=GateReadyExisting");
            return true;
        }

        internal static string ValidateGateReadyRoot(string canonicalRoot)
        {
            TrustedCandidateInventory unused;
            return ValidateGateReadyRoot(canonicalRoot, out unused);
        }

        internal static string ValidateGateReadyRoot(
            string canonicalRoot, out TrustedCandidateInventory outInventory)
        {
            outInventory = null;
            try
            {
                var repo = new PersonalizationCandidateRepository("local_learner", canonicalRoot);
                TrustedCandidateInventory inventory = repo.ReloadTrustedAndVerify();
                if (!inventory.OverallEnrollmentGateReady)
                    return "OverallEnrollmentGateReady=false G=" +
                        inventory.CountForWord("GREETING") + " L=" +
                        inventory.CountForWord("LOVE") + " T=" +
                        inventory.CountForWord("THANK_YOU");
                int g = inventory.CountForWord("GREETING");
                int l = inventory.CountForWord("LOVE");
                int t = inventory.CountForWord("THANK_YOU");
                if (g < 3 || l < 3 || t < 3)
                    return "individual trusted count below minimum: G=" +
                        g + " L=" + l + " T=" + t;
                outInventory = inventory;
                return null;
            }
            catch (Exception exception)
            {
                return exception.GetType().Name + ": " + exception.Message;
            }
        }

        [MenuItem(ArmMenu)]
        static void ArmNewCleanCandidateRoot()
        {
            string projectRoot = Directory.GetParent(Application.dataPath).FullName;
            string validationRoot = Path.Combine(projectRoot, "Validation");
            string uniqueName = "WordsInSilence_step10k_v2_2_clean_enrollment_" +
                DateTime.UtcNow.ToString("yyyyMMddTHHmmssfffZ") + "_" +
                Guid.NewGuid().ToString("N").Substring(0, 12);
            string candidateRoot = Path.Combine(validationRoot, uniqueName);
            Directory.CreateDirectory(candidateRoot);

            string failure;
            if (!TryArmExistingEmptyRoot(candidateRoot, out failure))
            {
                Debug.LogError(LogPrefix + " arm_rejected root=" + candidateRoot + " reason=" + failure);
                return;
            }

            Selection.activeObject = AssetDatabase.LoadAssetAtPath<UnityEngine.Object>("Assets");
            Debug.Log(LogPrefix + " next PlayMode is armed; root remains evidence and will not be deleted");
        }

        [MenuItem(ArmMenu, true)]
        static bool CanArmNewCleanCandidateRoot() =>
            !EditorApplication.isPlayingOrWillChangePlaymode;

        [MenuItem(GateReadyArmMenu)]
        static void ArmExistingGateReadyRoot()
        {
            string defaultBrowsePath = Path.Combine(
                Directory.GetParent(Application.dataPath).FullName, "Validation");
            string selected = EditorUtility.OpenFolderPanel(
                "Select Gate-Ready Candidate Root", defaultBrowsePath, string.Empty);
            if (string.IsNullOrEmpty(selected)) return;

            string failure;
            TrustedCandidateInventory inventory;
            if (!TryArmExistingGateReadyRoot(selected, out failure, out inventory))
            {
                Debug.LogError(LogPrefix + " gate_ready_arm_rejected root=" +
                    selected + " reason=" + failure);
                return;
            }

            Debug.Log(LogPrefix + " gate_ready_armed root=" +
                Path.GetFullPath(selected) +
                " mode=GateReadyExisting" +
                " trusted G=" + inventory.CountForWord("GREETING") +
                " L=" + inventory.CountForWord("LOVE") +
                " T=" + inventory.CountForWord("THANK_YOU") +
                " Gate=" + inventory.OverallEnrollmentGateReady);
        }

        [MenuItem(GateReadyArmMenu, true)]
        static bool CanArmExistingGateReadyRoot() =>
            !EditorApplication.isPlayingOrWillChangePlaymode;

        [MenuItem(StatusMenu)]
        static void LogCandidateRootState()
        {
            Debug.Log(
                LogPrefix +
                " state armed=" + SessionState.GetString(ArmedRootKey, string.Empty) +
                " active=" + SessionState.GetString(ActiveRootKey, string.Empty) +
                " mode=" + SessionState.GetString(ArmedModeKey, string.Empty) +
                " last_completed=" + LastCompletedRoot);
        }

        static void OnPlayModeStateChanged(PlayModeStateChange change)
        {
            if (change == PlayModeStateChange.ExitingEditMode)
            {
                string armedRoot = SessionState.GetString(ArmedRootKey, string.Empty);
                string armedModeStr = SessionState.GetString(ArmedModeKey, string.Empty);
                SessionState.EraseString(ActiveRootKey);
                if (string.IsNullOrEmpty(armedRoot))
                {
                    Debug.Log(LogPrefix + " session_begin mode=production_default");
                    return;
                }

                bool isGateReady = armedModeStr ==
                    DeveloperCandidateRootArmMode.GateReadyExisting.ToString();
                bool isEmptyClean = armedModeStr ==
                    DeveloperCandidateRootArmMode.EmptyClean.ToString();

                if (!isGateReady && !isEmptyClean)
                {
                    SessionState.EraseString(ArmedRootKey);
                    SessionState.EraseString(ActiveRootKey);
                    SessionState.EraseString(ArmedModeKey);
                    Debug.LogError(LogPrefix +
                        " session_blocked reason=unknown_or_missing_mode mode=" + armedModeStr);
                    EditorApplication.isPlaying = false;
                    return;
                }

                bool requireEmpty = isEmptyClean;

                string canonicalRoot;
                string failure;
                if (!DeveloperCleanCandidateRootPathPolicy.TryValidate(
                        armedRoot,
                        PersonalizationCandidateRepository.DefaultRootDirectory,
                        Application.dataPath,
                        requireEmpty,
                        out canonicalRoot,
                        out failure))
                {
                    SessionState.EraseString(ArmedRootKey);
                    SessionState.EraseString(ActiveRootKey);
                    SessionState.EraseString(ArmedModeKey);
                    Debug.LogError(LogPrefix + " session_blocked reason=" + failure);
                    EditorApplication.isPlaying = false;
                    return;
                }

                if (isGateReady)
                {
                    string gateFailure = ValidateGateReadyRoot(canonicalRoot);
                    if (!string.IsNullOrEmpty(gateFailure))
                    {
                        SessionState.EraseString(ArmedRootKey);
                        SessionState.EraseString(ActiveRootKey);
                        SessionState.EraseString(ArmedModeKey);
                        Debug.LogError(LogPrefix +
                            " session_blocked_gate_failed reason=" + gateFailure);
                        EditorApplication.isPlaying = false;
                        return;
                    }
                }

                SessionState.SetString(ActiveRootKey, canonicalRoot);
                SessionState.EraseString(ArmedRootKey);
                Debug.Log(LogPrefix + " session_begin mode=" +
                    (isGateReady ? "developer_gate_ready" : "developer_clean") +
                    " root=" + canonicalRoot);
            }
            else if (change == PlayModeStateChange.EnteredPlayMode)
            {
                string activeRoot = SessionState.GetString(ActiveRootKey, string.Empty);
                string modeStr = SessionState.GetString(ArmedModeKey, string.Empty);
                Debug.Log(LogPrefix + (string.IsNullOrEmpty(activeRoot)
                    ? " entered_playmode mode=production_default"
                    : " entered_playmode mode=" + modeStr + " root=" + activeRoot));
            }
            else if (change == PlayModeStateChange.EnteredEditMode)
            {
                DisarmAfterPlayMode("entered_edit_mode");
            }
        }

        static void DisarmAfterPlayMode(string reason)
        {
            string activeRoot = SessionState.GetString(ActiveRootKey, string.Empty);
            if (!string.IsNullOrEmpty(activeRoot))
                SessionState.SetString(LastCompletedRootKey, activeRoot);
            SessionState.EraseString(ActiveRootKey);
            SessionState.EraseString(ArmedRootKey);
            SessionState.EraseString(ArmedModeKey);
            Debug.Log(LogPrefix + " session_end disarmed=true reason=" + reason +
                " completed_root=" + activeRoot);
        }
    }

    /// <summary>Pure state model used to prove one-shot semantics without entering PlayMode.</summary>
    public sealed class DeveloperCleanCandidateRootSession
    {
        string armedRoot;
        string activeRoot;
        DeveloperCandidateRootArmMode mode;

        public bool IsArmed => !string.IsNullOrEmpty(armedRoot);
        public bool HasActiveRoot => !string.IsNullOrEmpty(activeRoot);
        public string ActiveRoot => activeRoot ?? string.Empty;
        public DeveloperCandidateRootArmMode ArmMode => mode;

        public bool TryArm(
            string candidateRoot,
            string productionRoot,
            string assetsRoot,
            out string failure)
        {
            string canonicalRoot;
            if (!DeveloperCleanCandidateRootPathPolicy.TryValidate(
                    candidateRoot, productionRoot, assetsRoot, true,
                    out canonicalRoot, out failure))
                return false;
            if (HasActiveRoot)
            {
                failure = "an active PlayMode session cannot be re-armed";
                return false;
            }
            armedRoot = canonicalRoot;
            mode = DeveloperCandidateRootArmMode.EmptyClean;
            return true;
        }

        public bool TryArmGateReady(
            string candidateRoot,
            string productionRoot,
            string assetsRoot,
            Func<string, string> gateVerifier,
            out string failure)
        {
            failure = string.Empty;
            string canonicalRoot;
            if (!DeveloperCleanCandidateRootPathPolicy.TryValidate(
                    candidateRoot, productionRoot, assetsRoot, false,
                    out canonicalRoot, out failure))
            {
                armedRoot = null;
                mode = DeveloperCandidateRootArmMode.None;
                return false;
            }
            if (HasActiveRoot)
            {
                failure = "an active PlayMode session cannot be re-armed";
                return false;
            }
            if (gateVerifier == null)
            {
                failure = "gate verifier is required for gate-ready mode";
                armedRoot = null;
                mode = DeveloperCandidateRootArmMode.None;
                return false;
            }
            string gateFailure = gateVerifier(canonicalRoot);
            if (!string.IsNullOrEmpty(gateFailure))
            {
                failure = "gate-ready validation failed: " + gateFailure;
                armedRoot = null;
                mode = DeveloperCandidateRootArmMode.None;
                return false;
            }
            armedRoot = canonicalRoot;
            mode = DeveloperCandidateRootArmMode.GateReadyExisting;
            return true;
        }

        public bool BeginNextPlayMode(out string root)
        {
            if (!IsArmed)
            {
                activeRoot = null;
                root = string.Empty;
                return false;
            }
            activeRoot = armedRoot;
            armedRoot = null;
            root = activeRoot;
            return true;
        }

        public bool BeginNextPlayMode(
            string productionRoot,
            string assetsRoot,
            Func<string, string> gateVerifier,
            out string root,
            out string failure)
        {
            failure = string.Empty;
            if (!IsArmed)
            {
                activeRoot = null;
                root = string.Empty;
                return false;
            }

            bool isGateReady = mode == DeveloperCandidateRootArmMode.GateReadyExisting;
            bool isEmptyClean = mode == DeveloperCandidateRootArmMode.EmptyClean;

            if (!isGateReady && !isEmptyClean)
            {
                failure = "unknown or missing arm mode: " + mode;
                armedRoot = null;
                activeRoot = null;
                mode = DeveloperCandidateRootArmMode.None;
                root = string.Empty;
                return false;
            }

            bool requireEmpty = isEmptyClean;

            string canonicalRoot;
            if (!DeveloperCleanCandidateRootPathPolicy.TryValidate(
                    armedRoot, productionRoot, assetsRoot, requireEmpty,
                    out canonicalRoot, out failure))
            {
                armedRoot = null;
                mode = DeveloperCandidateRootArmMode.None;
                root = string.Empty;
                return false;
            }

            if (isGateReady)
            {
                if (gateVerifier == null)
                {
                    failure = "gate verifier is required for gate-ready re-validation";
                    armedRoot = null;
                    mode = DeveloperCandidateRootArmMode.None;
                    root = string.Empty;
                    return false;
                }
                string gateFailure = gateVerifier(canonicalRoot);
                if (!string.IsNullOrEmpty(gateFailure))
                {
                    failure = "gate-ready re-validation failed: " + gateFailure;
                    armedRoot = null;
                    mode = DeveloperCandidateRootArmMode.None;
                    root = string.Empty;
                    return false;
                }
            }

            activeRoot = canonicalRoot;
            armedRoot = null;
            root = activeRoot;
            return true;
        }

        public string ResolveCandidateRoot(string productionDefaultRoot) =>
            HasActiveRoot ? activeRoot : productionDefaultRoot;

        public string ResolveCandidateRoot(
            string productionDefaultRoot, Func<string, string> gateVerifier)
        {
            if (!HasActiveRoot) return productionDefaultRoot;

            if (mode == DeveloperCandidateRootArmMode.GateReadyExisting)
            {
                if (gateVerifier == null)
                    throw new InvalidOperationException(
                        "gate verifier is required for gate-ready resolve");
                string gateFailure = gateVerifier(activeRoot);
                if (!string.IsNullOrEmpty(gateFailure))
                    throw new InvalidOperationException(
                        "gate-ready resolve re-validation failed: " + gateFailure);
            }
            else if (mode != DeveloperCandidateRootArmMode.EmptyClean)
            {
                throw new InvalidOperationException(
                    "active root has unknown or missing mode: " + mode);
            }

            return activeRoot;
        }

        public string EndPlayMode()
        {
            string completedRoot = activeRoot ?? string.Empty;
            activeRoot = null;
            armedRoot = null;
            mode = DeveloperCandidateRootArmMode.None;
            return completedRoot;
        }
    }

    public static class DeveloperCleanCandidateRootPathPolicy
    {
        public static bool TryValidate(
            string candidateRoot,
            string productionRoot,
            string assetsRoot,
            bool requireEmpty,
            out string canonicalRoot,
            out string failure)
        {
            return TryValidate(
                candidateRoot, productionRoot, assetsRoot, requireEmpty,
                IsReparsePoint, out canonicalRoot, out failure);
        }

        public static bool TryValidate(
            string candidateRoot,
            string productionRoot,
            string assetsRoot,
            bool requireEmpty,
            Func<string, bool> reparsePointProbe,
            out string canonicalRoot,
            out string failure)
        {
            canonicalRoot = string.Empty;
            failure = string.Empty;
            if (string.IsNullOrWhiteSpace(candidateRoot) ||
                string.IsNullOrWhiteSpace(productionRoot) ||
                string.IsNullOrWhiteSpace(assetsRoot))
                return Fail("candidate, production, and Assets roots are required", out failure);

            try
            {
                canonicalRoot = Canonical(candidateRoot);
                string canonicalProduction = Canonical(productionRoot);
                string canonicalAssets = Canonical(assetsRoot);

                if (IsSameOrDescendant(canonicalRoot, canonicalProduction) ||
                    IsSameOrDescendant(canonicalProduction, canonicalRoot))
                    return Fail("candidate root cannot equal, contain, or be contained by production root", out failure);
                if (IsSameOrDescendant(canonicalRoot, canonicalAssets))
                    return Fail("candidate root cannot be inside project Assets", out failure);
                if (File.Exists(canonicalRoot))
                    return Fail("candidate root resolves to a file", out failure);

                DirectoryInfo cursor = Directory.Exists(canonicalRoot)
                    ? new DirectoryInfo(canonicalRoot)
                    : new DirectoryInfo(canonicalRoot).Parent;
                while (cursor != null)
                {
                    if (cursor.Exists && reparsePointProbe(cursor.FullName))
                        return Fail("candidate root path traverses a symbolic link, junction, or reparse point", out failure);
                    cursor = cursor.Parent;
                }

                if (requireEmpty && Directory.Exists(canonicalRoot) &&
                    Directory.EnumerateFileSystemEntries(canonicalRoot).GetEnumerator().MoveNext())
                    return Fail("candidate root must not exist or must be completely empty", out failure);
                return true;
            }
            catch (Exception exception)
            {
                canonicalRoot = string.Empty;
                return Fail(exception.GetType().Name + ": " + exception.Message, out failure);
            }
        }

        static bool IsReparsePoint(string path) =>
            (File.GetAttributes(path) & FileAttributes.ReparsePoint) != 0;

        static string Canonical(string path)
        {
            string full = Path.GetFullPath(path);
            string root = Path.GetPathRoot(full);
            return string.Equals(full, root, StringComparison.OrdinalIgnoreCase)
                ? full
                : full.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        }

        static bool IsSameOrDescendant(string path, string possibleAncestor)
        {
            if (string.Equals(path, possibleAncestor, StringComparison.OrdinalIgnoreCase)) return true;
            string prefix = possibleAncestor.TrimEnd(
                Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar) + Path.DirectorySeparatorChar;
            return path.StartsWith(prefix, StringComparison.OrdinalIgnoreCase);
        }

        static bool Fail(string message, out string failure)
        {
            failure = message;
            return false;
        }
    }
}
#endif
