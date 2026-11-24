using System;
using System.Globalization;
using System.IO;
using System.Text;
using System.Text.RegularExpressions;

namespace WordsInSilence.Practice.PetRobotVerticalSlice
{
    /// <summary>
    /// Immutable identity of the currently active demo participant. A new identity is a new
    /// GUID namespace, so it is always empty by construction: nothing is copied into it and
    /// nothing is removed from any previous participant's directories.
    /// </summary>
    public sealed class DemoParticipantIdentity
    {
        internal DemoParticipantIdentity(string learnerId, string candidateRoot, string profilesRoot)
        {
            LearnerId = learnerId;
            CandidateRoot = candidateRoot;
            ProfilesRoot = profilesRoot;
        }

        public string LearnerId { get; }
        public string CandidateRoot { get; }
        public string ProfilesRoot { get; }
    }

    /// <summary>
    /// Owns exactly two things inside its dedicated root: <c>active.json</c> and its own unique
    /// staging temp files. It never enumerates, reads, copies or deletes personalization,
    /// training, AIHub or three-view source data; "reset" only means committing a new active
    /// learner id, after which previous data simply stops being addressed.
    /// </summary>
    public sealed class DemoParticipantStore
    {
        const string ManifestFileName = "active.json";
        const string LearnerIdPrefix = "demo_";
        const int SchemaVersion = 1;
        const int MaxIdAttempts = 8;
        const int MaxManifestBytes = 4096;

        /// <summary>The only accepted manifest grammar; see README. Anchored, fixed key order,
        /// schema exactly the integer 1, learnerId exactly demo_ + 32 lowercase hex.</summary>
        static readonly Regex ManifestGrammar = new Regex(
            "^\\{\\s*\"schema\"\\s*:\\s*1\\s*,\\s*\"learnerId\"\\s*:\\s*\"demo_[0-9a-f]{32}\"\\s*\\}\\s*\\z",
            RegexOptions.CultureInvariant);

        // Deliberately generic: these strings can surface in UI, so they carry no paths.
        const string ErrRoot = "demo participant storage root is not usable";
        const string ErrManifest = "demo participant manifest is not valid";
        const string ErrCommit = "demo participant identity could not be saved";
        const string ErrAllocate = "demo participant identity could not be allocated";

        readonly string root;

        public DemoParticipantStore(string explicitRoot)
        {
            if (string.IsNullOrWhiteSpace(explicitRoot)) throw new ArgumentException(ErrRoot, nameof(explicitRoot));
            // fully qualified only: this rejects "C:relative" and rooted-current-drive "\dir"
            if (!Path.IsPathFullyQualified(explicitRoot)) throw new InvalidOperationException(ErrRoot);
            foreach (string segment in explicitRoot.Split('/', '\\'))
                if (segment == "." || segment == "..") throw new InvalidOperationException(ErrRoot);

            string full;
            try { full = Path.GetFullPath(explicitRoot); }
            catch (Exception) { throw new InvalidOperationException(ErrRoot); }

            root = full.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
            if (root.Length == 0) throw new InvalidOperationException(ErrRoot);

            CandidateRoot = Path.Combine(root, "candidates");
            ProfilesRoot = Path.Combine(root, "profiles");
            RejectReparsePoints();
        }

        public string CandidateRoot { get; }
        public string ProfilesRoot { get; }

        string ManifestPath => Path.Combine(root, ManifestFileName);

        /// <summary>
        /// Returns the persisted active identity, so the same participant survives a process
        /// relaunch. A missing manifest creates a fresh identity; a corrupt one fails loudly
        /// instead of silently degrading to a legacy learner id.
        /// </summary>
        public DemoParticipantIdentity GetOrCreate()
        {
            EnsureRoot();
            string manifest = ManifestPath;
            if (!File.Exists(manifest)) return CommitNewIdentity();

            string text;
            try
            {
                var info = new FileInfo(manifest);
                if (info.Length > MaxManifestBytes) throw new InvalidOperationException(ErrManifest);
                text = File.ReadAllText(manifest, Encoding.UTF8);
            }
            catch (InvalidOperationException) { throw; }
            catch (Exception) { throw new InvalidOperationException(ErrManifest); }

            return BuildIdentity(ParseLearnerId(text));
        }

        /// <summary>Commits a brand new participant id atomically. A failed transaction leaves
        /// the previous manifest byte-identical and only removes this store's own staging file.</summary>
        public DemoParticipantIdentity StartNew()
        {
            EnsureRoot();
            return CommitNewIdentity();
        }

        DemoParticipantIdentity CommitNewIdentity()
        {
            string learnerId = AllocateFreeLearnerId();
            string staging = Path.Combine(root, "active." + Guid.NewGuid().ToString("N") + ".tmp");
            string manifest = ManifestPath;
            try
            {
                File.WriteAllText(staging, BuildManifestText(learnerId), new UTF8Encoding(false));
                if (File.Exists(manifest)) File.Replace(staging, manifest, null);
                else File.Move(staging, manifest);
            }
            catch (Exception)
            {
                TryCleanStaging(staging);
                throw new InvalidOperationException(ErrCommit);
            }
            return BuildIdentity(learnerId);
        }

        string AllocateFreeLearnerId()
        {
            for (int attempt = 0; attempt < MaxIdAttempts; attempt++)
            {
                string candidate = LearnerIdPrefix + Guid.NewGuid().ToString("N");
                bool taken;
                try
                {
                    string candidateEntry = Path.Combine(CandidateRoot, candidate);
                    taken = Directory.Exists(candidateEntry) || File.Exists(candidateEntry) ||
                            File.Exists(ProfilePathFor(candidate));
                }
                catch (Exception) { throw new InvalidOperationException(ErrAllocate); }
                if (!taken) return candidate;
            }
            throw new InvalidOperationException(ErrAllocate);
        }

        void TryCleanStaging(string staging)
        {
            try { if (File.Exists(staging)) File.Delete(staging); }
            catch (Exception) { /* best effort on our own temp file only */ }
        }

        /// <summary>Per-participant neutral calibration profile file, as written under ProfilesRoot.</summary>
        public string ProfilePathFor(string learnerId) =>
            Path.Combine(ProfilesRoot, learnerId + ".json");

        DemoParticipantIdentity BuildIdentity(string learnerId) =>
            new DemoParticipantIdentity(learnerId, CandidateRoot, ProfilesRoot);

        static string BuildManifestText(string learnerId) =>
            "{\"schema\":" + SchemaVersion.ToString(CultureInfo.InvariantCulture) +
            ",\"learnerId\":\"" + learnerId + "\"}";

        /// <summary>Accepts only the exact grammar this class writes. Anything else - duplicate
        /// keys, reordered keys, trailing junk, missing braces, schema 1.5, a legacy learner id -
        /// is corruption and fails closed.</summary>
        static string ParseLearnerId(string text)
        {
            if (text == null || !ManifestGrammar.IsMatch(text))
                throw new InvalidOperationException(ErrManifest);
            int idAt = text.IndexOf('"', text.IndexOf("\"learnerId\"", StringComparison.Ordinal) + 11) + 1;
            return text.Substring(idAt, LearnerIdPrefix.Length + 32);
        }

        void EnsureRoot()
        {
            RejectReparsePoints();
            try
            {
                Directory.CreateDirectory(root);
                Directory.CreateDirectory(CandidateRoot);
                Directory.CreateDirectory(ProfilesRoot);
            }
            catch (Exception) { throw new InvalidOperationException(ErrRoot); }
            RejectReparsePoints();
        }

        /// <summary>Refuses to operate under a junction/symlink, so a redirected ancestor or an
        /// own directory can never turn "write my own manifest" into a write somewhere else.
        /// Only the store's own three directories and their ancestors are inspected.</summary>
        void RejectReparsePoints()
        {
            RejectReparsePointChain(root);
            RejectReparsePointChain(CandidateRoot);
            RejectReparsePointChain(ProfilesRoot);
            RejectReparsePointFile(ManifestPath);
        }

        static void RejectReparsePointChain(string path)
        {
            try
            {
                for (var dir = new DirectoryInfo(path); dir != null; dir = dir.Parent)
                {
                    if (!dir.Exists) continue;
                    if ((dir.Attributes & FileAttributes.ReparsePoint) != 0)
                        throw new InvalidOperationException(ErrRoot);
                }
            }
            catch (InvalidOperationException) { throw; }
            catch (Exception) { throw new InvalidOperationException(ErrRoot); }
        }

        static void RejectReparsePointFile(string path)
        {
            try
            {
                var file = new FileInfo(path);
                if (file.Exists && (file.Attributes & FileAttributes.ReparsePoint) != 0)
                    throw new InvalidOperationException(ErrRoot);
            }
            catch (InvalidOperationException) { throw; }
            catch (Exception) { throw new InvalidOperationException(ErrRoot); }
        }
    }

    /// <summary>
    /// Process-wide active participant. Nothing activates this by default: the legacy default
    /// path is untouched until <see cref="DemoParticipantSessionHost"/> (or a test) activates it.
    /// Once activation has been requested, a failure never falls back to the legacy identity -
    /// <see cref="Current"/> throws and every caller fails closed.
    /// </summary>
    public static class DemoParticipantSession
    {
        static readonly object gate = new object();
        static bool activationRequested;
        static DemoParticipantStore store;
        static DemoParticipantIdentity current;

        /// <summary>True as soon as activation is attempted, including a failed attempt.</summary>
        public static bool IsActive
        {
            get { lock (gate) return activationRequested; }
        }

        public static DemoParticipantIdentity Current
        {
            get
            {
                lock (gate)
                {
                    if (current == null)
                        throw new InvalidOperationException("demo participant session is not available");
                    return current;
                }
            }
        }

        /// <summary>Profiles root of the active participant. Null <em>only</em> when the session was
        /// never activated; if activation was requested but the identity is unavailable this throws,
        /// because a null here would silently mean "use the legacy profiles location".</summary>
        public static string ProfilesRootOrNull
        {
            get
            {
                lock (gate)
                {
                    if (!activationRequested) return null;
                    if (current == null)
                        throw new InvalidOperationException("demo participant session is not available");
                    return current.ProfilesRoot;
                }
            }
        }

        public static void Activate(string root)
        {
            lock (gate)
            {
                if (activationRequested && current != null) return;
                activationRequested = true;
                var opened = new DemoParticipantStore(root);
                DemoParticipantIdentity identity = opened.GetOrCreate();
                store = opened;
                current = identity;
            }
        }

        /// <summary>Commits the new identity to disk first; <see cref="Current"/> only moves once
        /// the commit succeeded, so a storage failure leaves the running participant intact.</summary>
        public static DemoParticipantIdentity StartNew()
        {
            lock (gate)
            {
                if (store == null)
                    throw new InvalidOperationException("demo participant session is not available");
                DemoParticipantIdentity next = store.StartNew();
                current = next;
                return next;
            }
        }

#if UNITY_5_3_OR_NEWER
        [UnityEngine.RuntimeInitializeOnLoadMethod(
            UnityEngine.RuntimeInitializeLoadType.SubsystemRegistration)]
        static void ClearForNewPlayDomain() => ResetStaticForTests();
#endif

        /// <summary>Test/domain-reload seam only. Public because the edit-mode tests live in a
        /// separate assembly; it performs no I/O and touches no participant data on disk.</summary>
        public static void ResetStaticForTests()
        {
            lock (gate)
            {
                activationRequested = false;
                store = null;
                current = null;
            }
        }
    }
}
