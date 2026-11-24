using System;
using System.IO;
using NUnit.Framework;
using UnityEngine;
using WordsInSilence.Practice.PetRobotVerticalSlice;

namespace WordsInSilence.Tests.EditMode
{
    public sealed class DeveloperCleanCandidateRootOverrideV1Tests
    {
        // ── Existing EmptyClean tests (unchanged) ──────────────────────────

        [Test]
        public void UnarmedStateResolvesProductionDefault()
        {
            var state = new DeveloperCleanCandidateRootSession();
            string productionRoot = Path.Combine(Path.GetTempPath(), "wis_production_default");

            Assert.IsFalse(state.IsArmed);
            Assert.IsFalse(state.HasActiveRoot);
            Assert.AreEqual(productionRoot, state.ResolveCandidateRoot(productionRoot));
        }

        [TestCase(false)]
        [TestCase(true)]
        public void ExplicitArmAllowsOnlyValidMissingOrEmptyExternalRoot(bool precreateCandidateRoot)
        {
            string testRoot = NewTestRoot();
            try
            {
                string productionRoot = Path.Combine(testRoot, "production", "default");
                string assetsRoot = Path.Combine(testRoot, "project", "Assets");
                string candidateRoot = Path.Combine(testRoot, "validation", "clean");
                Directory.CreateDirectory(productionRoot);
                Directory.CreateDirectory(assetsRoot);
                if (precreateCandidateRoot) Directory.CreateDirectory(candidateRoot);

                var state = new DeveloperCleanCandidateRootSession();
                string failure;
                Assert.IsTrue(state.TryArm(candidateRoot, productionRoot, assetsRoot, out failure), failure);
                Assert.IsTrue(state.IsArmed);
                Assert.IsFalse(state.HasActiveRoot);
            }
            finally
            {
                DeleteTestRoot(testRoot);
            }
        }

        [TestCase("same")]
        [TestCase("ancestor")]
        [TestCase("descendant")]
        [TestCase("assets")]
        [TestCase("nonempty")]
        [TestCase("reparse")]
        public void UnsafeProductionAssetsNonemptyAndReparseRootsAreRejected(string scenario)
        {
            string testRoot = NewTestRoot();
            try
            {
                string productionRoot = Path.Combine(testRoot, "production", "default");
                string assetsRoot = Path.Combine(testRoot, "project", "Assets");
                Directory.CreateDirectory(productionRoot);
                Directory.CreateDirectory(assetsRoot);

                string candidateRoot;
                switch (scenario)
                {
                    case "same": candidateRoot = productionRoot; break;
                    case "ancestor": candidateRoot = Directory.GetParent(productionRoot).FullName; break;
                    case "descendant": candidateRoot = Path.Combine(productionRoot, "child"); break;
                    case "assets": candidateRoot = Path.Combine(assetsRoot, "validation"); break;
                    default: candidateRoot = Path.Combine(testRoot, "validation", scenario); break;
                }
                Directory.CreateDirectory(candidateRoot);
                if (scenario == "nonempty")
                    File.WriteAllText(Path.Combine(candidateRoot, "existing.txt"), "occupied");

                string canonical;
                string failure;
                Func<string, bool> reparseProbe = path => scenario == "reparse" &&
                    string.Equals(Path.GetFullPath(path), Path.GetFullPath(candidateRoot),
                        StringComparison.OrdinalIgnoreCase);
                Assert.IsFalse(DeveloperCleanCandidateRootPathPolicy.TryValidate(
                    candidateRoot, productionRoot, assetsRoot, true, reparseProbe,
                    out canonical, out failure));
                Assert.IsNotEmpty(failure);
            }
            finally
            {
                DeleteTestRoot(testRoot);
            }
        }

        [Test]
        public void ActivePlayModeSessionRetainsSameRootAcrossBootstrapRecreation()
        {
            string testRoot = NewTestRoot();
            try
            {
                string productionRoot = Path.Combine(testRoot, "production", "default");
                string assetsRoot = Path.Combine(testRoot, "project", "Assets");
                string candidateRoot = Path.Combine(testRoot, "validation", "clean");
                Directory.CreateDirectory(productionRoot);
                Directory.CreateDirectory(assetsRoot);
                Directory.CreateDirectory(candidateRoot);

                var state = new DeveloperCleanCandidateRootSession();
                string failure;
                Assert.IsTrue(state.TryArm(candidateRoot, productionRoot, assetsRoot, out failure), failure);
                string activated;
                Assert.IsTrue(state.BeginNextPlayMode(out activated));

                Assert.AreEqual(activated, state.ResolveCandidateRoot(productionRoot));
                Assert.AreEqual(activated, state.ResolveCandidateRoot(productionRoot));
                Assert.IsTrue(state.HasActiveRoot);
                Assert.IsFalse(state.IsArmed);
            }
            finally
            {
                DeleteTestRoot(testRoot);
            }
        }

        [Test]
        public void EndingPlayModeDisarmsOneShotAndNextUnarmedSessionUsesDefault()
        {
            string testRoot = NewTestRoot();
            try
            {
                string productionRoot = Path.Combine(testRoot, "production", "default");
                string assetsRoot = Path.Combine(testRoot, "project", "Assets");
                string candidateRoot = Path.Combine(testRoot, "validation", "clean");
                Directory.CreateDirectory(productionRoot);
                Directory.CreateDirectory(assetsRoot);
                Directory.CreateDirectory(candidateRoot);

                var state = new DeveloperCleanCandidateRootSession();
                string failure;
                Assert.IsTrue(state.TryArm(candidateRoot, productionRoot, assetsRoot, out failure), failure);
                string activated;
                Assert.IsTrue(state.BeginNextPlayMode(out activated));
                Assert.AreEqual(activated, state.EndPlayMode());
                Assert.IsFalse(state.IsArmed);
                Assert.IsFalse(state.HasActiveRoot);
                Assert.AreEqual(productionRoot, state.ResolveCandidateRoot(productionRoot));
                string unused;
                Assert.IsFalse(state.BeginNextPlayMode(out unused));
            }
            finally
            {
                DeleteTestRoot(testRoot);
            }
        }

        [Test]
        public void CandidateOverrideDoesNotChangeNeutralProfilePathOrLearnerIdentity()
        {
            string testRoot = NewTestRoot();
            try
            {
                const string learnerId = "local_learner";
                string productionRoot = Path.Combine(testRoot, "production", "default");
                string assetsRoot = Path.Combine(testRoot, "project", "Assets");
                string candidateRoot = Path.Combine(testRoot, "validation", "clean");
                string profileRoot = Path.Combine(testRoot, "profiles");
                Directory.CreateDirectory(productionRoot);
                Directory.CreateDirectory(assetsRoot);
                Directory.CreateDirectory(candidateRoot);

                var profileBefore = new PetRobotNeutralCalibrationProfileService(learnerId, profileRoot);
                var state = new DeveloperCleanCandidateRootSession();
                string failure;
                Assert.IsTrue(state.TryArm(candidateRoot, productionRoot, assetsRoot, out failure), failure);
                string activated;
                Assert.IsTrue(state.BeginNextPlayMode(out activated));
                var profileDuring = new PetRobotNeutralCalibrationProfileService(learnerId, profileRoot);

                Assert.AreNotEqual(productionRoot, state.ResolveCandidateRoot(productionRoot));
                Assert.AreEqual(profileBefore.ProfilePath, profileDuring.ProfilePath);
                Assert.AreEqual(profileBefore.LearnerProfileId, profileDuring.LearnerProfileId);
                Assert.AreEqual(learnerId, profileDuring.LearnerProfileId);
            }
            finally
            {
                DeleteTestRoot(testRoot);
            }
        }

        [Test]
        public void SeamIsEditorOnlyAndAddsNoPlayerFacingResetProfileOrClearSurface()
        {
            string helper = Source("PetRobotDeveloperCleanCandidateRootOverride.cs");
            string bootstrap = Source("PetRobotPracticeVerticalSliceBootstrap.cs");

            StringAssert.StartsWith("#if UNITY_EDITOR", helper.TrimStart('\ufeff'));
            StringAssert.Contains("[MenuItem(ArmMenu)]", helper);
            StringAssert.Contains("[MenuItem(GateReadyArmMenu)]", helper);
            StringAssert.Contains("Developer Validation", helper);
            StringAssert.DoesNotContain("UnityEngine.UI", helper);
            StringAssert.DoesNotContain("Button", helper);
            StringAssert.DoesNotContain("SceneManager.LoadScene", helper);
            StringAssert.DoesNotContain("새로 시작하기", helper);
            StringAssert.DoesNotContain("사용자 선택", helper);
            StringAssert.DoesNotContain("데이터 초기화", helper);
            StringAssert.Contains("#if UNITY_EDITOR", bootstrap);
            StringAssert.Contains("ResolveForBootstrap()", bootstrap);
            StringAssert.Contains("new PersonalizationCandidateRepository(learnerProfileId)", bootstrap);
            StringAssert.Contains("new PetRobotNeutralCalibrationProfileService(learnerProfileId)", bootstrap);
        }

        // ── EmptyClean mode tracking ───────────────────────────────────────

        [Test]
        public void EmptyCleanArmSetsEmptyCleanMode()
        {
            string testRoot = NewTestRoot();
            try
            {
                SetupTestDirs(testRoot, out var prod, out var assets, out var candidate);
                var state = new DeveloperCleanCandidateRootSession();
                string failure;
                Assert.IsTrue(state.TryArm(candidate, prod, assets, out failure), failure);
                Assert.AreEqual(DeveloperCandidateRootArmMode.EmptyClean, state.ArmMode);
            }
            finally { DeleteTestRoot(testRoot); }
        }

        [Test]
        public void EmptyCleanRequireEmptyRejectsNonEmptyRoot()
        {
            string testRoot = NewTestRoot();
            try
            {
                SetupTestDirs(testRoot, out var prod, out var assets, out var candidate);
                File.WriteAllText(Path.Combine(candidate, "occupied.txt"), "data");
                var state = new DeveloperCleanCandidateRootSession();
                string failure;
                Assert.IsFalse(state.TryArm(candidate, prod, assets, out failure));
                Assert.IsNotEmpty(failure);
                Assert.AreEqual(DeveloperCandidateRootArmMode.None, state.ArmMode);
            }
            finally { DeleteTestRoot(testRoot); }
        }

        // ── GateReadyExisting: arm ─────────────────────────────────────────

        [Test]
        public void GateReadyArmSucceedsWithPassingGateVerifier()
        {
            string testRoot = NewTestRoot();
            try
            {
                SetupTestDirs(testRoot, out var prod, out var assets, out var candidate);
                File.WriteAllText(Path.Combine(candidate, "data.txt"), "content");

                var state = new DeveloperCleanCandidateRootSession();
                string failure;
                Assert.IsTrue(state.TryArmGateReady(candidate, prod, assets,
                    root => null, out failure), failure);
                Assert.IsTrue(state.IsArmed);
                Assert.IsFalse(state.HasActiveRoot);
                Assert.AreEqual(DeveloperCandidateRootArmMode.GateReadyExisting, state.ArmMode);
            }
            finally { DeleteTestRoot(testRoot); }
        }

        [Test]
        public void GateReadyArmRejectsWhenGateVerifierReportsTrusted221()
        {
            string testRoot = NewTestRoot();
            try
            {
                SetupTestDirs(testRoot, out var prod, out var assets, out var candidate);
                File.WriteAllText(Path.Combine(candidate, "data.txt"), "content");

                var state = new DeveloperCleanCandidateRootSession();
                string failure;
                Assert.IsFalse(state.TryArmGateReady(candidate, prod, assets,
                    root => "OverallEnrollmentGateReady=false G=2 L=2 T=1",
                    out failure));
                Assert.IsFalse(state.IsArmed);
                Assert.AreEqual(DeveloperCandidateRootArmMode.None, state.ArmMode);
                StringAssert.Contains("gate-ready validation failed", failure);
            }
            finally { DeleteTestRoot(testRoot); }
        }

        [Test]
        public void GateReadyArmRejectsCorruptRoot()
        {
            string testRoot = NewTestRoot();
            try
            {
                SetupTestDirs(testRoot, out var prod, out var assets, out var candidate);
                File.WriteAllText(Path.Combine(candidate, "data.txt"), "content");

                var state = new DeveloperCleanCandidateRootSession();
                string failure;
                Assert.IsFalse(state.TryArmGateReady(candidate, prod, assets,
                    root => "InvalidDataException: candidate index schema or learner ID mismatch",
                    out failure));
                Assert.IsFalse(state.IsArmed);
                StringAssert.Contains("gate-ready validation failed", failure);
            }
            finally { DeleteTestRoot(testRoot); }
        }

        [Test]
        public void GateReadyArmRejectsDefaultProductionRoot()
        {
            string testRoot = NewTestRoot();
            try
            {
                SetupTestDirs(testRoot, out var prod, out var assets, out var candidate);
                var state = new DeveloperCleanCandidateRootSession();
                string failure;
                Assert.IsFalse(state.TryArmGateReady(prod, prod, assets,
                    root => null, out failure));
                Assert.IsNotEmpty(failure);
            }
            finally { DeleteTestRoot(testRoot); }
        }

        [Test]
        public void GateReadyArmRejectsAssetsInternalPath()
        {
            string testRoot = NewTestRoot();
            try
            {
                SetupTestDirs(testRoot, out var prod, out var assets, out var candidate);
                string insideAssets = Path.Combine(assets, "validation");
                Directory.CreateDirectory(insideAssets);
                var state = new DeveloperCleanCandidateRootSession();
                string failure;
                Assert.IsFalse(state.TryArmGateReady(insideAssets, prod, assets,
                    root => null, out failure));
                StringAssert.Contains("Assets", failure);
            }
            finally { DeleteTestRoot(testRoot); }
        }

        // ── GateReadyExisting: ExitingEditMode re-validation ───────────────

        [Test]
        public void GateReadyBeginPlayModeRevalidatesAndActivates()
        {
            string testRoot = NewTestRoot();
            try
            {
                SetupTestDirs(testRoot, out var prod, out var assets, out var candidate);
                File.WriteAllText(Path.Combine(candidate, "data.txt"), "content");

                var state = new DeveloperCleanCandidateRootSession();
                string failure;
                Assert.IsTrue(state.TryArmGateReady(candidate, prod, assets,
                    root => null, out failure), failure);

                string root;
                Assert.IsTrue(state.BeginNextPlayMode(prod, assets,
                    r => null, out root, out failure), failure);
                Assert.IsTrue(state.HasActiveRoot);
                Assert.AreEqual(Path.GetFullPath(candidate), root);
            }
            finally { DeleteTestRoot(testRoot); }
        }

        [Test]
        public void GateReadyBeginPlayModeFailsAndClearsOnRevalidationFailure()
        {
            string testRoot = NewTestRoot();
            try
            {
                SetupTestDirs(testRoot, out var prod, out var assets, out var candidate);
                File.WriteAllText(Path.Combine(candidate, "data.txt"), "content");

                var state = new DeveloperCleanCandidateRootSession();
                string failure;
                Assert.IsTrue(state.TryArmGateReady(candidate, prod, assets,
                    root => null, out failure), failure);

                string root;
                Assert.IsFalse(state.BeginNextPlayMode(prod, assets,
                    r => "gate verification failed at PlayMode entry",
                    out root, out failure));
                Assert.IsFalse(state.HasActiveRoot);
                Assert.IsFalse(state.IsArmed);
                Assert.AreEqual(DeveloperCandidateRootArmMode.None, state.ArmMode);
                StringAssert.Contains("re-validation failed", failure);
            }
            finally { DeleteTestRoot(testRoot); }
        }

        // ── ResolveForBootstrap: consecutive calls, no consumption ─────────

        [Test]
        public void GateReadyResolveBootstrapConsecutiveCallsReturnSameRootNotConsumed()
        {
            string testRoot = NewTestRoot();
            try
            {
                SetupTestDirs(testRoot, out var prod, out var assets, out var candidate);
                File.WriteAllText(Path.Combine(candidate, "data.txt"), "content");

                var state = new DeveloperCleanCandidateRootSession();
                string failure;
                Assert.IsTrue(state.TryArmGateReady(candidate, prod, assets,
                    root => null, out failure), failure);
                string activatedRoot;
                Assert.IsTrue(state.BeginNextPlayMode(prod, assets,
                    r => null, out activatedRoot, out failure), failure);

                string resolve1 = state.ResolveCandidateRoot(prod);
                string resolve2 = state.ResolveCandidateRoot(prod);
                Assert.AreEqual(activatedRoot, resolve1);
                Assert.AreEqual(activatedRoot, resolve2);
                Assert.IsTrue(state.HasActiveRoot, "root must not be consumed by resolve");
            }
            finally { DeleteTestRoot(testRoot); }
        }

        // ── EndPlayMode: path and mode cleanup ─────────────────────────────

        [Test]
        public void GateReadyEndPlayModeClearsPathAndMode()
        {
            string testRoot = NewTestRoot();
            try
            {
                SetupTestDirs(testRoot, out var prod, out var assets, out var candidate);
                File.WriteAllText(Path.Combine(candidate, "data.txt"), "content");

                var state = new DeveloperCleanCandidateRootSession();
                string failure;
                Assert.IsTrue(state.TryArmGateReady(candidate, prod, assets,
                    root => null, out failure), failure);
                string activatedRoot;
                Assert.IsTrue(state.BeginNextPlayMode(prod, assets,
                    r => null, out activatedRoot, out failure), failure);

                string completed = state.EndPlayMode();
                Assert.AreEqual(activatedRoot, completed);
                Assert.IsFalse(state.IsArmed);
                Assert.IsFalse(state.HasActiveRoot);
                Assert.AreEqual(DeveloperCandidateRootArmMode.None, state.ArmMode);
                Assert.AreEqual(prod, state.ResolveCandidateRoot(prod));

                string nextUnused;
                Assert.IsFalse(state.BeginNextPlayMode(out nextUnused),
                    "next unarmed session must not activate");
            }
            finally { DeleteTestRoot(testRoot); }
        }

        // ── v2.3f: Re-arm stale state cleanup ─────────────────────────────

        [Test]
        public void GateReadyReArmFailureOnInvalidPathClearsPreviousArmedState()
        {
            string testRoot = NewTestRoot();
            try
            {
                SetupTestDirs(testRoot, out var prod, out var assets, out var candidate);
                File.WriteAllText(Path.Combine(candidate, "data.txt"), "content");

                var state = new DeveloperCleanCandidateRootSession();
                string failure;
                Assert.IsTrue(state.TryArmGateReady(candidate, prod, assets,
                    root => null, out failure), failure);
                Assert.IsTrue(state.IsArmed, "first arm must succeed");

                Assert.IsFalse(state.TryArmGateReady(prod, prod, assets,
                    root => null, out failure),
                    "re-arm with production root must fail");
                Assert.IsFalse(state.IsArmed, "stale armed state must be cleared");
                Assert.AreEqual(DeveloperCandidateRootArmMode.None, state.ArmMode);
            }
            finally { DeleteTestRoot(testRoot); }
        }

        [Test]
        public void GateReadyReArmFailureOnGateFalseClearsPreviousArmedState()
        {
            string testRoot = NewTestRoot();
            try
            {
                SetupTestDirs(testRoot, out var prod, out var assets, out var candidate);
                File.WriteAllText(Path.Combine(candidate, "data.txt"), "content");

                var state = new DeveloperCleanCandidateRootSession();
                string failure;
                Assert.IsTrue(state.TryArmGateReady(candidate, prod, assets,
                    root => null, out failure), failure);
                Assert.IsTrue(state.IsArmed);

                string candidate2 = Path.Combine(testRoot, "validation", "gate_ready_2");
                Directory.CreateDirectory(candidate2);
                File.WriteAllText(Path.Combine(candidate2, "data.txt"), "content");

                Assert.IsFalse(state.TryArmGateReady(candidate2, prod, assets,
                    root => "G=2 L=1 T=0", out failure));
                Assert.IsFalse(state.IsArmed, "stale armed state must be cleared");
                Assert.AreEqual(DeveloperCandidateRootArmMode.None, state.ArmMode);
            }
            finally { DeleteTestRoot(testRoot); }
        }

        [Test]
        public void GateReadyReArmFailureOnVerifierExceptionClearsPreviousArmedState()
        {
            string testRoot = NewTestRoot();
            try
            {
                SetupTestDirs(testRoot, out var prod, out var assets, out var candidate);
                File.WriteAllText(Path.Combine(candidate, "data.txt"), "content");

                var state = new DeveloperCleanCandidateRootSession();
                string failure;
                Assert.IsTrue(state.TryArmGateReady(candidate, prod, assets,
                    root => null, out failure), failure);
                Assert.IsTrue(state.IsArmed);

                Assert.IsFalse(state.TryArmGateReady(candidate, prod, assets,
                    null, out failure),
                    "null verifier must fail");
                Assert.IsFalse(state.IsArmed, "stale armed state must be cleared");
                Assert.AreEqual(DeveloperCandidateRootArmMode.None, state.ArmMode);
            }
            finally { DeleteTestRoot(testRoot); }
        }

        // ── v2.3f: Null/exception verifier fail-closed ───────────────────

        [Test]
        public void GateReadyBeginPlayModeNullVerifierFailsClosed()
        {
            string testRoot = NewTestRoot();
            try
            {
                SetupTestDirs(testRoot, out var prod, out var assets, out var candidate);
                File.WriteAllText(Path.Combine(candidate, "data.txt"), "content");

                var state = new DeveloperCleanCandidateRootSession();
                string failure;
                Assert.IsTrue(state.TryArmGateReady(candidate, prod, assets,
                    root => null, out failure), failure);

                string root;
                Assert.IsFalse(state.BeginNextPlayMode(prod, assets,
                    null, out root, out failure),
                    "null verifier must fail for GateReady");
                Assert.IsFalse(state.HasActiveRoot);
                Assert.IsFalse(state.IsArmed);
                Assert.AreEqual(DeveloperCandidateRootArmMode.None, state.ArmMode);
                StringAssert.Contains("gate verifier is required", failure);
            }
            finally { DeleteTestRoot(testRoot); }
        }

        [Test]
        public void GateReadyBeginPlayModeVerifierExceptionFailsClosed()
        {
            string testRoot = NewTestRoot();
            try
            {
                SetupTestDirs(testRoot, out var prod, out var assets, out var candidate);
                File.WriteAllText(Path.Combine(candidate, "data.txt"), "content");

                var state = new DeveloperCleanCandidateRootSession();
                string failure;
                Assert.IsTrue(state.TryArmGateReady(candidate, prod, assets,
                    root => null, out failure), failure);

                string activatedRoot;
                Assert.IsFalse(state.BeginNextPlayMode(prod, assets,
                    root => "repository corruption detected",
                    out activatedRoot, out failure));
                Assert.IsFalse(state.HasActiveRoot);
                Assert.IsFalse(state.IsArmed);
                Assert.AreEqual(DeveloperCandidateRootArmMode.None, state.ArmMode);
                StringAssert.Contains("re-validation failed", failure);
            }
            finally { DeleteTestRoot(testRoot); }
        }

        // ── v2.3f: Active GateReady resolve re-validates gate ─────────────

        [Test]
        public void GateReadyResolveWithVerifierRevalidatesGate()
        {
            string testRoot = NewTestRoot();
            try
            {
                SetupTestDirs(testRoot, out var prod, out var assets, out var candidate);
                File.WriteAllText(Path.Combine(candidate, "data.txt"), "content");

                var state = new DeveloperCleanCandidateRootSession();
                string failure;
                Assert.IsTrue(state.TryArmGateReady(candidate, prod, assets,
                    root => null, out failure), failure);
                string activatedRoot;
                Assert.IsTrue(state.BeginNextPlayMode(prod, assets,
                    r => null, out activatedRoot, out failure), failure);

                string resolve1 = state.ResolveCandidateRoot(prod, r => null);
                Assert.AreEqual(activatedRoot, resolve1);

                Assert.Throws<InvalidOperationException>(() =>
                    state.ResolveCandidateRoot(prod, r => "gate degraded G=2"));
            }
            finally { DeleteTestRoot(testRoot); }
        }

        // ── v2.3f: Unknown mode fail-closed ───────────────────────────────

        [Test]
        public void UnknownModeBeginPlayModeFailsClosed()
        {
            string testRoot = NewTestRoot();
            try
            {
                SetupTestDirs(testRoot, out var prod, out var assets, out var candidate);
                var state = new DeveloperCleanCandidateRootSession();
                string failure;
                Assert.IsTrue(state.TryArm(candidate, prod, assets, out failure), failure);

                typeof(DeveloperCleanCandidateRootSession)
                    .GetField("mode", System.Reflection.BindingFlags.NonPublic |
                        System.Reflection.BindingFlags.Instance)
                    .SetValue(state, DeveloperCandidateRootArmMode.None);

                string root;
                Assert.IsFalse(state.BeginNextPlayMode(prod, assets,
                    r => null, out root, out failure));
                Assert.IsFalse(state.IsArmed);
                Assert.IsFalse(state.HasActiveRoot);
                Assert.AreEqual(DeveloperCandidateRootArmMode.None, state.ArmMode);
                StringAssert.Contains("unknown", failure);
            }
            finally { DeleteTestRoot(testRoot); }
        }

        // ── v2.3f: EmptyClean regression ──────────────────────────────────

        [Test]
        public void EmptyCleanBeginPlayModeDoesNotRequireGateVerifier()
        {
            string testRoot = NewTestRoot();
            try
            {
                SetupTestDirs(testRoot, out var prod, out var assets, out var candidate);
                var state = new DeveloperCleanCandidateRootSession();
                string failure;
                Assert.IsTrue(state.TryArm(candidate, prod, assets, out failure), failure);
                Assert.AreEqual(DeveloperCandidateRootArmMode.EmptyClean, state.ArmMode);

                string root;
                Assert.IsTrue(state.BeginNextPlayMode(prod, assets,
                    null, out root, out failure), failure);
                Assert.IsTrue(state.HasActiveRoot);
                Assert.AreEqual(Path.GetFullPath(candidate), root);
            }
            finally { DeleteTestRoot(testRoot); }
        }

        // ── Source structure assertions ─────────────────────────────────────

        [Test]
        public void GateReadyMenuAndModeExistInOverrideSource()
        {
            string helper = Source("PetRobotDeveloperCleanCandidateRootOverride.cs");
            StringAssert.Contains("[MenuItem(GateReadyArmMenu)]", helper);
            StringAssert.Contains("Step 10M", helper);
            StringAssert.Contains("GateReadyExisting", helper);
            StringAssert.Contains("ArmedModeKey", helper);
            StringAssert.Contains("ValidateGateReadyRoot", helper);
            StringAssert.Contains("OverallEnrollmentGateReady", helper);
            StringAssert.Contains("PersonalizationCandidateRepository", helper);
            StringAssert.Contains("ReloadTrustedAndVerify", helper);
            StringAssert.DoesNotContain("File.Copy", helper);
            StringAssert.DoesNotContain("File.Move", helper);
            StringAssert.DoesNotContain("Directory.Move", helper);
        }

        // ── Helpers ────────────────────────────────────────────────────────

        static string Source(string fileName)
        {
            string path = Path.Combine(
                Application.dataPath,
                "_Project", "Scripts", "Practice", "PetRobotVerticalSlice", fileName);
            return File.ReadAllText(path);
        }

        static string NewTestRoot()
        {
            return Path.Combine(
                Path.GetTempPath(), "wis_step10k_v2_2_clean_root_fixture_" + Guid.NewGuid().ToString("N"));
        }

        static void DeleteTestRoot(string path)
        {
            if (Directory.Exists(path)) Directory.Delete(path, true);
        }

        static void SetupTestDirs(
            string testRoot,
            out string productionRoot,
            out string assetsRoot,
            out string candidateRoot)
        {
            productionRoot = Path.Combine(testRoot, "production", "default");
            assetsRoot = Path.Combine(testRoot, "project", "Assets");
            candidateRoot = Path.Combine(testRoot, "validation", "gate_ready");
            Directory.CreateDirectory(productionRoot);
            Directory.CreateDirectory(assetsRoot);
            Directory.CreateDirectory(candidateRoot);
        }
    }
}
