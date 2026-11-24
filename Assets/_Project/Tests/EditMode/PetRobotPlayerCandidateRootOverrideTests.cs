// M3_ROUTE_SOURCE_PREP_A1 (review fix) - draft, NOT installed. New behaviour only.
//
// These drive the SAME decision code the player uses: Resolve(string[], learnerId, probe).
// The Unity entry point differs only in that it supplies Environment.GetCommandLineArgs()
// and the production gate probe, so nothing here is a copy of the logic.
//
// The gate probe is injected. Cases that must not depend on real candidate data use a
// probe that records its arguments; the one case that exercises the REAL repository gate
// is Explicit and requires an operator-supplied working root, so an unexecuted "valid"
// case is never reported as a pass. Step 10X and the frozen 30-case fixture are not
// re-run.
using NUnit.Framework;
using System.IO;
using UnityEngine;
using UnityEngine.TestTools;
using WordsInSilence.Practice.PetRobotVerticalSlice;
using Failure = WordsInSilence.Practice.PetRobotVerticalSlice
    .PetRobotPlayerCandidateRootOverride.RootFailure;

namespace WordsInSilence.Tests.EditMode
{
    public sealed class PetRobotPlayerCandidateRootOverrideTests
    {
        const string Learner = "local_learner";
        const string Flag = "-wisCandidateRoot";

        static PetRobotPlayerCandidateRootOverride.Resolution Run(string[] arguments,
            bool gateReady = true, string learner = Learner)
        {
            return PetRobotPlayerCandidateRootOverride.Resolve(arguments, learner,
                (id, root) => gateReady);
        }

        static string ExistingDirectory()
        {
            return Path.GetFullPath(Path.GetTempPath());
        }

        [SetUp]
        public void Reset()
        {
            PetRobotPlayerCandidateRootOverride.ResetForTests();
        }

        [Test]
        public void FlagAbsent_KeepsDefaultBehaviourWithNoRoot()
        {
            var result = Run(new[] { "player.exe", "-someOtherFlag" });
            Assert.That(result.Requested, Is.False);
            Assert.That(result.Allowed, Is.True);
            Assert.That(result.Root, Is.Null);
            Assert.That(result.Failure, Is.EqualTo(Failure.None));
        }

        [Test]
        public void FlagWithoutValue_FailsClosed()
        {
            var result = Run(new[] { "player.exe", Flag });
            Assert.That(result.Allowed, Is.False);
            Assert.That(result.Root, Is.Null, "there must be no fallback root");
            Assert.That(result.Failure, Is.EqualTo(Failure.FlagValueMissing));
        }

        [Test]
        public void FlagFollowedByAnotherFlag_FailsClosed()
        {
            var result = Run(new[] { "player.exe", Flag, "-batchmode" });
            Assert.That(result.Allowed, Is.False);
            Assert.That(result.Failure, Is.EqualTo(Failure.FlagValueMissing));
        }

        [Test]
        public void DuplicateFlag_FailsClosed()
        {
            string root = ExistingDirectory();
            var result = Run(new[] { "player.exe", Flag, root, Flag, root });
            Assert.That(result.Allowed, Is.False);
            Assert.That(result.Root, Is.Null);
            Assert.That(result.Failure, Is.EqualTo(Failure.FlagDuplicated));
        }

        [Test]
        public void RelativePath_FailsClosed()
        {
            var result = Run(new[] { "player.exe", Flag, "candidates" });
            Assert.That(result.Allowed, Is.False);
            Assert.That(result.Failure, Is.EqualTo(Failure.PathNotFullyQualified));
        }

        [Test]
        public void DriveRelativePath_FailsClosed()
        {
            // "C:candidates" is rooted but NOT fully qualified
            var result = Run(new[] { "player.exe", Flag, "C:candidates" });
            Assert.That(result.Allowed, Is.False);
            Assert.That(result.Root, Is.Null);
            Assert.That(result.Failure, Is.EqualTo(Failure.PathNotFullyQualified));
        }

        [Test]
        public void RootRelativePath_FailsClosed()
        {
            var result = Run(new[] { "player.exe", Flag, "\\candidates" });
            Assert.That(result.Allowed, Is.False);
            Assert.That(result.Failure, Is.EqualTo(Failure.PathNotFullyQualified));
        }

        [Test]
        public void NonexistentDirectory_FailsClosed()
        {
            string missing = Path.Combine(ExistingDirectory(),
                "wis_m3_missing_" + System.Guid.NewGuid().ToString("N"));
            var result = Run(new[] { "player.exe", Flag, missing });
            Assert.That(result.Allowed, Is.False);
            Assert.That(result.Root, Is.Null);
            Assert.That(result.Failure, Is.EqualTo(Failure.PathNotFound));
        }

        [Test]
        public void GateNotReady_FailsClosedAndSelectsNothing()
        {
            var result = Run(new[] { "player.exe", Flag, ExistingDirectory() },
                gateReady: false);
            Assert.That(result.Allowed, Is.False);
            Assert.That(result.Root, Is.Null, "a failing gate must not select the root");
            Assert.That(result.Failure, Is.EqualTo(Failure.EnrollmentGateNotReady));
        }

        [Test]
        public void ThrowingGateProbe_FailsClosedAsUnreadable()
        {
            var result = PetRobotPlayerCandidateRootOverride.Resolve(
                new[] { "player.exe", Flag, ExistingDirectory() }, Learner,
                (id, root) => { throw new IOException("probe failed"); });
            Assert.That(result.Allowed, Is.False);
            Assert.That(result.Failure, Is.EqualTo(Failure.RepositoryUnreadable));
        }

        [Test]
        public void PassingGate_SelectsTheGivenRootAndNeverTheDefault()
        {
            string root = ExistingDirectory();
            string probedRoot = null;
            string probedLearner = null;
            var result = PetRobotPlayerCandidateRootOverride.Resolve(
                new[] { "player.exe", Flag, root }, Learner,
                (id, candidate) => { probedLearner = id; probedRoot = candidate; return true; });
            Assert.That(result.Allowed, Is.True);
            Assert.That(result.Root, Is.EqualTo(Path.GetFullPath(root)));
            Assert.That(probedRoot, Is.EqualTo(Path.GetFullPath(root)),
                "the gate must be probed with the selected root, not the default");
            Assert.That(probedLearner, Is.EqualTo(Learner));
        }

        [Test]
        public void SecondConsumerWithADifferentLearnerId_FailsClosedAndStaysFailed()
        {
            PetRobotPlayerCandidateRootOverride.SeedForTests(
                Run(new[] { "player.exe", Flag, ExistingDirectory() }));
            string root;
            Assert.That(PetRobotPlayerCandidateRootOverride
                .TryResolveForPlayer(Learner, out root), Is.True);

            LogAssert.Expect(LogType.Error, new System.Text.RegularExpressions.Regex(
                "LearnerIdentityMismatch"));
            Assert.That(PetRobotPlayerCandidateRootOverride
                .TryResolveForPlayer("other_learner", out root), Is.False);
            Assert.That(root, Is.Null);

            // the failure sticks: the first consumer must not be allowed to continue
            Assert.That(PetRobotPlayerCandidateRootOverride
                .TryResolveForPlayer(Learner, out root), Is.False);
            Assert.That(PetRobotPlayerCandidateRootOverride.Blocked, Is.True);
        }

        [Test]
        public void PublicFailureMessages_CarryNoPathOrLearnerId()
        {
            foreach (Failure value in System.Enum.GetValues(typeof(Failure)))
            {
                string message = PetRobotPlayerCandidateRootOverride.MessageFor(value);
                Assert.That(message, Does.Not.Contain("\\"));
                Assert.That(message, Does.Not.Contain("/"));
                Assert.That(message, Does.Not.Contain(Learner));
                if (value != Failure.None)
                    Assert.That(message, Is.Not.Empty);
            }
        }

        /// <summary>
        /// The only case that touches the real PersonalizationCandidateRepository gate.
        /// It runs in the ordinary set: if the operator did not set WIS_M3_TEST_ROOT for
        /// this process it is Ignored - never silently passed - and when the variable is
        /// present it must actually pass or fail on the real probe.
        /// </summary>
        [Test]
        public void RealRepositoryGate_AcceptsTheOperatorWorkingCopy()
        {
            string root = System.Environment.GetEnvironmentVariable("WIS_M3_TEST_ROOT");
            if (string.IsNullOrEmpty(root))
            {
                Assert.Ignore("WIS_M3_TEST_ROOT is not set for this process; the real "
                              + "repository gate was NOT exercised.");
            }
            var result = PetRobotPlayerCandidateRootOverride.Resolve(
                new[] { "player.exe", Flag, root }, Learner,
                PetRobotPlayerCandidateRootOverride.ProductionGateProbe);
            Assert.That(result.Failure,
                Is.EqualTo(PetRobotPlayerCandidateRootOverride.RootFailure.None),
                "the supplied working copy did not pass the real enrollment gate");
            Assert.That(result.Allowed, Is.True);
            Assert.That(result.Root, Is.EqualTo(Path.GetFullPath(root)));
        }
    }
}
