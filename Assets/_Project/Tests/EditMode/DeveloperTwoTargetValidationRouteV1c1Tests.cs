using NUnit.Framework;
using WordsInSilence.Recognition.LiveInput;

namespace WordsInSilence.Tests.EditMode
{
    /// <summary>
    /// Step 10P v1.0c1: the developer-only live validation plan runs GREETING once and
    /// THANK_YOU once, never LOVE, never a retry, and owns nothing that can write.
    /// </summary>
    public sealed class DeveloperTwoTargetValidationRouteV1c1Tests
    {
        [Test]
        public void TargetsAreExactlyGreetingThenThankYou()
        {
            var plan = new CommonSignalTwoTargetValidationPlan();
            Assert.AreEqual(2, plan.TargetCount);
            Assert.AreEqual("GREETING", plan.Targets[0]);
            Assert.AreEqual("THANK_YOU", plan.Targets[1]);
            Assert.AreEqual("GREETING", plan.CurrentTarget);
        }

        [Test]
        public void HappyPathFinalizesEachTargetExactlyOnceInOrder()
        {
            var plan = new CommonSignalTwoTargetValidationPlan();

            Assert.IsTrue(plan.TryBegin("GREETING"));
            Assert.IsTrue(plan.TryFinalize("GREETING", "live-1"));
            Assert.AreEqual("THANK_YOU", plan.CurrentTarget);
            Assert.IsTrue(plan.TryBegin("THANK_YOU"));
            Assert.IsTrue(plan.TryFinalize("THANK_YOU", "live-2"));

            Assert.IsTrue(plan.Completed);
            Assert.AreEqual(2, plan.FinalizedCount);
            Assert.AreEqual("live-1", plan.FinalizedSessionIds[0]);
            Assert.AreEqual("live-2", plan.FinalizedSessionIds[1]);
        }

        [Test]
        public void LoveIsNeverBegunOrFinalized()
        {
            var plan = new CommonSignalTwoTargetValidationPlan();
            Assert.IsTrue(CommonSignalTwoTargetValidationPlan.IsForbiddenTarget("LOVE"));
            Assert.IsFalse(plan.TryBegin("LOVE"));
            StringAssert.Contains("LOVE", plan.RejectionReason);
            Assert.IsFalse(plan.TryFinalize("LOVE", "live-x"));
            Assert.AreEqual(0, plan.FinalizedCount);
            Assert.AreEqual("GREETING", plan.CurrentTarget);
        }

        [Test]
        public void OutOfOrderTargetIsRejected()
        {
            var plan = new CommonSignalTwoTargetValidationPlan();
            Assert.IsFalse(plan.TryBegin("THANK_YOU"));
            StringAssert.Contains("out of order", plan.RejectionReason);
        }

        [Test]
        public void NoRetryOrRepeatIsPossible()
        {
            var plan = new CommonSignalTwoTargetValidationPlan();
            Assert.IsTrue(plan.TryBegin("GREETING"));
            Assert.IsFalse(plan.TryBegin("GREETING"), "a second begin is a retry");
            Assert.IsTrue(plan.TryFinalize("GREETING", "live-1"));
            Assert.IsFalse(plan.TryFinalize("GREETING", "live-1b"),
                "a second finalize is a repeat");
            Assert.AreEqual(1, plan.FinalizedCount);
        }

        [Test]
        public void ABlankOrRepeatedCaptureSessionIdIsRejected()
        {
            var plan = new CommonSignalTwoTargetValidationPlan();
            Assert.IsTrue(plan.TryBegin("GREETING"));
            Assert.IsFalse(plan.TryFinalize("GREETING", "  "));
            Assert.IsTrue(plan.TryFinalize("GREETING", "live-1"));
            Assert.IsTrue(plan.TryBegin("THANK_YOU"));
            Assert.IsFalse(plan.TryFinalize("THANK_YOU", "live-1"),
                "the same capture session id must not be finalized twice");
            Assert.IsTrue(plan.TryFinalize("THANK_YOU", "live-2"));
        }

        [Test]
        public void NothingIsAcceptedAfterBothTargetsAreDone()
        {
            var plan = new CommonSignalTwoTargetValidationPlan();
            plan.TryBegin("GREETING");
            plan.TryFinalize("GREETING", "live-1");
            plan.TryBegin("THANK_YOU");
            plan.TryFinalize("THANK_YOU", "live-2");

            Assert.IsFalse(plan.TryBegin("GREETING"));
            Assert.IsFalse(plan.TryBegin("THANK_YOU"));
            Assert.IsFalse(plan.TryFinalize("THANK_YOU", "live-3"));
            Assert.AreEqual(string.Empty, plan.CurrentTarget);
            Assert.AreEqual(2, plan.FinalizedCount);
        }

        [Test]
        public void ThePlanOwnsNoRepositoryOrSessionType()
        {
            // The plan's public surface is deliberately free of any persistence or
            // progression type: a route built on it cannot store a candidate, activate a
            // profile or complete a quest, because it holds no such collaborator.
            foreach (var member in typeof(CommonSignalTwoTargetValidationPlan).GetMembers())
            {
                string text = member.ToString();
                Assert.IsFalse(text.Contains("Repository"), "plan exposes " + text);
                Assert.IsFalse(text.Contains("PetRobotSignPracticeSession"), "plan exposes " + text);
                Assert.IsFalse(text.Contains("Candidate"), "plan exposes " + text);
                Assert.IsFalse(text.Contains("Quest"), "plan exposes " + text);
            }
        }
    }
}
