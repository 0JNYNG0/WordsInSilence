// F58 EditMode tests for the bounded frame pacing lease. No camera, scene, model or user data.
using NUnit.Framework;
using WordsInSilence.Core;

namespace WordsInSilence.Tests.EditMode
{
    public sealed class CameraFramePacingLeaseTests
    {
        int value;
        int writes;

        void Use(bool enabled, int initial)
        {
            value = initial;
            writes = 0;
            CameraFramePacingLease.ConfigureForTests(
                enabled, () => value, v => { value = v; writes++; }, () => 0);
        }

        [TearDown]
        public void TearDown() { CameraFramePacingLease.ResetToProductionDefaults(); }

        [Test]
        public void DisabledLeaseChangesNothing()
        {
            Use(false, -1);
            bool held = false;
            CameraFramePacingLease.Acquire(ref held);
            Assert.IsFalse(held);
            Assert.AreEqual(-1, value);
            Assert.AreEqual(0, writes, "a disabled lease must not write the setting");
            Assert.AreEqual(0, CameraFramePacingLease.ActiveLeases);
            CameraFramePacingLease.Release(ref held);
            Assert.AreEqual(0, writes);
            Assert.AreEqual(-1, value);
        }

        [Test]
        public void UncappedIsBoundedToSixtyAndRestoredToMinusOne()
        {
            Use(true, -1);
            bool held = false;
            CameraFramePacingLease.Acquire(ref held);
            Assert.IsTrue(held);
            Assert.AreEqual(60, value, "an uncapped player is bounded while the camera is owned");

            CameraFramePacingLease.Release(ref held);
            Assert.AreEqual(-1, value, "the previous value is restored exactly, including -1");
            Assert.AreEqual(0, CameraFramePacingLease.ActiveLeases);
        }

        [Test]
        public void AnExistingLowerCapIsPreserved()
        {
            Use(true, 30);
            bool held = false;
            CameraFramePacingLease.Acquire(ref held);
            Assert.AreEqual(30, value, "a lower positive cap must never be raised");
            Assert.AreEqual(0, writes, "nothing needs to be written when the cap already binds");

            CameraFramePacingLease.Release(ref held);
            Assert.AreEqual(30, value);
            Assert.AreEqual(0, writes);
        }

        [Test]
        public void AHigherCapIsBoundedToSixtyAndRestored()
        {
            Use(true, 120);
            bool held = false;
            CameraFramePacingLease.Acquire(ref held);
            Assert.AreEqual(60, value);

            CameraFramePacingLease.Release(ref held);
            Assert.AreEqual(120, value, "the previous higher cap is restored exactly");
        }

        [Test]
        public void AnOverlappingOwnerKeepsTheLeaseUntilTheLastRelease()
        {
            Use(true, -1);
            bool outgoing = false, incoming = false;
            CameraFramePacingLease.Acquire(ref outgoing);
            CameraFramePacingLease.Acquire(ref incoming);      // new scene owner starts first
            Assert.AreEqual(2, CameraFramePacingLease.ActiveLeases);
            Assert.AreEqual(-1, CameraFramePacingLease.CapturedPreviousValue,
                "the previous value is captured once, by the first owner only");

            CameraFramePacingLease.Release(ref outgoing);       // old owner disposes afterwards
            Assert.AreEqual(60, value, "the departing owner must not clear the new owner's lease");
            Assert.AreEqual(1, CameraFramePacingLease.ActiveLeases);

            CameraFramePacingLease.Release(ref incoming);
            Assert.AreEqual(-1, value);
            Assert.AreEqual(0, CameraFramePacingLease.ActiveLeases);
        }

        [Test]
        public void DoubleAcquireAndDoubleReleaseByTheSameHolderAreIdempotent()
        {
            Use(true, -1);
            bool held = false;
            CameraFramePacingLease.Release(ref held);           // never acquired, e.g. camera absent
            Assert.AreEqual(0, writes);
            Assert.AreEqual(0, CameraFramePacingLease.ActiveLeases);

            CameraFramePacingLease.Acquire(ref held);
            CameraFramePacingLease.Acquire(ref held);
            Assert.AreEqual(1, CameraFramePacingLease.ActiveLeases, "re-acquire is a no-op");
            Assert.AreEqual(1, writes);

            CameraFramePacingLease.Release(ref held);
            int after = writes;
            CameraFramePacingLease.Release(ref held);           // second dispose of the same owner
            Assert.AreEqual(after, writes, "a double release must not write again");
            Assert.AreEqual(0, CameraFramePacingLease.ActiveLeases);
            Assert.AreEqual(-1, value);
        }
    }
}
