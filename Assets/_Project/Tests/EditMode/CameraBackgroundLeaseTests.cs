// F56 EditMode tests for the runInBackground lease. No camera, scene, model or user data.
using NUnit.Framework;
using WordsInSilence.Core;

namespace WordsInSilence.Tests.EditMode
{
    public sealed class CameraBackgroundLeaseTests
    {
        bool value;
        int writes;

        void Use(bool enabled, bool initial)
        {
            value = initial;
            writes = 0;
            CameraBackgroundLease.ConfigureForTests(enabled, () => value, v => { value = v; writes++; });
        }

        [TearDown]
        public void TearDown() { CameraBackgroundLease.ResetToProductionDefaults(); }

        [Test]
        public void DisabledLeaseChangesNothing()
        {
            Use(false, false);
            bool held = false;
            CameraBackgroundLease.Acquire(ref held);
            Assert.IsFalse(held);
            Assert.IsFalse(value);
            Assert.AreEqual(0, writes, "a disabled lease must not write the setting");
            Assert.AreEqual(0, CameraBackgroundLease.ActiveLeases);
            CameraBackgroundLease.Release(ref held);
            Assert.AreEqual(0, writes);
        }

        [TestCase(false)]
        [TestCase(true)]
        public void ThePreviousValueIsRestoredOnRelease(bool initial)
        {
            Use(true, initial);
            bool held = false;
            CameraBackgroundLease.Acquire(ref held);
            Assert.IsTrue(held);
            Assert.IsTrue(value, "the lease forces background running while the camera is owned");
            Assert.AreEqual(1, CameraBackgroundLease.ActiveLeases);

            CameraBackgroundLease.Acquire(ref held);
            Assert.AreEqual(1, CameraBackgroundLease.ActiveLeases, "re-acquire is a no-op");

            CameraBackgroundLease.Release(ref held);
            Assert.IsFalse(held);
            Assert.AreEqual(initial, value, "the previous value is restored exactly");
            Assert.AreEqual(0, CameraBackgroundLease.ActiveLeases);
        }

        [Test]
        public void AnOverlappingOwnerKeepsTheLeaseUntilTheLastRelease()
        {
            Use(true, false);
            bool outgoing = false, incoming = false;
            CameraBackgroundLease.Acquire(ref outgoing);
            CameraBackgroundLease.Acquire(ref incoming);      // new scene owner starts first
            Assert.AreEqual(2, CameraBackgroundLease.ActiveLeases);

            CameraBackgroundLease.Release(ref outgoing);       // old owner disposes afterwards
            Assert.IsTrue(value, "the departing owner must not clear the new owner's lease");
            Assert.AreEqual(1, CameraBackgroundLease.ActiveLeases);

            CameraBackgroundLease.Release(ref incoming);
            Assert.IsFalse(value);
            Assert.AreEqual(0, CameraBackgroundLease.ActiveLeases);
        }

        [Test]
        public void DoubleReleaseAndReleaseWithoutAcquireAreHarmless()
        {
            Use(true, false);
            bool held = false;
            CameraBackgroundLease.Release(ref held);           // never acquired, e.g. camera absent
            Assert.AreEqual(0, writes);
            Assert.AreEqual(0, CameraBackgroundLease.ActiveLeases);

            CameraBackgroundLease.Acquire(ref held);
            CameraBackgroundLease.Release(ref held);
            int after = writes;
            CameraBackgroundLease.Release(ref held);           // second dispose of the same owner
            Assert.AreEqual(after, writes, "a double release must not write again");
            Assert.AreEqual(0, CameraBackgroundLease.ActiveLeases);
            Assert.IsFalse(value);
        }
    }
}
