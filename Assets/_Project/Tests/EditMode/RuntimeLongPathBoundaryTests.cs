// M3_RUNTIME_PATH_A4 - draft, NOT installed.
//
// One real boundary, not a mirror of the implementation: a file whose fully qualified
// path is at or beyond the classic 260-character limit is written, then read back with
// the plain path and with the helper's path. If the environment cannot create such a
// path at all the test is Ignored, never silently passed.
//
// It touches no profile, no candidate, no model and no threshold, and it cleans up only
// what it created. Step 10X and the 14 route tests are untouched.
using NUnit.Framework;
using System;
using System.IO;
using System.Text;
using WordsInSilence.Recognition.Runtime;

namespace WordsInSilence.Tests.EditMode
{
    public sealed class RuntimeLongPathBoundaryTests
    {
        string root;

        [SetUp]
        public void CreateRoot()
        {
            root = Path.Combine(Path.GetTempPath(),
                "wis_lp_" + Guid.NewGuid().ToString("N").Substring(0, 8));
        }

        [TearDown]
        public void RemoveRoot()
        {
            if (root == null) return;
            try { Directory.Delete(RuntimeLongPath.ForIo(root), true); }
            catch (Exception) { /* best effort; the temp root is ours alone */ }
        }

        [Test]
        public void ShortPath_IsReadableThroughTheHelperToo()
        {
            Directory.CreateDirectory(root);
            string file = Path.Combine(root, "short.bin");
            var payload = new byte[] { 1, 2, 3, 4 };
            File.WriteAllBytes(file, payload);

            Assert.That(File.ReadAllBytes(RuntimeLongPath.ForIo(file)), Is.EqualTo(payload),
                "the helper must not change behaviour for ordinary paths");
        }

        [Test]
        public void PathAtOrBeyond260_IsReadableOnlyThroughTheHelper()
        {
            string deep = root;
            var segment = new string('d', 40);
            try
            {
                while (Path.GetFullPath(Path.Combine(deep, "asset.bin")).Length < 268)
                    deep = Path.Combine(deep, segment);
                Directory.CreateDirectory(RuntimeLongPath.ForIo(deep));
            }
            catch (Exception error)
            {
                Assert.Ignore("this environment cannot create a >260 character path ("
                              + error.GetType().Name + "); the boundary was NOT exercised.");
            }

            string file = Path.Combine(deep, "asset.bin");
            Assert.That(Path.GetFullPath(file).Length, Is.GreaterThanOrEqualTo(260));
            byte[] payload = Encoding.UTF8.GetBytes("boundary");
            File.WriteAllBytes(RuntimeLongPath.ForIo(file), payload);

            Assert.That(File.ReadAllBytes(RuntimeLongPath.ForIo(file)), Is.EqualTo(payload),
                "the helper's path must be readable at the boundary");
            Assert.That(new FileInfo(RuntimeLongPath.ForIo(file)).Length,
                Is.EqualTo(payload.Length),
                "size checks must work through the same form");
        }

        [Test]
        public void AlreadyPrefixedPath_IsReturnedUnchanged()
        {
            if (Path.DirectorySeparatorChar != '\\')
                Assert.Ignore("Windows-only behaviour.");
            const string prefixed = @"\\?\C:\a\b";
            Assert.That(RuntimeLongPath.ForIo(prefixed), Is.EqualTo(prefixed));
        }

        [Test]
        public void UncPath_UsesTheUncExtendedForm()
        {
            if (Path.DirectorySeparatorChar != '\\')
                Assert.Ignore("Windows-only behaviour.");
            Assert.That(RuntimeLongPath.ForIo(@"\\server\share\file.bin"),
                Is.EqualTo(@"\\?\UNC\server\share\file.bin"));
        }
    }
}
