using System.IO;
using NUnit.Framework;
using UnityEngine;
using WordsInSilence.Pose;

namespace WordsInSilence.Tests.EditMode
{
    public class HandPoseRecordingRepositoryTests
    {
        const string TestMotionId = "test_repo_01";
        string _saveDir;

        [SetUp]
        public void SetUp()
        {
            _saveDir = Path.Combine(Application.persistentDataPath, "Recordings");
            // 이전 테스트 잔여 파일 제거
            string path = Path.Combine(_saveDir, $"{TestMotionId}.json");
            if (File.Exists(path)) File.Delete(path);
        }

        [TearDown]
        public void TearDown()
        {
            string path = Path.Combine(_saveDir, $"{TestMotionId}.json");
            if (File.Exists(path)) File.Delete(path);
        }

        // ─── 1 ───────────────────────────────────────────────────────────────

        [Test]
        public void Save_ThenLoad_ReturnsEqualSchemaAndMotionId()
        {
            var recording = MakeRecording(TestMotionId);

            HandPoseRecordingRepository.Save(recording);
            var loaded = HandPoseRecordingRepository.Load(TestMotionId);

            Assert.IsNotNull(loaded);
            Assert.AreEqual(2, loaded.SchemaVersion);
            Assert.AreEqual(TestMotionId, loaded.MotionId);
        }

        // ─── 2 ───────────────────────────────────────────────────────────────

        [Test]
        public void Save_CreatesJsonFileInPersistentPath()
        {
            var recording = MakeRecording(TestMotionId);

            HandPoseRecordingRepository.Save(recording);

            string expected = Path.Combine(_saveDir, $"{TestMotionId}.json");
            Assert.IsTrue(File.Exists(expected));
        }

        // ─── 3 ───────────────────────────────────────────────────────────────

        [Test]
        public void Load_NonexistentMotionId_ReturnsNull()
        {
            var result = HandPoseRecordingRepository.Load("__nonexistent_id__");

            Assert.IsNull(result);
        }

        // ─── 4 ───────────────────────────────────────────────────────────────

        [Test]
        public void ListAll_AfterSave_ContainsMotionId()
        {
            var recording = MakeRecording(TestMotionId);
            HandPoseRecordingRepository.Save(recording);

            var list = HandPoseRecordingRepository.ListAll();

            Assert.IsNotNull(list);
            Assert.Contains(TestMotionId, list);
        }

        // ─── 5 ───────────────────────────────────────────────────────────────

        [Test]
        public void Save_Overwrites_ExistingFile()
        {
            var first = MakeRecording(TestMotionId);
            first.DisplayName = "First";
            HandPoseRecordingRepository.Save(first);

            var second = MakeRecording(TestMotionId);
            second.DisplayName = "Second";
            HandPoseRecordingRepository.Save(second);

            var loaded = HandPoseRecordingRepository.Load(TestMotionId);
            Assert.AreEqual("Second", loaded.DisplayName);
        }

        // ─── Helpers ─────────────────────────────────────────────────────────

        static HandPoseRecording MakeRecording(string motionId)
        {
            return new HandPoseRecording
            {
                MotionId            = motionId,
                DisplayName         = motionId,
                DurationMs          = 3000f,
                SourceType          = "hand_pose",
                Frames              = new HandPoseRecordingFrame[0],
                TotalFrames         = 0,
                ValidHandFrames     = 0,
                ValidPoseFrames     = 0,
                ValidCombinedFrames = 0
            };
        }
    }
}
