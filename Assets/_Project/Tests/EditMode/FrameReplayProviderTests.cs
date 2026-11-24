using System;
using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;
using WordsInSilence.Pose;

namespace WordsInSilence.Tests.EditMode
{
    /// <summary>
    /// FrameReplayProvider 테스트: 타임스탬프 순서, 재생 진행, 프레임 발행.
    /// </summary>
    public class FrameReplayProviderTests
    {
        static HandPoseRecording MakeRecording(params long[] timestampsMs)
        {
            var frames = new HandPoseRecordingFrame[timestampsMs.Length];
            for (int i = 0; i < timestampsMs.Length; i++)
            {
                frames[i] = new HandPoseRecordingFrame
                {
                    TimestampMs = timestampsMs[i],
                    HasLeftHand = true,
                    HasRightHand = true,
                    HasPose = true,
                    HasValidShoulders = true,
                    ShoulderWidth = 0.2f,
                    ShoulderCenter = new Vector2(0.5f, 0.3f),
                    Quality = FrameDataQuality.Good,
                };
            }
            return new HandPoseRecording
            {
                SchemaVersion = 2,
                MotionId = "test",
                Frames = frames,
                TotalFrames = frames.Length,
            };
        }

        // ─── Load ───────────────────────────────────────────────────────────

        [Test]
        public void Load_NullRecording_Throws()
        {
            var provider = new FrameReplayProvider();
            Assert.Throws<ArgumentNullException>(() => provider.Load(null));
        }

        [Test]
        public void Load_EmptyFrames_Throws()
        {
            var rec = new HandPoseRecording
            {
                Frames = Array.Empty<HandPoseRecordingFrame>(),
                TotalFrames = 0,
            };
            var provider = new FrameReplayProvider();
            Assert.Throws<ArgumentException>(() => provider.Load(rec));
        }

        [Test]
        public void Load_NonMonotonicTimestamp_Throws()
        {
            var rec = MakeRecording(100, 200, 150, 300);
            var provider = new FrameReplayProvider();
            Assert.Throws<ArgumentException>(() => provider.Load(rec));
        }

        [Test]
        public void Load_ValidRecording_SetsState()
        {
            var rec = MakeRecording(0, 33, 66, 100);
            var provider = new FrameReplayProvider();
            provider.Load(rec);

            Assert.AreEqual(4, provider.TotalFrames);
            Assert.AreEqual(0, provider.CurrentFrameIndex);
            Assert.IsFalse(provider.IsPlaying);
        }

        // ─── Play/Stop ──────────────────────────────────────────────────────

        [Test]
        public void Play_WithoutLoad_Throws()
        {
            var provider = new FrameReplayProvider();
            Assert.Throws<InvalidOperationException>(() => provider.Play());
        }

        [Test]
        public void Play_SetsIsPlaying()
        {
            var provider = new FrameReplayProvider();
            provider.Load(MakeRecording(0, 33, 66));
            provider.Play();
            Assert.IsTrue(provider.IsPlaying);
        }

        [Test]
        public void Stop_ClearsIsPlaying()
        {
            var provider = new FrameReplayProvider();
            provider.Load(MakeRecording(0, 33, 66));
            provider.Play();
            provider.Stop();
            Assert.IsFalse(provider.IsPlaying);
        }

        // ─── Update & Frame Emission ────────────────────────────────────────

        [Test]
        public void Update_EmitsFramesInTimestampOrder()
        {
            var rec = MakeRecording(0, 100, 200, 300);
            var provider = new FrameReplayProvider();
            provider.Load(rec);
            provider.Play();

            var emitted = new List<long>();
            provider.FrameReplayed += f => emitted.Add(f.TimestampMs);

            // Advance 150ms — should emit frames at 0 and 100
            provider.Update(150);
            Assert.AreEqual(2, emitted.Count);
            Assert.AreEqual(0, emitted[0]);
            Assert.AreEqual(100, emitted[1]);

            // Advance another 200ms — should emit 200 and 300
            provider.Update(200);
            Assert.AreEqual(4, emitted.Count);
            Assert.AreEqual(200, emitted[2]);
            Assert.AreEqual(300, emitted[3]);
        }

        [Test]
        public void Update_StopsAtEnd()
        {
            var rec = MakeRecording(0, 50, 100);
            var provider = new FrameReplayProvider();
            provider.Load(rec);
            provider.Play();

            var count = 0;
            provider.FrameReplayed += _ => count++;

            provider.Update(200); // all frames emitted
            Assert.AreEqual(3, count);
            Assert.IsFalse(provider.IsPlaying);

            // Further updates should not emit
            provider.Update(100);
            Assert.AreEqual(3, count);
        }

        [Test]
        public void Update_WhenNotPlaying_DoesNothing()
        {
            var rec = MakeRecording(0, 50, 100);
            var provider = new FrameReplayProvider();
            provider.Load(rec);

            var count = 0;
            provider.FrameReplayed += _ => count++;

            provider.Update(200);
            Assert.AreEqual(0, count);
        }

        [Test]
        public void Update_Progress_IncrementsCorrectly()
        {
            var rec = MakeRecording(0, 100, 200, 300);
            var provider = new FrameReplayProvider();
            provider.Load(rec);
            provider.Play();

            Assert.AreEqual(0f, provider.Progress, 0.001f);

            provider.Update(150); // frames 0, 100 emitted → index 2
            Assert.AreEqual(0.5f, provider.Progress, 0.001f);
        }

        // ─── EmitFrameAt ────────────────────────────────────────────────────

        [Test]
        public void EmitFrameAt_ValidIndex_EmitsFrame()
        {
            var rec = MakeRecording(0, 100, 200);
            var provider = new FrameReplayProvider();
            provider.Load(rec);

            HandPoseFrame emitted = null;
            provider.FrameReplayed += f => emitted = f;

            provider.EmitFrameAt(1);
            Assert.IsNotNull(emitted);
            Assert.AreEqual(100, emitted.TimestampMs);
        }

        [Test]
        public void EmitFrameAt_InvalidIndex_Throws()
        {
            var rec = MakeRecording(0, 100);
            var provider = new FrameReplayProvider();
            provider.Load(rec);

            Assert.Throws<ArgumentOutOfRangeException>(() => provider.EmitFrameAt(-1));
            Assert.Throws<ArgumentOutOfRangeException>(() => provider.EmitFrameAt(2));
        }

        [Test]
        public void EmitFrameAt_WithoutLoad_Throws()
        {
            var provider = new FrameReplayProvider();
            Assert.Throws<InvalidOperationException>(() => provider.EmitFrameAt(0));
        }

        // ─── Frame Content ──────────────────────────────────────────────────

        [Test]
        public void EmittedFrame_CopiesRecordingFields()
        {
            var rec = MakeRecording(500);
            rec.Frames[0].ShoulderCenter = new Vector2(0.48f, 0.32f);
            rec.Frames[0].ShoulderWidth = 0.19f;
            rec.Frames[0].HasLeftHand = true;
            rec.Frames[0].HasRightHand = false;
            rec.Frames[0].Quality = FrameDataQuality.HandOnly;

            var provider = new FrameReplayProvider();
            provider.Load(rec);

            HandPoseFrame emitted = null;
            provider.FrameReplayed += f => emitted = f;
            provider.EmitFrameAt(0);

            Assert.AreEqual(500, emitted.TimestampMs);
            Assert.AreEqual(0.5f, emitted.Timestamp, 0.001f);
            Assert.AreEqual(0.48f, emitted.ShoulderCenter.x, 0.001f);
            Assert.AreEqual(0.19f, emitted.ShoulderWidth, 0.001f);
            Assert.IsTrue(emitted.HasLeftHand);
            Assert.IsFalse(emitted.HasRightHand);
            Assert.AreEqual(FrameDataQuality.HandOnly, emitted.Quality);
        }

        [Test]
        public void EmittedFrame_IsValidForBodyRelative_WhenPoseAndShoulders()
        {
            var rec = MakeRecording(0);
            rec.Frames[0].HasPose = true;
            rec.Frames[0].HasValidShoulders = true;
            rec.Frames[0].ShoulderWidth = 0.2f;

            var provider = new FrameReplayProvider();
            provider.Load(rec);

            HandPoseFrame emitted = null;
            provider.FrameReplayed += f => emitted = f;
            provider.EmitFrameAt(0);

            Assert.IsTrue(emitted.IsValidForBodyRelativeEvaluation);
        }

        [Test]
        public void EmittedFrame_NotValidForBodyRelative_WhenNoPose()
        {
            var rec = MakeRecording(0);
            rec.Frames[0].HasPose = false;
            rec.Frames[0].HasValidShoulders = false;
            rec.Frames[0].ShoulderWidth = 0f;

            var provider = new FrameReplayProvider();
            provider.Load(rec);

            HandPoseFrame emitted = null;
            provider.FrameReplayed += f => emitted = f;
            provider.EmitFrameAt(0);

            Assert.IsFalse(emitted.IsValidForBodyRelativeEvaluation);
        }

        // ─── Replay restart ─────────────────────────────────────────────────

        [Test]
        public void Play_AfterCompletion_ReplaysFromStart()
        {
            var rec = MakeRecording(0, 50);
            var provider = new FrameReplayProvider();
            provider.Load(rec);
            provider.Play();

            var count = 0;
            provider.FrameReplayed += _ => count++;

            provider.Update(100);
            Assert.AreEqual(2, count);
            Assert.IsFalse(provider.IsPlaying);

            // Replay
            provider.Play();
            Assert.IsTrue(provider.IsPlaying);
            Assert.AreEqual(0, provider.CurrentFrameIndex);

            provider.Update(100);
            Assert.AreEqual(4, count); // 2 more
        }
    }
}
