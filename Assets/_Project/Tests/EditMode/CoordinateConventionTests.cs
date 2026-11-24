using NUnit.Framework;
using UnityEngine;
using WordsInSilence.Pose;
using WordsInSilence.Recognition.Canonical;

namespace WordsInSilence.Tests.EditMode
{
    /// <summary>
    /// CoordinateConvention / CoordinateTransformDescriptor / FrameCanonicalizer 단위 테스트.
    /// </summary>
    public class CoordinateConventionTests
    {
        // ─── CoordinateTransformDescriptor.For ─────────────────────────────

        [Test]
        public void For_UnityRuntime_NoFlips()
        {
            var d = CoordinateTransformDescriptor.For(CoordinateConvention.UnityRuntime);
            Assert.AreEqual(CoordinateConvention.UnityRuntime, d.Source);
            Assert.IsFalse(d.FlipBodyRelativeX);
            Assert.IsFalse(d.FlipBodyRelativeY);
            Assert.IsFalse(d.FlipHandScreenX);
            Assert.IsFalse(d.FlipHandScreenY);
        }

        [Test]
        public void For_OfflineMediaPipe_FlipsBodyRelativeXY()
        {
            var d = CoordinateTransformDescriptor.For(CoordinateConvention.OfflineMediaPipe);
            Assert.AreEqual(CoordinateConvention.OfflineMediaPipe, d.Source);
            Assert.IsTrue(d.FlipBodyRelativeX);
            Assert.IsTrue(d.FlipBodyRelativeY);
            Assert.IsFalse(d.FlipHandScreenX);
            Assert.IsFalse(d.FlipHandScreenY);
        }

        // ─── FrameCanonicalizer.Canonicalize ────────────────────────────────

        [Test]
        public void Canonicalize_NullRecording_ReturnsEmpty()
        {
            var result = FrameCanonicalizer.Canonicalize(
                (HandPoseRecording)null, CoordinateConvention.UnityRuntime);
            Assert.IsNotNull(result);
            Assert.AreEqual(0, result.Length);
        }

        [Test]
        public void Canonicalize_EmptyFrames_ReturnsEmpty()
        {
            var rec = new HandPoseRecording { Frames = new HandPoseRecordingFrame[0] };
            var result = FrameCanonicalizer.Canonicalize(rec, CoordinateConvention.UnityRuntime);
            Assert.AreEqual(0, result.Length);
        }

        [Test]
        public void Canonicalize_UnityRuntime_TimestampZeroBased()
        {
            var frames = new[]
            {
                MakeFrame(100L, hasHand: true),
                MakeFrame(133L, hasHand: true),
                MakeFrame(166L, hasHand: true),
            };
            var result = FrameCanonicalizer.Canonicalize(frames, CoordinateConvention.UnityRuntime);

            Assert.AreEqual(3, result.Length);
            Assert.AreEqual(0L,  result[0].TimestampMs);
            Assert.AreEqual(33L, result[1].TimestampMs);
            Assert.AreEqual(66L, result[2].TimestampMs);
        }

        [Test]
        public void Canonicalize_UnityRuntime_OriginalTimestampPreserved()
        {
            var frames = new[] { MakeFrame(99999L, hasHand: false) };
            var result = FrameCanonicalizer.Canonicalize(frames, CoordinateConvention.UnityRuntime);
            Assert.AreEqual(99999L, result[0].OriginalTimestampMs);
        }

        [Test]
        public void Canonicalize_UnityRuntime_NoBodyRelativeFlip()
        {
            // UnityRuntime: Left.x = -0.5, Right.x = +0.5 → そのまま
            var frame = MakeFrame(0L, hasHand: false);
            frame.LeftWristBodyRelative  = new Vector2(-0.5f, -0.2f);
            frame.RightWristBodyRelative = new Vector2( 0.5f, -0.2f);
            frame.HasValidShoulders = true;
            frame.ShoulderWidth     = 0.2f;
            frame.HasPose           = true;

            var result = FrameCanonicalizer.Canonicalize(new[] { frame }, CoordinateConvention.UnityRuntime);
            var cf = result[0];
            Assert.AreEqual(-0.5f, cf.LeftWristBodyRelative.x,  1e-5f);
            Assert.AreEqual( 0.5f, cf.RightWristBodyRelative.x, 1e-5f);
        }

        [Test]
        public void Canonicalize_OfflineMediaPipe_FlipsBodyRelativeXY()
        {
            // OfflineMediaPipe: Left.x > 0 → flip → Left.x < 0
            var frame = MakeFrame(0L, hasHand: false);
            frame.LeftWristBodyRelative  = new Vector2(0.5f,  0.2f);
            frame.RightWristBodyRelative = new Vector2(-0.5f, 0.2f);
            frame.HasValidShoulders = true;
            frame.ShoulderWidth     = 0.2f;
            frame.HasPose           = true;

            var result = FrameCanonicalizer.Canonicalize(new[] { frame }, CoordinateConvention.OfflineMediaPipe);
            var cf = result[0];
            Assert.AreEqual(-0.5f, cf.LeftWristBodyRelative.x,  1e-5f);
            Assert.AreEqual( 0.5f, cf.RightWristBodyRelative.x, 1e-5f);
            Assert.AreEqual(-0.2f, cf.LeftWristBodyRelative.y,  1e-5f);
        }

        [Test]
        public void Canonicalize_EmptyHandPoints_BecomeNull()
        {
            var frame = MakeFrame(0L, hasHand: false);
            frame.LeftHandPoints  = new Vector3[0];  // 빈 배열 → null 통일
            frame.RightHandPoints = null;

            var result = FrameCanonicalizer.Canonicalize(new[] { frame }, CoordinateConvention.UnityRuntime);
            Assert.IsNull(result[0].LeftHandPoints);
            Assert.IsNull(result[0].RightHandPoints);
        }

        [Test]
        public void Canonicalize_InvalidShoulder_WristBodyRelativeNaN()
        {
            var frame = MakeFrame(0L, hasHand: false);
            frame.HasValidShoulders = false;   // 어깨 미검출
            frame.ShoulderWidth     = 0f;
            frame.LeftWristBodyRelative  = new Vector2(-0.5f, -0.2f);
            frame.RightWristBodyRelative = new Vector2( 0.5f, -0.2f);

            var result = FrameCanonicalizer.Canonicalize(new[] { frame }, CoordinateConvention.UnityRuntime);
            Assert.IsTrue(float.IsNaN(result[0].LeftWristBodyRelative.x));
            Assert.IsTrue(float.IsNaN(result[0].RightWristBodyRelative.x));
        }

        // ─── FrameCanonicalizer.ResampleToFps ───────────────────────────────

        [Test]
        public void ResampleToFps_ZeroFps_ReturnsSameReference()
        {
            var frames = new[]
            {
                MakeCanonical(0L),
                MakeCanonical(100L),
            };
            var result = FrameCanonicalizer.ResampleToFps(frames, 0f);
            Assert.AreSame(frames, result);
        }

        [Test]
        public void ResampleToFps_SingleFrame_ReturnsNewFrameAtTimestampZero()
        {
            // 단일 프레임 → TimestampMs=0 인 새 인스턴스 반환
            var frames = new[] { MakeCanonical(500L) };
            var result = FrameCanonicalizer.ResampleToFps(frames, 30f);
            Assert.AreEqual(1, result.Length);
            Assert.AreEqual(0L, result[0].TimestampMs);
            Assert.AreNotSame(frames[0], result[0]);
        }

        [Test]
        public void ResampleToFps_OutputTimestampStartsAtZero()
        {
            var frames = new[]
            {
                MakeCanonical(0L),
                MakeCanonical(500L),
                MakeCanonical(1000L),
            };
            var result = FrameCanonicalizer.ResampleToFps(frames, 30f);
            Assert.AreEqual(0L, result[0].TimestampMs);
        }

        // ─── 헬퍼 ────────────────────────────────────────────────────────────

        static HandPoseRecordingFrame MakeFrame(long timestampMs, bool hasHand)
        {
            return new HandPoseRecordingFrame
            {
                TimestampMs      = timestampMs,
                HasLeftHand      = hasHand,
                HasRightHand     = hasHand,
                HasPose          = false,
                HasValidShoulders = false,
                ShoulderWidth    = 0f,
                Quality          = FrameDataQuality.Good,
                LeftHandPoints   = null,
                RightHandPoints  = null,
                PosePoints       = null,
            };
        }

        static CanonicalFrame MakeCanonical(long timestampMs)
        {
            return new CanonicalFrame { TimestampMs = timestampMs };
        }
    }
}
