using NUnit.Framework;
using UnityEngine;
using WordsInSilence.Pose;
using WordsInSilence.Recognition.Canonical;

namespace WordsInSilence.Tests.EditMode
{
    /// <summary>
    /// FrameCanonicalizer 리샘플링 및 보간 테스트.
    /// CoordinateConventionTests 와 보완적 관계.
    /// </summary>
    public class FrameCanonicalizerTests
    {
        // ─── ResampleToFps — 기본 경계값 ────────────────────────────────────

        [Test]
        public void ResampleToFps_Null_ReturnsEmpty()
        {
            var result = FrameCanonicalizer.ResampleToFps(null, 30f);
            Assert.IsNotNull(result);
            Assert.AreEqual(0, result.Length);
        }

        [Test]
        public void ResampleToFps_ZeroFps_ReturnsSameReference()
        {
            var frames = new[] { MakeCanonical(0L), MakeCanonical(100L) };
            var result = FrameCanonicalizer.ResampleToFps(frames, 0f);
            Assert.AreSame(frames, result);
        }

        [Test]
        public void ResampleToFps_NegativeFps_ReturnsSameReference()
        {
            var frames = new[] { MakeCanonical(0L), MakeCanonical(100L) };
            var result = FrameCanonicalizer.ResampleToFps(frames, -1f);
            Assert.AreSame(frames, result);
        }

        [Test]
        public void ResampleToFps_SingleFrame_ReturnsNewFrameAtZero()
        {
            var frames = new[] { MakeCanonical(999L) };
            var result = FrameCanonicalizer.ResampleToFps(frames, 30f);
            Assert.AreEqual(1, result.Length);
            Assert.AreEqual(0L, result[0].TimestampMs);
            // 새 인스턴스여야 함
            Assert.AreNotSame(frames[0], result[0]);
        }

        // ─── 출력 타임스탬프 ─────────────────────────────────────────────────

        [Test]
        public void ResampleToFps_OutputTimestampsAreUniform()
        {
            // 원본: 0ms, 100ms, 200ms, ..., 900ms (10fps, 10 frames)
            var frames = new CanonicalFrame[10];
            for (int i = 0; i < 10; i++)
                frames[i] = MakeCanonical(i * 100L);

            var result = FrameCanonicalizer.ResampleToFps(frames, 30f);
            // stepMs = 1000/30 ≈ 33.33ms
            float step = 1000f / 30f;
            for (int i = 1; i < result.Length; i++)
            {
                long expected = (long)System.Math.Round(i * step);
                Assert.AreEqual(expected, result[i].TimestampMs,
                    $"Frame[{i}].TimestampMs mismatch");
            }
            // 첫 프레임은 항상 0
            Assert.AreEqual(0L, result[0].TimestampMs);
        }

        [Test]
        public void ResampleToFps_LastFrameIncluded()
        {
            // 총 990ms, stepMs=33.33 → floor(990/33.33)+1 = 30+1 = 31 frames
            var frames = new CanonicalFrame[10];
            for (int i = 0; i < 10; i++)
                frames[i] = MakeCanonical(i * 110L);  // 0, 110, ..., 990

            var result = FrameCanonicalizer.ResampleToFps(frames, 30f);
            // 마지막 출력 프레임 타임스탬프 ≤ 990ms
            Assert.LessOrEqual(result[result.Length - 1].TimestampMs, 990L);
            // 마지막 프레임이 990ms에 가까워야 함
            Assert.GreaterOrEqual(result[result.Length - 1].TimestampMs, 900L);
        }

        [Test]
        public void ResampleToFps_DuplicateTimestamps_Handled()
        {
            // 중복 타임스탬프 입력
            var frames = new[]
            {
                MakeCanonical(0L),
                MakeCanonical(0L),   // 중복
                MakeCanonical(100L),
            };
            Assert.DoesNotThrow(() => FrameCanonicalizer.ResampleToFps(frames, 30f));
        }

        [Test]
        public void ResampleToFps_NonZeroStart_HandledCorrectly()
        {
            // 시작 타임스탬프가 0이 아닌 경우
            var frames = new[]
            {
                MakeCanonical(5000L),
                MakeCanonical(5100L),
                MakeCanonical(5200L),
            };
            var result = FrameCanonicalizer.ResampleToFps(frames, 30f);
            // 출력 첫 프레임은 0
            Assert.AreEqual(0L, result[0].TimestampMs);
        }

        // ─── 출력 프레임은 새 객체 ───────────────────────────────────────────

        [Test]
        public void ResampleToFps_OutputFramesAreNewInstances()
        {
            var frames = new[]
            {
                MakeCanonical(0L),
                MakeCanonical(100L),
                MakeCanonical(200L),
            };
            var result = FrameCanonicalizer.ResampleToFps(frames, 30f);
            // 출력 프레임은 원본 배열 인스턴스와 달라야 함
            foreach (var r in result)
                foreach (var src in frames)
                    Assert.AreNotSame(src, r);
        }

        [Test]
        public void ResampleToFps_HandPoints_NotSameArrayReference()
        {
            var pts = new[] { new Vector3(0.1f, 0.2f, 0.01f) };
            var frames = new[]
            {
                new CanonicalFrame { TimestampMs = 0L,   RightHandPoints = pts },
                new CanonicalFrame { TimestampMs = 100L, RightHandPoints = pts },
            };
            var result = FrameCanonicalizer.ResampleToFps(frames, 30f);
            foreach (var r in result)
                if (r.RightHandPoints != null)
                    Assert.AreNotSame(pts, r.RightHandPoints);
        }

        // ─── 보간 ────────────────────────────────────────────────────────────

        [Test]
        public void ResampleToFps_BodyRelative_InterpolatedAtMidpoint()
        {
            // 두 프레임: t=0 (x=0), t=200ms (x=1.0)
            // 30fps 두 번째 슬롯 (t≈33ms) → x ≈ 0.165
            var L = new CanonicalFrame
            {
                TimestampMs            = 0L,
                HasValidShoulders      = true,
                ShoulderWidth          = 0.3f,
                HasPose                = true,
                RightWristBodyRelative = new Vector2(0f, 0f),
                LeftWristBodyRelative  = new Vector2(float.NaN, float.NaN),
            };
            var R = new CanonicalFrame
            {
                TimestampMs            = 200L,
                HasValidShoulders      = true,
                ShoulderWidth          = 0.3f,
                HasPose                = true,
                RightWristBodyRelative = new Vector2(1f, 0f),
                LeftWristBodyRelative  = new Vector2(float.NaN, float.NaN),
            };

            var result = FrameCanonicalizer.ResampleToFps(new[] { L, R }, 30f);
            // result[1].TimestampMs ≈ 33ms → 보간 위치 33/200 = 0.165
            Assert.Greater(result.Length, 1);
            float x1 = result[1].RightWristBodyRelative.x;
            Assert.Greater(x1, 0f);
            Assert.Less(x1, 0.5f);
        }

        [Test]
        public void ResampleToFps_OneValidOneNaN_UsesValidValue()
        {
            // L.Left = NaN, R.Left = (0.3, 0.1) → 출력은 R의 값
            var L = new CanonicalFrame
            {
                TimestampMs           = 0L,
                LeftWristBodyRelative = new Vector2(float.NaN, float.NaN),
            };
            var R = new CanonicalFrame
            {
                TimestampMs           = 100L,
                LeftWristBodyRelative = new Vector2(-0.3f, 0.1f),
            };

            var result = FrameCanonicalizer.ResampleToFps(new[] { L, R }, 30f);
            // 마지막 프레임은 R에 가까우므로 NaN이 아니어야 함
            var last = result[result.Length - 1];
            Assert.IsFalse(float.IsNaN(last.LeftWristBodyRelative.x));
        }

        // ─── Canonicalize — 기존 기능 유지 확인 ─────────────────────────────

        [Test]
        public void Canonicalize_MultipleFrames_TransformApplied()
        {
            var frames = new[]
            {
                MakeFrame(0L),
                MakeFrame(33L),
            };
            var result = FrameCanonicalizer.Canonicalize(frames, CoordinateConvention.OfflineMediaPipe);
            foreach (var f in result)
                Assert.AreEqual(CoordinateConvention.OfflineMediaPipe, f.AppliedTransform.Source);
        }

        [Test]
        public void Canonicalize_HandPoints_Preserved()
        {
            var pts = new[] { new Vector3(0.1f, 0.2f, 0.01f), new Vector3(0.3f, 0.4f, 0.02f) };
            var frame = MakeFrame(0L);
            frame.RightHandPoints = pts;
            frame.HasRightHand    = true;

            var result = FrameCanonicalizer.Canonicalize(new[] { frame }, CoordinateConvention.UnityRuntime);
            Assert.IsNotNull(result[0].RightHandPoints);
            Assert.AreEqual(pts.Length, result[0].RightHandPoints.Length);
            Assert.AreEqual(pts[0].x, result[0].RightHandPoints[0].x, 1e-5f);
        }

        [Test]
        public void Canonicalize_ValidShoulder_BodyRelativeSet()
        {
            var frame = MakeFrame(0L);
            frame.HasValidShoulders      = true;
            frame.ShoulderWidth          = 0.3f;
            frame.HasPose                = true;
            frame.LeftWristBodyRelative  = new Vector2(-0.3f, 0.1f);
            frame.RightWristBodyRelative = new Vector2( 0.3f, 0.1f);

            var result = FrameCanonicalizer.Canonicalize(new[] { frame }, CoordinateConvention.UnityRuntime);
            Assert.IsFalse(float.IsNaN(result[0].LeftWristBodyRelative.x));
            Assert.IsFalse(float.IsNaN(result[0].RightWristBodyRelative.x));
        }

        [Test]
        public void Canonicalize_TooNarrowShoulder_BodyRelativeNaN()
        {
            var frame = MakeFrame(0L);
            frame.HasValidShoulders      = true;
            frame.ShoulderWidth          = 0.01f;   // MinShoulderWidth(0.05) 미만
            frame.HasPose                = true;
            frame.LeftWristBodyRelative  = new Vector2(-0.3f, 0.1f);
            frame.RightWristBodyRelative = new Vector2( 0.3f, 0.1f);

            var result = FrameCanonicalizer.Canonicalize(new[] { frame }, CoordinateConvention.UnityRuntime);
            Assert.IsTrue(float.IsNaN(result[0].LeftWristBodyRelative.x));
        }

        // ─── 헬퍼 ────────────────────────────────────────────────────────────

        static CanonicalFrame MakeCanonical(long timestampMs)
            => new CanonicalFrame { TimestampMs = timestampMs };

        static HandPoseRecordingFrame MakeFrame(long timestampMs)
        {
            return new HandPoseRecordingFrame
            {
                TimestampMs       = timestampMs,
                HasLeftHand       = false,
                HasRightHand      = false,
                HasPose           = false,
                HasValidShoulders = false,
                ShoulderWidth     = 0f,
                Quality           = FrameDataQuality.Good,
                LeftHandPoints    = null,
                RightHandPoints   = null,
                PosePoints        = null,
            };
        }
    }
}
