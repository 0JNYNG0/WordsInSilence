using NUnit.Framework;
using UnityEngine;
using WordsInSilence.Recognition.Canonical;
using WordsInSilence.Recognition.Quality;

namespace WordsInSilence.Tests.EditMode
{
    /// <summary>
    /// MotionSegmentDetector 단위 테스트.
    ///
    /// 각 테스트는 독립적인 SegmentationConfig 인스턴스를 사용한다.
    /// SegmentationConfig.Default 정적 인스턴스를 공유하면 테스트 순서에 따라
    /// 결과가 달라질 수 있으므로 절대 수정하지 않는다.
    /// </summary>
    public class MotionSegmentDetectorTests
    {
        // ─── 공통 헬퍼: 독립 config 생성 ────────────────────────────────────

        /// <summary>
        /// 테스트용 느슨한 설정 (새 인스턴스).
        /// 합성 데이터를 통과시키기 위해 임계값을 낮춘다.
        /// </summary>
        static SegmentationConfig MakeLooseCfg()
        {
            var cfg = SegmentationConfig.CreateInstance<SegmentationConfig>();
            cfg.MinHandFrameRatio     = 0.0f;
            cfg.MinPoseFrameRatio     = 0.0f;
            cfg.MinTotalFrames        = 1;
            cfg.MinActiveFrames       = 1;
            cfg.MinSegmentDurationMs  = 0f;
            cfg.StartPaddingFrames    = 0;
            cfg.EndPaddingFrames      = 0;
            cfg.SmoothingWindowFrames = 1;
            cfg.MaxInactiveGapFrames  = 0;
            cfg.WristVelocityThreshold    = 0.001f;
            cfg.LandmarkEnergyThreshold   = 0.0001f;
            cfg.TwoHandedThreshold        = 0.40f;
            return cfg;
        }

        // ─── 기본 실패 케이스 ─────────────────────────────────────────────────

        [Test]
        public void Detect_Null_ReturnsRecordingTooShort()
        {
            var r = MotionSegmentDetector.Detect(null);
            Assert.AreEqual(FrameQualityStatus.RecordingTooShort, r.Status);
        }

        [Test]
        public void Detect_Empty_ReturnsRecordingTooShort()
        {
            var r = MotionSegmentDetector.Detect(new CanonicalFrame[0]);
            Assert.AreEqual(FrameQualityStatus.RecordingTooShort, r.Status);
        }

        [Test]
        public void Detect_TooFewFrames_ReturnsRecordingTooShort()
        {
            var cfg = SegmentationConfig.CreateInstance<SegmentationConfig>();
            cfg.MinTotalFrames = 20;

            var frames = MakeFrames(5, velocity: 0f);
            var r = MotionSegmentDetector.Detect(frames, cfg);
            Assert.AreEqual(FrameQualityStatus.RecordingTooShort, r.Status);
        }

        [Test]
        public void Detect_InsufficientHandFrames_ReturnsInsufficientHandFrames()
        {
            var cfg = SegmentationConfig.CreateInstance<SegmentationConfig>();
            cfg.MinHandFrameRatio  = 0.50f;
            cfg.MinTotalFrames     = 1;

            // 10 frames, hands absent → handRatio=0
            var frames = MakeFrames(10, velocity: 0f, handPresent: false);
            var r = MotionSegmentDetector.Detect(frames, cfg);
            Assert.AreEqual(FrameQualityStatus.InsufficientHandFrames, r.Status);
        }

        [Test]
        public void Detect_InsufficientPoseFrames_ReturnsInsufficientPoseFrames()
        {
            var cfg = SegmentationConfig.CreateInstance<SegmentationConfig>();
            cfg.MinHandFrameRatio  = 0.0f;
            cfg.MinPoseFrameRatio  = 0.50f;
            cfg.MinTotalFrames     = 1;

            // 10 frames, hands present but no pose
            var frames = MakeFrames(10, velocity: 0f, handPresent: true, posePresent: false);
            var r = MotionSegmentDetector.Detect(frames, cfg);
            Assert.AreEqual(FrameQualityStatus.InsufficientPoseFrames, r.Status);
        }

        // ─── 핵심 동작 수정 확인 ─────────────────────────────────────────────

        [Test]
        public void Detect_HandPresentButNoVelocity_ReturnsSegmentNotFound()
        {
            // 손이 검출되지만 움직이지 않음 → SegmentNotFound (과거 버그: Analyzable 반환)
            var cfg = SegmentationConfig.CreateInstance<SegmentationConfig>();
            cfg.MinHandFrameRatio     = 0.0f;
            cfg.MinPoseFrameRatio     = 0.0f;
            cfg.MinTotalFrames        = 1;
            cfg.MinActiveFrames       = 1;
            cfg.WristVelocityThreshold    = 1.0f;   // 높은 임계값
            cfg.LandmarkEnergyThreshold   = 1.0f;
            cfg.SmoothingWindowFrames     = 1;
            cfg.MaxInactiveGapFrames      = 0;

            // 손이 있지만 body-relative 좌표가 전혀 안 바뀜 → 속도=0
            var frames = MakeFrames(20, velocity: 0f, handPresent: true);
            var r = MotionSegmentDetector.Detect(frames, cfg);
            Assert.AreEqual(FrameQualityStatus.SegmentNotFound, r.Status);
        }

        [Test]
        public void Detect_MovingHand_ReturnsAnalyzable()
        {
            var cfg = MakeLooseCfg();
            // 손이 매 프레임마다 이동 → 속도 > 0 → active
            var frames = MakeMovingFrames(20);
            var r = MotionSegmentDetector.Detect(frames, cfg);
            Assert.AreEqual(FrameQualityStatus.Analyzable, r.Status);
            Assert.IsNotNull(r.Segment);
            Assert.Greater(r.Segment.Length, 0);
        }

        [Test]
        public void Detect_PartialMotion_SegmentCoversOnlyMovingPart()
        {
            var cfg = MakeLooseCfg();
            cfg.WristVelocityThreshold = 0.05f;
            cfg.StartPaddingFrames     = 0;
            cfg.EndPaddingFrames       = 0;

            // 처음 10프레임: 정지 (velocity=0), 마지막 10프레임: 이동
            var frames = new CanonicalFrame[20];
            for (int i = 0; i < 10; i++)
                frames[i] = MakeStaticFrame(i * 33L);
            for (int i = 10; i < 20; i++)
                frames[i] = MakeMovingFrameAt(i * 33L, i * 0.1f);

            var r = MotionSegmentDetector.Detect(frames, cfg);
            Assert.AreEqual(FrameQualityStatus.Analyzable, r.Status);
            // 구간 시작은 10 이후여야 함 (정지 구간 제외)
            Assert.GreaterOrEqual(r.SegmentStartIndex, 10);
        }

        [Test]
        public void Detect_GapFilling_MergesShortInactiveGap()
        {
            var cfg = MakeLooseCfg();
            cfg.WristVelocityThreshold = 0.05f;
            cfg.MaxInactiveGapFrames   = 5;    // 5 이하 간격 연결

            // 0-9: 이동, 10-12: 정지 (gap=3), 13-19: 이동
            // 갭 연결 후 하나의 구간이 되어야 함
            var frames = new CanonicalFrame[20];
            for (int i = 0; i < 10; i++)
                frames[i] = MakeMovingFrameAt(i * 33L, i * 0.1f);
            for (int i = 10; i < 13; i++)
                frames[i] = MakeStaticFrame(i * 33L);
            for (int i = 13; i < 20; i++)
                frames[i] = MakeMovingFrameAt(i * 33L, i * 0.1f);

            var r = MotionSegmentDetector.Detect(frames, cfg);
            Assert.AreEqual(FrameQualityStatus.Analyzable, r.Status);
            // 구간이 0~19 를 포함해야 함 (갭 연결로 합쳐짐)
            Assert.LessOrEqual(r.SegmentStartIndex, 2);
            Assert.GreaterOrEqual(r.SegmentEndIndex, 17);
        }

        [Test]
        public void Detect_GapTooLarge_NotMerged()
        {
            var cfg = MakeLooseCfg();
            cfg.WristVelocityThreshold = 0.05f;
            cfg.MaxInactiveGapFrames   = 2;    // 2 이하만 연결
            cfg.MinActiveFrames        = 5;

            // 0-4: 이동, 5-14: 정지 (gap=10 > MaxInactiveGapFrames=2), 15-19: 이동
            // 갭이 너무 커서 두 개의 독립 구간이 됨
            // 가장 긴 구간 길이 = 5 (0-4 또는 15-19)
            var frames = new CanonicalFrame[20];
            for (int i = 0; i < 5; i++)
                frames[i] = MakeMovingFrameAt(i * 33L, i * 0.1f);
            for (int i = 5; i < 15; i++)
                frames[i] = MakeStaticFrame(i * 33L);
            for (int i = 15; i < 20; i++)
                frames[i] = MakeMovingFrameAt(i * 33L, i * 0.1f);

            var r = MotionSegmentDetector.Detect(frames, cfg);
            // minActiveFrames=5, 두 구간 각각 5프레임 → 첫 번째 구간 선택
            Assert.AreEqual(FrameQualityStatus.Analyzable, r.Status);
            // 선택된 구간은 두 이동 구간 중 하나여야 함 (합쳐지면 안 됨)
            int segLen = r.SegmentEndIndex - r.SegmentStartIndex + 1;
            Assert.LessOrEqual(segLen, 7); // padding 없으면 5, padding 있어도 7 이하
        }

        [Test]
        public void Detect_Analyzable_SegmentIndicesWithinBounds()
        {
            var cfg = MakeLooseCfg();
            var frames = MakeMovingFrames(30);
            var r = MotionSegmentDetector.Detect(frames, cfg);
            Assert.AreEqual(FrameQualityStatus.Analyzable, r.Status);
            Assert.GreaterOrEqual(r.SegmentStartIndex, 0);
            Assert.Less(r.SegmentEndIndex, frames.Length);
            Assert.LessOrEqual(r.SegmentStartIndex, r.SegmentEndIndex);
        }

        [Test]
        public void Detect_Analyzable_HandFrameRatioCorrect()
        {
            var cfg = MakeLooseCfg();
            // 20 frames, 처음 10만 손 있음, 나머지 10은 이동하는 손 없음
            var frames = new CanonicalFrame[20];
            for (int i = 0; i < 10; i++)
            {
                var f = MakeMovingFrameAt(i * 33L, i * 0.1f);
                f.HasRightHand = true;
                frames[i] = f;
            }
            for (int i = 10; i < 20; i++)
            {
                var f = MakeMovingFrameAt(i * 33L, i * 0.1f);
                f.HasRightHand = false;
                frames[i] = f;
            }

            var r = MotionSegmentDetector.Detect(frames, cfg);
            Assert.AreEqual(0.5f, r.HandFrameRatio, 0.01f);
        }

        [Test]
        public void Detect_TwoHanded_EstimatedFromSegment()
        {
            var cfg = MakeLooseCfg();
            cfg.TwoHandedThreshold = 0.40f;

            // 양손으로 이동
            var frames = new CanonicalFrame[20];
            for (int i = 0; i < 20; i++)
            {
                frames[i] = MakeMovingFrameAt(i * 33L, i * 0.1f);
                frames[i].HasLeftHand  = true;
                frames[i].HasRightHand = true;
                frames[i].LeftWristBodyRelative = new Vector2(-0.3f - i * 0.01f, 0f);
            }
            var r = MotionSegmentDetector.Detect(frames, cfg);
            Assert.AreEqual(FrameQualityStatus.Analyzable, r.Status);
            Assert.IsTrue(r.EstimatedTwoHanded);
        }

        // ─── 헬퍼 ────────────────────────────────────────────────────────────

        /// <summary>정적 손목 좌표 (이동 없음) 프레임 배열.</summary>
        static CanonicalFrame[] MakeFrames(int count, float velocity,
            bool handPresent = true, bool posePresent = false)
        {
            var frames = new CanonicalFrame[count];
            for (int i = 0; i < count; i++)
            {
                frames[i] = new CanonicalFrame
                {
                    TimestampMs  = i * 33L,
                    HasRightHand = handPresent,
                    HasLeftHand  = false,
                    HasPose      = posePresent,
                    RightWristBodyRelative = handPresent
                        ? new Vector2(0.3f, -0.2f)   // 고정 좌표 → 속도=0
                        : new Vector2(float.NaN, float.NaN),
                    LeftWristBodyRelative  = new Vector2(float.NaN, float.NaN),
                };
            }
            return frames;
        }

        /// <summary>매 프레임 손목이 이동하는 프레임 배열.</summary>
        static CanonicalFrame[] MakeMovingFrames(int count)
        {
            var frames = new CanonicalFrame[count];
            for (int i = 0; i < count; i++)
                frames[i] = MakeMovingFrameAt(i * 33L, i * 0.05f);
            return frames;
        }

        static CanonicalFrame MakeMovingFrameAt(long ts, float offset)
        {
            return new CanonicalFrame
            {
                TimestampMs  = ts,
                HasRightHand = true,
                HasLeftHand  = false,
                HasPose      = false,
                RightWristBodyRelative = new Vector2(0.3f + offset, -0.2f),
                LeftWristBodyRelative  = new Vector2(float.NaN, float.NaN),
            };
        }

        static CanonicalFrame MakeStaticFrame(long ts)
        {
            return new CanonicalFrame
            {
                TimestampMs  = ts,
                HasRightHand = true,
                HasLeftHand  = false,
                HasPose      = false,
                RightWristBodyRelative = new Vector2(0.3f, -0.2f),
                LeftWristBodyRelative  = new Vector2(float.NaN, float.NaN),
            };
        }
    }
}
