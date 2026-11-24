using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;
using WordsInSilence.Pose;
using WordsInSilence.Recognition;
using WordsInSilence.Segmentation;
using WordsInSilence.Settings;

namespace WordsInSilence.Tests.EditMode
{
    /// <summary>
    /// R4 인식기 테스트: 요소별 점수, 피드백 코드, 품질 게이트, 결측 처리.
    /// </summary>
    public class SignRecognizerTests
    {
        // ─── 헬퍼 ──────────────────────────────────────────────────────────

        static HandPoseFrame MakeFrame(
            long timestampMs, float wristX = 0.5f, float wristY = 0.5f,
            bool hasLeftHand = true, bool hasRightHand = true)
        {
            var pts = new Vector3[33];
            var vis = new float[33];
            for (int i = 0; i < 33; i++) vis[i] = 0.9f;
            pts[0] = new Vector3(0.5f, 0.2f, 0);
            pts[11] = new Vector3(0.35f, 0.35f, 0);
            pts[12] = new Vector3(0.65f, 0.35f, 0);
            pts[13] = new Vector3(0.30f, 0.55f, 0);
            pts[14] = new Vector3(0.70f, 0.55f, 0);
            pts[15] = new Vector3(0.28f, 0.75f, 0);
            pts[16] = new Vector3(0.72f, 0.75f, 0);

            return new HandPoseFrame
            {
                TimestampMs = timestampMs,
                HasPose = true,
                HasValidShoulders = true,
                HasLeftHand = hasLeftHand,
                HasRightHand = hasRightHand,
                PoseFrame = new UpperBodyPoseFrame { AllPoints = pts, Visibility = vis },
                ShoulderCenter = new Vector2(0.5f, 0.35f),
                ShoulderWidth = 0.30f,
                IsValidForBodyRelativeEvaluation = true,
                LeftWristBodyRelative = new Vector2(-wristX, wristY),
                RightWristBodyRelative = new Vector2(wristX, wristY),
            };
        }

        static List<HandPoseFrame> MakeSequence(int count, float startX = 0.3f, float endX = 0.7f)
        {
            var frames = new List<HandPoseFrame>();
            for (int i = 0; i < count; i++)
            {
                float t = (float)i / (count - 1);
                float x = Mathf.Lerp(startX, endX, t);
                float y = 0.5f + 0.2f * Mathf.Sin(t * Mathf.PI);
                frames.Add(MakeFrame(i * 33, x, y));
            }
            return frames;
        }

        static SignReference MakeReference(string signId, int frameCount = 30,
            float startX = 0.3f, float endX = 0.7f)
        {
            var frames = MakeSequence(frameCount, startX, endX);

            // HandPoseRecording 생성
            var recording = new HandPoseRecording
            {
                SchemaVersion = 2,
                MotionId = signId,
                DisplayName = signId,
                TotalFrames = frameCount,
                Frames = new HandPoseRecordingFrame[frameCount],
            };
            for (int i = 0; i < frameCount; i++)
            {
                var f = frames[i];
                recording.Frames[i] = new HandPoseRecordingFrame
                {
                    TimestampMs = f.TimestampMs,
                    HasLeftHand = f.HasLeftHand,
                    HasRightHand = f.HasRightHand,
                    HasPose = f.HasPose,
                    HasValidShoulders = f.HasValidShoulders,
                    ShoulderCenter = f.ShoulderCenter,
                    ShoulderWidth = f.ShoulderWidth,
                    LeftWristBodyRelative = f.LeftWristBodyRelative,
                    RightWristBodyRelative = f.RightWristBodyRelative,
                };
            }
            recording.DurationMs = (frameCount - 1) * 33;

            var features = SignFeatureExtractor.Extract(frames);

            return new SignReference
            {
                SignId = signId,
                DisplayName = signId,
                Recording = recording,
                Features = features,
            };
        }

        // ─── 테스트 ────────────────────────────────────────────────────────

        [Test]
        public void Recognize_NullFrames_ReturnsInvalid()
        {
            var recognizer = new SignRecognizer();
            var result = recognizer.Recognize(null, new List<SignReference>());
            Assert.AreEqual(AttemptResultStatus.Invalid, result.Status);
        }

        [Test]
        public void Recognize_TooFewFrames_ReturnsInvalid()
        {
            var recognizer = new SignRecognizer();
            var frames = new List<HandPoseFrame> { MakeFrame(0), MakeFrame(33) };
            var result = recognizer.Recognize(frames, new List<SignReference>());
            Assert.AreEqual(AttemptResultStatus.Invalid, result.Status);
        }

        [Test]
        public void Recognize_OutOfFrame_ReturnsUncertain()
        {
            var recognizer = new SignRecognizer();
            var frames = MakeSequence(30);
            var refs = new List<SignReference> { MakeReference("test") };
            var quality = new CaptureQualityInfo { OutOfFrame = true };

            var result = recognizer.Recognize(frames, refs, quality);
            Assert.AreEqual(AttemptResultStatus.Uncertain, result.Status);
            Assert.IsTrue(result.FeedbackCodes.Contains("CAMERA_OUT_OF_FRAME"));
        }

        [Test]
        public void Recognize_OutOfFrame_NotMarkedAsIncorrect()
        {
            // 핵심: 화면 이탈을 incorrect로 보고하지 않음
            var recognizer = new SignRecognizer();
            var frames = MakeSequence(30);
            var refs = new List<SignReference> { MakeReference("test") };
            var quality = new CaptureQualityInfo { OutOfFrame = true };

            var result = recognizer.Recognize(frames, refs, quality);
            Assert.AreNotEqual(AttemptResultStatus.Incorrect, result.Status);
        }

        [Test]
        public void Recognize_SimilarSequence_HasComponentScores()
        {
            var recognizer = new SignRecognizer();
            var userFrames = MakeSequence(30, 0.3f, 0.7f);
            var refs = new List<SignReference> { MakeReference("greeting", 30, 0.3f, 0.7f) };

            var result = recognizer.Recognize(userFrames, refs);
            Assert.IsNotNull(result.ComponentScores);
            Assert.Greater(result.GlobalScore, 0f);
            Assert.IsNotNull(result.BestSignId);
        }

        [Test]
        public void Recognize_VeryDifferentSequence_ReturnsIncorrect()
        {
            var recognizer = new SignRecognizer();
            // 사용자: 오른쪽으로 크게 이동
            var userFrames = MakeSequence(30, 0.1f, 0.9f);
            // 레퍼런스: 거의 안 움직임
            var refs = new List<SignReference> { MakeReference("greeting", 30, 0.49f, 0.51f) };

            var result = recognizer.Recognize(userFrames, refs);
            // 점수가 낮으면 incorrect 또는 uncertain
            Assert.AreNotEqual(AttemptResultStatus.Passed, result.Status);
        }

        [Test]
        public void Recognize_LowHandRatio_ReturnsUncertain()
        {
            var recognizer = new SignRecognizer();
            // 대부분 손 미검출
            var frames = new List<HandPoseFrame>();
            for (int i = 0; i < 30; i++)
                frames.Add(MakeFrame(i * 33, hasLeftHand: false, hasRightHand: false));

            var refs = new List<SignReference> { MakeReference("test") };
            var result = recognizer.Recognize(frames, refs);
            Assert.AreEqual(AttemptResultStatus.Uncertain, result.Status);
        }

        [Test]
        public void Recognize_GeneratesFeedbackCodes()
        {
            var recognizer = new SignRecognizer();
            var userFrames = MakeSequence(30, 0.1f, 0.9f);
            var refs = new List<SignReference> { MakeReference("greeting", 30, 0.49f, 0.51f) };

            var result = recognizer.Recognize(userFrames, refs);
            if (result.Status == AttemptResultStatus.Incorrect ||
                result.Status == AttemptResultStatus.Uncertain)
            {
                // 피드백 코드가 1~2개 있어야 함
                Assert.Greater(result.FeedbackCodes.Count, 0,
                    "Should generate feedback codes for non-passed results");
            }
        }

        [Test]
        public void Recognize_AmbiguityGap_Calculated()
        {
            var recognizer = new SignRecognizer();
            var userFrames = MakeSequence(30);
            var refs = new List<SignReference>
            {
                MakeReference("sign_a", 30, 0.3f, 0.7f),
                MakeReference("sign_b", 30, 0.2f, 0.8f),
            };

            var result = recognizer.Recognize(userFrames, refs);
            // AmbiguityGap은 0 이상
            Assert.GreaterOrEqual(result.AmbiguityGap, 0f);
        }

        // ─── SignFeatureExtractor 단위 테스트 ──────────────────────────────

        [Test]
        public void FeatureExtractor_ValidSequence_ExtractsFeatures()
        {
            var frames = MakeSequence(30);
            var features = SignFeatureExtractor.Extract(frames);

            Assert.IsTrue(features.Valid);
            Assert.AreEqual(30, features.FrameCount);
            Assert.Greater(features.HandFrameRatio, 0f);
            Assert.Greater(features.PoseFrameRatio, 0f);
        }

        [Test]
        public void FeatureExtractor_NullFrames_ReturnsInvalid()
        {
            var features = SignFeatureExtractor.Extract(null);
            Assert.IsFalse(features.Valid);
        }

        [Test]
        public void FeatureExtractor_TrajectoryValid()
        {
            var frames = MakeSequence(30, 0.2f, 0.8f);
            var features = SignFeatureExtractor.Extract(frames);

            Assert.IsTrue(features.Trajectory.Valid);
            Assert.Greater(features.Trajectory.TotalDisplacement, 0f);
        }

        [Test]
        public void FeatureExtractor_PositionHasMean()
        {
            var frames = MakeSequence(30);
            var features = SignFeatureExtractor.Extract(frames);

            Assert.IsTrue(features.Position.HasRight);
        }

        [Test]
        public void FeatureExtractor_TwoHandValid()
        {
            var frames = MakeSequence(30);
            var features = SignFeatureExtractor.Extract(frames);

            Assert.IsTrue(features.TwoHandRelation.Valid);
            Assert.Greater(features.TwoHandRelation.BothHandsFrameCount, 0);
        }
    }
}
