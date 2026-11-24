using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;
using WordsInSilence.Personalization;
using WordsInSilence.Pose;
using WordsInSilence.Recognition;
using WordsInSilence.Segmentation;
using WordsInSilence.Settings;

namespace WordsInSilence.Tests.EditMode
{
    /// <summary>
    /// R5 개인화 테스트: 등록 흐름, 품질 게이트, 가중치, 공용 하한.
    /// </summary>
    public class PersonalizationTests
    {
        // ─── 헬퍼 ──────────────────────────────────────────────────────────

        static SignRecognitionResult MakePassedResult(float globalScore = 0.75f)
        {
            return new SignRecognitionResult
            {
                Status = AttemptResultStatus.Passed,
                GlobalScore = globalScore,
                BestSignId = "SIGN002",
                UserFeatures = new SignFeatureSet
                {
                    Valid = true,
                    FrameCount = 30,
                    HandFrameRatio = 0.9f,
                    PoseFrameRatio = 0.9f,
                    LeftHandFrameRatio = 0.8f,
                    RightHandFrameRatio = 0.8f,
                    Position = new PositionFeatures
                    {
                        HasRight = true,
                        RightMeanPosition = new Vector2(0.3f, 0.5f),
                    },
                    Trajectory = new TrajectoryFeatures
                    {
                        Valid = true,
                        TotalDisplacement = 1.5f,
                        DirectionChangesY = 2,
                    },
                    TwoHandRelation = new TwoHandFeatures
                    {
                        Valid = true,
                        BothHandsFrameCount = 20,
                        MeanDistance = 0.3f,
                    },
                },
            };
        }

        static SignRecognitionResult MakeFailedResult(float globalScore = 0.30f)
        {
            return new SignRecognitionResult
            {
                Status = AttemptResultStatus.Incorrect,
                GlobalScore = globalScore,
                BestSignId = "SIGN002",
                UserFeatures = new SignFeatureSet
                {
                    Valid = true,
                    FrameCount = 30,
                    HandFrameRatio = 0.9f,
                },
            };
        }

        static List<HandPoseFrame> MakeDummyFrames(int count = 30)
        {
            var frames = new List<HandPoseFrame>();
            for (int i = 0; i < count; i++)
                frames.Add(new HandPoseFrame { TimestampMs = i * 33 });
            return frames;
        }

        // ─── 등록 흐름 ────────────────────────────────────────────────────

        [Test]
        public void Enrollment_NotStarted_Initially()
        {
            var svc = new PersonalizationService("test_user");
            var enrollment = svc.GetEnrollment("SIGN002");
            Assert.AreEqual("not_started", enrollment.EnrollmentStatus);
            Assert.AreEqual(0, enrollment.AcceptedCount);
            Assert.IsFalse(enrollment.IsReady);
        }

        [Test]
        public void Enrollment_AcceptsPassedAttempt()
        {
            var svc = new PersonalizationService("test_user");
            var result = svc.TryEnroll("SIGN002", MakePassedResult(), MakeDummyFrames(), "session1");
            Assert.IsTrue(result.Accepted);
            Assert.AreEqual(1, result.AcceptedTotal);
            Assert.AreEqual(3, result.RequiredTotal);
            Assert.IsFalse(result.EnrollmentReady);
        }

        [Test]
        public void Enrollment_3Accepted_BecomesReady()
        {
            var svc = new PersonalizationService("test_user");

            for (int i = 0; i < 3; i++)
            {
                var result = svc.TryEnroll("SIGN002", MakePassedResult(), MakeDummyFrames(), $"session{i}");
                Assert.IsTrue(result.Accepted, $"Attempt {i + 1} should be accepted");
            }

            Assert.IsTrue(svc.IsSignReady("SIGN002"));
            var enrollment = svc.GetEnrollment("SIGN002");
            Assert.AreEqual("ready", enrollment.EnrollmentStatus);
            Assert.AreEqual(3, enrollment.AcceptedCount);
        }

        [Test]
        public void Enrollment_RejectsLowGlobalScore()
        {
            var svc = new PersonalizationService("test_user");
            // 공용 하한(0.45) 미만
            var lowResult = MakePassedResult(0.30f);
            lowResult.Status = AttemptResultStatus.Passed;
            lowResult.GlobalScore = 0.30f;

            var result = svc.TryEnroll("SIGN002", lowResult, MakeDummyFrames(), "session1");
            Assert.IsFalse(result.Accepted);
            Assert.AreEqual("global_score_below_minimum", result.Reason);
        }

        [Test]
        public void Enrollment_RejectsIncorrectResult()
        {
            var svc = new PersonalizationService("test_user");
            var result = svc.TryEnroll("SIGN002", MakeFailedResult(), MakeDummyFrames(), "session1");
            Assert.IsFalse(result.Accepted);
            Assert.AreEqual("recognition_not_passed", result.Reason);
        }

        [Test]
        public void Enrollment_WrongAction_3Times_NotRegistered()
        {
            // 핵심: 잘못된 동작을 3번 반복해도 등록 안 됨
            var svc = new PersonalizationService("test_user");
            for (int i = 0; i < 3; i++)
            {
                var result = svc.TryEnroll("SIGN002", MakeFailedResult(), MakeDummyFrames(), $"s{i}");
                Assert.IsFalse(result.Accepted);
            }
            Assert.IsFalse(svc.IsSignReady("SIGN002"));
        }

        [Test]
        public void Enrollment_MaxAttempts_Exceeded()
        {
            var svc = new PersonalizationService("test_user");
            // 5회 시도 (2 수락 + 3 거절)
            svc.TryEnroll("SIGN002", MakePassedResult(), MakeDummyFrames(), "s1");
            svc.TryEnroll("SIGN002", MakePassedResult(), MakeDummyFrames(), "s2");
            svc.TryEnroll("SIGN002", MakeFailedResult(), MakeDummyFrames(), "s3");
            svc.TryEnroll("SIGN002", MakeFailedResult(), MakeDummyFrames(), "s4");
            svc.TryEnroll("SIGN002", MakeFailedResult(), MakeDummyFrames(), "s5");

            var result = svc.TryEnroll("SIGN002", MakePassedResult(), MakeDummyFrames(), "s6");
            Assert.IsFalse(result.Accepted);
            Assert.AreEqual("max_attempts_exceeded", result.Reason);
        }

        // ─── 개인 점수 ─────────────────────────────────────────────────────

        [Test]
        public void PersonalScore_NoExemplars_ReturnsNaN()
        {
            var svc = new PersonalizationService("test_user");
            float ps = svc.ComputePersonalScore("SIGN002", new SignFeatureSet { Valid = true }, 0.7f);
            Assert.IsTrue(float.IsNaN(ps));
        }

        [Test]
        public void CombinedScore_GlobalBelowFloor_IgnoresPersonal()
        {
            // 공용 절대 하한(0.35) 미만 → 개인 점수가 높아도 거부
            var svc = new PersonalizationService("test_user");
            float combined = svc.ComputeCombinedScore("SIGN002", 0.20f, 0.95f);
            Assert.AreEqual(0.20f, combined, 0.01f,
                "Global score below floor should ignore personal score");
        }

        [Test]
        public void CombinedScore_NoPersonal_ReturnsGlobal()
        {
            var svc = new PersonalizationService("test_user");
            float combined = svc.ComputeCombinedScore("SIGN002", 0.70f, float.NaN);
            Assert.AreEqual(0.70f, combined, 0.01f);
        }

        [Test]
        public void CombinedScore_WithPersonal_UsesWeights()
        {
            var svc = new PersonalizationService("test_user");
            // 3개 등록 후 → 75/25 가중치
            for (int i = 0; i < 3; i++)
                svc.TryEnroll("SIGN002", MakePassedResult(), MakeDummyFrames(), $"s{i}");

            float combined = svc.ComputeCombinedScore("SIGN002", 0.60f, 0.80f);
            // 3개 exemplars → gw=0.75, pw=0.25
            float expected = 0.60f * 0.75f + 0.80f * 0.25f;
            Assert.AreEqual(expected, combined, 0.01f);
        }

        // ─── 재학습/초기화 ─────────────────────────────────────────────────

        [Test]
        public void ResetEnrollment_ClearsSign()
        {
            var svc = new PersonalizationService("test_user");
            svc.TryEnroll("SIGN002", MakePassedResult(), MakeDummyFrames(), "s1");
            Assert.AreEqual(1, svc.GetEnrollment("SIGN002").AcceptedCount);

            svc.ResetEnrollment("SIGN002");
            Assert.AreEqual(0, svc.GetEnrollment("SIGN002").AcceptedCount);
            Assert.IsFalse(svc.IsSignReady("SIGN002"));
        }

        [Test]
        public void ResetAll_ClearsEverything()
        {
            var svc = new PersonalizationService("test_user");
            svc.TryEnroll("SIGN002", MakePassedResult(), MakeDummyFrames(), "s1");
            svc.TryEnroll("SIGN003", MakePassedResult(), MakeDummyFrames(), "s1");

            svc.ResetAll();
            Assert.AreEqual(0, svc.GetEnrollment("SIGN002").AcceptedCount);
            Assert.AreEqual(0, svc.GetEnrollment("SIGN003").AcceptedCount);
        }

        // ─── 가중치 테이블 확인 ────────────────────────────────────────────

        [Test]
        public void Weights_0Exemplars_FullGlobal()
        {
            var settings = RecognitionSettings.Default;
            settings.GetWeights(0, out float gw, out float pw);
            Assert.AreEqual(1.0f, gw, 0.01f);
            Assert.AreEqual(0.0f, pw, 0.01f);
        }

        [Test]
        public void Weights_3Exemplars_75_25()
        {
            var settings = RecognitionSettings.Default;
            settings.GetWeights(3, out float gw, out float pw);
            Assert.AreEqual(0.75f, gw, 0.01f);
            Assert.AreEqual(0.25f, pw, 0.01f);
        }

        [Test]
        public void Weights_10Exemplars_60_40()
        {
            var settings = RecognitionSettings.Default;
            settings.GetWeights(10, out float gw, out float pw);
            Assert.AreEqual(0.60f, gw, 0.01f);
            Assert.AreEqual(0.40f, pw, 0.01f);
        }
    }
}
