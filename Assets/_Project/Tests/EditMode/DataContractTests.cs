using NUnit.Framework;
using UnityEngine;
using WordsInSilence.Calibration;
using WordsInSilence.Segmentation;
using WordsInSilence.Personalization;

namespace WordsInSilence.Tests.EditMode
{
    /// <summary>
    /// R1 데이터 계약 직렬화, null/valid 처리, schema_version 검증.
    /// </summary>
    public class DataContractTests
    {
        // ─── UserCalibrationProfile ─────────────────────────────────────────

        [Test]
        public void UserCalibrationProfile_DefaultSchemaVersion_Is1()
        {
            var p = new UserCalibrationProfile();
            Assert.AreEqual(1, p.SchemaVersion);
        }

        [Test]
        public void UserCalibrationProfile_RoundTrip_PreservesAllFields()
        {
            var p = new UserCalibrationProfile
            {
                UserProfileId = "test-user-001",
                CreatedUtc = "2026-08-01T00:00:00Z",
                UpdatedUtc = "2026-08-01T01:00:00Z",
            };
            p.BodyScale.ShoulderWidthNorm = 0.25f;
            p.BodyScale.HasUpperArmRatio = true;
            p.BodyScale.LeftUpperArmOverShoulder = 0.8f;
            p.Neutral.HasWristBaseline = true;
            p.Neutral.LeftWristNeutralBR = new Vector2(-0.3f, 0.5f);
            p.Neutral.RightWristNeutralBR = new Vector2(0.3f, 0.5f);
            p.Neutral.HasArmAngleBaseline = true;
            p.Neutral.LeftArmAngleNeutral = 90f;
            p.Neutral.RightArmAngleNeutral = 90f;
            p.Neutral.MotionNoiseFloor = 0.005f;
            p.Quality.Status = "good";

            string json = JsonUtility.ToJson(p);
            var loaded = JsonUtility.FromJson<UserCalibrationProfile>(json);

            Assert.AreEqual("test-user-001", loaded.UserProfileId);
            Assert.AreEqual(0.25f, loaded.BodyScale.ShoulderWidthNorm, 0.001f);
            Assert.IsTrue(loaded.BodyScale.HasUpperArmRatio);
            Assert.AreEqual(0.8f, loaded.BodyScale.LeftUpperArmOverShoulder, 0.001f);
            Assert.IsTrue(loaded.Neutral.HasWristBaseline);
            Assert.AreEqual(-0.3f, loaded.Neutral.LeftWristNeutralBR.x, 0.001f);
            Assert.IsTrue(loaded.Neutral.HasArmAngleBaseline);
            Assert.AreEqual(90f, loaded.Neutral.LeftArmAngleNeutral, 0.1f);
            Assert.AreEqual(0.005f, loaded.Neutral.MotionNoiseFloor, 0.001f);
            Assert.AreEqual("good", loaded.Quality.Status);
        }

        [Test]
        public void UserCalibrationProfile_MissingLandmark_NotZeroVector()
        {
            var p = new UserCalibrationProfile();
            // 보정 전 기본값 확인
            Assert.IsFalse(p.Neutral.HasWristBaseline);
            Assert.IsFalse(p.Neutral.HasArmAngleBaseline);
            Assert.IsFalse(p.Neutral.HasArmVerticalBaseline);
        }

        [Test]
        public void UserCalibrationProfile_BodyScale_OptionalRatios()
        {
            var p = new UserCalibrationProfile();
            Assert.IsFalse(p.BodyScale.HasUpperArmRatio);
            Assert.IsFalse(p.BodyScale.HasForearmRatio);
            Assert.IsFalse(p.BodyScale.HasTorsoRatio);
            // 비율이 없을 때 0으로 처리하지 않음 — Has 플래그로 확인
        }

        // ─── ActiveSessionCalibration ───────────────────────────────────────

        [Test]
        public void ActiveSessionCalibration_DefaultSchemaVersion_Is1()
        {
            var s = new ActiveSessionCalibration();
            Assert.AreEqual(1, s.SchemaVersion);
        }

        [Test]
        public void ActiveSessionCalibration_RoundTrip()
        {
            var s = new ActiveSessionCalibration
            {
                SessionId = "session-abc",
                UserProfileId = "user-001",
                CreatedUtc = "2026-08-01T12:00:00Z",
                AppVersion = "0.1.0",
            };
            s.CurrentScale.ShoulderWidthNorm = 0.22f;
            s.CurrentScale.HasBodyCenter = true;
            s.CurrentScale.BodyCenter = new Vector3(0.5f, 0.4f, 0f);
            s.Neutral.BottomExitValidated = true;
            s.Quality.EligibleForStableBaselineUpdate = true;

            string json = JsonUtility.ToJson(s);
            var loaded = JsonUtility.FromJson<ActiveSessionCalibration>(json);

            Assert.AreEqual("session-abc", loaded.SessionId);
            Assert.AreEqual(0.22f, loaded.CurrentScale.ShoulderWidthNorm, 0.001f);
            Assert.IsTrue(loaded.CurrentScale.HasBodyCenter);
            Assert.AreEqual(0.5f, loaded.CurrentScale.BodyCenter.x, 0.001f);
            Assert.IsTrue(loaded.Neutral.BottomExitValidated);
            Assert.IsTrue(loaded.Quality.EligibleForStableBaselineUpdate);
        }

        [Test]
        public void ActiveSessionCalibration_BodyCenter_OptionalWithFlag()
        {
            var s = new ActiveSessionCalibration();
            Assert.IsFalse(s.CurrentScale.HasBodyCenter);
        }

        // ─── SignAttemptRecord ───────────────────────────────────────────────

        [Test]
        public void SignAttemptRecord_DefaultSchemaVersion_Is1()
        {
            var r = new SignAttemptRecord();
            Assert.AreEqual(1, r.SchemaVersion);
        }

        [Test]
        public void SignAttemptRecord_RoundTrip()
        {
            var r = new SignAttemptRecord
            {
                AttemptId = "attempt-xyz",
                UserProfileId = "user-001",
                SignId = "SIGN002_GREETING",
                SceneContext = "tutorial",
            };
            r.Scores.GlobalScore = 0.82f;
            r.Scores.Handshape = 0.90f;
            r.Scores.Position = 0.75f;
            r.Result.Status = "passed";
            r.Result.FeedbackCodes = new[] { "POSITION_RIGHT_HAND_TOO_LOW" };
            r.Personalization.EligibleForEnrollment = true;

            string json = JsonUtility.ToJson(r);
            var loaded = JsonUtility.FromJson<SignAttemptRecord>(json);

            Assert.AreEqual("SIGN002_GREETING", loaded.SignId);
            Assert.AreEqual(0.82f, loaded.Scores.GlobalScore, 0.001f);
            Assert.AreEqual(0.90f, loaded.Scores.Handshape, 0.001f);
            Assert.AreEqual("passed", loaded.Result.Status);
            Assert.AreEqual(1, loaded.Result.FeedbackCodes.Length);
            Assert.IsTrue(loaded.Personalization.EligibleForEnrollment);
        }

        [Test]
        public void SignAttemptRecord_NaN_Scores_Preserved()
        {
            var r = new SignAttemptRecord();
            // PersonalScore는 NaN이어야 함 (아직 개인 점수 없음)
            Assert.IsTrue(float.IsNaN(r.Scores.PersonalScore));
            Assert.IsFalse(r.Scores.HasPersonalScore);

            // GlobalScore 설정 후에도 PersonalScore는 NaN 유지
            r.Scores.GlobalScore = 0.7f;
            Assert.IsFalse(r.Scores.HasPersonalScore);
        }

        [Test]
        public void SignAttemptRecord_FeedbackCodes_DefaultEmpty()
        {
            var r = new SignAttemptRecord();
            Assert.IsNotNull(r.Result.FeedbackCodes);
            Assert.AreEqual(0, r.Result.FeedbackCodes.Length);
        }

        // ─── PersonalSignEnrollment ─────────────────────────────────────────

        [Test]
        public void PersonalSignEnrollment_DefaultSchemaVersion_Is1()
        {
            var e = new PersonalSignEnrollment();
            Assert.AreEqual(1, e.SchemaVersion);
        }

        [Test]
        public void PersonalSignEnrollment_IsReady_RequiresStatusAndCount()
        {
            var e = new PersonalSignEnrollment
            {
                SignId = "SIGN002_GREETING",
                EnrollmentStatus = "not_started",
                RequiredAcceptedRepetitions = 3,
            };
            Assert.IsFalse(e.IsReady);

            e.EnrollmentStatus = "ready";
            Assert.IsFalse(e.IsReady); // AcceptedCount = 0

            e.AcceptedAttemptIds = new[] { "a1", "a2", "a3" };
            Assert.IsTrue(e.IsReady);
        }

        [Test]
        public void PersonalSignEnrollment_AcceptedCount()
        {
            var e = new PersonalSignEnrollment();
            Assert.AreEqual(0, e.AcceptedCount);

            e.AcceptedAttemptIds = new[] { "a1", "a2" };
            Assert.AreEqual(2, e.AcceptedCount);
        }

        [Test]
        public void PersonalSignEnrollment_RoundTrip()
        {
            var e = new PersonalSignEnrollment
            {
                UserProfileId = "user-001",
                SignId = "SIGN002_GREETING",
                EnrollmentStatus = "in_progress",
                CaptureViewPolicy = "front",
                AcceptedAttemptIds = new[] { "att-1" },
                RejectedAttemptIds = new[] { "att-2", "att-3" },
            };
            e.Ranges.DurationMedianMs = 1500f;
            e.Quality.GlobalAnchorScore = 0.78f;

            string json = JsonUtility.ToJson(e);
            var loaded = JsonUtility.FromJson<PersonalSignEnrollment>(json);

            Assert.AreEqual("SIGN002_GREETING", loaded.SignId);
            Assert.AreEqual("in_progress", loaded.EnrollmentStatus);
            Assert.AreEqual(1, loaded.AcceptedAttemptIds.Length);
            Assert.AreEqual(2, loaded.RejectedAttemptIds.Length);
            Assert.AreEqual(1500f, loaded.Ranges.DurationMedianMs, 0.1f);
            Assert.AreEqual(0.78f, loaded.Quality.GlobalAnchorScore, 0.001f);
        }

        // ─── AttemptState & CaptureViewPolicy enums ─────────────────────────

        [Test]
        public void AttemptState_HasExpectedValues()
        {
            Assert.AreEqual(0, (int)AttemptState.Inactive);
            Assert.AreEqual(5, (int)AttemptState.Capturing);
            Assert.AreEqual(6, (int)AttemptState.EndCandidate);
            Assert.AreEqual(7, (int)AttemptState.Evaluating);
        }

        [Test]
        public void AttemptResultStatus_HasFourValues()
        {
            Assert.AreEqual(4, System.Enum.GetValues(typeof(AttemptResultStatus)).Length);
        }

        [Test]
        public void CaptureViewPolicy_DefaultIsFront()
        {
            Assert.AreEqual(CaptureViewPolicy.Front, default(CaptureViewPolicy));
        }
    }
}
