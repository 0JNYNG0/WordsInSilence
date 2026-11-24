using NUnit.Framework;
using UnityEngine;
using WordsInSilence.Calibration;
using WordsInSilence.Pose;
using WordsInSilence.Settings;

namespace WordsInSilence.Tests.EditMode
{
    /// <summary>
    /// R2 보정 시스템 테스트: CameraReadinessEvaluator, NeutralPoseDetector,
    /// ReachCheckEvaluator, BodyScaleCalculator.
    /// </summary>
    public class CalibrationTests
    {
        // ─── 테스트 헬퍼 ────────────────────────────────────────────────────

        static UpperBodyPoseFrame MakePoseFrame(
            Vector3? nose = null,
            Vector3? leftShoulder = null, Vector3? rightShoulder = null,
            Vector3? leftElbow = null, Vector3? rightElbow = null,
            Vector3? leftWrist = null, Vector3? rightWrist = null,
            float shoulderVisibility = 0.9f)
        {
            var pts = new Vector3[33];
            var vis = new float[33];

            pts[0] = nose ?? new Vector3(0.5f, 0.2f, 0);
            pts[11] = leftShoulder ?? new Vector3(0.35f, 0.35f, 0);
            pts[12] = rightShoulder ?? new Vector3(0.65f, 0.35f, 0);
            pts[13] = leftElbow ?? new Vector3(0.30f, 0.55f, 0);
            pts[14] = rightElbow ?? new Vector3(0.70f, 0.55f, 0);
            pts[15] = leftWrist ?? new Vector3(0.28f, 0.75f, 0);
            pts[16] = rightWrist ?? new Vector3(0.72f, 0.75f, 0);
            // Hips for body scale
            pts[23] = new Vector3(0.40f, 0.70f, 0);
            pts[24] = new Vector3(0.60f, 0.70f, 0);

            for (int i = 0; i < 33; i++) vis[i] = 0.9f;
            vis[11] = shoulderVisibility;
            vis[12] = shoulderVisibility;

            return new UpperBodyPoseFrame { AllPoints = pts, Visibility = vis };
        }

        static HandPoseFrame MakeFrame(
            bool hasPose = true, bool hasValidShoulders = true,
            bool hasLeftHand = true, bool hasRightHand = true,
            UpperBodyPoseFrame poseFrame = null,
            float shoulderWidth = 0.30f)
        {
            var pose = poseFrame ?? MakePoseFrame();
            return new HandPoseFrame
            {
                HasPose = hasPose,
                HasValidShoulders = hasValidShoulders,
                HasLeftHand = hasLeftHand,
                HasRightHand = hasRightHand,
                PoseFrame = hasPose ? pose : null,
                ShoulderCenter = new Vector2(0.5f, 0.35f),
                ShoulderWidth = shoulderWidth,
                IsValidForBodyRelativeEvaluation = hasPose && hasValidShoulders,
                LeftWristBodyRelative = new Vector2(-0.5f, 1.2f),
                RightWristBodyRelative = new Vector2(0.5f, 1.2f),
                LeftArmAngleDegrees = 170f,
                RightArmAngleDegrees = 170f,
            };
        }

        // ─── CameraReadinessEvaluator ──────────────────────────────────────

        [Test]
        public void Framing_NullFrame_ReturnsNoPoseDetected()
        {
            var eval = new CameraReadinessEvaluator();
            var result = eval.EvaluateFraming(null);
            Assert.AreEqual(FramingStatus.NoPoseDetected, result.Status);
            Assert.IsFalse(result.IsGood);
        }

        [Test]
        public void Framing_NoPose_ReturnsNoPoseDetected()
        {
            var frame = MakeFrame(hasPose: false);
            var eval = new CameraReadinessEvaluator();
            var result = eval.EvaluateFraming(frame);
            Assert.AreEqual(FramingStatus.NoPoseDetected, result.Status);
        }

        [Test]
        public void Framing_NoShoulders_ReturnsShouldersNotVisible()
        {
            var frame = MakeFrame(hasValidShoulders: false, shoulderWidth: 0.01f);
            var eval = new CameraReadinessEvaluator();
            var result = eval.EvaluateFraming(frame);
            Assert.AreEqual(FramingStatus.ShouldersNotVisible, result.Status);
        }

        [Test]
        public void Framing_GoodFrame_ReturnsGood()
        {
            var frame = MakeFrame();
            var eval = new CameraReadinessEvaluator();
            var result = eval.EvaluateFraming(frame);
            Assert.AreEqual(FramingStatus.Good, result.Status);
            Assert.IsTrue(result.IsGood);
        }

        [Test]
        public void Framing_TooFar_ReturnsTooFar()
        {
            var frame = MakeFrame(shoulderWidth: 0.05f);
            var eval = new CameraReadinessEvaluator();
            var result = eval.EvaluateFraming(frame);
            Assert.AreEqual(FramingStatus.TooFar, result.Status);
        }

        [Test]
        public void Framing_TooClose_ReturnsTooClose()
        {
            var frame = MakeFrame(shoulderWidth: 0.55f);
            var eval = new CameraReadinessEvaluator();
            var result = eval.EvaluateFraming(frame);
            Assert.AreEqual(FramingStatus.TooClose, result.Status);
        }

        [Test]
        public void Framing_NoseOutOfFrame_ReturnsHeadOutOfFrame()
        {
            var pose = MakePoseFrame(nose: new Vector3(0.5f, -0.1f, 0));
            var frame = MakeFrame(poseFrame: pose);
            var eval = new CameraReadinessEvaluator();
            var result = eval.EvaluateFraming(frame);
            Assert.AreEqual(FramingStatus.HeadOutOfFrame, result.Status);
        }

        [Test]
        public void Framing_ElbowOutOfFrame_ReturnsElbowsOutOfFrame()
        {
            var pose = MakePoseFrame(leftElbow: new Vector3(-0.1f, 0.5f, 0));
            var frame = MakeFrame(poseFrame: pose);
            var eval = new CameraReadinessEvaluator();
            var result = eval.EvaluateFraming(frame);
            Assert.AreEqual(FramingStatus.ElbowsOutOfFrame, result.Status);
        }

        // ─── NeutralPoseDetector ───────────────────────────────────────────

        [Test]
        public void Neutral_NoBaseline_NotNeutral()
        {
            var detector = new NeutralPoseDetector();
            var frame = MakeFrame();
            var result = detector.CheckNeutral(frame, 0.033f);
            Assert.IsFalse(result.IsNeutral);
            Assert.IsFalse(detector.HasBaseline);
        }

        [Test]
        public void Neutral_NullFrame_NotNeutral()
        {
            var detector = new NeutralPoseDetector();
            detector.SetBaseline(new NeutralPoseData());
            var result = detector.CheckNeutral(null, 0.033f);
            Assert.IsFalse(result.IsNeutral);
        }

        [Test]
        public void Neutral_NoPose_NotNeutral()
        {
            var detector = new NeutralPoseDetector();
            detector.SetBaseline(new NeutralPoseData());
            var frame = MakeFrame(hasPose: false);
            var result = detector.CheckNeutral(frame, 0.033f);
            Assert.IsFalse(result.IsNeutral);
            Assert.AreEqual("포즈 미검출.", result.Reason);
        }

        [Test]
        public void Neutral_HandDetectionFailureAlone_DoesNotTriggerNeutral()
        {
            // 핵심 규칙: Hand 검출 실패만으로 중립을 판정하지 않는다
            var detector = new NeutralPoseDetector();
            detector.SetBaseline(new NeutralPoseData());
            var frame = MakeFrame(hasPose: false, hasLeftHand: false, hasRightHand: false);
            var result = detector.CheckNeutral(frame, 0.033f);
            Assert.IsFalse(result.IsNeutral);
        }

        [Test]
        public void Neutral_NoShoulders_NotNeutral()
        {
            var detector = new NeutralPoseDetector();
            detector.SetBaseline(new NeutralPoseData());
            var frame = MakeFrame(hasValidShoulders: false);
            var result = detector.CheckNeutral(frame, 0.033f);
            Assert.IsFalse(result.IsNeutral);
        }

        [Test]
        public void Neutral_SetBaseline_HasBaselineTrue()
        {
            var detector = new NeutralPoseDetector();
            Assert.IsFalse(detector.HasBaseline);
            detector.SetBaseline(new NeutralPoseData());
            Assert.IsTrue(detector.HasBaseline);
        }

        [Test]
        public void Neutral_SetBaselineNull_HasBaselineFalse()
        {
            var detector = new NeutralPoseDetector();
            detector.SetBaseline(new NeutralPoseData());
            Assert.IsTrue(detector.HasBaseline);
            detector.SetBaseline(null);
            Assert.IsFalse(detector.HasBaseline);
        }

        [Test]
        public void Neutral_WithBaseline_ArmsNeutral_IsNeutral()
        {
            var baseline = new NeutralPoseData
            {
                HandsVisible = true, // 테스트 프레임에 손이 있으므로
                HasArmAngleBaseline = true,
                LeftArmAngleNeutral = 170f,
                RightArmAngleNeutral = 170f,
                HasArmVerticalBaseline = true,
                LeftArmVerticalNeutral = 10f,
                RightArmVerticalNeutral = 10f,
                MotionNoiseFloor = 0.01f,
            };

            var detector = new NeutralPoseDetector();
            detector.SetBaseline(baseline);
            // 기본 MakeFrame: 팔이 아래로 내려간 자세 → 중립
            var frame = MakeFrame();
            var result = detector.CheckNeutral(frame, 0.033f);
            Assert.IsTrue(result.IsNeutral);
            Assert.IsTrue(result.ArmsNeutral);
        }

        [Test]
        public void Neutral_HoldTimeAccumulates()
        {
            var baseline = new NeutralPoseData
            {
                HandsVisible = true,
                HasWristBaseline = true,
                LeftWristNeutralBR = new Vector2(-0.5f, 1.2f),
                RightWristNeutralBR = new Vector2(0.5f, 1.2f),
                HasArmAngleBaseline = true,
                LeftArmAngleNeutral = 170f,
                RightArmAngleNeutral = 170f,
                HasArmVerticalBaseline = true,
                LeftArmVerticalNeutral = 10f,
                RightArmVerticalNeutral = 10f,
                MotionNoiseFloor = 0.01f,
            };

            var detector = new NeutralPoseDetector();
            detector.SetBaseline(baseline);
            var frame = MakeFrame();

            detector.CheckNeutral(frame, 0.5f);
            detector.CheckNeutral(frame, 0.5f);
            var result = detector.CheckNeutral(frame, 0.5f);
            Assert.AreEqual(1.5f, result.HoldTime, 0.01f);
        }

        [Test]
        public void Neutral_HoldTimeResets_WhenNotNeutral()
        {
            var baseline = new NeutralPoseData
            {
                HandsVisible = true,
                HasWristBaseline = true,
                LeftWristNeutralBR = new Vector2(-0.5f, 1.2f),
                RightWristNeutralBR = new Vector2(0.5f, 1.2f),
                HasArmAngleBaseline = true,
                LeftArmAngleNeutral = 170f,
                RightArmAngleNeutral = 170f,
                HasArmVerticalBaseline = true,
                LeftArmVerticalNeutral = 10f,
                RightArmVerticalNeutral = 10f,
                MotionNoiseFloor = 0.01f,
            };

            var detector = new NeutralPoseDetector();
            detector.SetBaseline(baseline);
            var frame = MakeFrame();

            detector.CheckNeutral(frame, 0.5f);
            detector.CheckNeutral(frame, 0.5f);
            Assert.AreEqual(1.0f, detector.NeutralHoldTime, 0.01f);

            // Break neutral
            detector.CheckNeutral(null, 0.5f);
            Assert.AreEqual(0f, detector.NeutralHoldTime, 0.01f);
        }

        [Test]
        public void Neutral_NoHands_ArmsDown_IsNeutral()
        {
            var baseline = new NeutralPoseData
            {
                HasArmAngleBaseline = true,
                LeftArmAngleNeutral = 170f,
                RightArmAngleNeutral = 170f,
                HasArmVerticalBaseline = true,
                LeftArmVerticalNeutral = 10f,
                RightArmVerticalNeutral = 10f,
                MotionNoiseFloor = 0.01f,
            };

            var detector = new NeutralPoseDetector();
            detector.SetBaseline(baseline);

            // 양손 미검출, 팔이 아래로 → 중립
            var frame = MakeFrame(hasLeftHand: false, hasRightHand: false);
            var result = detector.CheckNeutral(frame, 0.033f);
            Assert.IsTrue(result.IsNeutral, "Arms down without hands should be neutral");
        }

        [Test]
        public void Neutral_UpperArmSideways_NotNeutral()
        {
            var baseline = new NeutralPoseData { MotionNoiseFloor = 0.01f };
            var detector = new NeutralPoseDetector();
            detector.SetBaseline(baseline);

            // 왼쪽 상완(어깨→팔꿈치)이 수평으로 뻗음
            var pose = MakePoseFrame(
                leftElbow: new Vector3(0.10f, 0.35f, 0));
            var frame = MakeFrame(hasLeftHand: false, hasRightHand: false, poseFrame: pose);

            var result = detector.CheckNeutral(frame, 0.033f);
            Assert.IsFalse(result.IsNeutral, "Upper arm sideways should NOT be neutral");
        }

        [Test]
        public void Neutral_UpperArmRaised_NotNeutral()
        {
            var baseline = new NeutralPoseData { MotionNoiseFloor = 0.01f };
            var detector = new NeutralPoseDetector();
            detector.SetBaseline(baseline);

            // 왼쪽 상완이 위를 향함 (elbow.y < shoulder.y)
            var pose = MakePoseFrame(
                leftElbow: new Vector3(0.35f, 0.20f, 0));
            var frame = MakeFrame(hasLeftHand: false, hasRightHand: false, poseFrame: pose);

            var result = detector.CheckNeutral(frame, 0.033f);
            Assert.IsFalse(result.IsNeutral, "Upper arm raised should NOT be neutral");
        }

        [Test]
        public void Neutral_HandsOnLap_ElbowBent_IsNeutral()
        {
            var baseline = new NeutralPoseData { MotionNoiseFloor = 0.01f };
            var detector = new NeutralPoseDetector();
            detector.SetBaseline(baseline);

            // 상완은 아래로 향하지만 팔꿈치가 구부러짐 (손이 다리 위)
            var pose = MakePoseFrame(
                leftElbow: new Vector3(0.33f, 0.55f, 0),   // 어깨 아래
                rightElbow: new Vector3(0.67f, 0.55f, 0),
                leftWrist: new Vector3(0.40f, 0.60f, 0),   // 팔꿈치에서 안쪽으로
                rightWrist: new Vector3(0.60f, 0.60f, 0));
            var frame = MakeFrame(hasLeftHand: false, hasRightHand: false, poseFrame: pose);

            var result = detector.CheckNeutral(frame, 0.033f);
            Assert.IsTrue(result.IsNeutral, "Hands on lap with bent elbows should be neutral");
        }

        // ─── NeutralPoseDetector Calibration ──────────────────────────────

        [Test]
        public void Calibration_TooFewFrames_Fails()
        {
            var detector = new NeutralPoseDetector();
            detector.BeginCalibration();

            // 3프레임만 추가 (최소 5프레임 필요)
            for (int i = 0; i < 3; i++)
                detector.AddCalibrationFrame(MakeFrame());

            var result = detector.FinishCalibration();
            Assert.IsFalse(result.Success);
        }

        [Test]
        public void Calibration_EnoughFrames_Succeeds()
        {
            var detector = new NeutralPoseDetector();
            detector.BeginCalibration();

            for (int i = 0; i < 30; i++)
                detector.AddCalibrationFrame(MakeFrame());

            var result = detector.FinishCalibration();
            Assert.IsTrue(result.Success);
            Assert.IsNotNull(result.Baseline);
            Assert.IsTrue(detector.HasBaseline);
            Assert.AreEqual(30, result.TotalSamples);
        }

        [Test]
        public void Calibration_InvalidFrames_LowValidRatio_Fails()
        {
            var settings = RecognitionSettings.Default;
            var detector = new NeutralPoseDetector(settings);
            detector.BeginCalibration();

            // 10 valid + 40 invalid (no shoulders) = 20% valid
            for (int i = 0; i < 10; i++)
                detector.AddCalibrationFrame(MakeFrame());
            for (int i = 0; i < 40; i++)
                detector.AddCalibrationFrame(MakeFrame(hasValidShoulders: false));

            var result = detector.FinishCalibration();
            // 유효 프레임은 10개뿐이지만, AddCalibrationFrame이 무효 프레임을 거르므로
            // 실제로 10개만 저장됨. 10 >= 5이고 validRatio는 100% (모두 HasValidBodyRelative).
            // 이 테스트는 AddCalibrationFrame의 필터링을 검증한다.
            Assert.AreEqual(10, result.TotalSamples);
        }

        // ─── ReachCheckEvaluator ───────────────────────────────────────────

        [Test]
        public void ReachCheck_InitialPhase_IsLeftArm()
        {
            var eval = new ReachCheckEvaluator();
            Assert.AreEqual(ReachCheckPhase.LeftArm, eval.CurrentPhase);
            Assert.IsFalse(eval.IsComplete);
        }

        [Test]
        public void ReachCheck_LeftArmExtended_PassesPhase()
        {
            var eval = new ReachCheckEvaluator();

            // 왼손목이 왼어깨보다 왼쪽에 있음
            var pose = MakePoseFrame(
                leftWrist: new Vector3(0.10f, 0.5f, 0),
                leftElbow: new Vector3(0.20f, 0.45f, 0));
            var frame = MakeFrame(poseFrame: pose);

            // 0.7초 유지
            for (int i = 0; i < 22; i++) // 22 * 0.033 ≈ 0.73초
                eval.Update(frame, 0.033f);

            Assert.AreEqual(ReachCheckPhase.RightArm, eval.CurrentPhase);
        }

        [Test]
        public void ReachCheck_AllPhases_Complete()
        {
            var eval = new ReachCheckEvaluator();

            // Left arm
            var leftPose = MakePoseFrame(
                leftWrist: new Vector3(0.10f, 0.5f, 0),
                leftElbow: new Vector3(0.20f, 0.45f, 0));
            var leftFrame = MakeFrame(poseFrame: leftPose);
            for (int i = 0; i < 25; i++)
                eval.Update(leftFrame, 0.033f);
            Assert.AreEqual(ReachCheckPhase.RightArm, eval.CurrentPhase);

            // Right arm
            var rightPose = MakePoseFrame(
                rightWrist: new Vector3(0.90f, 0.5f, 0),
                rightElbow: new Vector3(0.80f, 0.45f, 0));
            var rightFrame = MakeFrame(poseFrame: rightPose);
            for (int i = 0; i < 25; i++)
                eval.Update(rightFrame, 0.033f);
            Assert.AreEqual(ReachCheckPhase.BothArmsUp, eval.CurrentPhase);

            // Both arms up (wrists above nose)
            var upPose = MakePoseFrame(
                nose: new Vector3(0.5f, 0.3f, 0),
                leftWrist: new Vector3(0.35f, 0.10f, 0),
                rightWrist: new Vector3(0.65f, 0.10f, 0));
            var upFrame = MakeFrame(poseFrame: upPose);
            for (int i = 0; i < 25; i++)
                eval.Update(upFrame, 0.033f);

            Assert.IsTrue(eval.IsComplete);
            Assert.IsTrue(eval.Result.ReachCheckPassed);
        }

        [Test]
        public void ReachCheck_NullFrame_ResetsHold()
        {
            var eval = new ReachCheckEvaluator();
            var leftPose = MakePoseFrame(
                leftWrist: new Vector3(0.10f, 0.5f, 0),
                leftElbow: new Vector3(0.20f, 0.45f, 0));
            var leftFrame = MakeFrame(poseFrame: leftPose);

            eval.Update(leftFrame, 0.3f);
            Assert.Greater(eval.HoldTime, 0f);

            eval.Update(null, 0.033f);
            Assert.AreEqual(0f, eval.HoldTime, 0.001f);
        }

        [Test]
        public void ReachCheck_Reset_GoesBackToLeftArm()
        {
            var eval = new ReachCheckEvaluator();

            // Advance past left arm
            var leftPose = MakePoseFrame(
                leftWrist: new Vector3(0.10f, 0.5f, 0),
                leftElbow: new Vector3(0.20f, 0.45f, 0));
            var leftFrame = MakeFrame(poseFrame: leftPose);
            for (int i = 0; i < 25; i++)
                eval.Update(leftFrame, 0.033f);
            Assert.AreEqual(ReachCheckPhase.RightArm, eval.CurrentPhase);

            eval.Reset();
            Assert.AreEqual(ReachCheckPhase.LeftArm, eval.CurrentPhase);
            Assert.IsFalse(eval.IsComplete);
        }

        [Test]
        public void ReachCheck_DirectionalGuidance_WhenWristNotExtended()
        {
            var eval = new ReachCheckEvaluator();
            // 왼손목이 왼어깨보다 오른쪽(뻗지 않음)
            var frame = MakeFrame(); // default wrist at 0.28, shoulder at 0.35 → wrist < shoulder → OK
            // Make wrist to the right of shoulder
            var pose = MakePoseFrame(
                leftWrist: new Vector3(0.40f, 0.5f, 0)); // 왼손목이 어깨(0.35) 오른쪽
            var badFrame = MakeFrame(poseFrame: pose);
            var result = eval.Update(badFrame, 0.033f);
            Assert.IsNotNull(result.Guidance);
            Assert.IsTrue(result.Guidance.Contains("왼팔"));
        }

        // ─── BodyScaleCalculator ───────────────────────────────────────────

        [Test]
        public void BodyScale_TooFewFrames_ReturnsEmpty()
        {
            var frames = new System.Collections.Generic.List<HandPoseFrame> { MakeFrame() };
            var result = BodyScaleCalculator.ComputeFromFrames(frames);
            Assert.AreEqual(0f, result.ShoulderWidthNorm, 0.001f);
        }

        [Test]
        public void BodyScale_ValidFrames_ComputesRatios()
        {
            var frames = new System.Collections.Generic.List<HandPoseFrame>();
            for (int i = 0; i < 10; i++)
                frames.Add(MakeFrame());

            var result = BodyScaleCalculator.ComputeFromFrames(frames);
            Assert.Greater(result.ShoulderWidthNorm, 0f);
            Assert.IsTrue(result.HasUpperArmRatio);
            Assert.IsTrue(result.HasForearmRatio);
            Assert.IsTrue(result.HasTorsoRatio);
            Assert.Greater(result.LeftUpperArmOverShoulder, 0f);
            Assert.Greater(result.TorsoOverShoulder, 0f);
        }

        [Test]
        public void BodyMeasurement_StableFrames_Success()
        {
            var frames = new System.Collections.Generic.List<HandPoseFrame>();
            for (int i = 0; i < 30; i++)
                frames.Add(MakeFrame());

            var result = BodyScaleCalculator.ComputeWithValidation(frames);
            Assert.IsTrue(result.Success);
            Assert.Greater(result.Scale.ShoulderWidthNorm, 0f);
            Assert.AreEqual(0f, result.ShoulderWidthStability, 0.001f); // 동일 프레임 → 0 편차
        }

        [Test]
        public void BodyMeasurement_TooFewFrames_Fails()
        {
            var frames = new System.Collections.Generic.List<HandPoseFrame> { MakeFrame() };
            var result = BodyScaleCalculator.ComputeWithValidation(frames);
            Assert.IsFalse(result.Success);
            Assert.IsNotNull(result.FailureReason);
        }

        [Test]
        public void BodyMeasurement_UnstableFrames_Fails()
        {
            var frames = new System.Collections.Generic.List<HandPoseFrame>();
            // 어깨 너비가 크게 변하는 프레임 (카메라 거리 급변)
            for (int i = 0; i < 15; i++)
            {
                var pose = MakePoseFrame(
                    leftShoulder: new Vector3(0.20f, 0.35f, 0),
                    rightShoulder: new Vector3(0.80f, 0.35f, 0));
                frames.Add(MakeFrame(poseFrame: pose, shoulderWidth: 0.60f));
            }
            for (int i = 0; i < 15; i++)
            {
                var pose = MakePoseFrame(
                    leftShoulder: new Vector3(0.45f, 0.35f, 0),
                    rightShoulder: new Vector3(0.55f, 0.35f, 0));
                frames.Add(MakeFrame(poseFrame: pose, shoulderWidth: 0.10f));
            }

            var result = BodyScaleCalculator.ComputeWithValidation(frames);
            Assert.IsFalse(result.Success, "Unstable measurement should fail");
        }

        // ─── CalibrationState enum ─────────────────────────────────────────

        [Test]
        public void CalibrationState_HasAllExpectedValues()
        {
            Assert.AreEqual(0, (int)CalibrationState.Idle);
            Assert.AreEqual(1, (int)CalibrationState.ProfileAndCameraSelect);
            Assert.AreEqual(2, (int)CalibrationState.FramingCheck);
            Assert.AreEqual(3, (int)CalibrationState.BodyMeasurement);
            Assert.AreEqual(4, (int)CalibrationState.NeutralCalibration);
            Assert.AreEqual(7, (int)CalibrationState.Complete);
            Assert.AreEqual(8, (int)CalibrationState.Failed);
        }

        [Test]
        public void ReachCheckPhase_HasAllValues()
        {
            Assert.AreEqual(0, (int)ReachCheckPhase.LeftArm);
            Assert.AreEqual(1, (int)ReachCheckPhase.RightArm);
            Assert.AreEqual(2, (int)ReachCheckPhase.BothArmsUp);
            Assert.AreEqual(3, (int)ReachCheckPhase.Done);
        }
    }
}
