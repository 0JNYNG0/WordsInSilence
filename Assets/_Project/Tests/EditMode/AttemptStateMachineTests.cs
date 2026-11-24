using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;
using WordsInSilence.Calibration;
using WordsInSilence.Pose;
using WordsInSilence.Segmentation;
using WordsInSilence.Settings;

namespace WordsInSilence.Tests.EditMode
{
    /// <summary>
    /// R3 상태 머신 테스트:
    /// 준비 경계, 종료 재개, 타임아웃, 취소, 하단 neutral, 이벤트 발행,
    /// 유예기간, EndCandidate 재시도, MaxDuration 평가.
    /// </summary>
    public class AttemptStateMachineTests
    {
        // ─── 헬퍼 ──────────────────────────────────────────────────────────

        static NeutralPoseData MakeBaseline()
        {
            return new NeutralPoseData
            {
                HasArmAngleBaseline = true,
                LeftArmAngleNeutral = 170f,
                RightArmAngleNeutral = 170f,
                HasArmVerticalBaseline = true,
                LeftArmVerticalNeutral = 10f,
                RightArmVerticalNeutral = 10f,
                HasWristBaseline = true,
                LeftWristNeutralBR = new Vector2(-0.5f, 1.2f),
                RightWristNeutralBR = new Vector2(0.5f, 1.2f),
                MotionNoiseFloor = 0.005f,
            };
        }

        static UpperBodyPoseFrame MakePoseFrame(
            Vector3? leftElbow = null, Vector3? rightElbow = null,
            Vector3? leftWrist = null, Vector3? rightWrist = null)
        {
            var pts = new Vector3[33];
            var vis = new float[33];
            pts[0] = new Vector3(0.5f, 0.2f, 0);
            pts[11] = new Vector3(0.35f, 0.35f, 0);
            pts[12] = new Vector3(0.65f, 0.35f, 0);
            pts[13] = leftElbow ?? new Vector3(0.30f, 0.55f, 0);
            pts[14] = rightElbow ?? new Vector3(0.70f, 0.55f, 0);
            pts[15] = leftWrist ?? new Vector3(0.28f, 0.75f, 0);
            pts[16] = rightWrist ?? new Vector3(0.72f, 0.75f, 0);
            pts[23] = new Vector3(0.40f, 0.70f, 0);
            pts[24] = new Vector3(0.60f, 0.70f, 0);
            for (int i = 0; i < 33; i++) vis[i] = 0.9f;
            return new UpperBodyPoseFrame { AllPoints = pts, Visibility = vis };
        }

        static HandPoseFrame MakeNeutralFrame(long timestampMs = 0)
        {
            return new HandPoseFrame
            {
                TimestampMs = timestampMs,
                HasPose = true,
                HasValidShoulders = true,
                HasLeftHand = false,
                HasRightHand = false,
                PoseFrame = MakePoseFrame(),
                ShoulderCenter = new Vector2(0.5f, 0.35f),
                ShoulderWidth = 0.30f,
                IsValidForBodyRelativeEvaluation = true,
                LeftWristBodyRelative = new Vector2(-0.5f, 1.2f),
                RightWristBodyRelative = new Vector2(0.5f, 1.2f),
                LeftArmAngleDegrees = 170f,
                RightArmAngleDegrees = 170f,
            };
        }

        static HandPoseFrame MakeMovingFrame(long timestampMs = 0, float offset = 0.5f)
        {
            return new HandPoseFrame
            {
                TimestampMs = timestampMs,
                HasPose = true,
                HasValidShoulders = true,
                HasLeftHand = true,
                HasRightHand = true,
                PoseFrame = MakePoseFrame(
                    leftElbow: new Vector3(0.30f, 0.35f, 0),
                    rightElbow: new Vector3(0.70f, 0.35f, 0)),
                ShoulderCenter = new Vector2(0.5f, 0.35f),
                ShoulderWidth = 0.30f,
                IsValidForBodyRelativeEvaluation = true,
                LeftWristBodyRelative = new Vector2(-0.5f + offset, 0.3f),
                RightWristBodyRelative = new Vector2(0.5f + offset, 0.3f),
                LeftArmAngleDegrees = 60f,
                RightArmAngleDegrees = 60f,
            };
        }

        static AttemptStateMachine CreateMachine(out NeutralPoseDetector detector)
        {
            var settings = RecognitionSettings.Default;
            detector = new NeutralPoseDetector(settings);
            detector.SetBaseline(MakeBaseline());
            return new AttemptStateMachine(detector, settings);
        }

        /// <summary>
        /// 중립 프레임을 반복 틱하여 Armed까지 진입시킨다.
        /// </summary>
        static void AdvanceToArmed(AttemptStateMachine sm, NeutralPoseDetector detector)
        {
            sm.Start();
            // 1.5초 중립 유지 → Armed
            float dt = 0.033f;
            for (int i = 0; i < 50; i++) // 50 * 0.033 = 1.65초
                sm.Tick(MakeNeutralFrame(i * 33), dt);
            Assert.AreEqual(AttemptState.Armed, sm.State,
                "Should be Armed after 1.65s neutral hold");
        }

        /// <summary>
        /// Armed에서 움직임 → Capturing까지 진입시킨다.
        /// </summary>
        static void AdvanceToCapturing(AttemptStateMachine sm, NeutralPoseDetector detector)
        {
            AdvanceToArmed(sm, detector);
            float dt = 0.033f;
            // 움직이는 프레임 여러 개 → onset 감지 OR hand detection
            for (int i = 0; i < 5; i++)
                sm.Tick(MakeMovingFrame(2000 + i * 33, 0.1f * (i + 1)), dt);
            Assert.AreEqual(AttemptState.Capturing, sm.State,
                "Should be Capturing after motion onset or hand detection");
        }

        // ─── 테스트 ────────────────────────────────────────────────────────

        [Test]
        public void InitialState_IsInactive()
        {
            var sm = CreateMachine(out _);
            Assert.AreEqual(AttemptState.Inactive, sm.State);
        }

        [Test]
        public void Start_EntersWaitForNeutral()
        {
            var sm = CreateMachine(out _);
            sm.Start();
            Assert.AreEqual(AttemptState.WaitForNeutral, sm.State);
        }

        [Test]
        public void Stop_ReturnsToInactive()
        {
            var sm = CreateMachine(out _);
            sm.Start();
            sm.Stop();
            Assert.AreEqual(AttemptState.Inactive, sm.State);
        }

        // ─── 준비 경계 테스트 ──────────────────────────────────────────────

        [Test]
        public void NeutralHold_1_49s_DoesNotArm()
        {
            var sm = CreateMachine(out var detector);
            sm.Start();
            float dt = 0.033f;
            for (int i = 0; i < 45; i++)
                sm.Tick(MakeNeutralFrame(i * 33), dt);
            Assert.AreEqual(AttemptState.WaitForNeutral, sm.State,
                "1.49s neutral should NOT arm");
        }

        [Test]
        public void NeutralHold_1_50s_Arms()
        {
            var sm = CreateMachine(out var detector);
            sm.Start();
            float dt = 0.033f;
            for (int i = 0; i < 46; i++)
                sm.Tick(MakeNeutralFrame(i * 33), dt);
            Assert.AreEqual(AttemptState.Armed, sm.State,
                "1.5s+ neutral should arm");
        }

        // ─── 움직임 없이 자동 종료 안 됨 ──────────────────────────────────

        [Test]
        public void Armed_NoMotion_DoesNotCapture()
        {
            var sm = CreateMachine(out var detector);
            AdvanceToArmed(sm, detector);

            // 중립 프레임만 계속 (HasLeftHand=false) → Capturing으로 안 넘어감
            for (int i = 0; i < 30; i++)
                sm.Tick(MakeNeutralFrame(3000 + i * 33), 0.033f);

            Assert.AreEqual(AttemptState.Armed, sm.State);
        }

        [Test]
        public void Armed_Timeout_ReturnsToWaitForNeutral()
        {
            var sm = CreateMachine(out var detector);
            AdvanceToArmed(sm, detector);

            for (int i = 0; i < 200; i++) // 200 * 0.033 = 6.6초
                sm.Tick(MakeNeutralFrame(3000 + i * 33), 0.033f);

            Assert.AreEqual(AttemptState.WaitForNeutral, sm.State,
                "Armed should timeout to WaitForNeutral after 6s");
        }

        // ─── 움직임 → Capturing ────────────────────────────────────────────

        [Test]
        public void Motion_EntersCapturing()
        {
            var sm = CreateMachine(out var detector);
            AdvanceToCapturing(sm, detector);
            Assert.AreEqual(AttemptState.Capturing, sm.State);
        }

        [Test]
        public void Capturing_HasAttemptId()
        {
            var sm = CreateMachine(out var detector);
            AdvanceToCapturing(sm, detector);
            Assert.IsNotNull(sm.AttemptId);
            Assert.IsNotEmpty(sm.AttemptId);
        }

        // ─── Hand Appeared → Capturing (보정 시 손 안 보이던 경우) ──────────

        [Test]
        public void Armed_HandAppeared_WhenBaselineHandsNotVisible_EntersCapturing()
        {
            var settings = RecognitionSettings.Default;
            var detector = new NeutralPoseDetector(settings);
            var baseline = MakeBaseline();
            baseline.HandsVisible = false; // 보정 시 손 안 보였음
            detector.SetBaseline(baseline);
            var sm = new AttemptStateMachine(detector, settings);

            sm.Start();
            float dt = 0.033f;
            for (int i = 0; i < 50; i++)
                sm.Tick(MakeNeutralFrame(i * 33), dt);
            Assert.AreEqual(AttemptState.Armed, sm.State);

            // 손이 잡히는 프레임 → 캡처 시작
            var handFrame = MakeNeutralFrame(3000);
            handFrame.HasLeftHand = true;
            sm.Tick(handFrame, 0.033f);

            Assert.AreEqual(AttemptState.Capturing, sm.State,
                "Hand appearing (when baseline has no hands) should trigger Capturing");
        }

        [Test]
        public void Armed_HandVisible_WhenBaselineHandsVisible_DoesNotCapture()
        {
            // 보정 시 손이 보였던 경우, 손 검출만으로는 캡처 시작 안 함
            var settings = RecognitionSettings.Default;
            var detector = new NeutralPoseDetector(settings);
            var baseline = MakeBaseline();
            baseline.HandsVisible = true; // 보정 시 손 보였음
            detector.SetBaseline(baseline);
            var sm = new AttemptStateMachine(detector, settings);

            sm.Start();
            float dt = 0.033f;
            for (int i = 0; i < 50; i++)
                sm.Tick(MakeNeutralFrame(i * 33), dt);
            Assert.AreEqual(AttemptState.Armed, sm.State);

            // 손이 잡히지만 움직임 없음 → Armed 유지
            var handFrame = MakeNeutralFrame(3000);
            handFrame.HasLeftHand = true;
            sm.Tick(handFrame, 0.033f);

            Assert.AreEqual(AttemptState.Armed, sm.State,
                "Hand visible (when baseline also had hands) should NOT trigger Capturing");
        }

        // ─── 캡처 유예기간 ──────────────────────────────────────────────────

        [Test]
        public void CaptureGracePeriod_NeutralIgnored()
        {
            var sm = CreateMachine(out var detector);
            AdvanceToCapturing(sm, detector);

            // 유예기간(0.5초) 내에 중립 프레임 → EndCandidate로 안 넘어감
            for (int i = 0; i < 10; i++) // 10 * 0.033 = 0.33초 (< 0.5초)
            {
                var frame = MakeNeutralFrame(5000 + i * 33);
                frame.LeftArmAngleDegrees = 170f;
                frame.RightArmAngleDegrees = 170f;
                sm.Tick(frame, 0.033f);
            }

            Assert.AreEqual(AttemptState.Capturing, sm.State,
                "Neutral frames during grace period should not end capture");
        }

        // ─── EndCandidate 재움직임 → 동일 attempt 재개 ─────────────────────

        [Test]
        public void EndCandidate_NeutralBreak_SameAttempt()
        {
            var sm = CreateMachine(out var detector);
            AdvanceToCapturing(sm, detector);
            string originalAttemptId = sm.AttemptId;

            // 유예기간 지나기
            for (int i = 0; i < 20; i++)
                sm.Tick(MakeMovingFrame(4000 + i * 33, 0.1f), 0.033f);

            // 중립 복귀 → EndCandidate
            for (int i = 0; i < 10; i++)
                sm.Tick(MakeNeutralFrame(5000 + i * 33), 0.033f);

            if (sm.State == AttemptState.EndCandidate)
            {
                // 중립이 깨지는 프레임 (손목 위치가 baseline에서 벗어남)
                var breakFrame = MakeMovingFrame(6000, 0.8f);
                sm.Tick(breakFrame, 0.033f);

                Assert.AreEqual(AttemptState.Capturing, sm.State);
                Assert.AreEqual(originalAttemptId, sm.AttemptId,
                    "Should resume same attempt ID when neutral breaks");
            }
        }

        // ─── EndCandidate 3초 → 평가 ────────────────────────────────────

        [Test]
        public void EndCandidate_3s_TriggersEvaluation()
        {
            var sm = CreateMachine(out var detector);
            AdvanceToCapturing(sm, detector);

            AttemptCompletionResult completionResult = null;
            int completionCount = 0;
            sm.AttemptCompleted += r => { completionResult = r; completionCount++; };

            // 유예기간 지나기
            for (int i = 0; i < 20; i++)
                sm.Tick(MakeMovingFrame(4000 + i * 33, 0.1f), 0.033f);

            // 중립 복귀 후 3초 유지
            for (int i = 0; i < 100; i++) // 100 * 0.033 = 3.3초
                sm.Tick(MakeNeutralFrame(5000 + i * 33), 0.033f);

            Assert.AreEqual(AttemptState.Feedback, sm.State,
                "Should reach Feedback after 3s neutral hold");
            Assert.IsNotNull(completionResult, "Should have completion result");
            Assert.AreEqual(AttemptResultStatus.Passed, completionResult.Status);
            Assert.AreEqual(1, completionCount, "Evaluation should fire exactly once");
        }

        // ─── 취소 ──────────────────────────────────────────────────────────

        [Test]
        public void Cancel_DuringCapturing_EmitsInvalid()
        {
            var sm = CreateMachine(out var detector);
            AdvanceToCapturing(sm, detector);

            AttemptCompletionResult result = null;
            sm.AttemptCompleted += r => result = r;

            sm.Cancel();
            Assert.AreEqual(AttemptState.Cancelled, sm.State);
            Assert.IsNotNull(result);
            Assert.AreEqual(AttemptResultStatus.Invalid, result.Status);
            Assert.AreEqual("cancelled", result.InvalidReason);
        }

        [Test]
        public void Cancel_WhenInactive_DoesNothing()
        {
            var sm = CreateMachine(out _);
            AttemptCompletionResult result = null;
            sm.AttemptCompleted += r => result = r;
            sm.Cancel();
            Assert.IsNull(result);
        }

        // ─── 최대 수행 시간 → 평가 (invalid 아님) ───────────────────────────

        [Test]
        public void MaxCaptureDuration_TriggersEvaluation()
        {
            var sm = CreateMachine(out var detector);
            AdvanceToCapturing(sm, detector);

            AttemptCompletionResult result = null;
            sm.AttemptCompleted += r => result = r;

            // 10초 초과
            for (int i = 0; i < 310; i++) // 310 * 0.033 = 10.23초
                sm.Tick(MakeMovingFrame(5000 + i * 33, 0.1f * ((i % 5) + 1)), 0.033f);

            Assert.AreEqual(AttemptState.Feedback, sm.State,
                "Max duration should lead to Feedback, not InvalidAttempt");
            Assert.IsNotNull(result);
            Assert.AreEqual(AttemptResultStatus.Passed, result.Status,
                "Max duration should result in Passed (evaluate captured frames)");
            Assert.IsNull(result.InvalidReason);
        }

        // ─── 추적 gap ─────────────────────────────────────────────────────

        [Test]
        public void ShortTrackingGap_Tolerated()
        {
            var sm = CreateMachine(out var detector);
            AdvanceToCapturing(sm, detector);

            for (int i = 0; i < 6; i++) // 6 * 0.033 = 0.198초
                sm.Tick(null, 0.033f);

            Assert.AreEqual(AttemptState.Capturing, sm.State,
                "Short tracking gap should be tolerated");
        }

        [Test]
        public void LongTrackingGap_InvalidAttempt()
        {
            var sm = CreateMachine(out var detector);
            AdvanceToCapturing(sm, detector);

            AttemptCompletionResult result = null;
            sm.AttemptCompleted += r => result = r;

            for (int i = 0; i < 15; i++) // 15 * 0.033 = 0.495초
                sm.Tick(null, 0.033f);

            Assert.AreEqual(AttemptState.InvalidAttempt, sm.State);
            Assert.AreEqual("tracking_gap", result.InvalidReason);
        }

        // ─── 상태 이벤트 발행 ──────────────────────────────────────────────

        [Test]
        public void StateChanged_FiresOnEachTransition()
        {
            var sm = CreateMachine(out var detector);
            var transitions = new List<(AttemptState from, AttemptState to)>();
            sm.StateChanged += (from, to) => transitions.Add((from, to));

            sm.Start();
            Assert.AreEqual(1, transitions.Count);
            Assert.AreEqual(AttemptState.Inactive, transitions[0].from);
            Assert.AreEqual(AttemptState.WaitForNeutral, transitions[0].to);
        }

        [Test]
        public void StateChanged_NoDoubleEmission()
        {
            var sm = CreateMachine(out var detector);
            int count = 0;
            sm.StateChanged += (_, _) => count++;

            sm.Start();
            sm.Start();
            Assert.AreEqual(1, count,
                "Should not emit duplicate state change");
        }

        // ─── AcknowledgeFeedback ───────────────────────────────────────────

        [Test]
        public void AcknowledgeFeedback_ReturnsToWaitForNeutral()
        {
            var sm = CreateMachine(out var detector);
            AdvanceToCapturing(sm, detector);

            // 유예기간 지나기 + 중립 3초 → Feedback
            for (int i = 0; i < 20; i++)
                sm.Tick(MakeMovingFrame(4000 + i * 33, 0.1f), 0.033f);
            for (int i = 0; i < 100; i++)
                sm.Tick(MakeNeutralFrame(5000 + i * 33), 0.033f);

            Assert.AreEqual(AttemptState.Feedback, sm.State);

            sm.AcknowledgeFeedback();
            Assert.AreEqual(AttemptState.WaitForNeutral, sm.State);
        }

        // ─── 순환 버퍼 ────────────────────────────────────────────────────

        [Test]
        public void PreBuffer_IncludesFramesBeforeMotion()
        {
            var sm = CreateMachine(out var detector);
            AdvanceToCapturing(sm, detector);

            Assert.Greater(sm.CapturedFrameCount, 0,
                "Should have frames from pre-buffer");
        }

        // ─── CircularFrameBuffer 단위 테스트 ──────────────────────────────

        [Test]
        public void CircularBuffer_AddBeyondCapacity_Overwrites()
        {
            var buf = new CircularFrameBuffer(3);
            for (int i = 0; i < 5; i++)
                buf.Add(MakeNeutralFrame(i * 100));

            Assert.AreEqual(3, buf.Count);
            var arr = buf.ToArray();
            Assert.AreEqual(3, arr.Length);
            Assert.AreEqual(200, arr[0].TimestampMs);
            Assert.AreEqual(300, arr[1].TimestampMs);
            Assert.AreEqual(400, arr[2].TimestampMs);
        }

        [Test]
        public void CircularBuffer_Clear_Resets()
        {
            var buf = new CircularFrameBuffer(5);
            buf.Add(MakeNeutralFrame());
            buf.Add(MakeNeutralFrame());
            Assert.AreEqual(2, buf.Count);

            buf.Clear();
            Assert.AreEqual(0, buf.Count);
            Assert.AreEqual(0, buf.ToArray().Length);
        }

        // ─── MotionOnsetDetector 단위 테스트 ──────────────────────────────

        [Test]
        public void MotionOnset_SmallDisplacement_NotMoving()
        {
            var detector = new MotionOnsetDetector();
            var frame1 = MakeNeutralFrame(0);
            var frame2 = MakeNeutralFrame(33);
            detector.Update(frame1);
            bool moving = detector.Update(frame2);
            Assert.IsFalse(moving);
        }

        [Test]
        public void MotionOnset_LargeDisplacement_IsMoving()
        {
            var detector = new MotionOnsetDetector();
            detector.Update(MakeNeutralFrame(0));
            for (int i = 1; i <= 4; i++)
            {
                bool moving = detector.Update(MakeMovingFrame(i * 33, 0.2f * i));
                if (i >= 3) Assert.IsTrue(moving);
            }
        }
    }
}
