using System.Collections.Generic;
using NUnit.Framework;
using WordsInSilence.Core;
using WordsInSilence.Recognition;
using WordsInSilence.Segmentation;

namespace WordsInSilence.Tests.EditMode
{
    /// <summary>
    /// R6/R7 피드백 매핑, 게임 브릿지, 이벤트 테스트.
    /// </summary>
    public class SignGameBridgeTests
    {
        // ─── SignFeedbackMapper ────────────────────────────────────────────

        [Test]
        public void FeedbackMapper_CameraIssue_Detected()
        {
            Assert.IsTrue(SignFeedbackMapper.IsCameraIssue("CAMERA_OUT_OF_FRAME"));
            Assert.IsFalse(SignFeedbackMapper.IsCameraIssue("HANDSHAPE_MISMATCH"));
        }

        [Test]
        public void FeedbackMapper_GetUserMessage_ReturnsKorean()
        {
            string msg = SignFeedbackMapper.GetUserMessage("HANDSHAPE_MISMATCH");
            Assert.IsNotNull(msg);
            Assert.IsTrue(msg.Contains("손 모양"));
        }

        [Test]
        public void FeedbackMapper_UnknownCode_ReturnsCodeItself()
        {
            string msg = SignFeedbackMapper.GetUserMessage("UNKNOWN_CODE");
            Assert.AreEqual("UNKNOWN_CODE", msg);
        }

        [Test]
        public void FeedbackMapper_BuildMessages_SeparatesCameraAndPerformance()
        {
            var result = new SignRecognitionResult
            {
                Status = AttemptResultStatus.Incorrect,
            };
            result.AddFeedback("CAMERA_OUT_OF_FRAME");
            result.AddFeedback("HANDSHAPE_MISMATCH");

            var msgs = SignFeedbackMapper.BuildMessages(result);
            Assert.IsTrue(msgs.HasCameraIssues);
            Assert.IsTrue(msgs.HasPerformanceIssues);
            Assert.AreEqual(1, msgs.CameraMessages.Count);
            Assert.AreEqual(1, msgs.PerformanceMessages.Count);
        }

        [Test]
        public void FeedbackMapper_StatusMessage_Passed()
        {
            string msg = SignFeedbackMapper.GetStatusMessage(AttemptResultStatus.Passed);
            Assert.IsTrue(msg.Contains("잘했습니다"));
        }

        // ─── SignGameBridge ────────────────────────────────────────────────

        [Test]
        public void Bridge_Success_FiresExactlyOnce()
        {
            var bridge = new SignGameBridge();
            bridge.SetAllowedSigns(new[] { "SIGN002" });

            int successCount = 0;
            bridge.OnSignSuccess += _ => successCount++;

            var evt = new SignAttemptEvent
            {
                AttemptId = "attempt1",
                SignId = "SIGN002",
                Status = AttemptResultStatus.Passed,
            };

            bridge.ProcessResult(evt, 0f);
            bridge.ProcessResult(evt, 0f); // 중복 — 무시

            Assert.AreEqual(1, successCount, "Success should fire exactly once");
        }

        [Test]
        public void Bridge_Incorrect_DoesNotAdvanceQuest()
        {
            var bridge = new SignGameBridge();
            bridge.SetAllowedSigns(new[] { "SIGN002" });

            int successCount = 0;
            bridge.OnSignSuccess += _ => successCount++;

            var evt = new SignAttemptEvent
            {
                AttemptId = "attempt1",
                SignId = "SIGN002",
                Status = AttemptResultStatus.Incorrect,
            };

            var action = bridge.ProcessResult(evt, 0f);
            Assert.AreEqual(SignGameAction.Retry, action);
            Assert.AreEqual(0, successCount);
        }

        [Test]
        public void Bridge_Uncertain_TriggersCameraRecovery()
        {
            var bridge = new SignGameBridge();
            bridge.SetAllowedSigns(new[] { "SIGN002" });

            var evt = new SignAttemptEvent
            {
                AttemptId = "attempt1",
                SignId = "SIGN002",
                Status = AttemptResultStatus.Uncertain,
            };

            var action = bridge.ProcessResult(evt, 0f);
            Assert.AreEqual(SignGameAction.CameraRecovery, action);
        }

        [Test]
        public void Bridge_Cooldown_BlocksDuplicate()
        {
            var bridge = new SignGameBridge(cooldownSeconds: 5f);
            bridge.SetAllowedSigns(new[] { "SIGN002" });

            var evt1 = new SignAttemptEvent
            {
                AttemptId = "attempt1",
                SignId = "SIGN002",
                Status = AttemptResultStatus.Passed,
            };
            var evt2 = new SignAttemptEvent
            {
                AttemptId = "attempt2",
                SignId = "SIGN002",
                Status = AttemptResultStatus.Passed,
            };

            bridge.ProcessResult(evt1, 0f); // 성공
            var action = bridge.ProcessResult(evt2, 2f); // 2초 후 → cooldown 내
            Assert.AreEqual(SignGameAction.None, action);
        }

        [Test]
        public void Bridge_AfterCooldown_AllowsAgain()
        {
            var bridge = new SignGameBridge(cooldownSeconds: 3f);
            bridge.SetAllowedSigns(new[] { "SIGN002" });

            var evt1 = new SignAttemptEvent
            {
                AttemptId = "attempt1",
                SignId = "SIGN002",
                Status = AttemptResultStatus.Passed,
            };
            var evt2 = new SignAttemptEvent
            {
                AttemptId = "attempt2",
                SignId = "SIGN002",
                Status = AttemptResultStatus.Passed,
            };

            bridge.ProcessResult(evt1, 0f);
            var action = bridge.ProcessResult(evt2, 5f); // 5초 후 → cooldown 지남
            Assert.AreEqual(SignGameAction.Success, action);
        }

        [Test]
        public void Bridge_NotAllowedSign_Ignored()
        {
            var bridge = new SignGameBridge();
            bridge.SetAllowedSigns(new[] { "SIGN002" });

            var evt = new SignAttemptEvent
            {
                AttemptId = "attempt1",
                SignId = "SIGN003",
                Status = AttemptResultStatus.Passed,
            };

            var action = bridge.ProcessResult(evt, 0f);
            Assert.AreEqual(SignGameAction.None, action);
        }

        [Test]
        public void Bridge_Reset_ClearsState()
        {
            var bridge = new SignGameBridge();
            bridge.SetAllowedSigns(new[] { "SIGN002" });
            Assert.IsTrue(bridge.IsSignAllowed("SIGN002"));

            bridge.Reset();
            Assert.IsFalse(bridge.IsSignAllowed("SIGN002"));
        }

        [Test]
        public void Bridge_RetryAfterIncorrect_Works()
        {
            var bridge = new SignGameBridge();
            bridge.SetAllowedSigns(new[] { "SIGN002" });

            // 1차: 오답
            bridge.ProcessResult(new SignAttemptEvent
            {
                AttemptId = "attempt1",
                SignId = "SIGN002",
                Status = AttemptResultStatus.Incorrect,
            }, 0f);

            // 2차: 성공 (다른 attemptId)
            var action = bridge.ProcessResult(new SignAttemptEvent
            {
                AttemptId = "attempt2",
                SignId = "SIGN002",
                Status = AttemptResultStatus.Passed,
            }, 1f);

            Assert.AreEqual(SignGameAction.Success, action);
        }
    }
}
