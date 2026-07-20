using NUnit.Framework;
using WordsInSilence.Camera;
using WordsInSilence.Debugging;
using WordsInSilence.Settings;
using UnityEngine;

namespace WordsInSilence.Tests.EditMode
{
    public class CameraDeviceInfoTests
    {
        [Test]
        public void CameraDeviceInfo_Fields_SetCorrectly()
        {
            var info = new CameraDeviceInfo
            {
                Name = "TestCam",
                Index = 2,
                IsFrontFacing = true
            };

            Assert.AreEqual("TestCam", info.Name);
            Assert.AreEqual(2, info.Index);
            Assert.IsTrue(info.IsFrontFacing);
        }

        [Test]
        public void CameraDeviceInfo_DefaultValues_AreNull()
        {
            var info = new CameraDeviceInfo();
            Assert.IsNull(info.Name);
            Assert.AreEqual(0, info.Index);
            Assert.IsFalse(info.IsFrontFacing);
        }

        [Test]
        public void CameraStartResult_Success_HasCorrectProperties()
        {
            var result = CameraStartResult.Success("WebCam0", 640, 480, 30);

            Assert.IsTrue(result.IsSuccess);
            Assert.AreEqual("WebCam0", result.DeviceName);
            Assert.AreEqual(640, result.ActualWidth);
            Assert.AreEqual(480, result.ActualHeight);
            Assert.AreEqual(30, result.ActualFps);
            Assert.IsNull(result.ErrorCode);
        }

        [Test]
        public void CameraStartResult_Failure_HasErrorCode()
        {
            var result = CameraStartResult.Failure("CAMERA_NOT_FOUND", "카메라 없음");

            Assert.IsFalse(result.IsSuccess);
            Assert.AreEqual("CAMERA_NOT_FOUND", result.ErrorCode);
            Assert.AreEqual("카메라 없음", result.ErrorDetail);
            Assert.IsNull(result.DeviceName);
        }

        [Test]
        public void CameraStartRequest_DefaultValues_AreValid()
        {
            var req = new CameraStartRequest();

            Assert.IsNull(req.DeviceName);
            Assert.AreEqual(640, req.RequestedWidth);
            Assert.AreEqual(480, req.RequestedHeight);
            Assert.AreEqual(30, req.RequestedFps);
            Assert.IsTrue(req.MirrorDisplay);
        }

        [Test]
        public void CaptureDiagnostics_CanBeInstantiated()
        {
            var diag = new CaptureDiagnostics
            {
                DeviceName = "TestCam",
                ActualWidth = 640,
                ActualHeight = 480,
                ActualFps = 30,
                IsRunning = true,
                LastErrorCode = null
            };

            Assert.IsNotNull(diag);
            Assert.IsTrue(diag.IsRunning);
            Assert.AreEqual(640, diag.ActualWidth);
        }

        [Test]
        public void CaptureDiagnostics_ToString_ContainsKeyInfo()
        {
            var diag = new CaptureDiagnostics
            {
                DeviceName = "MyCam",
                ActualWidth = 1280,
                ActualHeight = 720,
                ActualFps = 30,
                IsRunning = true
            };

            string s = diag.ToString();
            Assert.IsTrue(s.Contains("MyCam"));
            Assert.IsTrue(s.Contains("1280"));
        }

        [Test]
        public void CaptureConfig_CanBeCreated()
        {
            // ScriptableObject는 CreateInstance로 생성한다.
            var config = ScriptableObject.CreateInstance<CaptureConfig>();

            Assert.IsNotNull(config);
            Assert.AreEqual(640, config.requestedWidth);
            Assert.AreEqual(480, config.requestedHeight);
            Assert.AreEqual(30, config.requestedFps);
            Assert.IsTrue(config.mirrorDisplay);
            Assert.Greater(config.countdownSeconds, 0f);
            Assert.Greater(config.maxRecordingSeconds, 0f);
            Assert.GreaterOrEqual(config.minValidFrameRatio, 0f);
            Assert.LessOrEqual(config.minValidFrameRatio, 1f);

            Object.DestroyImmediate(config);
        }
    }
}
