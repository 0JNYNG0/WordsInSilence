using System.Collections;
using System.Threading;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;
using WordsInSilence.Camera;

namespace WordsInSilence.Tests.PlayMode
{
    /// <summary>
    /// UnityWebCamFrameSource PlayMode 테스트.
    /// 실제 웹캠에 의존하는 테스트는 [Ignore]로 표시하고 수동 체크리스트로 관리한다.
    /// </summary>
    public class WebCamFrameSourceTests
    {
        UnityWebCamFrameSource _source;

        [SetUp]
        public void SetUp()
        {
            _source = new UnityWebCamFrameSource();
        }

        [TearDown]
        public void TearDown()
        {
            _source?.Dispose();
            _source = null;
        }

        [UnityTest]
        public IEnumerator StartAsync_WhenNoCamerasAvailable_ReturnsCameraNotFound()
        {
            var devices = WebCamTexture.devices;
            if (devices.Length > 0)
            {
                Assert.Ignore("웹캠이 연결되어 있어 이 테스트를 건너뜁니다. 카메라 없는 환경에서 실행하세요.");
                yield break;
            }

            var request = new CameraStartRequest();
            var task = _source.StartAsync(request, CancellationToken.None);
            yield return new WaitUntil(() => task.IsCompleted);

            Assert.IsNotNull(task.Result);
            Assert.IsFalse(task.Result.IsSuccess);
            Assert.AreEqual("CAMERA_NOT_FOUND", task.Result.ErrorCode);
        }

        [UnityTest]
        public IEnumerator StartAsync_WhenAlreadyRunning_ReturnsCameraAlreadyRunning()
        {
            Assert.Ignore("이 테스트는 실제 웹캠이 필요합니다. 수동 체크리스트 MT-4를 참고하세요.");
            yield return null;
        }

        [UnityTest]
        public IEnumerator StopAsync_WhenNotRunning_DoesNotThrow()
        {
            bool threw = false;

            var task = _source.StopAsync(CancellationToken.None);
            yield return new WaitUntil(() => task.IsCompleted);

            if (task.IsFaulted)
                threw = true;

            Assert.IsFalse(threw, "StopAsync가 미실행 상태에서 예외를 발생시켰습니다.");
            Assert.IsFalse(_source.IsRunning);
        }

        [UnityTest]
        public IEnumerator IsRunning_Initially_IsFalse()
        {
            Assert.IsFalse(_source.IsRunning);
            yield return null;
        }

        [UnityTest]
        public IEnumerator ActiveTexture_Initially_IsNull()
        {
            Assert.IsNull(_source.ActiveTexture);
            yield return null;
        }

        [UnityTest]
        public IEnumerator ActiveDeviceName_Initially_IsNull()
        {
            Assert.IsNull(_source.ActiveDeviceName);
            yield return null;
        }

        [UnityTest]
        public IEnumerator GetAvailableDevices_ReturnsNonNullArray()
        {
            var devices = _source.GetAvailableDevices();
            Assert.IsNotNull(devices);
            yield return null;
        }

        [UnityTest]
        public IEnumerator StartAsync_WithUnknownDeviceName_ReturnsCameraDeviceNotFound()
        {
            var devices = WebCamTexture.devices;
            if (devices.Length == 0)
            {
                Assert.Ignore("웹캠이 없어 이 테스트를 건너뜁니다.");
                yield break;
            }

            var request = new CameraStartRequest { DeviceName = "존재하지않는카메라_XYZ_12345" };
            var task = _source.StartAsync(request, CancellationToken.None);
            yield return new WaitUntil(() => task.IsCompleted);

            Assert.IsNotNull(task.Result);
            Assert.IsFalse(task.Result.IsSuccess);
            Assert.AreEqual("CAMERA_DEVICE_NOT_FOUND", task.Result.ErrorCode);
        }

        [Test]
        [Ignore("실제 웹캠 필요 — 수동 체크리스트 MT-1 참고")]
        public void StartAsync_WithRealCamera_ShouldSucceed()
        {
            // 수동 테스트: Docs/Testing/MANUAL_TEST_CHECKLIST.md MT-1 항목 참고
        }
    }
}
