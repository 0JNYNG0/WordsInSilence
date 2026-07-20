using System.Collections;
using System.Threading;
using System.Threading.Tasks;
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
            // 이 테스트는 WebCamTexture.devices가 비어있는 환경에서만 통과한다.
            // 웹캠이 연결된 환경에서는 이 테스트가 실패할 수 있으므로,
            // 실제 카메라 없는 CI 환경에서 실행을 권장한다.
            var devices = WebCamTexture.devices;
            if (devices.Length > 0)
            {
                Assert.Ignore("웹캠이 연결되어 있어 이 테스트를 건너뜁니다. 카메라 없는 환경에서 실행하세요.");
                yield break;
            }

            var cts = new CancellationTokenSource();
            Task<CameraStartResult> task = null;

            var request = new CameraStartRequest();

            // 코루틴에서 Task를 실행하기 위해 래퍼 사용
            bool done = false;
            CameraStartResult result = null;
            _ = Task.Run(async () =>
            {
                result = await _source.StartAsync(request, cts.Token);
                done = true;
            });

            yield return new WaitUntil(() => done);

            Assert.IsNotNull(result);
            Assert.IsFalse(result.IsSuccess);
            Assert.AreEqual("CAMERA_NOT_FOUND", result.ErrorCode);
        }

        [UnityTest]
        public IEnumerator StartAsync_WhenAlreadyRunning_ReturnsCameraAlreadyRunning()
        {
            // 실제 웹캠 없이 테스트하기 위해 내부 상태를 직접 조작할 수 없으므로,
            // 두 번 StartAsync를 호출하는 시나리오를 시뮬레이션한다.
            // 웹캠이 없는 환경에서 첫 StartAsync는 CAMERA_NOT_FOUND를 반환하므로
            // 두 번째 호출 테스트는 웹캠이 있는 환경을 요구한다.
            Assert.Ignore("이 테스트는 실제 웹캠이 필요합니다. 수동 체크리스트 MT-4를 참고하세요.");
            yield return null;
        }

        [UnityTest]
        public IEnumerator StopAsync_WhenNotRunning_DoesNotThrow()
        {
            bool threw = false;
            bool done = false;

            _ = Task.Run(async () =>
            {
                try
                {
                    await _source.StopAsync(CancellationToken.None);
                }
                catch
                {
                    threw = true;
                }
                finally
                {
                    done = true;
                }
            });

            yield return new WaitUntil(() => done);

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

            bool done = false;
            CameraStartResult result = null;

            var request = new CameraStartRequest { DeviceName = "존재하지않는카메라_XYZ_12345" };
            _ = Task.Run(async () =>
            {
                result = await _source.StartAsync(request, CancellationToken.None);
                done = true;
            });

            yield return new WaitUntil(() => done);

            Assert.IsNotNull(result);
            Assert.IsFalse(result.IsSuccess);
            Assert.AreEqual("CAMERA_DEVICE_NOT_FOUND", result.ErrorCode);
        }

        [Test]
        [Ignore("실제 웹캠 필요 — 수동 체크리스트 MT-1 참고")]
        public void StartAsync_WithRealCamera_ShouldSucceed()
        {
            // 수동 테스트: Docs/Testing/MANUAL_TEST_CHECKLIST.md MT-1 항목 참고
        }
    }
}
