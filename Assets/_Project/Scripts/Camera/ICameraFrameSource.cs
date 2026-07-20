using System;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using WordsInSilence.Debugging;

namespace WordsInSilence.Camera
{
    /// <summary>
    /// 카메라 프레임 소스 추상화 인터페이스.
    /// 게임 로직은 WebCamTexture를 직접 참조하지 않고 이 인터페이스를 통해서만 접근한다.
    /// </summary>
    public interface ICameraFrameSource
    {
        bool IsRunning { get; }
        string ActiveDeviceName { get; }
        WebCamTexture ActiveTexture { get; }

        /// <summary>
        /// 진단 정보 업데이트 시 발생. 매 프레임이 아닌 일정 주기로 발생해야 한다.
        /// </summary>
        event Action<CaptureDiagnostics> DiagnosticsUpdated;

        /// <summary>
        /// 카메라를 시작한다. 실패해도 예외를 던지지 않고 CameraStartResult로 반환한다.
        /// </summary>
        Task<CameraStartResult> StartAsync(CameraStartRequest request, CancellationToken ct);

        /// <summary>
        /// 카메라를 정지하고 리소스를 해제한다.
        /// </summary>
        Task StopAsync(CancellationToken ct);

        /// <summary>
        /// 연결된 카메라 장치 목록을 반환한다.
        /// </summary>
        CameraDeviceInfo[] GetAvailableDevices();
    }
}
