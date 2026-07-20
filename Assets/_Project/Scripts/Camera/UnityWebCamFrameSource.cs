using System;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using WordsInSilence.Debugging;

namespace WordsInSilence.Camera
{
    /// <summary>
    /// Unity WebCamTexture 기반 ICameraFrameSource 구현체.
    /// MonoBehaviour가 아니므로 소유자가 OnDestroy에서 반드시 StopAsync 또는 Dispose를 호출해야 한다.
    /// </summary>
    public sealed class UnityWebCamFrameSource : ICameraFrameSource, IDisposable
    {
        const float InitTimeoutSeconds = 3f;
        const int InitPollIntervalMs = 50;

        WebCamTexture _texture;
        bool _isRunning;
        string _activeDeviceName;
        bool _disposed;

        public bool IsRunning => _isRunning;
        public string ActiveDeviceName => _activeDeviceName;
        public WebCamTexture ActiveTexture => _texture;

        public event Action<CaptureDiagnostics> DiagnosticsUpdated;

        public CameraDeviceInfo[] GetAvailableDevices()
        {
            var devices = WebCamTexture.devices;
            var result = new CameraDeviceInfo[devices.Length];
            for (int i = 0; i < devices.Length; i++)
            {
                result[i] = new CameraDeviceInfo
                {
                    Name = devices[i].name,
                    Index = i,
                    IsFrontFacing = devices[i].isFrontFacing
                };
            }
            return result;
        }

        public async Task<CameraStartResult> StartAsync(CameraStartRequest request, CancellationToken ct)
        {
            if (_disposed)
                return CameraStartResult.Failure("CAMERA_UNKNOWN_ERROR", "이미 해제된 인스턴스입니다.");

            if (_isRunning)
                return CameraStartResult.Failure("CAMERA_ALREADY_RUNNING", "카메라가 이미 실행 중입니다.");

            var devices = WebCamTexture.devices;
            if (devices == null || devices.Length == 0)
                return CameraStartResult.Failure("CAMERA_NOT_FOUND", "연결된 카메라 장치가 없습니다.");

            string deviceName = request.DeviceName;
            if (string.IsNullOrEmpty(deviceName))
            {
                deviceName = devices[0].name;
            }
            else
            {
                bool found = false;
                foreach (var d in devices)
                {
                    if (d.name == deviceName) { found = true; break; }
                }
                if (!found)
                    return CameraStartResult.Failure("CAMERA_DEVICE_NOT_FOUND",
                        $"장치 '{deviceName}'를 찾을 수 없습니다.");
            }

            try
            {
                _texture = new WebCamTexture(
                    deviceName,
                    request.RequestedWidth,
                    request.RequestedHeight,
                    request.RequestedFps);

                _texture.Play();

                // WebCamTexture.Play() 직후에는 width가 16으로 반환될 수 있다.
                // 실제 해상도가 확정될 때까지 대기한다.
                float elapsed = 0f;
                while (_texture.width <= 16)
                {
                    if (ct.IsCancellationRequested)
                    {
                        CleanupTexture();
                        return CameraStartResult.Failure("CAMERA_UNKNOWN_ERROR", "시작이 취소되었습니다.");
                    }

                    elapsed += InitPollIntervalMs / 1000f;
                    if (elapsed >= InitTimeoutSeconds)
                    {
                        CleanupTexture();
                        return CameraStartResult.Failure("CAMERA_TIMEOUT",
                            $"카메라 초기화 타임아웃 ({InitTimeoutSeconds}초).");
                    }

                    await Task.Delay(InitPollIntervalMs, ct).ConfigureAwait(false);
                }

                _isRunning = true;
                _activeDeviceName = deviceName;

                EmitDiagnostics(null);

                return CameraStartResult.Success(
                    deviceName,
                    _texture.width,
                    _texture.height,
                    (int)_texture.requestedFPS);
            }
            catch (OperationCanceledException)
            {
                CleanupTexture();
                return CameraStartResult.Failure("CAMERA_UNKNOWN_ERROR", "시작이 취소되었습니다.");
            }
            catch (Exception ex)
            {
                CleanupTexture();
                return CameraStartResult.Failure("CAMERA_UNKNOWN_ERROR", ex.Message);
            }
        }

        public Task StopAsync(CancellationToken ct)
        {
            CleanupTexture();
            return Task.CompletedTask;
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            CleanupTexture();
        }

        void CleanupTexture()
        {
            if (_texture != null)
            {
                if (_texture.isPlaying)
                    _texture.Stop();

                UnityEngine.Object.Destroy(_texture);
                _texture = null;
            }
            _isRunning = false;
            _activeDeviceName = null;
        }

        void EmitDiagnostics(string errorCode)
        {
            DiagnosticsUpdated?.Invoke(new CaptureDiagnostics
            {
                DeviceName = _activeDeviceName,
                ActualWidth = _texture != null ? _texture.width : 0,
                ActualHeight = _texture != null ? _texture.height : 0,
                ActualFps = _texture != null ? (int)_texture.requestedFPS : 0,
                IsRunning = _isRunning,
                LastErrorCode = errorCode
            });
        }
    }
}
