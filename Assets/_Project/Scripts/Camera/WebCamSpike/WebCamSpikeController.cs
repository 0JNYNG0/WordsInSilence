using System;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using WordsInSilence.Camera;
using WordsInSilence.Debugging;
using WordsInSilence.Settings;

namespace WordsInSilence.Camera.WebCamSpike
{
    /// <summary>
    /// 01_WebCamSpike Scene의 메인 컨트롤러.
    /// WebCamSpikeUI와 함께 동작하며, ICameraFrameSource를 통해 카메라를 제어한다.
    /// </summary>
    public sealed class WebCamSpikeController : MonoBehaviour
    {
        [Header("References")]
        [SerializeField] WebCamSpikeUI _ui;
        [SerializeField] UnityEngine.UI.RawImage _rawImage;
        [SerializeField] CaptureConfig _config;

        ICameraFrameSource _cameraSource;
        CancellationTokenSource _cts;
        bool _mirrorDisplay;

        void Awake()
        {
            _cameraSource = new UnityWebCamFrameSource();
            _cameraSource.DiagnosticsUpdated += OnDiagnosticsUpdated;

            _mirrorDisplay = _config != null && _config.mirrorDisplay;

            PopulateDeviceDropdown();
        }

        void Start()
        {
            _ui.SetStatus("준비됨. Start 버튼을 눌러 카메라를 시작하세요.");
            _ui.SetButtons(startEnabled: true, stopEnabled: false, restartEnabled: false);
        }

        void OnDestroy()
        {
            _cts?.Cancel();
            _cts?.Dispose();

            if (_cameraSource is IDisposable disposable)
                disposable.Dispose();
        }

        void PopulateDeviceDropdown()
        {
            var devices = _cameraSource.GetAvailableDevices();
            var names = new System.Collections.Generic.List<string>();

            if (devices.Length == 0)
            {
                names.Add("(카메라 없음)");
            }
            else
            {
                foreach (var d in devices)
                    names.Add(d.Name);
            }

            _ui.SetDeviceOptions(names);
        }

        public void OnStartClicked()
        {
            _cts?.Cancel();
            _cts?.Dispose();
            _cts = new CancellationTokenSource();
            _ = StartCameraAsync(_cts.Token);
        }

        public void OnStopClicked()
        {
            _cts?.Cancel();
            _ = StopCameraAsync();
        }

        public void OnRestartClicked()
        {
            _cts?.Cancel();
            _cts?.Dispose();
            _cts = new CancellationTokenSource();
            _ = RestartCameraAsync(_cts.Token);
        }

        public void OnMirrorToggled(bool value)
        {
            _mirrorDisplay = value;
            ApplyUvRect();
        }

        async Task StartCameraAsync(CancellationToken ct)
        {
            _ui.SetStatus("카메라 시작 중...");
            _ui.SetButtons(startEnabled: false, stopEnabled: false, restartEnabled: false);

            var request = BuildRequest();
            var result = await _cameraSource.StartAsync(request, ct);

            if (ct.IsCancellationRequested) return;

            if (!result.IsSuccess)
            {
                _ui.SetStatus($"오류: {result.ErrorDetail} [{result.ErrorCode}]");
                _ui.SetButtons(startEnabled: true, stopEnabled: false, restartEnabled: false);
                return;
            }

            _rawImage.texture = _cameraSource.ActiveTexture;
            ApplyRotation();
            ApplyUvRect();

            _ui.SetStatus($"카메라 실행 중: {result.DeviceName} ({result.ActualWidth}x{result.ActualHeight} @ {result.ActualFps}fps)");
            _ui.SetButtons(startEnabled: false, stopEnabled: true, restartEnabled: true);
        }

        async Task StopCameraAsync()
        {
            _ui.SetStatus("카메라 정지 중...");
            _ui.SetButtons(startEnabled: false, stopEnabled: false, restartEnabled: false);

            await _cameraSource.StopAsync(CancellationToken.None);

            _rawImage.texture = null;

            _ui.SetStatus("카메라 정지됨.");
            _ui.SetButtons(startEnabled: true, stopEnabled: false, restartEnabled: false);
        }

        async Task RestartCameraAsync(CancellationToken ct)
        {
            await _cameraSource.StopAsync(ct);
            if (!ct.IsCancellationRequested)
                await StartCameraAsync(ct);
        }

        CameraStartRequest BuildRequest()
        {
            string selectedDevice = _ui.GetSelectedDeviceName();
            return new CameraStartRequest
            {
                DeviceName = selectedDevice,
                RequestedWidth = _config != null ? _config.requestedWidth : 640,
                RequestedHeight = _config != null ? _config.requestedHeight : 480,
                RequestedFps = _config != null ? _config.requestedFps : 30,
                MirrorDisplay = _mirrorDisplay
            };
        }

        void ApplyUvRect()
        {
            if (_rawImage == null) return;

            var tex = _cameraSource.ActiveTexture;
            bool verticallyMirrored = tex != null && tex.videoVerticallyMirrored;

            float u = _mirrorDisplay ? 1f : 0f;
            float v = verticallyMirrored ? 1f : 0f;
            float w = _mirrorDisplay ? -1f : 1f;
            float h = verticallyMirrored ? -1f : 1f;

            _rawImage.uvRect = new Rect(u, v, w, h);
        }

        void ApplyRotation()
        {
            if (_rawImage == null || _cameraSource.ActiveTexture == null) return;
            int rotation = _cameraSource.ActiveTexture.videoRotationAngle;
            _rawImage.rectTransform.localRotation = Quaternion.Euler(0f, 0f, -rotation);
        }

        void OnDiagnosticsUpdated(CaptureDiagnostics diag)
        {
            _ui.SetDiagnostics(diag.ToString());
        }
    }
}
