namespace WordsInSilence.Debugging
{
    /// <summary>
    /// 카메라 캡처 실시간 진단 데이터.
    /// ICameraFrameSource.DiagnosticsUpdated 이벤트로 전달된다.
    /// </summary>
    public sealed class CaptureDiagnostics
    {
        public string DeviceName;
        public int RequestedWidth;
        public int RequestedHeight;
        public int RequestedFps;
        public int ActualWidth;
        public int ActualHeight;
        public int ActualFps;
        public float FrameProcessingMs;
        public string LastErrorCode;
        public bool IsRunning;

        public override string ToString() =>
            $"{DeviceName} | {ActualWidth}x{ActualHeight}@{ActualFps}fps | {FrameProcessingMs:F1}ms | {(IsRunning ? "Running" : "Stopped")}";
    }
}
