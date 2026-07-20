namespace WordsInSilence.Camera
{
    /// <summary>
    /// 카메라 시작 요청 데이터.
    /// </summary>
    public sealed class CameraStartRequest
    {
        /// <summary>사용할 카메라 장치 이름. null이면 기본(첫 번째) 장치를 사용한다.</summary>
        public string DeviceName;

        /// <summary>요청 해상도 (가로). 실제 해상도는 장치에 따라 다를 수 있다.</summary>
        public int RequestedWidth = 640;

        /// <summary>요청 해상도 (세로).</summary>
        public int RequestedHeight = 480;

        /// <summary>요청 프레임레이트.</summary>
        public int RequestedFps = 30;

        /// <summary>화면 표시 시 좌우 반전 여부 (셀카 방향).</summary>
        public bool MirrorDisplay = true;
    }
}
