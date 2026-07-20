namespace WordsInSilence.Camera
{
    /// <summary>
    /// 카메라 장치 정보.
    /// </summary>
    public sealed class CameraDeviceInfo
    {
        /// <summary>OS가 제공하는 장치 이름.</summary>
        public string Name;

        /// <summary>WebCamTexture.devices 배열 내 인덱스.</summary>
        public int Index;

        /// <summary>전면 카메라(셀카 카메라) 여부.</summary>
        public bool IsFrontFacing;
    }
}
