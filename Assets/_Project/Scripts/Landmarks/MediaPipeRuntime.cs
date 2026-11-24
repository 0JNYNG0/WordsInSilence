using Mediapipe;

namespace WordsInSilence.Landmarks
{
    /// <summary>
    /// MediaPipe Glog/Protobuf 초기화를 레퍼런스 카운터로 관리한다.
    /// 여러 Provider (Hand, Pose 등)가 동시에 사용해도 Glog가 중복 초기화되지 않도록 한다.
    /// </summary>
    internal static class MediaPipeRuntime
    {
        static int _refCount;

        public static void EnsureInitialized()
        {
            if (_refCount++ == 0)
            {
                Protobuf.SetLogHandler(Protobuf.DefaultLogHandler);
                Glog.Initialize("mediapipe");
            }
        }

        public static void Release()
        {
            if (_refCount <= 0)
            {
                // nothing is acquired; an unbalanced release must not shut down a
                // runtime another owner may still be using
                _refCount = 0;
                return;
            }
            if (--_refCount == 0)
            {
                Protobuf.ResetLogHandler();
                Glog.Shutdown();
            }
        }
    }
}
