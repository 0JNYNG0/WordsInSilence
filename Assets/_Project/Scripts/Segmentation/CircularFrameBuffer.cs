using WordsInSilence.Pose;

namespace WordsInSilence.Segmentation
{
    /// <summary>
    /// 고정 크기 순환 버퍼. Armed 상태에서 시작 전 프레임을 보존한다.
    /// </summary>
    public sealed class CircularFrameBuffer
    {
        readonly HandPoseFrame[] _buffer;
        int _head;
        int _count;

        public int Count => _count;
        public int Capacity => _buffer.Length;

        public CircularFrameBuffer(int capacity)
        {
            _buffer = new HandPoseFrame[capacity];
        }

        public void Add(HandPoseFrame frame)
        {
            _buffer[_head] = frame;
            _head = (_head + 1) % _buffer.Length;
            if (_count < _buffer.Length) _count++;
        }

        public void Clear()
        {
            _head = 0;
            _count = 0;
            for (int i = 0; i < _buffer.Length; i++)
                _buffer[i] = null;
        }

        /// <summary>
        /// 가장 오래된 것부터 최신 순서로 배열을 반환한다.
        /// </summary>
        public HandPoseFrame[] ToArray()
        {
            var result = new HandPoseFrame[_count];
            int start = (_head - _count + _buffer.Length) % _buffer.Length;
            for (int i = 0; i < _count; i++)
                result[i] = _buffer[(start + i) % _buffer.Length];
            return result;
        }
    }
}
