using System;
using UnityEngine;

namespace WordsInSilence.Recognition
{
    [Serializable]
    public struct SignEntry
    {
        public string signId;          // 고유 ID (예: "A", "B", "peace")
        public string displayName;     // UI 표시명
        [Tooltip("21개 원시 랜드마크 좌표 (Console 출력값 그대로 입력 — 정규화는 런타임에 자동 처리)")]
        public Vector3[] landmarks;    // 길이 21
        public float threshold;        // 이 거리 이하면 매칭 (기본 0.3)
    }
}
