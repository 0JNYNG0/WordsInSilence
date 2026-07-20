using System;
using System.Threading;
using TMPro;
using UnityEngine;
using UnityEngine.UI;
using WordsInSilence.Core;

#if UNITY_EDITOR
using UnityEditor;
#endif

namespace WordsInSilence.Recognition
{
    /// <summary>
    /// Phase 3 인식 테스트 UI.
    /// Record → Save로 포즈를 바로 SignDictionary에 등록하고, 인식 결과를 확인한다.
    /// </summary>
    public sealed class RecognitionTestUI : MonoBehaviour
    {
        [Header("Recording")]
        [SerializeField] MotionRecorder _recorder;
        [SerializeField] SignDictionary _dictionary;
        [SerializeField] Button _btnRecord;
        [SerializeField] Button _btnStop;
        [SerializeField] TMP_Text _textResult;
        [SerializeField] float _recordDuration = 2f;

        [Header("Pose Registration")]
        [SerializeField] TMP_InputField _inputSignId;
        [SerializeField] TMP_InputField _inputDisplayName;
        [SerializeField] Button _btnSave;
        [SerializeField] float _defaultThreshold = 0.15f;

        CancellationTokenSource _cts;
        Vector3[] _lastAvgPose;

        void Start()
        {
            _btnRecord.onClick.AddListener(OnRecordClicked);
            _btnStop.onClick.AddListener(OnStopClicked);
            _btnStop.interactable = false;

            if (_btnSave != null)
            {
                _btnSave.onClick.AddListener(OnSaveClicked);
                _btnSave.interactable = false;
            }
        }

        void OnDestroy()
        {
            _cts?.Cancel();
            _cts?.Dispose();
        }

        async void OnRecordClicked()
        {
            if (_recorder.IsRecording) return;

            _cts?.Dispose();
            _cts = new CancellationTokenSource();

            _btnRecord.interactable = false;
            _btnStop.interactable = true;
            if (_btnSave != null) _btnSave.interactable = false;
            _textResult.text = $"녹화 중... ({_recordDuration}s)";

            try
            {
                var sample = await _recorder.RecordAsync(_recordDuration, _cts.Token);
                ProcessSample(sample);
            }
            catch (OperationCanceledException)
            {
                _textResult.text = "녹화 취소됨";
            }
            catch (Exception ex)
            {
                _textResult.text = "오류 발생";
                Debug.LogError($"[RecognitionTestUI] {ex}");
            }
            finally
            {
                _btnRecord.interactable = true;
                _btnStop.interactable = false;
            }
        }

        void OnStopClicked()
        {
            _cts?.Cancel();
        }

        void OnSaveClicked()
        {
            if (_lastAvgPose == null)
            {
                _textResult.text = "저장 실패: 녹화된 포즈가 없습니다";
                return;
            }

            if (_dictionary == null)
            {
                _textResult.text = "저장 실패: Dictionary가 연결되지 않았습니다";
                return;
            }

            string signId = _inputSignId != null ? _inputSignId.text.Trim() : "";
            string displayName = _inputDisplayName != null ? _inputDisplayName.text.Trim() : "";

            if (string.IsNullOrEmpty(signId))
            {
                _textResult.text = "저장 실패: Sign ID를 입력하세요";
                return;
            }

            if (string.IsNullOrEmpty(displayName))
                displayName = signId;

            // 같은 ID가 있으면 덮어쓰기
            int existingIdx = _dictionary.signs.FindIndex(e => e.signId == signId);
            var entry = new SignEntry
            {
                signId = signId,
                displayName = displayName,
                landmarks = _lastAvgPose,
                threshold = _defaultThreshold
            };

            if (existingIdx >= 0)
            {
                _dictionary.signs[existingIdx] = entry;
                _textResult.text = $"덮어씀: {displayName} (ID: {signId})";
            }
            else
            {
                _dictionary.signs.Add(entry);
                _textResult.text = $"저장 완료: {displayName} (ID: {signId}) — 총 {_dictionary.signs.Count}개";
            }

#if UNITY_EDITOR
            EditorUtility.SetDirty(_dictionary);
            AssetDatabase.SaveAssets();
            Debug.Log($"[RecognitionTestUI] '{signId}' 저장됨 → DefaultSignDictionary.asset");
#endif
            if (_btnSave != null) _btnSave.interactable = false;
            _lastAvgPose = null;
        }

        void ProcessSample(SignMotionSample sample)
        {
            if (sample == null || sample.Frames.Length == 0)
            {
                _textResult.text = "손이 감지되지 않았습니다";
                return;
            }

            var avgPose = PoseTemplateEvaluator.ComputeAveragePose(sample);
            if (avgPose == null)
            {
                _textResult.text = "평균 포즈 계산 실패";
                return;
            }

            _lastAvgPose = avgPose;
            if (_btnSave != null) _btnSave.interactable = true;

            var evaluator = new PoseTemplateEvaluator(_dictionary);
            var match = evaluator.Evaluate(avgPose);

            if (match.HasValue)
                _textResult.text = $"인식: {match.Value.displayName}";
            else
                _textResult.text = "인식 실패 — 포즈를 저장하려면 ID 입력 후 Save";
        }
    }
}
