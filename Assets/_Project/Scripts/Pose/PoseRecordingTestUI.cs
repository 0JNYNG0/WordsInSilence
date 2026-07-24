using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using TMPro;
using UnityEngine;
using UnityEngine.UI;
using WordsInSilence.Recognition;

namespace WordsInSilence.Pose
{
    /// <summary>
    /// 02_PoseIntegration 씬 전용 녹화 테스트 UI.
    /// Record → (카운트다운) → Compare / Save 흐름.
    /// Compare: Recordings/ 폴더의 기준 JSON 전체와 비교 → 가장 유사한 단어 + 점수 표시.
    /// </summary>
    public sealed class PoseRecordingTestUI : MonoBehaviour
    {
        [SerializeField] HandPoseMotionRecorder _recorder;
        [SerializeField] Button _btnRecord;
        [SerializeField] Button _btnStop;
        [SerializeField] Button _btnSave;
        [SerializeField] Button _btnCompare;
        [SerializeField] Button _btnSaveSample;
        [SerializeField] TMP_InputField _inputMotionId;
        [SerializeField] TMP_InputField _inputDisplayName;
        [SerializeField] TMP_Text _textStatus;

        [Header("Countdown")]
        [SerializeField] TMP_Text _textCountdown;
        [SerializeField] int _countdownSeconds = 3;
        [SerializeField] float _recordDuration = 10f;

        [Header("Keyboard")]
        [SerializeField] KeyCode _recordKey = KeyCode.Space;

        CancellationTokenSource _cts;
        HandPoseRecording _lastRecording;

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

            if (_btnCompare != null)
            {
                _btnCompare.onClick.AddListener(OnCompareClicked);
                _btnCompare.interactable = false;
            }

            if (_btnSaveSample != null)
            {
                _btnSaveSample.onClick.AddListener(OnSaveSampleClicked);
                _btnSaveSample.interactable = false;
            }

            // SVM 모델 로드 (StreamingAssets/svm_model.json)
            if (SVMClassifier.TryLoad())
                Debug.Log("[PoseRecordingTestUI] SVM 모델 로드 완료");
            else
                Debug.LogWarning("[PoseRecordingTestUI] SVM 모델 없음 — DTW만 사용");

            HideCountdown();
        }

        void Update()
        {
            if (!Input.GetKeyDown(_recordKey)) return;

            if (_btnRecord.interactable)
                OnRecordClicked();
            else if (_btnStop.interactable)
                OnStopClicked();
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

            try
            {
                // 카운트다운
                await RunCountdownAsync(_cts.Token);

                // 녹화
                if (_textStatus != null) _textStatus.text = "녹화 중... (동작 완료 후 STOP 누르세요)";
                _lastRecording = await _recorder.RecordAsync(_recordDuration, _cts.Token);

                if (_textStatus != null)
                    _textStatus.text = $"완료 — Total:{_lastRecording.TotalFrames} Hand:{_lastRecording.ValidHandFrames} Pose:{_lastRecording.ValidPoseFrames} Combined:{_lastRecording.ValidCombinedFrames}";
                if (_btnSave != null)       _btnSave.interactable       = true;
                if (_btnCompare != null)    _btnCompare.interactable    = true;
                if (_btnSaveSample != null) _btnSaveSample.interactable = true;
            }
            catch (OperationCanceledException)
            {
                if (_textStatus != null) _textStatus.text = "녹화 취소됨";
                if (_btnCompare != null)    _btnCompare.interactable    = false;
                if (_btnSaveSample != null) _btnSaveSample.interactable = false;
            }
            catch (Exception ex)
            {
                if (_textStatus != null) _textStatus.text = "오류 발생";
                Debug.LogError($"[PoseRecordingTestUI] {ex}");
            }
            finally
            {
                HideCountdown();
                _btnRecord.interactable = true;
                _btnStop.interactable = false;
            }
        }

        async Task RunCountdownAsync(CancellationToken ct)
        {
            for (int i = _countdownSeconds; i > 0; i--)
            {
                ShowCountdown(i.ToString());
                await Task.Delay(1000, ct);
            }
            HideCountdown();
        }

        void ShowCountdown(string text)
        {
            if (_textCountdown == null) return;
            _textCountdown.text = text;
            _textCountdown.gameObject.SetActive(true);
        }

        void HideCountdown()
        {
            if (_textCountdown == null) return;
            _textCountdown.gameObject.SetActive(false);
        }

        void OnStopClicked()
        {
            // 녹화 중이면 결과 반환하며 정상 종료, 카운트다운 중이면 취소
            if (_recorder.IsRecording)
                _recorder.StopRecording();
            else
                _cts?.Cancel();
        }

        void OnCompareClicked()
        {
            HandPoseEvaluator.DiagnosticMode = true;
            if (_lastRecording == null)
            {
                if (_textStatus != null) _textStatus.text = "비교 실패: 먼저 녹화하세요";
                return;
            }

            // ── DTW 비교 ─────────────────────────────────────────────────────
            string[] ids = HandPoseRecordingRepository.ListAll();
            string dtwLine;
            if (ids == null || ids.Length == 0)
            {
                dtwLine = "DTW: 기준 파일 없음";
            }
            else
            {
                var references = new List<WordsInSilence.Pose.HandPoseRecording>();
                foreach (var id in ids)
                {
                    var rec = HandPoseRecordingRepository.Load(id);
                    if (rec != null) references.Add(rec);
                }

                var dtw = HandPoseEvaluator.FindBestMatch(_lastRecording, references);
                if (string.IsNullOrEmpty(dtw.MotionId))
                    dtwLine = "DTW: 일치 없음";
                else
                    dtwLine = $"DTW: {dtw.DisplayName} ({Mathf.RoundToInt(dtw.Score * 100f)}점)";

                Debug.Log($"[DTW] {dtwLine}");
            }

            // ── SVM 분류 ──────────────────────────────────────────────────────
            string svmLine;
            if (!SVMClassifier.IsLoaded)
            {
                svmLine = "SVM: 모델 없음";
            }
            else
            {
                var svm = SVMClassifier.Classify(_lastRecording);
                if (!svm.HasValue)
                    svmLine = "SVM: 피처 추출 실패";
                else if (svm.Value.ClassName == "_unknown")
                    svmLine = $"SVM: 인식 불가 (margin={svm.Value.ConfidenceMargin:F2})";
                else
                    svmLine = $"SVM: {svm.Value.ClassName} (margin={svm.Value.ConfidenceMargin:F2})";

                Debug.Log($"[SVM] {svmLine}");
            }

            if (_textStatus != null)
                _textStatus.text = $"{dtwLine}\n{svmLine}";
        }

        void OnSaveSampleClicked()
        {
            if (_lastRecording == null)
            {
                if (_textStatus != null) _textStatus.text = "샘플 저장 실패: 먼저 녹화하세요";
                return;
            }

            string displayName = _inputDisplayName != null ? _inputDisplayName.text.Trim() : "";
            if (string.IsNullOrEmpty(displayName))
            {
                if (_textStatus != null) _textStatus.text = "샘플 저장 실패: DisplayName(단어 이름)을 입력하세요";
                return;
            }

            _lastRecording.DisplayName = displayName;
            HandPoseRecordingRepository.SaveTrainingSample(_lastRecording);

            int count = HandPoseRecordingRepository.ListTrainingSamples().Length;
            if (_textStatus != null)
                _textStatus.text = $"샘플 저장 완료: {displayName}  (총 {count}개)";
        }

        void OnSaveClicked()
        {
            if (_lastRecording == null)
            {
                if (_textStatus != null) _textStatus.text = "저장 실패: 녹화된 데이터가 없습니다";
                return;
            }

            string motionId = _inputMotionId != null ? _inputMotionId.text.Trim() : "";
            if (string.IsNullOrEmpty(motionId))
            {
                if (_textStatus != null) _textStatus.text = "저장 실패: MotionId를 입력하세요";
                return;
            }

            string displayName = _inputDisplayName != null ? _inputDisplayName.text.Trim() : "";
            if (string.IsNullOrEmpty(displayName)) displayName = motionId;

            _lastRecording.MotionId = motionId;
            _lastRecording.DisplayName = displayName;

            HandPoseRecordingRepository.Save(_lastRecording);

            if (_textStatus != null) _textStatus.text = $"저장 완료: {motionId}";
            if (_btnSave != null) _btnSave.interactable = false;
        }
    }
}
