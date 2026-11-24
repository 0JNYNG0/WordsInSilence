using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using WordsInSilence.Recognition.Runtime;

namespace WordsInSilence.Recognition.LiveInput
{
    public enum ManualDiagnosticCaptureState
    {
        Idle,
        Capturing,
        Processing,
        Completed,
        Cancelled,
        Failed,
    }

    public sealed class DiagnosticCaptureFailure
    {
        public string ExceptionType { get; }
        public string Message { get; }

        internal DiagnosticCaptureFailure(Exception exception)
        {
            ExceptionType = exception.GetType().FullName;
            Message = exception.Message;
        }
    }

    public sealed class DiagnosticRecognitionResult
    {
        public string CaptureId { get; }
        public int RawFrameCount { get; }
        public int NormalizedFrameCount { get; }
        public int ValidLandmarkCount { get; }
        public int InterpolatedLandmarkCount { get; }
        public string PredictedWord { get; }
        public string WinningPrototypeId { get; }
        public double GreetingDistance { get; }
        public double LoveDistance { get; }
        public double ThankYouDistance { get; }
        public double BestDistance { get; }
        public double Threshold { get; }
        public bool Accepted { get; }

        internal DiagnosticRecognitionResult(string captureId, ApprovedPreprocessingResult preprocessing, RuntimeRecognitionResult recognition)
        {
            CaptureId = captureId;
            RawFrameCount = preprocessing.RawFrameCount;
            NormalizedFrameCount = preprocessing.NormalizedFrameCount;
            ValidLandmarkCount = preprocessing.ValidLandmarkCount;
            InterpolatedLandmarkCount = preprocessing.InterpolatedLandmarkCount;
            PredictedWord = recognition.PredictedWord;
            WinningPrototypeId = recognition.WinningPrototypeId;
            GreetingDistance = recognition.GreetingDistance;
            LoveDistance = recognition.LoveDistance;
            ThankYouDistance = recognition.ThankYouDistance;
            BestDistance = recognition.BestDistance;
            Threshold = recognition.Threshold;
            Accepted = recognition.Accepted;
        }
    }

    /// <summary>Pure manual capture state machine over the approved adapter and Step 9A runtime service.</summary>
    public sealed class ManualDiagnosticCaptureController
    {
        readonly RuntimeRecognitionService service;
        readonly ApprovedLivePreprocessingAdapter adapter;
        readonly List<RawMediaPipeObservation> observations = new List<RawMediaPipeObservation>();
        RawObservationConvention convention;
        string captureId = string.Empty;

        public ManualDiagnosticCaptureState State { get; private set; } = ManualDiagnosticCaptureState.Idle;
        public string CaptureId => captureId;
        public int RawFrameCount => observations.Count;
        public DiagnosticRecognitionResult Result { get; private set; }
        public DiagnosticCaptureFailure Failure { get; private set; }
        public IReadOnlyList<RawMediaPipeObservation> Observations => new ReadOnlyCollection<RawMediaPipeObservation>(observations.ToArray());

        public ManualDiagnosticCaptureController(RuntimeRecognitionService service, ApprovedLivePreprocessingAdapter adapter)
        {
            this.service = service ?? throw new ArgumentNullException(nameof(service));
            this.adapter = adapter ?? throw new ArgumentNullException(nameof(adapter));
        }

        public void Begin(string newCaptureId, RawObservationConvention captureConvention)
        {
            if (State != ManualDiagnosticCaptureState.Idle) throw new InvalidOperationException($"begin is invalid in state {State}");
            if (string.IsNullOrWhiteSpace(newCaptureId)) throw new ArgumentException("capture ID is required", nameof(newCaptureId));
            captureId = newCaptureId;
            convention = captureConvention ?? throw new ArgumentNullException(nameof(captureConvention));
            observations.Clear();
            Result = null;
            Failure = null;
            State = ManualDiagnosticCaptureState.Capturing;
        }

        public void AddObservation(string observationCaptureId, RawMediaPipeObservation observation)
        {
            RequireCapture(observationCaptureId);
            if (State != ManualDiagnosticCaptureState.Capturing) throw new InvalidOperationException($"add-observation is invalid in state {State}");
            if (observation == null) throw new ArgumentNullException(nameof(observation));
            if (!convention.Equals(observation.Convention)) throw new ArgumentException("observation convention does not match capture metadata", nameof(observation));
            if (observations.Count > 0 && observation.TimestampSeconds <= observations[observations.Count - 1].TimestampSeconds)
                throw new ArgumentException("observation timestamps must be strictly increasing", nameof(observation));
            observations.Add(observation);
        }

        public DiagnosticRecognitionResult FinalizeCapture(string finalizeCaptureId)
        {
            RequireCapture(finalizeCaptureId);
            if (State != ManualDiagnosticCaptureState.Capturing) throw new InvalidOperationException($"finalize is invalid in state {State}");
            State = ManualDiagnosticCaptureState.Processing;
            try
            {
                if (observations.Count == 0) throw new InvalidOperationException("cannot finalize a capture with zero observations");
                ApprovedPreprocessingResult preprocessing = adapter.Adapt(observations);
                if (preprocessing.ValidLandmarkCount == 0) throw new InvalidOperationException("capture contains no valid normalized landmarks");

                var input = new RecognitionSequenceInputSession();
                input.Begin(RuntimeSequenceMetadata.RuntimeUnlabeled(captureId));
                foreach (NormalizedRecognitionFrame frame in preprocessing.Frames) input.AddFrame(captureId, frame);
                var sequence = input.FinalizeSequence(captureId);
                RuntimeRecognitionResult recognition = service.Recognize(captureId, sequence);
                Result = new DiagnosticRecognitionResult(captureId, preprocessing, recognition);
                State = ManualDiagnosticCaptureState.Completed;
                return Result;
            }
            catch (Exception exception)
            {
                Failure = new DiagnosticCaptureFailure(exception);
                State = ManualDiagnosticCaptureState.Failed;
                throw;
            }
        }

        public void Cancel(string cancelCaptureId)
        {
            RequireCapture(cancelCaptureId);
            if (State != ManualDiagnosticCaptureState.Capturing) throw new InvalidOperationException($"cancel is invalid in state {State}");
            State = ManualDiagnosticCaptureState.Cancelled;
        }

        public void Reset()
        {
            observations.Clear();
            convention = null;
            captureId = string.Empty;
            Result = null;
            Failure = null;
            State = ManualDiagnosticCaptureState.Idle;
        }

        void RequireCapture(string value)
        {
            if (State == ManualDiagnosticCaptureState.Idle || string.IsNullOrEmpty(captureId))
                throw new InvalidOperationException("no diagnostic capture has begun");
            if (!string.Equals(value, captureId, StringComparison.Ordinal))
                throw new ArgumentException("capture ID does not match the active capture", nameof(value));
        }
    }
}
