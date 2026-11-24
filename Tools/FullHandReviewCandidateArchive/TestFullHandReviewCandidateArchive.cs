// F11 console tests for the full-hand review-candidate archive.
//
// Build (.NET Framework, C# 5, no NUnit / no Unity / no other dependency):
//   csc.exe /langversion:5 /nologo /out:TestFullHandReviewCandidateArchive.exe ^
//       FullHandCommonCapture.cs FullHandReviewCandidateArchive.cs ^
//       TestFullHandReviewCandidateArchive.cs
//
// Run:
//   TestFullHandReviewCandidateArchive.exe <empty-existing-absolute-temp-dir>
//   TestFullHandReviewCandidateArchive.exe --decode <file.wisfhc>
//   TestFullHandReviewCandidateArchive.exe --emit   <existing-absolute-dir>
//
// Exit code 0 = every assertion held; non-zero = failure or bad usage.
// All capture data below is invented arithmetic. No learner data is used.
using System;
using System.IO;
using WordsInSilence.Recognition.LiveInput;
using WordsInSilence.Recognition.Personalization;

static class TestFullHandReviewCandidateArchive
{
    const int Points = 46;
    const int Axes = 3;

    // Per-slot source classification used to invent captures.
    const int SlotMissing = 0;            // no source, final invalid
    const int SlotOriginalFinal = 1;      // originally valid, final valid
    const int SlotInterpolatedFinal = 2;  // interpolated, final valid
    const int SlotOriginalDropped = 3;    // originally valid but final INVALID
                                          // (no torso transform for that frame)

    static int assertions;
    static int tests;
    static string workRoot;

    static int Main(string[] args)
    {
        try
        {
            if (args.Length == 2 && args[0] == "--decode") return Decode(args[1]);
            if (args.Length == 2 && args[0] == "--emit") return Emit(args[1]);
            if (args.Length != 1)
            {
                Console.Error.WriteLine(
                    "usage: <empty-temp-dir> | --decode <file.wisfhc> | --emit <dir>");
                return 2;
            }
            workRoot = Path.GetFullPath(args[0]);
            if (!Path.IsPathRooted(args[0]) || !Directory.Exists(workRoot))
            {
                Console.Error.WriteLine("Test directory must be an existing absolute path.");
                return 2;
            }
            if (Directory.GetFileSystemEntries(workRoot).Length != 0)
            {
                Console.Error.WriteLine("Test directory must be empty: " + workRoot);
                return 2;
            }

            Run("roundtrip_exact_with_nan_and_dropped_source", RoundTripExact);
            Run("observation_byte_layout", ObservationByteLayout);
            Run("one_hand_and_all_missing", OneHandAndAllMissing);
            Run("arbitrary_word_labels_and_determinism", ArbitraryWordLabels);
            Run("caller_arrays_are_copied", CallerArraysAreCopied);
            Run("corrupt_metadata_payload_and_digest", CorruptionDetected);
            Run("truncated_trailing_oversized_legacy", StructuralRejects);
            Run("malformed_capture_and_timestamps", MalformedCaptureRejects);
            Run("word_provenance_pairing", ProvenanceRejects);
            Run("quality_word_independence_never_eligible", QualityWordIndependence);
            Run("storage_paths_ids_and_duplicates", StorageRules);
            Run("production_count_semantics", ProductionCountSemantics);
            Run("fully_qualified_path_forms", FullyQualifiedPathForms);

            Console.WriteLine("F11 archive: " + tests + " tests, " + assertions +
                              " assertions, 0 failures.");
            return 0;
        }
        catch (Exception error)
        {
            Console.Error.WriteLine("FAILURE: " + error.Message);
            Console.Error.WriteLine(error.StackTrace);
            return 1;
        }
    }

    // ---------------------------------------------------------------- harness

    static void Run(string name, Action body)
    {
        tests++;
        try { body(); }
        catch (Exception error)
        {
            throw new Exception("test '" + name + "': " + error.Message, error);
        }
    }

    static void Check(bool condition, string label)
    {
        assertions++;
        if (!condition) throw new Exception("assertion failed: " + label);
    }

    static void Rejects<TException>(Action body, string label) where TException : Exception
    {
        assertions++;
        try { body(); }
        catch (TException) { return; }
        catch (Exception other)
        {
            throw new Exception("assertion failed: " + label + " threw " +
                                other.GetType().Name + " (" + other.Message + ")");
        }
        throw new Exception("assertion failed: " + label + " did not throw");
    }

    // ------------------------------------------------------------ synthetics

    static FullHandCommonCapture BuildCapture(int frames, Func<int, int, int> classify,
                                              int width, int height, string convention)
    {
        int slots = frames * Points;
        var coordinates = new double[slots * Axes];
        var finalMask = new bool[slots];
        var originalMask = new bool[slots];
        var interpolatedMask = new bool[slots];
        // Production count semantics (ApprovedLivePreprocessingAdapter.WriteComponent):
        // ValidPointCount = final-valid points (interpolated included);
        // InterpolatedPointCount = points that are final-valid AND interpolated.
        int finalCount = 0, finalInterpolatedCount = 0;

        for (int frame = 0; frame < frames; frame++)
            for (int point = 0; point < Points; point++)
            {
                int slot = frame * Points + point;
                int kind = classify(frame, point);
                bool isFinal = kind == SlotOriginalFinal || kind == SlotInterpolatedFinal;
                originalMask[slot] = kind == SlotOriginalFinal || kind == SlotOriginalDropped;
                interpolatedMask[slot] = kind == SlotInterpolatedFinal;
                finalMask[slot] = isFinal;
                if (isFinal)
                {
                    finalCount++;
                    if (interpolatedMask[slot]) finalInterpolatedCount++;
                }

                int at = slot * Axes;
                if (isFinal)
                {
                    coordinates[at] = 100.5 + frame * 0.25 + point;
                    coordinates[at + 1] = -20.125 + point * 0.5 - frame;
                    coordinates[at + 2] = 0.0;
                }
                else
                {
                    coordinates[at] = double.NaN;
                    coordinates[at + 1] = double.NaN;
                    coordinates[at + 2] = double.NaN;
                }
            }

        return new FullHandCommonCapture(frames, coordinates, finalMask, originalMask,
                                         interpolatedMask, finalCount, finalInterpolatedCount,
                                         width, height, convention);
    }

    // Mixed capture: valid, interpolated, missing, and originally-valid-but-dropped.
    static int MixedClassify(int frame, int point)
    {
        if (point == 3) return SlotOriginalDropped;   // source valid, final false
        if (point % 7 == 0) return SlotMissing;
        if (point % 5 == 0) return SlotInterpolatedFinal;
        if (frame == 1 && point > 40) return SlotMissing;
        return SlotOriginalFinal;
    }

    static double[] Times(int frames)
    {
        var times = new double[frames];
        for (int i = 0; i < frames; i++) times[i] = 1000.0 + i * 0.03125;
        return times;
    }

    static FullHandReviewCandidate MixedCandidate()
    {
        int frames = 4;
        return new FullHandReviewCandidate(
            BuildCapture(frames, MixedClassify, 1280, 720, "pixel_top_left_y_down_v1"),
            Times(frames), "sample-0001_ab", "target_word_alpha",
            FullHandWordEvidence.Scored("predicted_word_beta", -3.75, Sha256Label("model-A"),
                                        "wis_word_features_v3"),
            FullHandQualityEvidence.Confirmed("quality-note-77"));
    }

    static string Sha256Label(string seed)
    {
        // A deterministic 64-char lowercase hex label. Provenance metadata only:
        // it does not prove which model produced anything.
        using (var sha = System.Security.Cryptography.SHA256.Create())
        {
            byte[] hash = sha.ComputeHash(System.Text.Encoding.UTF8.GetBytes(seed));
            var text = new System.Text.StringBuilder(64);
            for (int i = 0; i < hash.Length; i++) text.Append(hash[i].ToString("x2"));
            return text.ToString();
        }
    }

    // ---------------------------------------------------------------- tests

    static void RoundTripExact()
    {
        FullHandReviewCandidate original = MixedCandidate();
        byte[] bytes = FullHandReviewCandidateCodec.Serialize(original);
        FullHandReviewCandidate decoded = FullHandReviewCandidateCodec.Deserialize(bytes);

        Check(decoded.SampleId == original.SampleId &&
              decoded.DeclaredTargetWord == original.DeclaredTargetWord &&
              decoded.FrameCount == original.FrameCount, "identity metadata survives");
        Check(decoded.Capture.ImageWidth == 1280 && decoded.Capture.ImageHeight == 720 &&
              decoded.Capture.ConventionId == "pixel_top_left_y_down_v1" &&
              decoded.Capture.SchemaId == original.Capture.SchemaId &&
              decoded.Capture.DomainContract == original.Capture.DomainContract,
              "convention, image size, schema and domain survive");
        Check(decoded.Capture.ValidPointCount == original.Capture.ValidPointCount &&
              decoded.Capture.InterpolatedPointCount ==
                  original.Capture.InterpolatedPointCount,
              "source counts survive");
        Check(decoded.WordEvidence.IsScored &&
              decoded.WordEvidence.PredictedWord == "predicted_word_beta" &&
              decoded.WordEvidence.WordScore == -3.75 &&
              decoded.WordEvidence.FeatureSchemaId == "wis_word_features_v3" &&
              decoded.WordEvidence.ModelIdentity == Sha256Label("model-A"),
              "word evidence survives");
        Check(decoded.QualityEvidence.State == FullHandReviewQualityState.Confirmed &&
              decoded.QualityEvidence.EvidenceId == "quality-note-77",
              "quality evidence survives");

        double[] wantTimes = original.CopyTimestamps(), gotTimes = decoded.CopyTimestamps();
        bool timesEqual = wantTimes.Length == gotTimes.Length;
        for (int i = 0; timesEqual && i < wantTimes.Length; i++)
            timesEqual = wantTimes[i] == gotTimes[i];
        Check(timesEqual, "timestamps survive bit-exactly");

        double[] wantCoords = original.Capture.CopyCoordinates();
        double[] gotCoords = decoded.Capture.CopyCoordinates();
        bool coordsEqual = wantCoords.Length == gotCoords.Length;
        int nanCount = 0;
        for (int i = 0; coordsEqual && i < wantCoords.Length; i++)
        {
            if (double.IsNaN(wantCoords[i])) { nanCount++; coordsEqual = double.IsNaN(gotCoords[i]); }
            else coordsEqual = wantCoords[i] == gotCoords[i];
        }
        Check(coordsEqual && nanCount > 0, "coordinates survive including NaN channels");

        bool[] wantFinal = original.Capture.CopyValidMask();
        bool[] gotFinal = decoded.Capture.CopyValidMask();
        bool[] gotOriginal = decoded.Capture.CopyOriginalValidMask();
        bool[] gotInterp = decoded.Capture.CopyInterpolatedMask();
        bool masksEqual = true;
        int droppedSource = 0;
        for (int i = 0; i < wantFinal.Length; i++)
        {
            if (wantFinal[i] != gotFinal[i]) masksEqual = false;
            if (gotOriginal[i] && gotInterp[i]) masksEqual = false;
            if (gotOriginal[i] && !gotFinal[i]) droppedSource++;
        }
        Check(masksEqual, "all three masks survive and stay disjoint");
        Check(droppedSource > 0,
              "originally-valid-but-final-invalid slots are preserved, not repaired");

        Check(decoded.Capture.PointCount == 46, "all 46 joints preserved");
        Check(decoded.IsReviewOnly && !decoded.IsEnrollmentEligible,
              "decoded record is review-only and not enrollment eligible");
    }

    static void ObservationByteLayout()
    {
        int frames = 3;
        FullHandCommonCapture capture =
            BuildCapture(frames, MixedClassify, 640, 480, "pixel_top_left_y_down_v1");
        double[] times = Times(frames);
        byte[] payload = FullHandObservationBinary.Write(capture, times);

        int slots = frames * Points;
        int expected = 24 + frames * 8 + slots * Axes * 8 + slots * 3;
        Check(payload.Length == expected && expected == 24 + 1250 * frames,
              "payload length is 24 + 1250*T");
        Check(payload[0] == 'W' && payload[1] == 'I' && payload[2] == 'S' &&
              payload[3] == 'F' && payload[4] == 'H' && payload[5] == 'O' &&
              payload[6] == 'B' && payload[7] == '1', "ASCII magic WISFHOB1");
        Check(ReadInt32(payload, 8) == 1 && ReadInt32(payload, 12) == frames &&
              ReadInt32(payload, 16) == 46 && ReadInt32(payload, 20) == 3,
              "int32LE version/T/46/3 header");

        Check(ReadDouble(payload, 24) == times[0] &&
              ReadDouble(payload, 24 + 8 * (frames - 1)) == times[frames - 1],
              "timestamps block");

        // Flattening is frame-major, then point, then axis.
        int coordsBase = 24 + frames * 8;
        int frame = 2, point = 11, axis = 1;
        int offset = coordsBase + ((frame * Points + point) * Axes + axis) * 8;
        Check(ReadDouble(payload, offset) == capture.Coordinate(frame, point, axis),
              "coordinate flattening is T-major/point/axis");

        int maskBase = coordsBase + slots * Axes * 8;
        int slot = frame * Points + point;
        Check(payload[maskBase + slot] == (capture.IsValid(frame, point) ? 1 : 0) &&
              payload[maskBase + slots + slot] ==
                  (capture.WasOriginallyValid(frame, point) ? 1 : 0) &&
              payload[maskBase + 2 * slots + slot] ==
                  (capture.WasInterpolated(frame, point) ? 1 : 0),
              "final, original-valid and interpolated mask blocks in order");

        // The record-level accessor emits exactly the same payload.
        var candidate = new FullHandReviewCandidate(capture, times, "layout-check", "w",
                                                    FullHandWordEvidence.Unscored(),
                                                    FullHandQualityEvidence.Unassessed());
        Check(BytesEqual(candidate.ToObservationBinary(), payload),
              "ToObservationBinary matches the standalone writer");
    }

    static void OneHandAndAllMissing()
    {
        // Right hand (points 25..45) entirely absent; nothing is filled in.
        int frames = 2;
        FullHandCommonCapture oneHand = BuildCapture(frames,
            delegate(int f, int p) { return p >= 25 ? SlotMissing : SlotOriginalFinal; },
            800, 600, "pixel_top_left_y_down_v1");
        var oneHandRecord = new FullHandReviewCandidate(oneHand, Times(frames), "one-hand",
            "target_word_alpha", FullHandWordEvidence.Unscored(),
            FullHandQualityEvidence.Unassessed());
        FullHandReviewCandidate decodedOneHand = FullHandReviewCandidateCodec.Deserialize(
            FullHandReviewCandidateCodec.Serialize(oneHandRecord));
        bool[] mask = decodedOneHand.Capture.CopyValidMask();
        double[] coords = decodedOneHand.Capture.CopyCoordinates();
        bool rightStillMissing = true;
        for (int f = 0; f < frames && rightStillMissing; f++)
            for (int p = 25; p < Points; p++)
            {
                int slot = f * Points + p;
                if (mask[slot] || !double.IsNaN(coords[slot * Axes]))
                {
                    rightStillMissing = false;
                    break;
                }
            }
        Check(decodedOneHand.Capture.PointCount == 46 && rightStillMissing,
              "one-hand capture keeps 46 slots with the absent hand still absent");

        // A completely empty final mask is legal review evidence.
        FullHandCommonCapture empty = BuildCapture(frames,
            delegate(int f, int p) { return SlotMissing; },
            800, 600, "pixel_top_left_y_down_v1");
        var emptyRecord = new FullHandReviewCandidate(empty, Times(frames), "all-missing",
            "target_word_alpha", FullHandWordEvidence.Unscored(),
            FullHandQualityEvidence.Confirmed("quality-note-1"));
        FullHandReviewCandidate decodedEmpty = FullHandReviewCandidateCodec.Deserialize(
            FullHandReviewCandidateCodec.Serialize(emptyRecord));
        Check(decodedEmpty.Capture.ValidPointCount == 0 &&
              decodedEmpty.Capture.InterpolatedPointCount == 0 &&
              !decodedEmpty.IsEnrollmentEligible,
              "all-missing capture archives as review evidence and stays ineligible");
    }

    static void ArbitraryWordLabels()
    {
        string[] labels = { "a", "hello world", "가나다", "word/with:odd-chars",
                            new string('x', 200) };
        bool allRoundTripped = true;
        for (int i = 0; i < labels.Length; i++)
        {
            FullHandReviewCandidate record = MixedCandidate()
                .WithSampleId("label-case-" + i)
                .WithWordEvidence(FullHandWordEvidence.Scored(labels[i], 0.0,
                    Sha256Label("model-B"), "wis_word_features_v3"));
            FullHandReviewCandidate back = FullHandReviewCandidateCodec.Deserialize(
                FullHandReviewCandidateCodec.Serialize(record));
            if (back.WordEvidence.PredictedWord != labels[i]) allRoundTripped = false;
        }
        Check(allRoundTripped, "arbitrary word labels round-trip with no per-word branch");

        // Determinism: identical inputs produce identical bytes.
        Check(BytesEqual(FullHandReviewCandidateCodec.Serialize(MixedCandidate()),
                         FullHandReviewCandidateCodec.Serialize(MixedCandidate())),
              "codec is deterministic for identical inputs");

        // A predicted word equal to the declared target changes nothing.
        FullHandReviewCandidate matched = MixedCandidate().WithWordEvidence(
            FullHandWordEvidence.Scored("target_word_alpha", 9.0, Sha256Label("model-B"),
                                        "wis_word_features_v3"));
        Check(matched.DeclaredTargetWord == matched.WordEvidence.PredictedWord &&
              matched.IsReviewOnly && !matched.IsEnrollmentEligible,
              "a matching prediction is never treated as a correctness verdict");

        Rejects<ArgumentException>(
            delegate { MixedCandidate().WithWordEvidence(
                FullHandWordEvidence.Scored(new string('y', 1100), 1.0,
                    Sha256Label("model-B"), "f")); },
            "over-long label rejected at 1024 UTF-8 bytes");
    }

    static void CallerArraysAreCopied()
    {
        int frames = 3;
        FullHandCommonCapture capture =
            BuildCapture(frames, MixedClassify, 320, 240, "pixel_top_left_y_down_v1");
        double[] times = Times(frames);
        var record = new FullHandReviewCandidate(capture, times, "copy-check", "w",
                                                 FullHandWordEvidence.Unscored(),
                                                 FullHandQualityEvidence.Unassessed());
        double originalFirst = times[0];
        times[0] = -999.0;
        Check(record.CopyTimestamps()[0] == originalFirst,
              "mutating the caller's timestamp array cannot reach the record");

        double[] copyA = record.CopyTimestamps();
        copyA[1] = -1.0;
        Check(record.CopyTimestamps()[1] != -1.0, "CopyTimestamps returns a fresh array");

        // Point 0 is a MASKED slot under MixedClassify, so its channels are NaN and
        // "== before" can never hold. Probe a known finite slot instead (point 1 of
        // frame 0 is final-valid) and additionally assert the masked slot stays NaN.
        double[] captureCoords = record.Capture.CopyCoordinates();
        int finiteChannel = (0 * Points + 1) * Axes;
        double before = captureCoords[finiteChannel];
        Check(!double.IsNaN(before), "the probed slot really is finite");
        captureCoords[finiteChannel] = -12345.0;
        captureCoords[0] = 7.0;
        double[] again = record.Capture.CopyCoordinates();
        Check(again[finiteChannel] == before && double.IsNaN(again[0]),
              "capture coordinate copies are independent");
    }

    static void CorruptionDetected()
    {
        byte[] good = FullHandReviewCandidateCodec.Serialize(MixedCandidate());
        Check(FullHandReviewCandidateCodec.Deserialize(good) != null,
              "baseline archive decodes");

        int labelAt = IndexOfAscii(good, "target_word_alpha");
        Check(labelAt > 0, "declared target label is present in the body");
        Rejects<FullHandArchiveFormatException>(
            delegate { FullHandReviewCandidateCodec.Deserialize(Flip(good, labelAt + 1)); },
            "edited metadata label detected");

        Rejects<FullHandArchiveFormatException>(
            delegate {
                FullHandReviewCandidateCodec.Deserialize(Flip(good, good.Length - 900)); },
            "edited coordinate payload detected");

        Rejects<FullHandArchiveFormatException>(
            delegate { FullHandReviewCandidateCodec.Deserialize(Flip(good, 16)); },
            "edited digest detected");
    }

    static void StructuralRejects()
    {
        byte[] good = FullHandReviewCandidateCodec.Serialize(MixedCandidate());

        byte[] truncated = new byte[good.Length - 10];
        Buffer.BlockCopy(good, 0, truncated, 0, truncated.Length);
        Rejects<FullHandArchiveFormatException>(
            delegate { FullHandReviewCandidateCodec.Deserialize(truncated); },
            "truncated archive rejected");

        byte[] trailing = new byte[good.Length + 1];
        Buffer.BlockCopy(good, 0, trailing, 0, good.Length);
        Rejects<FullHandArchiveFormatException>(
            delegate { FullHandReviewCandidateCodec.Deserialize(trailing); },
            "trailing byte rejected");

        Rejects<FullHandArchiveFormatException>(
            delegate {
                byte[] oversized = new byte[
                    FullHandReviewCandidateCodec.MaxArchiveBytes + 1];
                Buffer.BlockCopy(good, 0, oversized, 0, good.Length);
                FullHandReviewCandidateCodec.Deserialize(oversized);
            },
            "oversized archive rejected before parsing");

        Rejects<FullHandArchiveFormatException>(
            delegate {
                byte[] badVersion = (byte[])good.Clone();
                badVersion[8] = 2;
                FullHandReviewCandidateCodec.Deserialize(Reseal(badVersion));
            },
            "unsupported archive version rejected");

        Rejects<FullHandArchiveFormatException>(
            delegate {
                FullHandReviewCandidateCodec.Deserialize(
                    MixedCandidate().ToObservationBinary());
            },
            "a bare legacy WISFHOB1 payload is not accepted as an archive");

        // Digest recomputed after tampering: the body checks must still catch it.
        Rejects<FullHandArchiveFormatException>(
            delegate {
                byte[] badSchema = (byte[])good.Clone();
                int at = IndexOfAscii(badSchema, "wis_full_hand_review_candidate_v1");
                badSchema[at] = (byte)'X';
                FullHandReviewCandidateCodec.Deserialize(Reseal(badSchema));
            },
            "unknown record schema rejected even when the digest is recomputed");

        Rejects<FullHandArchiveFormatException>(
            delegate {
                byte[] badMask = (byte[])good.Clone();
                badMask[badMask.Length - 1] = 2;   // mask byte outside {0,1}
                FullHandReviewCandidateCodec.Deserialize(Reseal(badMask));
            },
            "mask byte outside {0,1} rejected even when the digest is recomputed");

        Rejects<FullHandArchiveFormatException>(
            delegate {
                byte[] badEnum = (byte[])good.Clone();
                int at = IndexOfAscii(badEnum, "quality-note-77") - 8;  // int32 quality state
                badEnum[at] = 7;
                FullHandReviewCandidateCodec.Deserialize(Reseal(badEnum));
            },
            "invalid quality enum rejected even when the digest is recomputed");
    }

    static void MalformedCaptureRejects()
    {
        int frames = 3;
        Func<int, int, int> mixed = MixedClassify;

        Rejects<ArgumentException>(
            delegate {
                double[] times = Times(frames);
                times[2] = times[1];   // not strictly increasing
                MakeRecord(BuildCapture(frames, mixed, 640, 480, "c"), times);
            },
            "non-increasing timestamps rejected");

        Rejects<ArgumentException>(
            delegate {
                double[] times = Times(frames);
                times[1] = double.NaN;
                MakeRecord(BuildCapture(frames, mixed, 640, 480, "c"), times);
            },
            "non-finite timestamp rejected");

        Rejects<ArgumentException>(
            delegate { MakeRecord(BuildCapture(frames, mixed, 640, 480, "c"),
                                  Times(frames + 1)); },
            "timestamp count mismatch rejected");

        Rejects<ArgumentException>(
            delegate { MakeRecord(BuildCapture(frames, mixed, 0, 480, "c"), Times(frames)); },
            "non-positive image width rejected");

        Rejects<ArgumentException>(
            delegate { MakeRecord(BuildCapture(frames, mixed, 640, 480, ""), Times(frames)); },
            "empty coordinate convention rejected");

        Rejects<ArgumentException>(
            delegate {
                FullHandCommonCapture bad = MutateCoordinate(
                    BuildCapture(frames, mixed, 640, 480, "c"), 1, 1, 2, 0.5);
                MakeRecord(bad, Times(frames));   // valid point with z != 0
            },
            "valid point with non-zero z rejected");

        Rejects<ArgumentException>(
            delegate {
                // point 1 of frame 0 is final-valid under MixedClassify
                FullHandCommonCapture bad = MutateCoordinate(
                    BuildCapture(frames, mixed, 640, 480, "c"), 0, 1, 0, double.NaN);
                MakeRecord(bad, Times(frames));
            },
            "valid point with NaN x rejected");

        Rejects<ArgumentException>(
            delegate {
                // point 0 of frame 0 is masked under MixedClassify: it must stay all-NaN
                FullHandCommonCapture bad = MutateCoordinate(
                    BuildCapture(frames, mixed, 640, 480, "c"), 0, 0, 1, 5.0);
                MakeRecord(bad, Times(frames));
            },
            "masked point with a non-NaN channel rejected");

        Rejects<ArgumentException>(
            delegate {
                // Source masks must be disjoint.
                FullHandCommonCapture source = BuildCapture(frames, mixed, 640, 480, "c");
                bool[] interpolated = source.CopyInterpolatedMask();
                bool[] original = source.CopyOriginalValidMask();
                int slot = IndexOfTrue(original);
                interpolated[slot] = true;
                MakeRecord(new FullHandCommonCapture(frames, source.CopyCoordinates(),
                    source.CopyValidMask(), original, interpolated,
                    source.ValidPointCount, source.InterpolatedPointCount + 1,
                    640, 480, "c"), Times(frames));
            },
            "overlapping original/interpolated source masks rejected");

        Rejects<ArgumentException>(
            delegate {
                FullHandCommonCapture source = BuildCapture(frames, mixed, 640, 480, "c");
                MakeRecord(new FullHandCommonCapture(frames, source.CopyCoordinates(),
                    source.CopyValidMask(), source.CopyOriginalValidMask(),
                    source.CopyInterpolatedMask(), source.ValidPointCount + 1,
                    source.InterpolatedPointCount, 640, 480, "c"), Times(frames));
            },
            "declared ValidPointCount inconsistent with the final mask rejected");

        Rejects<ArgumentException>(
            delegate {
                FullHandCommonCapture source = BuildCapture(frames, mixed, 640, 480, "c");
                MakeRecord(new FullHandCommonCapture(frames, source.CopyCoordinates(),
                    source.CopyValidMask(), source.CopyOriginalValidMask(),
                    source.CopyInterpolatedMask(), source.ValidPointCount,
                    source.InterpolatedPointCount + 1, 640, 480, "c"), Times(frames));
            },
            "declared InterpolatedPointCount inconsistent with final&interpolated rejected");

        Rejects<ArgumentOutOfRangeException>(
            delegate {
                int tooMany = 4097;
                MakeRecord(BuildCapture(tooMany,
                    delegate(int f, int p) { return SlotMissing; }, 640, 480, "c"),
                    Times(tooMany));
            },
            "more than 4096 frames rejected");
    }

    static void ProvenanceRejects()
    {
        FullHandWordEvidence unscored = FullHandWordEvidence.Unscored();
        Check(!unscored.IsScored && unscored.PredictedWord.Length == 0 &&
              unscored.ModelIdentity.Length == 0 && unscored.FeatureSchemaId.Length == 0,
              "unscored evidence carries no predicted word, model or feature schema");

        Rejects<ArgumentException>(
            delegate { FullHandWordEvidence.Scored("", 1.0, Sha256Label("m"), "f"); },
            "score without a predicted word rejected");
        Rejects<ArgumentOutOfRangeException>(
            delegate { FullHandWordEvidence.Scored("w", double.NaN, Sha256Label("m"), "f"); },
            "non-finite word score rejected");
        Rejects<ArgumentOutOfRangeException>(
            delegate {
                FullHandWordEvidence.Scored("w", double.PositiveInfinity,
                                            Sha256Label("m"), "f"); },
            "infinite word score rejected");
        Rejects<ArgumentException>(
            delegate { FullHandWordEvidence.Scored("w", 1.0, "", "f"); },
            "scored evidence without a model identity rejected");
        Rejects<ArgumentException>(
            delegate { FullHandWordEvidence.Scored("w", 1.0, "not-a-sha256", "f"); },
            "malformed model identity rejected");
        Rejects<ArgumentException>(
            delegate { FullHandWordEvidence.Scored("w", 1.0, Sha256Label("m"), ""); },
            "scored evidence without a feature schema rejected");
    }

    static void QualityWordIndependence()
    {
        FullHandReviewCandidate scoredButRejected = MixedCandidate()
            .WithQualityEvidence(FullHandQualityEvidence.Rejected("assessor-note-2"));
        FullHandReviewCandidate confirmedButUnscored = MixedCandidate()
            .WithWordEvidence(FullHandWordEvidence.Unscored())
            .WithQualityEvidence(FullHandQualityEvidence.Confirmed("assessor-note-3"));
        FullHandReviewCandidate unassessedUnscored = MixedCandidate()
            .WithWordEvidence(FullHandWordEvidence.Unscored())
            .WithQualityEvidence(FullHandQualityEvidence.Unassessed());

        FullHandReviewCandidate[] all = { MixedCandidate(), scoredButRejected,
                                          confirmedButUnscored, unassessedUnscored };
        bool allReviewOnly = true;
        for (int i = 0; i < all.Length; i++)
        {
            FullHandReviewCandidate back = FullHandReviewCandidateCodec.Deserialize(
                FullHandReviewCandidateCodec.Serialize(all[i]));
            if (!back.IsReviewOnly || back.IsEnrollmentEligible ||
                back.QualityEvidence.State != all[i].QualityEvidence.State ||
                back.WordEvidence.IsScored != all[i].WordEvidence.IsScored)
                allReviewOnly = false;
        }
        Check(allReviewOnly,
              "quality and word evidence vary independently; every state stays " +
              "review-only and enrollment-ineligible");

        Check(confirmedButUnscored.QualityEvidence.State ==
                  FullHandReviewQualityState.Confirmed &&
              !confirmedButUnscored.IsEnrollmentEligible,
              "Confirmed quality still yields IsEnrollmentEligible == false");

        Rejects<ArgumentException>(
            delegate { FullHandQualityEvidence.Confirmed(""); },
            "Confirmed without an evidence id rejected");
        Rejects<ArgumentException>(
            delegate { FullHandQualityEvidence.Create(
                FullHandReviewQualityState.Unassessed, "stray-id"); },
            "Unassessed with an evidence id rejected");
        Rejects<ArgumentOutOfRangeException>(
            delegate { FullHandQualityEvidence.Create((FullHandReviewQualityState)9, "x"); },
            "unknown quality state rejected");
    }

    static void StorageRules()
    {
        string saveDir = Path.Combine(workRoot, "save");
        Directory.CreateDirectory(saveDir);
        FullHandReviewCandidate record = MixedCandidate();
        string path = FullHandReviewCandidateStore.Save(record, saveDir);

        Check(File.Exists(path) &&
              path == Path.Combine(saveDir, record.SampleId + ".wisfhc"),
              "save writes <sampleId>.wisfhc into the supplied directory");
        Check(Directory.GetFiles(saveDir, "*.partial").Length == 0,
              "no partial file is left behind after a successful save");

        FullHandReviewCandidate loaded = FullHandReviewCandidateStore.Load(path);
        Check(loaded.SampleId == record.SampleId &&
              loaded.DeclaredTargetWord == record.DeclaredTargetWord &&
              loaded.FrameCount == record.FrameCount && !loaded.IsEnrollmentEligible,
              "load returns the same review-only record");

        byte[] onDisk = File.ReadAllBytes(path);
        FullHandReviewCandidate collidingOther = MixedCandidate()
            .WithQualityEvidence(FullHandQualityEvidence.Rejected("other-note"));
        Rejects<IOException>(
            delegate { FullHandReviewCandidateStore.Save(collidingOther, saveDir); },
            "duplicate save refuses to replace the existing file");
        Check(BytesEqual(File.ReadAllBytes(path), onDisk) &&
              Directory.GetFiles(saveDir, "*.partial").Length == 0,
              "the existing file is untouched and no partial is left after refusal");

        Rejects<ArgumentException>(
            delegate { FullHandReviewCandidateStore.Save(record, "relative-dir"); },
            "relative directory rejected");
        Rejects<DirectoryNotFoundException>(
            delegate { FullHandReviewCandidateStore.Save(
                record, Path.Combine(workRoot, "does-not-exist")); },
            "missing directory is never created");

        string assetsDir = Path.Combine(Path.Combine(workRoot, "Assets"), "StreamingAssets");
        Directory.CreateDirectory(assetsDir);
        Rejects<ArgumentException>(
            delegate { FullHandReviewCandidateStore.Save(record, assetsDir); },
            "a path under Assets/StreamingAssets is refused");
        Check(Directory.GetFiles(assetsDir).Length == 0,
              "nothing was written under Assets/StreamingAssets");

        Rejects<ArgumentException>(
            delegate { record.WithSampleId("../escape"); },
            "sample id with path separators rejected");
        Rejects<ArgumentException>(
            delegate { record.WithSampleId(new string('a', 65)); },
            "sample id longer than 64 characters rejected");
        Rejects<ArgumentException>(
            delegate { record.WithSampleId(""); }, "empty sample id rejected");

        Rejects<ArgumentException>(
            delegate { FullHandReviewCandidateStore.Load(
                Path.Combine(saveDir, "note.txt")); },
            "load refuses a non-.wisfhc extension");
    }

    // Independent probe of the REAL production count semantics: one frame where the
    // raw source-mask totals (3 original, 2 interpolated) deliberately differ from
    // the declared counters (2 final-valid, 1 final-valid-and-interpolated).
    static void ProductionCountSemantics()
    {
        var coordinates = new double[Points * Axes];
        var finalMask = new bool[Points];
        var originalMask = new bool[Points];
        var interpolatedMask = new bool[Points];
        for (int i = 0; i < coordinates.Length; i++) coordinates[i] = double.NaN;

        finalMask[0] = true; finalMask[1] = true;
        originalMask[0] = true; originalMask[2] = true; originalMask[4] = true;
        interpolatedMask[1] = true; interpolatedMask[3] = true;
        coordinates[0] = 12.5; coordinates[1] = -4.25; coordinates[2] = 0.0;
        coordinates[3] = 7.0; coordinates[4] = 9.75; coordinates[5] = 0.0;

        var capture = new FullHandCommonCapture(1, coordinates, finalMask, originalMask,
                                                interpolatedMask, 2, 1, 640, 480,
                                                "pixel_top_left_y_down_v1");
        FullHandReviewCandidate back = FullHandReviewCandidateCodec.Deserialize(
            FullHandReviewCandidateCodec.Serialize(MakeRecord(capture, Times(1))));

        bool[] outOriginal = back.Capture.CopyOriginalValidMask();
        bool[] outInterpolated = back.Capture.CopyInterpolatedMask();
        int rawOriginal = 0, rawInterpolated = 0;
        for (int i = 0; i < Points; i++)
        {
            if (outOriginal[i]) rawOriginal++;
            if (outInterpolated[i]) rawInterpolated++;
        }
        Check(back.Capture.ValidPointCount == 2 && back.Capture.InterpolatedPointCount == 1,
              "declared counters survive under production final-mask semantics");
        Check(rawOriginal == 3 && rawInterpolated == 2,
              "raw source-mask totals are preserved and legitimately differ from " +
              "the declared counters");
        Check(outOriginal[2] && !back.Capture.CopyValidMask()[2],
              "source provenance is retained where the final mask is false");

        Rejects<ArgumentException>(
            delegate {
                MakeRecord(new FullHandCommonCapture(1, coordinates, finalMask, originalMask,
                    interpolatedMask, 3, 1, 640, 480, "c"), Times(1));
            },
            "ValidPointCount set from the raw original mask is rejected");
        Rejects<ArgumentException>(
            delegate {
                MakeRecord(new FullHandCommonCapture(1, coordinates, finalMask, originalMask,
                    interpolatedMask, 2, 2, 640, 480, "c"), Times(1));
            },
            "InterpolatedPointCount set from the raw interpolated mask is rejected");
    }

    // Path.IsPathRooted accepts Windows root-relative and drive-relative forms, both
    // of which depend on ambient process state. Entry points must demand a fully
    // qualified path instead.
    static void FullyQualifiedPathForms()
    {
        FullHandReviewCandidate record = MixedCandidate();

        if (Path.DirectorySeparatorChar != '\\')
        {
            Rejects<ArgumentException>(
                delegate { FullHandReviewCandidateStore.ResolvePath("relative/dir", "id"); },
                "relative path rejected on POSIX");
            Check(FullHandReviewCandidateStore.ResolvePath(workRoot, "id")
                      .EndsWith("id.wisfhc", StringComparison.Ordinal),
                  "POSIX absolute path accepted");
            return;
        }

        Rejects<ArgumentException>(
            delegate { FullHandReviewCandidateStore.ResolvePath(@"\Users\Public", "id"); },
            "Windows root-relative '\\Users\\Public' rejected");
        Rejects<ArgumentException>(
            delegate { FullHandReviewCandidateStore.ResolvePath("C:.", "id"); },
            "Windows drive-relative 'C:.' rejected");
        Rejects<ArgumentException>(
            delegate { FullHandReviewCandidateStore.ResolvePath("C:work", "id"); },
            "Windows drive-relative 'C:work' rejected");
        Rejects<ArgumentException>(
            delegate { FullHandReviewCandidateStore.ResolvePath(@"\\server", "id"); },
            "incomplete UNC '\\\\server' rejected");
        Rejects<ArgumentException>(
            delegate { FullHandReviewCandidateStore.ResolvePath("file:///C:/temp", "id"); },
            "URI string rejected");
        Rejects<ArgumentException>(
            delegate { FullHandReviewCandidateStore.Save(record, @"\Users\Public"); },
            "Save refuses a root-relative directory");
        Rejects<ArgumentException>(
            delegate { FullHandReviewCandidateStore.Load(@"C:archive.wisfhc"); },
            "Load refuses a drive-relative file path");

        // A complete UNC share is accepted as a FORM: it fails later, on existence.
        Rejects<DirectoryNotFoundException>(
            delegate { FullHandReviewCandidateStore.ResolvePath(
                @"\\server\share\nowhere", "id"); },
            "complete UNC server/share passes the form check and fails on existence");

        // Drive form with either separator is accepted and normalized.
        string resolved = FullHandReviewCandidateStore.ResolvePath(workRoot, "path-form-id");
        Check(resolved == Path.Combine(Path.GetFullPath(workRoot), "path-form-id.wisfhc"),
              "drive-qualified directory is accepted and normalized");
        Check(FullHandReviewCandidateStore.ResolvePath(workRoot.Replace('\\', '/'),
                                                       "path-form-id") == resolved,
              "forward-slash drive form resolves to the same normalized path");
    }

    // ------------------------------------------------------------ console modes

    static int Decode(string path)
    {
        // Pass the argument through unchanged so Load's fully-qualified guard applies.
        FullHandReviewCandidate record = FullHandReviewCandidateStore.Load(path);
        FullHandWordEvidence word = record.WordEvidence;
        Console.WriteLine("record_schema       : " + FullHandReviewCandidate.RecordSchemaId);
        Console.WriteLine("sample_id           : " + record.SampleId);
        Console.WriteLine("declared_target     : " + record.DeclaredTargetWord);
        Console.WriteLine("frames              : " + record.FrameCount);
        Console.WriteLine("points              : " + record.Capture.PointCount);
        Console.WriteLine("capture_schema      : " + record.Capture.SchemaId);
        Console.WriteLine("domain_contract     : " + record.Capture.DomainContract);
        Console.WriteLine("convention          : " + record.Capture.ConventionId);
        Console.WriteLine("image               : " + record.Capture.ImageWidth + "x" +
                          record.Capture.ImageHeight);
        Console.WriteLine("source_valid_points : " + record.Capture.ValidPointCount);
        Console.WriteLine("interpolated_points : " + record.Capture.InterpolatedPointCount);
        Console.WriteLine("word_scored         : " + word.IsScored);
        Console.WriteLine("predicted_word      : " +
                          (word.IsScored ? word.PredictedWord : "(none)"));
        Console.WriteLine("word_score          : " +
                          (word.IsScored
                               ? word.WordScore.ToString("R",
                                     System.Globalization.CultureInfo.InvariantCulture) +
                                 " (opaque, no probability semantics)"
                               : "(none)"));
        Console.WriteLine("model_identity      : " +
                          (word.ModelIdentity.Length == 0 ? "(none)" : word.ModelIdentity));
        Console.WriteLine("feature_schema      : " +
                          (word.FeatureSchemaId.Length == 0 ? "(none)"
                                                            : word.FeatureSchemaId));
        Console.WriteLine("quality_state       : " + record.QualityEvidence.State);
        Console.WriteLine("quality_evidence_id : " +
                          (record.QualityEvidence.EvidenceId.Length == 0
                               ? "(none)" : record.QualityEvidence.EvidenceId));
        Console.WriteLine("observation_bytes   : " + record.ToObservationBinary().Length);
        Console.WriteLine("review_only         : " + record.IsReviewOnly);
        Console.WriteLine("enrollment_eligible : " + record.IsEnrollmentEligible);
        return 0;
    }

    static int Emit(string directory)
    {
        string full = Path.GetFullPath(directory);
        if (!Directory.Exists(full))
        {
            Console.Error.WriteLine("--emit needs an existing directory: " + full);
            return 2;
        }
        FullHandReviewCandidate record = MixedCandidate().WithSampleId("synthetic-emit-0001");
        string archivePath = FullHandReviewCandidateStore.Save(record, full);
        string observationPath = Path.Combine(full, "observation.bin");
        if (File.Exists(observationPath))
        {
            Console.Error.WriteLine("Refusing to replace " + observationPath);
            return 2;
        }
        File.WriteAllBytes(observationPath, record.ToObservationBinary());
        Console.WriteLine("archive     : " + archivePath + " (" +
                          new FileInfo(archivePath).Length + " bytes)");
        Console.WriteLine("observation : " + observationPath + " (" +
                          new FileInfo(observationPath).Length + " bytes, WISFHOB1)");
        Console.WriteLine("Synthetic invented data only; frames=" + record.FrameCount);
        return 0;
    }

    // ---------------------------------------------------------------- helpers

    static FullHandReviewCandidate MakeRecord(FullHandCommonCapture capture, double[] times)
    {
        return new FullHandReviewCandidate(capture, times, "malformed-case", "w",
                                           FullHandWordEvidence.Unscored(),
                                           FullHandQualityEvidence.Unassessed());
    }

    static FullHandCommonCapture MutateCoordinate(FullHandCommonCapture source, int frame,
                                                  int point, int axis, double value)
    {
        double[] coordinates = source.CopyCoordinates();
        coordinates[(frame * Points + point) * Axes + axis] = value;
        return new FullHandCommonCapture(source.FrameCount, coordinates,
            source.CopyValidMask(), source.CopyOriginalValidMask(),
            source.CopyInterpolatedMask(), source.ValidPointCount,
            source.InterpolatedPointCount, source.ImageWidth, source.ImageHeight,
            source.ConventionId);
    }

    static int IndexOfTrue(bool[] mask)
    {
        for (int i = 0; i < mask.Length; i++) if (mask[i]) return i;
        throw new Exception("no true slot in mask");
    }

    static byte[] Flip(byte[] source, int index)
    {
        byte[] copy = (byte[])source.Clone();
        copy[index] = (byte)(copy[index] ^ 0x40);
        return copy;
    }

    // Recompute the envelope digest so that body-level validation is exercised
    // independently of the corruption check.
    static byte[] Reseal(byte[] file)
    {
        byte[] copy = (byte[])file.Clone();
        int bodyLength = ReadInt32(copy, 12);
        var body = new byte[bodyLength];
        Buffer.BlockCopy(copy, 48, body, 0, bodyLength);
        using (var sha = System.Security.Cryptography.SHA256.Create())
            Buffer.BlockCopy(sha.ComputeHash(body), 0, copy, 16, 32);
        return copy;
    }

    static int IndexOfAscii(byte[] data, string needle)
    {
        for (int i = 0; i + needle.Length <= data.Length; i++)
        {
            bool hit = true;
            for (int j = 0; j < needle.Length; j++)
                if (data[i + j] != (byte)needle[j]) { hit = false; break; }
            if (hit) return i;
        }
        return -1;
    }

    static bool BytesEqual(byte[] left, byte[] right)
    {
        if (left.Length != right.Length) return false;
        for (int i = 0; i < left.Length; i++) if (left[i] != right[i]) return false;
        return true;
    }

    static int ReadInt32(byte[] data, int at)
    {
        return data[at] | (data[at + 1] << 8) | (data[at + 2] << 16) | (data[at + 3] << 24);
    }

    static double ReadDouble(byte[] data, int at)
    {
        long bits = 0;
        for (int i = 0; i < 8; i++) bits |= ((long)data[at + i]) << (8 * i);
        return BitConverter.Int64BitsToDouble(bits);
    }
}
