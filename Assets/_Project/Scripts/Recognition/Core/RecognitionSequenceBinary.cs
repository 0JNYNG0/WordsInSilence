using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;

namespace WordsInSilence.Recognition.Core
{
    /// <summary>Reader for the deterministic little-endian WISRCB01 sequence format.</summary>
    public static class RecognitionSequenceBinary
    {
        public const uint FormatVersion = 1;
        public const uint EndianMarker = 0x01020304;
        public const int PayloadHashBytes = 32;
        static readonly byte[] Magic = Encoding.ASCII.GetBytes("WISRCB01");

        public static RecognitionSequence LoadFile(string path)
        {
            if (string.IsNullOrWhiteSpace(path)) throw new ArgumentException("path is required", nameof(path));
            return Load(File.ReadAllBytes(path));
        }

        public static RecognitionSequence Load(byte[] data)
        {
            if (data == null) throw new ArgumentNullException(nameof(data));
            if (data.Length < Magic.Length + 4 + PayloadHashBytes) throw new InvalidDataException("runtime sequence is truncated");

            int payloadLength = data.Length - PayloadHashBytes;
            byte[] actualHash;
            using (var sha = SHA256.Create()) actualHash = sha.ComputeHash(data, 0, payloadLength);
            for (int i = 0; i < PayloadHashBytes; i++)
                if (actualHash[i] != data[payloadLength + i]) throw new InvalidDataException("runtime sequence payload SHA-256 mismatch");
            string payloadHash = ToLowerHex(actualHash);

            using (var stream = new MemoryStream(data, 0, payloadLength, false))
            using (var reader = new BinaryReader(stream, Encoding.UTF8, true))
            {
                byte[] magic = ReadExact(reader, Magic.Length, "magic");
                if (!magic.SequenceEqual(Magic)) throw new InvalidDataException("unsupported runtime sequence magic");
                uint version = reader.ReadUInt32();
                if (version != FormatVersion) throw new InvalidDataException($"unsupported runtime sequence version {version}");
                uint endian = reader.ReadUInt32();
                if (endian != EndianMarker) throw new InvalidDataException("unsupported byte order");
                uint landmarkCountRaw = reader.ReadUInt32();
                uint lengthRaw = reader.ReadUInt32();
                if (landmarkCountRaw > int.MaxValue || lengthRaw > int.MaxValue) throw new InvalidDataException("runtime sequence dimensions overflow");
                int landmarkCount = (int)landmarkCountRaw;
                int length = (int)lengthRaw;

                string sampleId = ReadString(reader, "sample ID");
                string word = ReadString(reader, "word");
                string speed = ReadString(reader, "speed");
                string split = ReadString(reader, "split");
                string sourceSha = ReadString(reader, "source NPZ SHA-256");
                string logicalSha = ReadString(reader, "logical payload SHA-256");

                ulong coordinateCountRaw = reader.ReadUInt64();
                int expectedCoordinates;
                int expectedMask;
                try
                {
                    expectedCoordinates = checked(length * landmarkCount * 3);
                    expectedMask = checked(length * landmarkCount);
                }
                catch (OverflowException exc)
                {
                    throw new InvalidDataException("runtime sequence dimensions overflow", exc);
                }
                if (coordinateCountRaw != (ulong)expectedCoordinates) throw new InvalidDataException("coordinate count does not match declared shape");
                var coordinates = new double[expectedCoordinates];
                for (int i = 0; i < coordinates.Length; i++) coordinates[i] = BitConverter.Int64BitsToDouble(reader.ReadInt64());

                ulong maskBitCount = reader.ReadUInt64();
                if (maskBitCount != (ulong)expectedMask) throw new InvalidDataException("mask count does not match declared shape");
                uint maskByteCountRaw = reader.ReadUInt32();
                int expectedMaskBytes = (expectedMask + 7) / 8;
                if (maskByteCountRaw != expectedMaskBytes) throw new InvalidDataException("packed mask byte count mismatch");
                byte[] packedMask = ReadExact(reader, expectedMaskBytes, "packed mask");
                if (expectedMaskBytes > 0 && expectedMask % 8 != 0)
                {
                    int usedBits = expectedMask % 8;
                    int unusedMask = 0xFF << usedBits;
                    if ((packedMask[packedMask.Length - 1] & unusedMask) != 0) throw new InvalidDataException("non-zero unused mask bits");
                }
                var validMask = new bool[expectedMask];
                for (int i = 0; i < expectedMask; i++) validMask[i] = (packedMask[i >> 3] & (1 << (i & 7))) != 0;
                if (stream.Position != payloadLength) throw new InvalidDataException("runtime sequence has unexpected trailing payload bytes");

                try
                {
                    return new RecognitionSequence(sampleId, word, speed, split, sourceSha, logicalSha,
                        length, landmarkCount, coordinates, validMask, payloadHash);
                }
                catch (ArgumentException exc)
                {
                    throw new InvalidDataException("runtime sequence metadata or payload is invalid", exc);
                }
            }
        }

        static string ReadString(BinaryReader reader, string field)
        {
            uint byteCount = reader.ReadUInt32();
            if (byteCount == 0 || byteCount > 4096) throw new InvalidDataException($"invalid {field} length");
            byte[] bytes = ReadExact(reader, (int)byteCount, field);
            string value = new UTF8Encoding(false, true).GetString(bytes);
            if (string.IsNullOrWhiteSpace(value)) throw new InvalidDataException($"{field} is empty");
            return value;
        }

        static byte[] ReadExact(BinaryReader reader, int count, string field)
        {
            byte[] bytes = reader.ReadBytes(count);
            if (bytes.Length != count) throw new InvalidDataException($"runtime sequence is truncated while reading {field}");
            return bytes;
        }

        public static string ToLowerHex(byte[] bytes)
        {
            var builder = new StringBuilder(bytes.Length * 2);
            for (int i = 0; i < bytes.Length; i++) builder.Append(bytes[i].ToString("x2"));
            return builder.ToString();
        }
    }

    public sealed class RecognitionTemplateLibrary
    {
        static readonly string[] RequiredWords = { "GREETING", "LOVE", "THANK_YOU" };
        static readonly string[] RequiredSpeeds = { "FAST", "NORMAL", "SLOW" };
        readonly List<RecognitionSequence> templates;

        public IReadOnlyList<RecognitionSequence> Templates => templates;

        public RecognitionTemplateLibrary(IEnumerable<RecognitionSequence> source)
        {
            if (source == null) throw new ArgumentNullException(nameof(source));
            templates = source.OrderBy(item => item.Word, StringComparer.Ordinal)
                              .ThenBy(item => item.Speed, StringComparer.Ordinal)
                              .ThenBy(item => item.SampleId, StringComparer.Ordinal)
                              .ToList();
            Validate();
        }

        public static RecognitionTemplateLibrary LoadDirectory(string directory)
        {
            if (string.IsNullOrWhiteSpace(directory)) throw new ArgumentException("directory is required", nameof(directory));
            if (!Directory.Exists(directory)) throw new DirectoryNotFoundException(directory);
            string[] paths = Directory.GetFiles(directory, "*.wisrt", SearchOption.TopDirectoryOnly);
            Array.Sort(paths, StringComparer.Ordinal);
            return new RecognitionTemplateLibrary(paths.Select(RecognitionSequenceBinary.LoadFile));
        }

        void Validate()
        {
            if (templates.Count != 9) throw new InvalidDataException($"exactly nine templates are required; found {templates.Count}");
            var sampleIds = new HashSet<string>(StringComparer.Ordinal);
            var groups = new HashSet<string>(StringComparer.Ordinal);
            foreach (RecognitionSequence template in templates)
            {
                if (!string.Equals(template.Split, "train", StringComparison.Ordinal))
                    throw new InvalidDataException($"template {template.SampleId} is not train split");
                if (!RequiredWords.Contains(template.Word, StringComparer.Ordinal))
                    throw new InvalidDataException($"unsupported template word {template.Word}");
                if (!RequiredSpeeds.Contains(template.Speed, StringComparer.Ordinal))
                    throw new InvalidDataException($"unsupported template speed {template.Speed}");
                if (!sampleIds.Add(template.SampleId)) throw new InvalidDataException($"duplicate template sample {template.SampleId}");
                string group = template.Word + "__" + template.Speed;
                if (!groups.Add(group)) throw new InvalidDataException($"duplicate template group {group}");
            }
            foreach (string word in RequiredWords)
            foreach (string speed in RequiredSpeeds)
                if (!groups.Contains(word + "__" + speed)) throw new InvalidDataException($"missing template group {word}__{speed}");
        }
    }
}
