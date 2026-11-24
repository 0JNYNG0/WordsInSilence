// Independent local timing experiment. Excludes camera, detection and source normalization.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using NUnit.Framework;
using WordsInSilence.Recognition.FullHandRuntime;
namespace WordsInSilence.Tests.EditMode
{
    public sealed class F44LocalLatencyAudit
    {
        static double Score(int method,FullHandFeatureSequence q,FullHandRuntimeBundle bank)
        {
            double best=double.NegativeInfinity;
            for(int j=0;j<bank.ReferenceCount;j++)
            {
                var reference=bank.ReferenceSequence(j);double value;
                if(method==0)
                {
                    double mean;var path=FullHandPhasePairFeatures.ConstrainedPath(FullHandPairGeometry.ComputeGeometry(q,reference).Cost,32,32,out mean);
                    value=path==null?double.NegativeInfinity:-mean;
                }
                else
                {
                    var pair=method==1?FullHandPhasePairFeatures.FixedTimeFeatures(q,reference):FullHandPhasePairFeatures.AlignedPairFeatures(q,reference);
                    value=pair.Ok?(method==1?bank.MlModel:bank.HybridModel).Compatibility(pair.CopyFeatures()):double.NegativeInfinity;
                }
                if(value>best)best=value;
            }
            return best;
        }
        [Test]
        public void TimeEachMethodWithRotatedOrderOnSameAvailableRecordings()
        {
            string root=Environment.GetEnvironmentVariable("WIS_F44_AUDIT_ROOT");Assert.IsFalse(string.IsNullOrEmpty(root));
            var bank=FullHandRuntimeBundle.Load(Path.Combine(root,"runtime/runtime_manifest.json"));
            var queries=new List<FullHandFeatureSequence>();var ids=new List<int>();
            using(var r=new BinaryReader(File.OpenRead(Path.Combine(root,"queries.private.bin"))))
            {
                Assert.AreEqual("WISF44Q1",Encoding.ASCII.GetString(r.ReadBytes(8)));int count=r.ReadInt32();
                for(int i=0;i<count;i++)
                {
                    int n=r.ReadInt32();byte[] b=r.ReadBytes(n*46*3*8);double[] a=new double[n*46*3];Buffer.BlockCopy(b,0,a,0,b.Length);bool[] mask=new bool[n*46];
                    for(int k=0;k<mask.Length;k++)mask[k]=r.ReadByte()==1;
                    var q=FullHandPairGeometry.BuildFeatures(a,mask,n);if(q.Usable){queries.Add(q);ids.Add(i);}
                }
            }
            Assert.AreEqual(43,queries.Count);
            for(int method=0;method<3;method++)Score(method,queries[0],bank);
            var report=new StringBuilder("query_id,repeat,method,milliseconds,score\n");
            for(int repeat=0;repeat<3;repeat++)for(int i=0;i<queries.Count;i++)for(int offset=0;offset<3;offset++)
            {
                int m=(i+repeat+offset)%3;var sw=System.Diagnostics.Stopwatch.StartNew();double value=Score(m,queries[i],bank);sw.Stop();
                Assert.IsFalse(double.IsNaN(value)||double.IsInfinity(value));
                report.Append(ids[i]).Append(',').Append(repeat).Append(',').Append(FullHandMethods.All[m]).Append(',').Append(sw.Elapsed.TotalMilliseconds.ToString("R",CultureInfo.InvariantCulture)).Append(',').Append(value.ToString("R",CultureInfo.InvariantCulture)).Append('\n');
            }
            string p=Path.Combine(root,"method_latency.csv");Assert.IsFalse(File.Exists(p));File.WriteAllText(p,report.ToString());
        }
    }
}
