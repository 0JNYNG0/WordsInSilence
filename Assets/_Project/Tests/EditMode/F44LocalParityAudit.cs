// Independent local audit adapter, not runtime implementation. No network or camera.
using System;
using System.IO;
using System.Text;
using NUnit.Framework;
using WordsInSilence.Recognition.FullHandRuntime;
namespace WordsInSilence.Tests.EditMode
{
    public sealed class F44LocalParityAudit
    {
        static double[] Doubles(BinaryReader r, int count)
        {
            byte[] b=r.ReadBytes(checked(count*8)); Assert.AreEqual(count*8,b.Length);
            var a=new double[count];Buffer.BlockCopy(b,0,a,0,b.Length);return a;
        }
        static void Doubles(BinaryWriter w,double[] a)
        {
            var b=new byte[a.Length*8];Buffer.BlockCopy(a,0,b,0,b.Length);w.Write(b);
        }
        static void Pair(BinaryWriter w,FullHandPairResult p,FullHandModel model)
        {
            w.Write(p.Ok);w.Write(p.Failure??"");
            if(!p.Ok)return;
            w.Write(p.DtwCost);w.Write(p.PathLength);Doubles(w,p.CopyFeatures());
            w.Write(model.Compatibility(p.CopyFeatures()));
        }
        [Test]
        public void ActualCandidateGatesFeedGameAuthorityWithoutPersonalRecords()
        {
            string root=Environment.GetEnvironmentVariable("WIS_F44_AUDIT_ROOT");
            Assert.IsFalse(string.IsNullOrEmpty(root));
            var bank=FullHandRuntimeBundle.Load(Path.Combine(root,"runtime/runtime_manifest.json"));
            FullHandGateConfig gates;string sourceReview,failure,detail;
            Assert.IsTrue(WordsInSilence.Recognition.LiveInput.FullHandGateDocument.TryParse(
                File.ReadAllBytes(Path.Combine(root,"runtime/research_gates_strings.json")),
                bank.PayloadSha256,out gates,out sourceReview,out failure,out detail),failure+" "+detail);
            const string participant="f44_audit_only_no_persistence";
            var pipeline=new FullHandRecognitionPipeline(bank,gates,participant,null);
            var report=new StringBuilder("capture_index,target,quest_success,known_negative,personal_references\n");
            using(var r=new BinaryReader(File.OpenRead(Path.Combine(root,"queries.private.bin"))))
            {
                Assert.AreEqual("WISF44Q1",Encoding.ASCII.GetString(r.ReadBytes(8)));int count=r.ReadInt32();
                for(int i=0;i<count;i++)
                {
                    int n=r.ReadInt32();double[] coords=Doubles(r,n*46*3);var mask=new bool[n*46];
                    for(int k=0;k<mask.Length;k++)mask[k]=r.ReadByte()==1;
                    if(i!=0 && i!=8 && i!=15 && i!=18 && i!=22 && i!=36)continue;
                    string target=i==15?"THANK_YOU":"GREETING",id="f44_wiring_"+i;
                    var q=FullHandPairGeometry.BuildFeatures(coords,mask,n);
                    var sample=new FullHandSample(id,participant,null,null,q);
                    var outcome=pipeline.EvaluateSample(sample,target);
                    Assert.AreEqual(0,outcome.Before.ParticipantReferenceCount);
                    Assert.AreEqual(0,outcome.After.ParticipantReferenceCount);
                    Assert.AreEqual(outcome.Before.EffectiveReferenceCount,outcome.After.EffectiveReferenceCount);
                    var evidence=FullHandGameCaptureResult.ForOutcome(id,n,outcome);
                    var game=FullHandGameplayEvaluator.DecideForCapture(evidence,target,id,true,true);
                    // Two requested-negative captures are KNOWN FALSE ACCEPTS. Verify faithful
                    // propagation, not semantic correctness, and record them explicitly.
                    bool knownNegative=i==8 || i==22;Assert.AreEqual(i!=18,game.GrantsQuestSuccess);
                    Assert.IsFalse(FullHandGameplayEvaluator.DecideForCapture(evidence,target,id,true,false).GrantsQuestSuccess);
                    Assert.IsFalse(FullHandGameplayEvaluator.DecideForCapture(evidence,target,"different_id",true,true).GrantsQuestSuccess);
                    report.Append(i).Append(',').Append(target).Append(',').Append(game.GrantsQuestSuccess).Append(',').Append(knownNegative).Append(",0\n");
                }
            }
            string output=Path.Combine(root,"game_wiring_audit.csv");Assert.IsFalse(File.Exists(output));File.WriteAllText(output,report.ToString());
        }
        [Test]
        public void ExportAllLocalPairsForIndependentPythonComparison()
        {
            Assert.IsTrue(BitConverter.IsLittleEndian);
            string root=Environment.GetEnvironmentVariable("WIS_F44_AUDIT_ROOT");
            Assert.IsFalse(string.IsNullOrEmpty(root),"Explicit local private fixture required");
            var bank=FullHandRuntimeBundle.Load(Path.Combine(root,"runtime/runtime_manifest.json"));
            Assert.AreEqual(432,bank.MlModel.FeatureCount);Assert.AreEqual(432,bank.HybridModel.FeatureCount);
            string target=Path.Combine(root,"csharp_pairs.private.bin");Assert.IsFalse(File.Exists(target));
            using(var r=new BinaryReader(File.OpenRead(Path.Combine(root,"queries.private.bin"))))
            using(var w=new BinaryWriter(new FileStream(target,FileMode.CreateNew)))
            {
                Assert.AreEqual("WISF44Q1",Encoding.ASCII.GetString(r.ReadBytes(8)));
                int count=r.ReadInt32();Assert.AreEqual(46,count);Assert.AreEqual(28,bank.ReferenceCount);
                w.Write(Encoding.ASCII.GetBytes("WISF44A1"));w.Write(count);w.Write(bank.ReferenceCount);
                for(int i=0;i<count;i++)
                {
                    int n=r.ReadInt32();Assert.Greater(n,0);Assert.Less(n,10000);
                    double[] coords=Doubles(r,n*46*3);var mask=new bool[n*46];
                    for(int k=0;k<mask.Length;k++){byte v=r.ReadByte();Assert.LessOrEqual(v,1);mask[k]=v==1;}
                    var q=FullHandPairGeometry.BuildFeatures(coords,mask,n);w.Write(q.Usable);w.Write(q.Failure??"");
                    if(!q.Usable)continue;
                    Doubles(w,q.CopyPoints());foreach(bool v in q.CopyMask())w.Write(v);
                    var t=FullHandPhasePairFeatures.BuildTemporal(q);Doubles(w,t.Values);foreach(bool v in t.Mask)w.Write(v);
                    for(int j=0;j<bank.ReferenceCount;j++)
                    {
                        var reference=bank.ReferenceSequence(j);
                        Pair(w,FullHandPhasePairFeatures.FixedTimeFeatures(q,reference),bank.MlModel);
                        Pair(w,FullHandPhasePairFeatures.AlignedPairFeatures(q,reference),bank.HybridModel);
                    }
                    var evaluation=FullHandScorer.Evaluate(bank,q);
                    foreach(string method in FullHandMethods.All)
                    {
                        var m=evaluation.Method(method);w.Write(m.Method);w.Write(m.Status??"");w.Write(m.Top1??"");w.Write(m.Score);w.Write(m.Margin);
                    }
                }
                Assert.AreEqual(r.BaseStream.Length,r.BaseStream.Position);
            }
        }
    }
}
