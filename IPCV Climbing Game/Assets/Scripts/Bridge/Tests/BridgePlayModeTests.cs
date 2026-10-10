using System;
using System.Collections;
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Reflection;
using System.Text;
using IPCV.Bridge;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;
using Debug = UnityEngine.Debug;

public sealed class BridgePlayModeTests
{
    private const string Session2 = "00000000000000000000000000000002";
    private GameObject obj;
    private UdpFrameReceiver receiver;
    private UdpFaceReceiver faces;
    private BridgeMetrics metrics;
    private int port, facePort;
    private static double Now() => BridgeClock.Now();
    private static TrackingFrame Fixture() => JsonUtility.FromJson<TrackingFrame>(Resources.Load<TextAsset>("python_frame").text);

    [Test]
    public void UnityJsonReadsAllPythonFields()
    {
        TrackingFrame f = Fixture();
        Assert.That(f.IsValid(), Is.True);
        Assert.That(f.players.Length, Is.EqualTo(2));
        Assert.That(f.players[0].pose.Length, Is.EqualTo(33));
        Assert.That(f.players[1].world_position_m, Is.EqualTo(new double[] { 0.4, 0, 2 }));
        Assert.That(f.players[0].head_rotation_deg.Length, Is.EqualTo(3));
        Assert.That(f.players[0].motion_signals[0].name, Is.EqualTo("left_reach"));
        Assert.That(f.processing_ms, Is.EqualTo(100).Within(0.001));
        f.players[0].pose[0].position[0] = double.NaN;
        Assert.That(f.IsValid(), Is.False);
    }

    [Test]
    public void StreamOrderingExpiryAndSessionReset()
    {
        var stream = new TrackingFrameStream(0.5, true);
        Assert.That(stream.TryAccept(Fixture(), 1000.2, 1000.2), Is.True);
        Assert.That(stream.TryAccept(Fixture(), 1000.21, 1000.21), Is.False);
        Assert.That(stream.GetCurrent(1000.601), Is.Null, "duplicate must not extend expiry");
        TrackingFrame f = Fixture();
        f.sequence = 2;
        Assert.That(stream.TryAccept(f, 1000.22, 1000.23), Is.True);
        Assert.That(stream.SequenceGaps, Is.EqualTo(1));
        f = Fixture(); f.sequence = 1;
        Assert.That(stream.TryAccept(f, 1000.24, 1000.24), Is.False);
        f = Fixture(); f.sequence = 3;
        Assert.That(stream.TryAccept(f, 1001.0, 1001.0), Is.False, "old capture");
        Assert.That(stream.TryAccept(f, 1000.2, 1001.0), Is.False, "old queue entry");
        f = Fixture(); f.session_id = Session2;
        Assert.That(stream.TryAccept(f, 1000.25, 1000.25), Is.True, "new session accepts sequence zero");
        f = Fixture(); f.sequence = 99;
        Assert.That(stream.TryAccept(f, 1000.26, 1000.26), Is.False, "retired session stays rejected");
        f = Fixture(); f.session_id = Session2; f.sequence = 1; f.players = new PlayerState[0];
        Assert.That(stream.TryAccept(f, 1000.27, 1000.27), Is.True);
        Assert.That(stream.GetCurrent(1000.27).players, Is.Empty);
    }

    private static void Set(object target, string field, object value) =>
        target.GetType().GetField(field, BindingFlags.Instance | BindingFlags.NonPublic).SetValue(target, value);

    private void CreateReceiver()
    {
        using (var a = new UdpClient(new IPEndPoint(IPAddress.Loopback, 0)))
        using (var b = new UdpClient(new IPEndPoint(IPAddress.Loopback, 0)))
        {
            port = ((IPEndPoint)a.Client.LocalEndPoint).Port;
            facePort = ((IPEndPoint)b.Client.LocalEndPoint).Port;
        }
        obj = new GameObject("Bridge test"); obj.SetActive(false);
        receiver = obj.AddComponent<UdpFrameReceiver>(); Set(receiver, "port", port);
        faces = obj.AddComponent<UdpFaceReceiver>(); Set(faces, "port", facePort);
        metrics = obj.AddComponent<BridgeMetrics>();
        obj.SetActive(true);
    }

    private void Send(TrackingFrame f)
    {
        f.clock = Environment.OSVersion.Platform == PlatformID.Win32NT ? "qpc" : "local";
        f.sent_time_s = Now(); f.captured_time_s = f.sent_time_s - 0.004; f.processing_ms = 4;
        SendBytes(Encoding.UTF8.GetBytes(JsonUtility.ToJson(f)), port);
    }

    private static void SendBytes(byte[] bytes, int targetPort)
    {
        using (var client = new UdpClient()) client.Send(bytes, bytes.Length, "127.0.0.1", targetPort);
    }

    private static IEnumerator Until(Func<bool> condition)
    {
        double deadline = Now() + 3;
        while (!condition() && Now() < deadline) yield return null;
        Assert.That(condition(), Is.True, "Timed out waiting for a bridge update");
    }

    [UnityTest]
    public IEnumerator RuntimeCropsLossMalformedDataAndRestart()
    {
        CreateReceiver();
        int mainThread = System.Threading.Thread.CurrentThread.ManagedThreadId;
        bool callbackOnMain = false;
        receiver.FrameReceived += _ => callbackOnMain = System.Threading.Thread.CurrentThread.ManagedThreadId == mainThread;
        Send(Fixture());
        yield return Until(() => receiver.TryGetPlayer(1, out _));
        Assert.That(callbackOnMain, Is.True);
        byte[] crop = Resources.Load<TextAsset>("python_face").bytes;
        SendBytes(crop, facePort);
        yield return Until(() => faces.TryGetTexture(1, out _));
        Assert.That(faces.TryGetTexture(1, out Texture2D texture), Is.True);
        Assert.That(texture.width, Is.EqualTo(256));
        Assert.That(texture.height, Is.EqualTo(256));
        SendBytes(Encoding.UTF8.GetBytes("{broken"), port);
        yield return Until(() => receiver.InvalidPackets > 0);

        TrackingFrame restarted = Fixture(); restarted.session_id = Session2;
        Send(restarted);
        yield return Until(() => receiver.CurrentFrame?.session_id == Session2);
        Assert.That(faces.TryGetTexture(1, out _), Is.False, "old-session image must disappear");
        SendBytes(crop, facePort);
        yield return Until(() => faces.RejectedCrops > 0);

        // A one-byte JPEG payload with an otherwise valid header must not throw.
        byte[] shortCrop = new byte[46]; Array.Copy(crop, shortCrop, 45);
        Encoding.ASCII.GetBytes(Session2).CopyTo(shortCrop, 4);
        long rejected = faces.RejectedCrops;
        SendBytes(shortCrop, facePort);
        yield return Until(() => faces.RejectedCrops > rejected);

        restarted.sequence = 1; restarted.players = new PlayerState[0]; Send(restarted);
        yield return Until(() => receiver.CurrentFrame?.players.Length == 0);
        Assert.That(receiver.TryGetPlayer(1, out _), Is.False);
        int lost = 0; receiver.TrackingLost += () => lost++;
        yield return Until(() => !receiver.HasFreshFrame);
        Assert.That(lost, Is.EqualTo(1));
        obj.SetActive(false);
        using (var probe = new UdpClient(new IPEndPoint(IPAddress.Loopback, port))) { }
        using (var probe = new UdpClient(new IPEndPoint(IPAddress.Loopback, facePort))) { }
    }

    [UnityTest]
    public IEnumerator PythonDemoRunsInsideUnityAndRecordsTiming()
    {
        CreateReceiver();
        string root = Environment.GetEnvironmentVariable("IPCV_BRIDGE_ROOT");
        string python = Environment.GetEnvironmentVariable("IPCV_BRIDGE_PYTHON");
        Assert.That(File.Exists(python), Is.True, "Set IPCV_BRIDGE_PYTHON or run the provided test script");
        string entry = Path.Combine(root ?? "", "Python", "main.py");
        Assert.That(File.Exists(entry), Is.True, "Set IPCV_BRIDGE_ROOT to the repository root");
        metrics.StartRecording();
        using (var process = Process.Start(new ProcessStartInfo(python,
            "\"" + entry + "\" --demo --duration 10 --port " + port + " --face-port " + facePort)
        { UseShellExecute = false, CreateNoWindow = true, RedirectStandardOutput = true, RedirectStandardError = true }))
        {
            try
            {
                yield return Until(() => receiver.TryGetPlayer(2, out _));
                yield return Until(() => faces.TryGetTexture(2, out _));
                double deadline = Now() + 15;
                while (!process.HasExited && Now() < deadline) yield return null;
                Assert.That(process.HasExited, Is.True);
                Assert.That(process.ExitCode, Is.EqualTo(0));
                Assert.That(receiver.AcceptedFrames, Is.GreaterThan(200));
                Assert.That(receiver.InvalidPackets, Is.Zero);
                Assert.That(faces.AcceptedCrops, Is.GreaterThan(20));
                yield return Until(() => !receiver.HasFreshFrame);
            }
            finally
            {
                if (!process.HasExited) { process.Kill(); process.WaitForExit(); }
                Debug.Log($"Python exit {process.ExitCode}; received {receiver.ReceivedPackets}; accepted {receiver.AcceptedFrames}; rejected {receiver.RejectedFrames}; invalid {receiver.InvalidPackets}; Unity clock {Now():F6}");
                Debug.Log(process.StandardOutput.ReadToEnd() + process.StandardError.ReadToEnd());
            }
        }
        metrics.StopRecording();
        Assert.That(File.ReadAllLines(metrics.RecordingPath).Length, Is.GreaterThan(200));
        Debug.Log("IPCV_EVALUATION_CSV=" + metrics.RecordingPath);
    }

    [TearDown]
    public void Cleanup() { if (obj != null) UnityEngine.Object.DestroyImmediate(obj); }
}
