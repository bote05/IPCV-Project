using System;
using System.Globalization;
using System.IO;
using UnityEngine;

namespace IPCV.Bridge
{
    [RequireComponent(typeof(UdpFrameReceiver))]
    public sealed class BridgeMetrics : MonoBehaviour
    {
        private UdpFrameReceiver receiver;
        private StreamWriter csv;
        private double windowStarted;
        private int unityFrames, trackingFrames;
        public double UnityFps { get; private set; }
        public double TrackingFps { get; private set; }
        public bool IsRecording => csv != null;
        public string RecordingPath { get; private set; }

        private void OnEnable()
        {
            receiver = GetComponent<UdpFrameReceiver>();
            receiver.FrameReceived += OnFrame;
            windowStarted = BridgeClock.Now();
            unityFrames = trackingFrames = 0;
        }

        private void Update()
        {
            unityFrames++;
            double elapsed = BridgeClock.Now() - windowStarted;
            if (elapsed < 1) return;
            UnityFps = unityFrames / elapsed;
            TrackingFps = trackingFrames / elapsed;
            windowStarted = BridgeClock.Now();
            unityFrames = trackingFrames = 0;
            csv?.Flush();
        }

        public void StartRecording()
        {
            if (csv != null) return;
            try
            {
                Directory.CreateDirectory(Application.persistentDataPath);
                RecordingPath = Path.Combine(Application.persistentDataPath,
                    "bridge-metrics-" + DateTime.Now.ToString("yyyyMMdd-HHmmss-fff") + ".csv");
                csv = new StreamWriter(RecordingPath);
                csv.WriteLine("time_s,session,sequence,processing_ms,encode_udp_ms,receive_apply_ms,capture_apply_ms,tracking_fps,unity_fps,sequence_gaps,queue_drops,invalid_json");
                Debug.Log("Bridge metrics CSV: " + RecordingPath, this);
            }
            catch (IOException e) { StopRecording(); Debug.LogError("Cannot record bridge metrics: " + e.Message, this); }
        }

        private void OnFrame(TrackingFrame frame)
        {
            trackingFrames++;
            if (csv == null) return;
            csv.WriteLine(string.Format(CultureInfo.InvariantCulture,
                "{0:F6},{1},{2},{3:F6},{4},{5:F6},{6},{7:F3},{8:F3},{9},{10},{11}",
                BridgeClock.Now(), frame.session_id, frame.sequence, frame.ProcessingMs,
                Metric(receiver.EncodeAndUdpMs), receiver.ReceiveToApplyMs, Metric(receiver.CaptureToApplyMs),
                TrackingFps, UnityFps, receiver.SequenceGaps, receiver.QueueDrops, receiver.InvalidPackets));
        }

        private static string Metric(double v) => double.IsNaN(v) ? "" : v.ToString("F6", CultureInfo.InvariantCulture);
        public void StopRecording() { csv?.Dispose(); csv = null; }
        private void OnDisable() { receiver.FrameReceived -= OnFrame; StopRecording(); }
    }
}
