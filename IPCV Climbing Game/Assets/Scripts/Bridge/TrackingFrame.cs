using System;
using System.Collections.Generic;

namespace IPCV.Bridge
{
    [Serializable]
    public sealed class TrackingFrame
    {
        public string session_id;
        public long sequence;
        public string clock;
        public double captured_time_s, sent_time_s, processing_ms;
        public PlayerState[] players;

        public bool IsValid()
        {
            Guid ignored;
            if (!Guid.TryParseExact(session_id, "N", out ignored) || sequence < 0
                || (clock != "qpc" && clock != "local") || !Numbers.Finite(captured_time_s)
                || !Numbers.Finite(sent_time_s) || captured_time_s <= 0 || sent_time_s < captured_time_s
                || !Numbers.Finite(processing_ms) || processing_ms < 0
                || Math.Abs(processing_ms - (sent_time_s - captured_time_s) * 1000) > 0.01
                || players == null || players.Length > 2) return false;
            int ids = 0;
            foreach (PlayerState p in players)
            {
                if (p == null || !p.IsValid() || (ids & (1 << p.id)) != 0) return false;
                ids |= 1 << p.id;
            }
            return true;
        }
    }

    [Serializable]
    public sealed class PlayerState
    {
        public int id;
        public bool tracked;
        public PoseLandmark[] pose;
        public double[] face_bbox, head_rotation_deg, world_position_m;
        public MotionSignal[] motion_signals;

        public bool IsValid()
        {
            if ((id != 1 && id != 2) || pose == null || (pose.Length != 0 && pose.Length != 33)
                || !Numbers.OptionalVector(face_bbox, 4) || !Numbers.OptionalVector(head_rotation_deg, 3)
                || !Numbers.OptionalVector(world_position_m, 3) || motion_signals == null) return false;
            if (!tracked && (pose.Length + face_bbox.Length + head_rotation_deg.Length
                + world_position_m.Length + motion_signals.Length != 0)) return false;
            foreach (PoseLandmark p in pose)
                if (p == null || !Numbers.Vector(p.position, 3) || !Numbers.Confidence(p.visibility)) return false;
            var names = new HashSet<string>();
            foreach (MotionSignal s in motion_signals)
                if (s == null || string.IsNullOrEmpty(s.name) || !names.Add(s.name)
                    || !Numbers.Confidence(s.confidence)) return false;
            return face_bbox.Length == 0 || (face_bbox[2] >= 0 && face_bbox[3] >= 0);
        }

        public bool TryGetSignal(string name, out MotionSignal signal)
        {
            foreach (MotionSignal s in motion_signals)
                if (s.name == name) { signal = s; return true; }
            signal = null;
            return false;
        }
    }

    [Serializable]
    public sealed class PoseLandmark { public double[] position; public double visibility; }

    [Serializable]
    public sealed class MotionSignal { public string name; public bool active; public double confidence; }

    internal static class Numbers
    {
        public static bool Finite(double v) => !double.IsNaN(v) && !double.IsInfinity(v);
        public static bool Confidence(double v) => Finite(v) && v >= 0 && v <= 1;
        public static bool Vector(double[] v, int size)
        {
            if (v == null || v.Length != size) return false;
            foreach (double n in v) if (!Finite(n)) return false;
            return true;
        }
        public static bool OptionalVector(double[] v, int size) => v != null && (v.Length == 0 || Vector(v, size));
    }

    // Keeps the newest frame. A new session (Python restarted) starts the counter again.
    public sealed class TrackingFrameStream
    {
        private readonly double timeout;
        private readonly bool compareQpc;
        private readonly HashSet<string> retired = new HashSet<string>();
        private TrackingFrame latest;
        private double receiptSeconds, ageAtReceipt;
        public long AcceptedFrames { get; private set; }
        public long RejectedFrames { get; private set; }
        public long SequenceGaps { get; private set; }
        public double ReceiveToApplyMs { get; private set; }
        public double EncodeAndUdpMs { get; private set; }
        public double CaptureToApplyMs { get; private set; }

        public TrackingFrameStream(double timeout, bool compareQpc)
        {
            if (!Numbers.Finite(timeout) || timeout <= 0) throw new ArgumentOutOfRangeException("timeout");
            this.timeout = timeout;
            this.compareQpc = compareQpc;
        }

        public bool TryAccept(TrackingFrame frame, double receipt, double now)
        {
            if (frame == null || !frame.IsValid()) return Reject();
            bool sharedClock = compareQpc && frame.clock == "qpc";
            double age = sharedClock ? receipt - frame.captured_time_s : frame.processing_ms / 1000;
            if (age < -0.001 || Math.Max(0, age) + now - receipt > timeout
                || retired.Contains(frame.session_id)) return Reject();
            if (latest != null && latest.session_id == frame.session_id)
            {
                if (frame.sequence <= latest.sequence) return Reject();
                SequenceGaps += frame.sequence - latest.sequence - 1;
            }
            else if (latest != null) retired.Add(latest.session_id);
            latest = frame;
            receiptSeconds = receipt;
            ageAtReceipt = Math.Max(0, age);
            ReceiveToApplyMs = Math.Max(0, (now - receipt) * 1000);
            EncodeAndUdpMs = sharedClock ? Math.Max(0, (receipt - frame.sent_time_s) * 1000) : double.NaN;
            CaptureToApplyMs = sharedClock ? Math.Max(0, (now - frame.captured_time_s) * 1000) : double.NaN;
            AcceptedFrames++;
            return true;
        }

        public TrackingFrame GetCurrent(double now) =>
            latest != null && ageAtReceipt + now - receiptSeconds <= timeout ? latest : null;

        private bool Reject() { RejectedFrames++; return false; }
    }
}
