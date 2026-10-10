using System;
using System.Collections.Generic;

namespace IPCV.Bridge
{
    public enum PoseJoint
    {
        Nose = 0, LeftShoulder = 1, RightShoulder = 2, LeftElbow = 3, RightElbow = 4,
        LeftWrist = 5, RightWrist = 6, LeftHip = 7, RightHip = 8,
        LeftKnee = 9, RightKnee = 10, LeftAnkle = 11, RightAnkle = 12
    }

    public enum JointState { Lost = 0, Acquiring = 1, Predicted = 2, Tracked = 3 }

    [Serializable]
    public sealed class TrackingFrame
    {
        public string session_id;
        public long sequence;
        public string clock;
        public double captured_time_s, sent_time_s;
        public PlayerState[] players;
        public double ProcessingMs => (sent_time_s - captured_time_s) * 1000;

        public bool IsValid()
        {
            Guid ignored;
            if (!Guid.TryParseExact(session_id, "N", out ignored) || sequence < 0
                || (clock != "qpc" && clock != "local") || !Numbers.Finite(captured_time_s)
                || !Numbers.Finite(sent_time_s) || captured_time_s <= 0 || sent_time_s < captured_time_s
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
        public bool[] hand_raised;
        public MotionEvent[] events;

        public bool IsValid()
        {
            if ((id != 1 && id != 2) || pose == null || (pose.Length != 0 && pose.Length != 13)
                || !Numbers.OptionalVector(face_bbox, 4) || !Numbers.OptionalVector(head_rotation_deg, 3)
                || !Numbers.OptionalVector(world_position_m, 3) || hand_raised == null
                || (hand_raised.Length != 0 && hand_raised.Length != 2) || events == null) return false;
            if (!tracked && (pose.Length + face_bbox.Length + head_rotation_deg.Length
                + world_position_m.Length + hand_raised.Length + events.Length != 0)) return false;
            foreach (PoseLandmark p in pose)
                if (p == null || !Numbers.Vector(p.position, 2) || (int)p.state < 0 || (int)p.state > 3) return false;
            long lastEventId = -1;
            foreach (MotionEvent e in events)
            {
                if (e == null || e.id <= lastEventId || !Numbers.Finite(e.strength)
                    || !Numbers.Finite(e.time) || string.IsNullOrEmpty(e.type) || e.side == null) return false;
                lastEventId = e.id;
            }
            return face_bbox.Length == 0 || (face_bbox[2] >= 0 && face_bbox[3] >= 0);
        }

        public bool TryGetJoint(PoseJoint joint, out PoseLandmark landmark)
        {
            int index = (int)joint;
            landmark = null;
            if (!tracked || pose == null || index < 0 || index >= pose.Length) return false;
            PoseLandmark point = pose[index];
            if (point == null || (point.state != JointState.Tracked && point.state != JointState.Predicted)) return false;
            landmark = point;
            return true;
        }
    }

    [Serializable]
    public sealed class PoseLandmark { public double[] position; public JointState state; }

    [Serializable]
    public sealed class MotionEvent
    {
        public long id;
        public string type, side;
        public double strength, time;
    }

    internal static class Numbers
    {
        public static bool Finite(double v) => !double.IsNaN(v) && !double.IsInfinity(v);
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
            double age = sharedClock ? receipt - frame.captured_time_s : frame.sent_time_s - frame.captured_time_s;
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
