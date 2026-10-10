using System;
using System.Text;
using UnityEngine;

namespace IPCV.Bridge
{
    [DefaultExecutionOrder(-20)]
    public sealed class UdpFrameReceiver : MonoBehaviour
    {
        [SerializeField] private int port = 5005;
        [SerializeField, Min(0.01f)] private float staleAfterSeconds = 0.5f;
        private static readonly UTF8Encoding Utf8 = new UTF8Encoding(false, true);
        private UdpInbox inbox;
        private TrackingFrameStream stream;
        private TrackingFrame lastPublished;
        private string eventSession;
        private readonly long[] lastEventIds = { -1, -1, -1 };

        public event Action<TrackingFrame> FrameReceived;
        public event Action<int, MotionEvent> MotionReceived;
        public event Action TrackingLost;
        public int Port => port;
        public TrackingFrame CurrentFrame => stream?.GetCurrent(UdpInbox.Now());
        public bool HasFreshFrame => CurrentFrame != null;
        public long ReceivedPackets => inbox?.ReceivedPackets ?? 0;
        public long QueueDrops => inbox?.Drops ?? 0;
        public long InvalidPackets { get; private set; }
        public long AcceptedFrames => stream?.AcceptedFrames ?? 0;
        public long RejectedFrames => stream?.RejectedFrames ?? 0;
        public long SequenceGaps => stream?.SequenceGaps ?? 0;
        public double ReceiveToApplyMs => stream?.ReceiveToApplyMs ?? 0;
        public double EncodeAndUdpMs => stream?.EncodeAndUdpMs ?? double.NaN;
        public double CaptureToApplyMs => stream?.CaptureToApplyMs ?? double.NaN;
        public string LastError { get; private set; }

        public bool TryGetPlayer(int id, out PlayerState player)
        {
            TrackingFrame frame = CurrentFrame;
            if (frame != null)
                foreach (PlayerState p in frame.players)
                    if (p.id == id && p.tracked) { player = p; return true; }
            player = null;
            return false;
        }

        private void OnEnable()
        {
            InvalidPackets = 0;
            LastError = null;
            lastPublished = null;
            try
            {
                stream = new TrackingFrameStream(staleAfterSeconds, Environment.OSVersion.Platform == PlatformID.Win32NT);
                inbox = new UdpInbox(port, 16384, 32);
            }
            catch (Exception e)
            {
                LastError = e.Message;
                Debug.LogError($"Tracking bridge: {LastError}", this);
                enabled = false;
            }
        }

        private void Update()
        {
            if (inbox == null) return;
            if (inbox.Error != null)
            {
                LastError = inbox.Error;
                Debug.LogError($"Tracking bridge: {LastError}", this);
                enabled = false;
                return;
            }
            foreach (UdpInbox.Packet packet in inbox.Drain())
            {
                TrackingFrame frame;
                try
                {
                    frame = JsonUtility.FromJson<TrackingFrame>(Utf8.GetString(packet.bytes));
                }
                catch (ArgumentException) { InvalidPackets++; continue; }
                if (stream.TryAccept(frame, packet.receiptSeconds, UdpInbox.Now())) PublishMotionEvents(frame);
            }
            TrackingFrame current = CurrentFrame;
            if (ReferenceEquals(current, lastPublished)) return;
            lastPublished = current;
            if (current == null) TrackingLost?.Invoke();
            else FrameReceived?.Invoke(current);
        }

        private void PublishMotionEvents(TrackingFrame frame)
        {
            if (eventSession != frame.session_id)
            {
                eventSession = frame.session_id;
                lastEventIds[1] = lastEventIds[2] = -1;
            }
            foreach (PlayerState player in frame.players)
                foreach (MotionEvent motion in player.events)
                    if (motion.id > lastEventIds[player.id])
                    {
                        lastEventIds[player.id] = motion.id;
                        MotionReceived?.Invoke(player.id, motion);
                    }
        }

        private void OnDisable()
        {
            inbox?.Dispose();
            inbox = null;
            stream = null;
            bool hadTracking = lastPublished != null;
            lastPublished = null;
            if (hadTracking) TrackingLost?.Invoke();
        }
    }
}
