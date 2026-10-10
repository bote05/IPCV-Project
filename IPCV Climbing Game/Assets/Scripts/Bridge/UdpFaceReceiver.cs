using System;
using System.Text;
using UnityEngine;

namespace IPCV.Bridge
{
    [DefaultExecutionOrder(-10), RequireComponent(typeof(UdpFrameReceiver))]
    public sealed class UdpFaceReceiver : MonoBehaviour
    {
        private const int HeaderSize = 45;
        [SerializeField] private int port = 5006;
        [SerializeField, Min(0.1f)] private float staleAfterSeconds = 2;
        private UdpFrameReceiver tracking;
        private UdpInbox inbox;
        private readonly Crop[] crops = new Crop[3];
        private sealed class Crop
        {
            public Texture2D texture;
            public string session;
            public ulong sequence;
            public double receipt;
            public bool valid;
        }
        public long AcceptedCrops { get; private set; }
        public long RejectedCrops { get; private set; }
        public string LastError { get; private set; }

        private void OnEnable()
        {
            tracking = GetComponent<UdpFrameReceiver>();
            LastError = null;
            AcceptedCrops = RejectedCrops = 0;
            try
            {
                if (port == tracking.Port) throw new ArgumentException("Face and tracking ports must differ");
                inbox = new UdpInbox(port, 60000, 4);
            }
            catch (Exception e) { LastError = e.Message; Debug.LogError("Face bridge: " + LastError, this); enabled = false; }
        }

        public bool TryGetTexture(int id, out Texture2D texture)
        {
            Crop crop = id >= 1 && id <= 2 ? crops[id] : null;
            TrackingFrame frame = tracking.CurrentFrame;
            if (crop != null && crop.valid && frame != null && crop.session == frame.session_id
                && UdpInbox.Now() - crop.receipt <= staleAfterSeconds
                && tracking.TryGetPlayer(id, out PlayerState player) && player.face_bbox.Length == 4)
            { texture = crop.texture; return true; }
            texture = null;
            return false;
        }

        private void Update()
        {
            if (inbox == null) return;
            if (inbox.Error != null)
            {
                LastError = inbox.Error;
                Debug.LogError("Face bridge: " + LastError, this);
                enabled = false;
                return;
            }
            foreach (UdpInbox.Packet packet in inbox.Drain())
            {
                byte[] b = packet.bytes;
                TrackingFrame frame = tracking.CurrentFrame;
                if (b.Length <= HeaderSize || b[0] != 'I' || b[1] != 'P' || b[2] != 'C' || b[3] != 'F'
                    || frame == null || UdpInbox.Now() - packet.receiptSeconds > staleAfterSeconds)
                { RejectedCrops++; continue; }
                int id = b[36];
                string session = Encoding.ASCII.GetString(b, 4, 32);
                if (session != frame.session_id || !tracking.TryGetPlayer(id, out PlayerState player)
                    || player.face_bbox.Length != 4)
                { RejectedCrops++; continue; }
                ulong sequence = 0;
                for (int i = 37; i < 45; i++) sequence = (sequence << 8) | b[i];
                Crop crop = crops[id];
                if (crop != null && crop.session == session && sequence <= crop.sequence)
                { RejectedCrops++; continue; }
                if (crop == null)
                    crops[id] = crop = new Crop { texture = new Texture2D(2, 2, TextureFormat.RGB24, false) };
                byte[] jpeg = new byte[b.Length - HeaderSize];
                Buffer.BlockCopy(b, HeaderSize, jpeg, 0, jpeg.Length);
                crop.valid = false;
                if (jpeg.Length < 4 || jpeg[0] != 255 || jpeg[1] != 216
                    || jpeg[jpeg.Length - 2] != 255 || jpeg[jpeg.Length - 1] != 217
                    || !ImageConversion.LoadImage(crop.texture, jpeg)
                    || crop.texture.width != 256 || crop.texture.height != 256)
                { RejectedCrops++; continue; }
                crop.session = session;
                crop.sequence = sequence;
                crop.receipt = packet.receiptSeconds;
                crop.valid = true;
                AcceptedCrops++;
            }
        }

        private void OnDisable()
        {
            inbox?.Dispose();
            inbox = null;
            for (int id = 1; id <= 2; id++)
            {
                if (crops[id] != null) Destroy(crops[id].texture);
                crops[id] = null;
            }
        }
    }
}
