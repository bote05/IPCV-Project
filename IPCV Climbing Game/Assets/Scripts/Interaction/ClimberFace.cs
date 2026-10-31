using IPCV.Bridge;
using UnityEngine;

namespace IPCV.Game
{
    // The receiver owns the texture. This component only puts it on the matching head.
    public sealed class ClimberFace : MonoBehaviour
    {
        public UdpFaceReceiver receiver;
        public int playerId = 1;
        public Renderer photo;
        public GameObject drawnFace;
        private MaterialPropertyBlock properties;
        private UdpFrameReceiver tracking;
        private Quaternion restRotation;

        private void Awake()
        {
            if (receiver != null) tracking = receiver.GetComponent<UdpFrameReceiver>();
            restRotation = photo.transform.localRotation;
        }

        private void LateUpdate()
        {
            Texture2D texture = null;
            bool ready = false;
            if (receiver != null)
            {
                ready = receiver.TryGetTexture(playerId, out texture);
            }

            // Show the photo when there is one, otherwise the model's own face.
            photo.enabled = ready;
            drawnFace.SetActive(!ready);
            float roll = 0;
            if (ready && tracking != null && tracking.TryGetPlayer(playerId, out PlayerState player)
                && player.head_rotation_deg.Length == 3)
            {
                roll = (float)player.head_rotation_deg[2];
            }
            // Image y points down; the photo's local y points up.
            photo.transform.localRotation = restRotation * Quaternion.Euler(0, 0, -roll);
            if (!ready)
            {
                return;
            }

            if (properties == null)
            {
                properties = new MaterialPropertyBlock();
            }
            properties.SetTexture("_BaseMap", texture);
            photo.SetPropertyBlock(properties);
        }
    }
}
