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
