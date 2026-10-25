using UnityEngine;

namespace IPCV.Game
{
    public sealed class ClimbingCamera : MonoBehaviour
    {
        public Transform climber;
        public float lowestHeight = 4.5f;
        private float speed;

        private void LateUpdate()
        {
            if (climber == null)
            {
                return;
            }

            // Stay 2 m above the climbers, but never go below the lowest height.
            float targetHeight = climber.position.y + 2f;
            if (targetHeight < lowestHeight)
            {
                targetHeight = lowestHeight;
            }

            // Only move up and down, smoothly.
            Vector3 position = transform.position;
            position.y = Mathf.SmoothDamp(position.y, targetHeight, ref speed, 0.3f);
            transform.position = position;
        }
    }
}
