using UnityEngine;

namespace IPCV.Game
{
    // Pose the downloaded rigs from the same shoulder, elbow and hand targets as physics.
    [DefaultExecutionOrder(110)]
    public sealed class ClimberRig : MonoBehaviour
    {
        public ClimberBody climber;
        public Transform[] upper = new Transform[4];
        public Transform[] lower = new Transform[4];
        public Transform[] wrists = new Transform[4];
        public Vector3[] upperDirection = new Vector3[4];
        public Vector3[] lowerDirection = new Vector3[4];
        public Quaternion[] upperRest = new Quaternion[4];
        public Quaternion[] lowerRest = new Quaternion[4];
        public Quaternion[] wristRest = new Quaternion[4];

        private void LateUpdate()
        {
            Pose();
        }

        public void Pose()
        {
            for (int i = 0; i < 4; i++)
            {
                // The three arm points that ClimberBody already worked out.
                LineRenderer path = climber.arms[i];
                Vector3 start = path.GetPosition(0);
                Vector3 elbow = path.GetPosition(1);
                Vector3 hand = path.GetPosition(2);

                Vector3 upperArm = elbow - start;
                Vector3 forearm = hand - elbow;

                // Turn each bone from its default direction to the new direction, on top of its rest pose.
                upper[i].position = start;
                upper[i].rotation = Quaternion.FromToRotation(upperDirection[i], upperArm) * upperRest[i];

                lower[i].position = elbow;
                lower[i].rotation = Quaternion.FromToRotation(lowerDirection[i], forearm) * lowerRest[i];

                wrists[i].position = hand;
                wrists[i].rotation = Quaternion.FromToRotation(lowerDirection[i], forearm) * wristRest[i];
            }
        }
    }
}
