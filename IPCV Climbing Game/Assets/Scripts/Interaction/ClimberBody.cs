using UnityEngine;
using UnityEngine.InputSystem;

namespace IPCV.Game
{
    [RequireComponent(typeof(Rigidbody))]
    public sealed class ClimberBody : MonoBehaviour
    {
        public ClimberInput input;
        public Transform[] shoulders = new Transform[4];
        public Transform[] hands = new Transform[4];
        public LineRenderer[] arms = new LineRenderer[4];
        [Tooltip("Spring strength of one holding hand. Lower = more sag.")]
        public float pullStrength = 16f;
        public float damping = 5f;
        public float maxPull = 80f;
        public float maxArmLength = 3.2f;
        public float finishHeight = 20.8f;
        [Tooltip("Player 1's and player 2's upper body. Each one can bend at the hips.")]
        public Transform[] upperBodies = new Transform[2];
        [Tooltip("The tube that joins the two bodies. It bends along with the hips.")]
        public MeshFilter waist;
        [Tooltip("Harness straps over the left and right join. Each one turns with its half.")]
        public Transform[] straps = new Transform[2];
        [Tooltip("How far (degrees) a half slumps when its player lets go while the other player holds on.")]
        public float droopAngle = 35f;
        [Tooltip("Spring strength of the hip. Lower = slower, floppier slump.")]
        public float droopStiffness = 30f;
        [Tooltip("Slows the hip swing down. Lower = more swinging before it settles.")]
        public float droopDamping = 4f;
        [Tooltip("How far the body may move sideways from where it started, so it cannot swing across the wall.")]
        public float maxSideSwing = 0.5f;
        [Tooltip("Length of a drawn arm from shoulder to hand. The elbow bends when the hand is closer than this.")]
        public float armLength = 1.7f;
        [Tooltip("How quickly the drawn hands follow their target. Higher is snappier.")]
        public float handSmoothing = 18f;
        public readonly ClimbingHold[] heldHolds = new ClimbingHold[4];
        public int GripCount { get; private set; }
        public float BestHeight { get; private set; }
        public bool Finished { get; private set; }
        public bool WaitingForPlayers { get; private set; }
        public int Falls { get; private set; }
        private Rigidbody body;
        private Vector3 start;
        private int inputVersion;
        private readonly bool[] mustRelease = new bool[4];
        // At the top: each player has lowered their hands since, so raising one now means "again".
        private readonly bool[] loweredAtTop = new bool[2];
        // Where each drawn hand is compared to its shoulder. It slides toward the real target.
        private readonly Vector3[] drawnOffsets = new Vector3[4];
        // Hip bending of each half: the angle in degrees and how fast it changes.
        private readonly float[] hipAngle = new float[2];
        private readonly float[] hipSpeed = new float[2];
        // The starting pose, so the hip can rotate it around the hip point.
        private readonly Vector3[] hipPoint = new Vector3[2];
        private readonly Vector3[] upperStartPosition = new Vector3[2];
        private readonly Quaternion[] upperStartRotation = new Quaternion[2];
        private readonly Vector3[] shoulderStart = new Vector3[4];
        private readonly Vector3[] strapStartPosition = new Vector3[2];
        private readonly Quaternion[] strapStartRotation = new Quaternion[2];
        // The waist tube: a copy we can bend, plus its original shape.
        private Mesh waistMesh;
        private Vector3[] waistStartVertices;
        private Vector3[] waistStartNormals;
        private Vector3[] waistVertices;
        private Vector3[] waistNormals;
        private AudioSource sound;
        private AudioClip grabSound;

        private void Awake()
        {
            body = GetComponent<Rigidbody>();
            start = transform.position;
            WaitingForPlayers = input != null && input.useTracking;
            body.isKinematic = WaitingForPlayers;

            // Remember how everything sits before any bending.
            for (int i = 0; i < 4; i++)
            {
                shoulderStart[i] = shoulders[i].localPosition;
            }
            for (int half = 0; half < 2; half++)
            {
                Transform upper = upperBodies[half];
                upperStartPosition[half] = upper.localPosition;
                upperStartRotation[half] = upper.localRotation;
                hipPoint[half] = FindHipPoint(upper);

                strapStartPosition[half] = straps[half].localPosition;
                strapStartRotation[half] = straps[half].localRotation;
            }

            // Bend a copy of the tube, so the saved mesh asset never changes.
            waistMesh = Instantiate(waist.sharedMesh);
            waist.mesh = waistMesh;
            waistStartVertices = waistMesh.vertices;
            waistStartNormals = waistMesh.normals;
            waistVertices = new Vector3[waistStartVertices.Length];
            waistNormals = new Vector3[waistStartNormals.Length];

            sound = gameObject.AddComponent<AudioSource>();
            sound.playOnAwake = false;
            sound.volume = 0.13f;

            // The grab sound is made in code: a sine wave that fades out.
            float[] samples = new float[2400];
            for (int i = 0; i < samples.Length; i++)
            {
                float wave = Mathf.Sin(i * 0.09f);
                float fade = Mathf.Exp(-i / 400f);
                samples[i] = wave * fade;
            }
            grabSound = AudioClip.Create("Grip", samples.Length, 1, 24000, false);
            grabSound.SetData(samples, 0);
        }

        private void Update()
        {
            if (Keyboard.current != null && Keyboard.current.rKey.wasPressedThisFrame)
            {
                Restart();
            }

            // With the camera, both players lower their hands and then raise one to climb again.
            if (Finished && input != null && input.useTracking)
            {
                for (int player = 0; player < 2; player++)
                {
                    if (!input.handsObserved[player])
                    {
                        loweredAtTop[player] = false;
                    }
                    else if (!input.handRaised[player])
                    {
                        loweredAtTop[player] = true;
                    }
                }
                bool bothRaised = BothReady();
                if (loweredAtTop[0] && loweredAtTop[1] && bothRaised)
                {
                    Restart();
                }
            }

            // Fell off the bottom of the room.
            if (transform.position.y < -4f)
            {
                Falls++;
                Restart();
            }

            float height = transform.position.y - start.y;
            if (height > BestHeight)
            {
                BestHeight = height;
            }

            if (transform.position.y >= finishHeight)
            {
                Finished = true;
            }
        }

        private void FixedUpdate()
        {
            if (input == null)
            {
                return;
            }

            // The input mode changed (keyboard or tracking), so let go of everything.
            if (inputVersion != input.ModeVersion)
            {
                inputVersion = input.ModeVersion;
                ReleaseAll();
                WaitingForPlayers = input.useTracking;
                loweredAtTop[0] = loweredAtTop[1] = false;
            }

            if (WaitingForPlayers && BothReady())
            {
                WaitingForPlayers = false;
                for (int i = 0; i < 4; i++) mustRelease[i] = false;
            }
            body.isKinematic = WaitingForPlayers || (Finished && input.useTracking);
            if (body.isKinematic) return;

            Vector3 targetSum = Vector3.zero;
            GripCount = 0;

            for (int i = 0; i < 4; i++)
            {
                // The hold broke. The player has to let go before this hand can grab again.
                if (heldHolds[i] != null && heldHolds[i].Broken)
                {
                    heldHolds[i] = null;
                    mustRelease[i] = true;
                }

                // The player let go of this hand.
                if (!input.wantsGrip[i])
                {
                    heldHolds[i] = null;
                    mustRelease[i] = false;
                }

                Vector3 offset = input.handOffsets[i];
                Vector3 shoulderOffset = shoulders[i].position - body.position;
                Vector3 reach = body.position + shoulderOffset + offset;

                // Try to grab a hold close to the hand.
                bool canTryGrab = heldHolds[i] == null && input.wantsGrip[i] && input.canGrab[i] && !mustRelease[i];
                if (canTryGrab)
                {
                    heldHolds[i] = ClimbingHold.FindNear(reach);
                    if (heldHolds[i] != null)
                    {
                        float volume = 1f;
                        if (i < 2)
                        {
                            volume = 0.8f;
                        }
                        sound.PlayOneShot(grabSound, volume);
                    }
                }

                if (heldHolds[i] == null)
                {
                    continue;
                }

                Vector3 anchor = heldHolds[i].transform.position;
                anchor.z = body.position.z;

                // The hold is too far away for the arm, so the hand slips off.
                float armLength = Vector2.Distance(shoulders[i].position, anchor);
                if (armLength > maxArmLength)
                {
                    heldHolds[i] = null;
                    mustRelease[i] = true;
                    continue;
                }

                // Where the body should be so that this hand sits exactly on its hold.
                Vector3 target = anchor - shoulderOffset - offset;
                targetSum += target;
                GripCount++;
            }

            UpdateHips();

            if (GripCount == 0)
            {
                return;
            }

            // Average the constraints: four grips should not produce four times the force.
            Vector3 averageTarget = targetSum / GripCount;

            // When one side lets go, the body only drifts a little instead of swinging across the wall.
            float leftLimit = start.x - maxSideSwing;
            float rightLimit = start.x + maxSideSwing;
            averageTarget.x = Mathf.Clamp(averageTarget.x, leftLimit, rightLimit);

            // Every holding hand pulls like a spring (Hooke's law). Gravity stays on, so the body
            // stretches the springs and sags below the holds. More hands = stiffer = less sag.
            float stiffness = pullStrength * GripCount;
            Vector3 spring = (averageTarget - body.position) * stiffness;
            Vector3 brake = body.linearVelocity * damping;
            Vector3 force = spring - brake;
            force.z = 0;
            force = Vector3.ClampMagnitude(force, maxPull);
            body.AddForce(force, ForceMode.Acceleration);
        }

        // Each half of the body bends at the hips like a spring. If one player lets go while the
        // other holds on, the free half slumps down and outward, swings a little and settles.
        private void UpdateHips()
        {
            for (int half = 0; half < 2; half++)
            {
                int other = 1 - half;
                bool holding = IsHolding(half);
                bool otherHolding = IsHolding(other);

                float target = 0f;
                if (!holding && otherHolding)
                {
                    // Player 1 slumps to the left (positive angle), player 2 to the right.
                    target = droopAngle;
                    if (half == 1)
                    {
                        target = -droopAngle;
                    }
                }

                // Spring toward the target angle, with damping so it does not swing forever.
                float acceleration = (target - hipAngle[half]) * droopStiffness - hipSpeed[half] * droopDamping;
                hipSpeed[half] += acceleration * Time.fixedDeltaTime;
                hipAngle[half] += hipSpeed[half] * Time.fixedDeltaTime;
            }
        }

        private bool IsHolding(int half)
        {
            int firstHand = half * 2;
            int secondHand = half * 2 + 1;
            return heldHolds[firstHand] != null || heldHolds[secondHand] != null;
        }

        // Rotate each upper body (and its two shoulders) around its hip point.
        private void PoseHips()
        {
            for (int half = 0; half < 2; half++)
            {
                Quaternion bend = Quaternion.AngleAxis(hipAngle[half], Vector3.forward);
                Vector3 pivot = hipPoint[half];

                Transform upper = upperBodies[half];
                upper.localPosition = pivot + bend * (upperStartPosition[half] - pivot);
                upper.localRotation = bend * upperStartRotation[half];

                for (int side = 0; side < 2; side++)
                {
                    int hand = half * 2 + side;
                    shoulders[hand].localPosition = pivot + bend * (shoulderStart[hand] - pivot);
                }

                straps[half].localPosition = pivot + bend * (strapStartPosition[half] - pivot);
                straps[half].localRotation = bend * strapStartRotation[half];
            }

            BendWaist();
        }

        // Bend the tube between the two bodies. Vertices in the middle of the tube do not turn,
        // vertices at a hip turn as far as that half, and the ones in between turn part of the way.
        private void BendWaist()
        {
            for (int v = 0; v < waistStartVertices.Length; v++)
            {
                Vector3 point = waistStartVertices[v];

                // Left of the middle belongs to player 1, right of the middle to player 2.
                int half = 0;
                if (point.x > 0)
                {
                    half = 1;
                }

                // 0 in the middle of the tube, 1 at the hip, smoothed so the curve has no kink.
                float amount = Mathf.Clamp01(Mathf.Abs(point.x) / Mathf.Abs(hipPoint[half].x));
                amount = amount * amount * (3f - 2f * amount);

                Quaternion bend = Quaternion.AngleAxis(hipAngle[half] * amount, Vector3.forward);
                Vector3 pivot = hipPoint[half];
                waistVertices[v] = pivot + bend * (point - pivot);
                waistNormals[v] = bend * waistStartNormals[v];
            }

            waistMesh.vertices = waistVertices;
            waistMesh.normals = waistNormals;
            waistMesh.RecalculateBounds();
        }

        // The hip point is where the upper body meets the fused waist: its "Hips" bone.
        private Vector3 FindHipPoint(Transform upper)
        {
            foreach (Transform part in upper.GetComponentsInChildren<Transform>(true))
            {
                if (part.name == "Hips")
                {
                    return transform.InverseTransformPoint(part.position);
                }
            }
            return upper.localPosition;
        }

        private void LateUpdate()
        {
            PoseHips();

            for (int i = 0; i < 4; i++)
            {
                Vector3 shoulder = shoulders[i].position;
                bool holding = heldHolds[i] != null;

                // The hand sits on its hold, or at its reach point when it is free.
                Vector3 end;
                if (holding)
                {
                    end = heldHolds[i].transform.position;
                }
                else
                {
                    end = shoulder + (Vector3)input.handOffsets[i];
                }
                end.z = shoulder.z - 0.08f;

                // Slide the drawn hand toward its target so the arm moves smoothly instead of jumping.
                Vector3 wanted = end - shoulder;
                float blend = 1f - Mathf.Exp(-handSmoothing * Time.deltaTime);
                drawnOffsets[i] = Vector3.Lerp(drawnOffsets[i], wanted, blend);
                end = shoulder + drawnOffsets[i];

                hands[i].position = end;

                if (holding)
                {
                    hands[i].localScale = Vector3.one * 0.26f;
                }
                else
                {
                    hands[i].localScale = Vector3.one * 0.19f;
                }

                Vector3 elbow = GetElbow(i, shoulder, end);

                arms[i].SetPosition(0, shoulder);
                arms[i].SetPosition(1, elbow);
                arms[i].SetPosition(2, end);
            }
        }

        // The upper arm and forearm are each half the arm length. When the hand is close to the
        // shoulder the elbow has to stick out, like a real arm. It bends down and away from the body.
        private Vector3 GetElbow(int i, Vector3 shoulder, Vector3 hand)
        {
            Vector3 middle = Vector3.Lerp(shoulder, hand, 0.5f);
            Vector3 arm = hand - shoulder;
            arm.z = 0;
            float distance = arm.magnitude;
            if (distance < 0.001f)
            {
                return middle;
            }
            Vector3 direction = arm / distance;

            // Player 1's arms are on the left, player 2's on the right.
            float outward = 1f;
            if (i < 2)
            {
                outward = -1f;
            }
            Vector3 hint = new Vector3(outward, -1f, 0f);

            // Keep only the part of the hint that is at a right angle to the arm.
            Vector3 bend = hint - direction * Vector3.Dot(hint, direction);
            if (bend.sqrMagnitude < 0.0001f)
            {
                bend = new Vector3(-direction.y, direction.x, 0f);
            }
            bend.Normalize();

            // Pythagoras: half the arm is the long side, half the shoulder-hand distance is one short side.
            float halfArm = armLength / 2f;
            float halfDistance = distance / 2f;
            float bendAmount = 0f;
            if (halfDistance < halfArm)
            {
                bendAmount = Mathf.Sqrt(halfArm * halfArm - halfDistance * halfDistance);
            }

            Vector3 elbow = middle + bend * bendAmount;
            elbow.z = middle.z - 0.12f;
            return elbow;
        }

        public void ReleaseAll()
        {
            for (int i = 0; i < 4; i++)
            {
                heldHolds[i] = null;
                mustRelease[i] = true;
            }
            GripCount = 0;
        }

        public void Restart()
        {
            ReleaseAll();
            foreach (ClimbingHold hold in ClimbingHold.All)
            {
                hold.ResetHold();
            }
            body.isKinematic = false;
            body.position = start;
            body.rotation = Quaternion.identity;
            body.linearVelocity = Vector3.zero;
            body.angularVelocity = Vector3.zero;
            for (int half = 0; half < 2; half++)
            {
                hipAngle[half] = 0f;
                hipSpeed[half] = 0f;
                loweredAtTop[half] = false;
            }
            Finished = false;
            BestHeight = 0;
            WaitingForPlayers = input != null && input.useTracking;
            body.isKinematic = WaitingForPlayers;
        }

        private bool BothReady()
        {
            return input.handsObserved[0] && input.handsObserved[1]
                && input.handRaised[0] && input.handRaised[1];
        }

        private void OnDestroy()
        {
            if (grabSound != null)
            {
                Destroy(grabSound);
            }
            if (waistMesh != null)
            {
                Destroy(waistMesh);
            }
        }
    }
}
