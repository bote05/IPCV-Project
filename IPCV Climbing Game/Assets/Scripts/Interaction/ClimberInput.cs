using IPCV.Bridge;
using UnityEngine;
using UnityEngine.InputSystem;

namespace IPCV.Game
{
    // Only this component knows about tracking or keyboard controls.
    [DefaultExecutionOrder(-5)]
    public sealed class ClimberInput : MonoBehaviour
    {
        public UdpFrameReceiver receiver;
        public bool useTracking;
        public float reachScale = 1.8f;
        [Tooltip("Camera distance in metres required before starting or restarting. Estimated from torso size; C in the webcam window calibrates it.")]
        public float minPlayingDistance = 1.5f;
        public float maxPlayingDistance = 3f;
        [Tooltip("Keyboard: extra sideways reach per hand. Hands on the inner shoulders (0 and 3) need more to reach the same holds.")]
        public float[] extraReach = { 0.55f, 0f, 0f, 0.55f };
        // Where each hand is drawn and pulls from.
        public readonly Vector2[] handOffsets = new Vector2[4];
        // Where each hand would grab if the player closed it now (used for the yellow ring).
        public readonly Vector2[] reachOffsets = new Vector2[4];
        public readonly bool[] wantsGrip = new bool[4];
        public readonly bool[] canGrab = new bool[4];
        public readonly bool[] playerSeen = new bool[2];
        // Pose's raised-hand signal, separate from the game's grip latch.
        public readonly bool[] handRaised = new bool[2];
        public readonly bool[] inPlayingArea = new bool[2];
        public readonly string[] positionHint = new string[2];
        // Keyboard: hand height above the shoulder. The arms start out to the side, a little raised.
        private readonly float[] handHeights = { 0.6f, 0.6f, 0.6f, 0.6f };
        private readonly float[] sideReach = { 1f, 1f };
        private readonly bool[] trackingGrip = new bool[4];
        // Release below the shoulder, measured in shoulder widths.
        private const float LetGo = -1f;
        // A straight raised arm reaches the hold column without crossing the face.
        private const float ColumnReach = 0.95f;
        private const float SideScale = 1.2f;
        // Keep the grip during brief arm or player loss.
        private const float LostGrace = 2f;
        private readonly float[] lastArmTime = new float[4];
        // Prevent raised hands from repeatedly taking over each other's grip.
        private readonly bool[] mustLower = new bool[4];
        private string session;
        public int ModeVersion { get; private set; }

        // Tracking only: opens one hand, e.g. after the same player's other hand grabbed a new hold.
        public void ReleaseHand(int hand)
        {
            if (!useTracking)
            {
                return;
            }
            trackingGrip[hand] = false;
            wantsGrip[hand] = false;
            mustLower[hand] = true;
        }

        public void SetTracking(bool value)
        {
            useTracking = value;
            ModeVersion++;
            for (int i = 0; i < 4; i++)
            {
                trackingGrip[i] = false;
                wantsGrip[i] = false;
            }
        }

        private void Update()
        {
            Keyboard keys = Keyboard.current;
            if (keys != null && keys.tKey.wasPressedThisFrame)
            {
                SetTracking(!useTracking);
            }

            if (useTracking)
            {
                ReadTracking();
            }
            else
            {
                ReadKeyboard(keys);
            }
        }

        private void ReadKeyboard(Keyboard keys)
        {
            playerSeen[0] = true;
            playerSeen[1] = true;

            for (int i = 0; i < 4; i++)
            {
                int player = i / 2;
                bool grip = IsGripKeyPressed(keys, i);
                float vertical = GetVertical(keys, player);
                float horizontal = GetHorizontal(keys, player);

                // Left/right changes how far out the arms reach (done once per player).
                if (i % 2 == 0)
                {
                    float newReach = sideReach[player] + horizontal * Time.deltaTime;
                    sideReach[player] = Mathf.Clamp(newReach, 0.4f, 1.6f);
                }

                // Up/down moves the hands. Hands never jump back: they stay where the player leaves them.
                // - A grabbing hand follows up/down: pulling down lifts the body.
                // - A free hand moves the other way while this player's other hand grabs,
                //   so pulling down with one hand reaches up with the other, like real climbing.
                // - With no hand grabbing, up/down simply aims both free hands.
                bool playerGrabs = IsGripKeyPressed(keys, player * 2) || IsGripKeyPressed(keys, player * 2 + 1);
                float move = vertical;
                if (!grip && playerGrabs)
                {
                    move = -vertical;
                }
                float newHeight = handHeights[i] + move * 1.4f * Time.deltaTime;
                handHeights[i] = Mathf.Clamp(newHeight, -0.6f, 2.1f);

                // Player 1 reaches to the left, player 2 to the right.
                float direction = 1;
                if (player == 0)
                {
                    direction = -1;
                }
                float sideways = sideReach[player] + extraReach[i];

                // The hand is drawn exactly where it would grab.
                handOffsets[i] = new Vector2(direction * sideways, handHeights[i]);
                reachOffsets[i] = handOffsets[i];

                wantsGrip[i] = grip;
                canGrab[i] = true;
            }
        }

        // Hands 0 and 1 are player 1 (Q, E). Hands 2 and 3 are player 2 (U, O).
        private static bool IsGripKeyPressed(Keyboard keys, int hand)
        {
            if (keys == null)
            {
                return false;
            }
            if (hand == 0)
            {
                return keys.qKey.isPressed;
            }
            if (hand == 1)
            {
                return keys.eKey.isPressed;
            }
            if (hand == 2)
            {
                return keys.uKey.isPressed;
            }
            return keys.oKey.isPressed;
        }

        // 1 for up, -1 for down, 0 for neither or both.
        private static float GetVertical(Keyboard keys, int player)
        {
            if (keys == null)
            {
                return 0;
            }

            float up;
            float down;
            if (player == 0)
            {
                up = KeyValue(keys.wKey.isPressed);
                down = KeyValue(keys.sKey.isPressed);
            }
            else
            {
                up = KeyValue(keys.upArrowKey.isPressed);
                down = KeyValue(keys.downArrowKey.isPressed);
            }
            return up - down;
        }

        // 1 for reaching further out, -1 for pulling the arms in.
        private static float GetHorizontal(Keyboard keys, int player)
        {
            if (keys == null)
            {
                return 0;
            }

            float outwards;
            float inwards;
            if (player == 0)
            {
                outwards = KeyValue(keys.aKey.isPressed);
                inwards = KeyValue(keys.dKey.isPressed);
            }
            else
            {
                outwards = KeyValue(keys.rightArrowKey.isPressed);
                inwards = KeyValue(keys.leftArrowKey.isPressed);
            }
            return outwards - inwards;
        }

        private static float KeyValue(bool pressed)
        {
            if (pressed)
            {
                return 1;
            }
            return 0;
        }

        private void ReadTracking()
        {
            TrackingFrame frame = null;
            if (receiver != null)
            {
                frame = receiver.CurrentFrame;
            }

            string nextSession = null;
            if (frame != null)
            {
                nextSession = frame.session_id;
            }

            // Python restarted or stopped, so old grips are no longer valid.
            if (nextSession != session)
            {
                session = nextSession;
                ModeVersion++;
                for (int i = 0; i < 4; i++)
                {
                    trackingGrip[i] = false;
                }
            }

            float aspect = 1f;
            if (frame != null)
            {
                aspect = frame.Aspect;
            }

            for (int player = 0; player < 2; player++)
            {
                PlayerState pose = null;
                playerSeen[player] = false;
                if (receiver != null)
                {
                    playerSeen[player] = receiver.TryGetPlayer(player + 1, out pose);
                }
                handRaised[player] = playerSeen[player] && pose.hand_raised.Length == 2
                    && (pose.hand_raised[0] || pose.hand_raised[1]);
                inPlayingArea[player] = false;
                if (!playerSeen[player])
                    positionHint[player] = "STEP INTO VIEW";
                else if (pose.world_position_m.Length != 3)
                    positionHint[player] = "SHOW SHOULDERS + HIPS";
                else if (pose.world_position_m[2] < minPlayingDistance)
                    positionHint[player] = "STEP BACK";
                else if (pose.world_position_m[2] > maxPlayingDistance)
                    positionHint[player] = "STEP CLOSER";
                else
                {
                    inPlayingArea[player] = true;
                    positionHint[player] = null;
                }

                for (int side = 0; side < 2; side++)
                {
                    int hand = player * 2 + side;
                    canGrab[hand] = false;

                    Vector2 offset;
                    bool fresh;
                    if (!playerSeen[player] || !TryArm(pose, side, aspect, out offset, out fresh))
                    {
                        // Not seen: the hand stays where it was and keeps holding for a moment, so the
                        // climbers do not fall. After that it lets go.
                        if (Time.time - lastArmTime[hand] > LostGrace)
                        {
                            trackingGrip[hand] = false;
                        }
                        wantsGrip[hand] = trackingGrip[hand];
                        continue;
                    }
                    lastArmTime[hand] = Time.time;

                    // Both arms point outwards from their own torso. The bridge already mirrors the image.
                    float direction = 1;
                    if (player == 0)
                    {
                        direction = -1;
                    }
                    // Outward (away from the other player) moves the hand past the holds, inward moves
                    // it back towards the middle.
                    float outward = offset.x * direction;
                    float x = Mathf.Clamp(ColumnReach + extraReach[hand] + outward * SideScale, -0.3f, 2.4f);
                    float y = Mathf.Clamp(offset.y * reachScale, -1.2f, 2.4f);
                    handOffsets[hand] = new Vector2(direction * x, y);
                    reachOffsets[hand] = handOffsets[hand];

                    // A hand above the release line is closed: it grabs the hold it touches (ClimberBody
                    // only takes holds above the shoulder). Dropping it below the line lets go.
                    if (offset.y < 0f)
                    {
                        mustLower[hand] = false;
                    }
                    if (offset.y < LetGo)
                    {
                        trackingGrip[hand] = false;
                    }
                    else if (!mustLower[hand])
                    {
                        trackingGrip[hand] = true;
                    }

                    wantsGrip[hand] = trackingGrip[hand];
                    canGrab[hand] = fresh;
                }
            }
        }

        // Finds where one wrist is compared to its shoulder, measured in shoulder widths.
        // aspect is the camera's width / height, so x and y use the same unit.
        public static bool TryArm(PlayerState pose, int side, float aspect, out Vector2 offset, out bool fresh)
        {
            offset = Vector2.zero;
            fresh = false;
            if (pose == null)
            {
                return false;
            }

            // The climbers face the players like a mirror image. Side 0 is the arm on the right
            // of the screen, so the player's right hand moves it.
            PoseJoint wristJoint = PoseJoint.LeftWrist;
            PoseJoint elbowJoint = PoseJoint.LeftElbow;
            if (side == 0)
            {
                wristJoint = PoseJoint.RightWrist;
                elbowJoint = PoseJoint.RightElbow;
            }

            PoseLandmark left;
            PoseLandmark right;
            if (!pose.TryGetJoint(PoseJoint.LeftShoulder, out left))
            {
                return false;
            }
            if (!pose.TryGetJoint(PoseJoint.RightShoulder, out right))
            {
                return false;
            }

            Vector2 leftShoulder = Point(left, aspect);
            Vector2 rightShoulder = Point(right, aspect);

            // Shoulder width is the unit, so the distance to the camera does not matter.
            float width = Vector2.Distance(leftShoulder, rightShoulder);
            if (width < 0.02f)
            {
                return false;
            }

            Vector2 shoulder = leftShoulder;
            if (side == 0)
            {
                shoulder = rightShoulder;
            }

            // Use the wrist when tracked, or extend a visible raised elbow.
            PoseLandmark wrist;
            PoseLandmark elbow;
            bool wristFound = pose.TryGetJoint(wristJoint, out wrist);
            bool wristSeen = wristFound && wrist.state == JointState.Tracked;
            // Image y points down, so a smaller y is higher.
            bool elbowRaised = pose.TryGetJoint(elbowJoint, out elbow) && elbow.state == JointState.Tracked
                && Point(elbow, aspect).y < shoulder.y;

            Vector2 wristPoint;
            if (wristSeen || (wristFound && !elbowRaised))
            {
                wristPoint = Point(wrist, aspect);
            }
            else if (elbowRaised)
            {
                // A raised wrist often leaves the top of the picture. The arm then goes on past the elbow
                // as far again as from the shoulder, which is about where the wrist of a straight arm is.
                Vector2 elbowPoint = Point(elbow, aspect);
                wristPoint = elbowPoint + (elbowPoint - shoulder);
                wristSeen = true;
            }
            else
            {
                return false;
            }

            // Flip y: a positive y means the wrist is above the shoulder.
            float x = wristPoint.x - shoulder.x;
            float y = shoulder.y - wristPoint.y;
            offset = new Vector2(x, y) / width;

            bool leftSeen = left.state == JointState.Tracked;
            bool rightSeen = right.state == JointState.Tracked;
            fresh = wristSeen && leftSeen && rightSeen;
            return true;
        }

        // A joint in the same unit on both axes: aspect is the camera's width / height.
        private static Vector2 Point(PoseLandmark joint, float aspect)
        {
            return new Vector2((float)joint.position[0] * aspect, (float)joint.position[1]);
        }
    }
}
