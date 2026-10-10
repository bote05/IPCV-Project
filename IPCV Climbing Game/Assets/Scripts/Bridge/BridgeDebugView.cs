using UnityEngine;

namespace IPCV.Bridge
{
    [RequireComponent(typeof(UdpFrameReceiver), typeof(UdpFaceReceiver), typeof(BridgeMetrics))]
    public sealed class BridgeDebugView : MonoBehaviour
    {
        private UdpFrameReceiver receiver;
        private UdpFaceReceiver faces;
        private BridgeMetrics metrics;
        private Vector2 scroll;
        private void Awake()
        {
            receiver = GetComponent<UdpFrameReceiver>();
            faces = GetComponent<UdpFaceReceiver>();
            metrics = GetComponent<BridgeMetrics>();
        }

        private void OnGUI()
        {
            GUILayout.BeginArea(new Rect(12, 12, Mathf.Min(550, Screen.width - 24), Screen.height - 24), GUI.skin.box);
            scroll = GUILayout.BeginScrollView(scroll);
            GUILayout.Label("IPCV bridge | --demo uses synthetic data");
            GUILayout.Label($"Tracking {metrics.TrackingFps:F1} FPS | Unity {metrics.UnityFps:F1} FPS");
            TrackingFrame frame = receiver.CurrentFrame;
            if (frame == null) GUILayout.Label("No fresh tracking. Run: python Python/main.py --demo");
            else
            {
                GUILayout.Label($"Frame {frame.sequence} | Python processing {frame.processing_ms:F3} ms");
                GUILayout.Label($"Receive -> apply {receiver.ReceiveToApplyMs:F3} ms");
                if (!double.IsNaN(receiver.CaptureToApplyMs))
                    GUILayout.Label($"Encode + UDP {receiver.EncodeAndUdpMs:F3} ms | capture -> apply {receiver.CaptureToApplyMs:F3} ms");
                foreach (PlayerState p in frame.players)
                {
                    GUILayout.Label($"Player {p.id}: {(p.tracked ? "tracked" : "lost")}");
                    if (!p.tracked) continue;
                    if (p.pose.Length == 33)
                        GUILayout.Label($"Wrists L ({p.pose[15].position[0]:F2}, {p.pose[15].position[1]:F2}) R ({p.pose[16].position[0]:F2}, {p.pose[16].position[1]:F2})");
                    if (p.world_position_m.Length == 3)
                        GUILayout.Label($"Position m: ({p.world_position_m[0]:F2}, {p.world_position_m[1]:F2}, {p.world_position_m[2]:F2})");
                    if (p.head_rotation_deg.Length == 3)
                        GUILayout.Label($"Head yaw/pitch/roll: {p.head_rotation_deg[0]:F1}, {p.head_rotation_deg[1]:F1}, {p.head_rotation_deg[2]:F1}");
                    foreach (MotionSignal s in p.motion_signals)
                        GUILayout.Label($"{s.name}: {s.active} (confidence {s.confidence:F2})");
                    if (faces.TryGetTexture(p.id, out Texture2D texture))
                        GUILayout.Box(texture, GUILayout.Width(128), GUILayout.Height(128));
                }
            }
            GUILayout.Label($"Accepted {receiver.AcceptedFrames} | rejected {receiver.RejectedFrames} | invalid JSON {receiver.InvalidPackets}");
            GUILayout.Label($"Sequence gaps {receiver.SequenceGaps} | queue/size drops {receiver.QueueDrops} | crops {faces.AcceptedCrops}");
            if (GUILayout.Button(metrics.IsRecording ? "Stop metrics CSV" : "Start metrics CSV"))
            {
                if (metrics.IsRecording) metrics.StopRecording(); else metrics.StartRecording();
            }
            if (metrics.RecordingPath != null) GUILayout.Label(metrics.RecordingPath);
            if (receiver.LastError != null) GUILayout.Label(receiver.LastError);
            if (faces.LastError != null) GUILayout.Label(faces.LastError);
            GUILayout.EndScrollView();
            GUILayout.EndArea();
        }
    }
}
