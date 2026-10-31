using System.Collections.Generic;
using TMPro;
using UnityEngine;

namespace IPCV.Game
{
    // Extends the silo by repeating its holds and wall sections.
    public sealed class SiloExtender : MonoBehaviour
    {
        public ClimberBody climber;
        [Tooltip("Rows of holds added above the 15 in the scene. Rows are 1.35 m apart; 8 rows make a climb of about 30 m.")]
        public int extraRows = 8;
        private const int SceneRows = 15;
        private const float RowSpacing = 1.35f;
        // Wall panels and ribs are 25 m tall; beams, light strips and height marks repeat every 4 m up to 24 m.
        private const float WallHeight = 25f;
        private const float FloorsHeight = 24f;
        // One copy reaches 48 m, which covers the view at a 30 m climb. Raise it for longer climbs.
        private const int Copies = 1;

        private void Awake()
        {
            var parts = new List<Transform>();
            foreach (Transform child in transform)
            {
                parts.Add(child);
            }

            var holds = new Dictionary<string, Transform>();
            var plates = new[] { new List<Transform>(), new List<Transform>() };
            Transform sign = null;
            foreach (Transform part in parts)
            {
                string name = part.name;
                if (name == "Hold Plate")
                {
                    plates[part.localPosition.x < 0 ? 0 : 1].Add(part);
                }
                else if (name.StartsWith("Hold "))
                {
                    holds[name.Substring(5)] = part;
                }
                else if (name == "Wall Panel" || name == "Vertical Rib")
                {
                    for (int c = 1; c <= Copies; c++)
                    {
                        Copy(part, c * WallHeight);
                    }
                }
                else if (name == "Ring Beam" || name == "Strip Housing" || name == "Strip Light" || name == "Level Mark")
                {
                    // The bottom row (floor level) is not repeated.
                    if (part.localPosition.y < 3f)
                    {
                        continue;
                    }
                    for (int c = 1; c <= Copies; c++)
                    {
                        Transform copy = Copy(part, c * FloorsHeight);
                        TMP_Text label = copy.GetComponent<TMP_Text>();
                        if (label != null && int.TryParse(label.text, out int metres))
                        {
                            label.text = (metres + (int)(c * FloorsHeight)).ToString("00");
                        }
                    }
                }
                else if (name == "Left Boundary" || name == "Right Boundary")
                {
                    // Keep the bottom where it is and reach the top of the copied walls.
                    Vector3 scale = part.localScale;
                    float bottom = part.localPosition.y - scale.y / 2f;
                    scale.y = (Copies + 1) * WallHeight - bottom;
                    part.localScale = scale;
                    part.localPosition = new Vector3(part.localPosition.x, bottom + scale.y / 2f, part.localPosition.z);
                }
                else if (name == "Exit Sign")
                {
                    sign = part;
                }
            }

            plates[0].Sort((a, b) => a.localPosition.y.CompareTo(b.localPosition.y));
            plates[1].Sort((a, b) => a.localPosition.y.CompareTo(b.localPosition.y));
            for (int row = SceneRows; row < SceneRows + extraRows; row++)
            {
                int source = row % SceneRows;
                int block = row / SceneRows;
                for (int side = 0; side < 2; side++)
                {
                    string label = source + (side == 0 ? " Left" : " Right");
                    if (holds.TryGetValue(label, out Transform hold))
                    {
                        Copy(hold, block * SceneRows * RowSpacing).name = "Hold " + row + (side == 0 ? " Left" : " Right");
                    }
                    if (source < plates[side].Count)
                    {
                        Copy(plates[side][source], block * SceneRows * RowSpacing);
                    }
                }
            }

            if (climber != null)
            {
                climber.finishHeight += extraRows * RowSpacing;
                // Above the climbers' heads at the finish and in front of the wall, so nothing covers it.
                if (sign != null)
                {
                    sign.localPosition = new Vector3(sign.localPosition.x, climber.finishHeight + 3.5f, -1f);
                }
            }
        }

        private static Transform Copy(Transform part, float up)
        {
            Transform copy = Instantiate(part.gameObject, part.parent).transform;
            copy.name = part.name;
            copy.localPosition = part.localPosition + Vector3.up * up;
            return copy;
        }
    }
}
