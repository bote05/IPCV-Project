using System;
using System.Collections.Generic;
using TMPro;
using UnityEngine;

namespace IPCV.Game
{
    // Keep only nearby sections alive. A section's seed makes falls/revisits reproduce its route.
    [DefaultExecutionOrder(-20)]
    public sealed class SiloExtender : MonoBehaviour
    {
        public ClimberBody climber;
        [Tooltip("Zero chooses a fresh route each time Play starts; otherwise reproducible.")]
        public int seed;
        private const float SectionHeight = 24f;
        private const int Rows = 16;
        private readonly Dictionary<int, GameObject> sections = new Dictionary<int, GameObject>();
        private readonly List<Transform> scenery = new List<Transform>();
        private readonly List<Transform>[] solid = { new List<Transform>(), new List<Transform>() };
        private readonly List<Transform>[] cracked = { new List<Transform>(), new List<Transform>() };
        private readonly Transform[] plates = new Transform[2];
        private readonly List<Transform> boundaries = new List<Transform>();
        private int routeSeed;
        private int currentSection = -1;

        private void Awake()
        {
            if (climber == null) return;
            routeSeed = seed == 0 ? Guid.NewGuid().GetHashCode() : seed;
            // Original objects are inactive templates: no runtime material/audio state is copied.
            foreach (Transform part in transform)
            {
                int side = part.localPosition.x < 0 ? 0 : 1;
                var hold = part.GetComponent<ClimbingHold>();
                if (hold != null)
                {
                    (hold.breaksWhenUsed ? cracked[side] : solid[side]).Add(part);
                    part.gameObject.SetActive(false);
                }
                else if (part.name == "Hold Plate")
                {
                    if (plates[side] == null) plates[side] = part;
                    part.gameObject.SetActive(false);
                }
                else if (part.name == "Wall Panel" || part.name == "Vertical Rib"
                    || part.name == "Ring Beam" || part.name == "Strip Housing"
                    || part.name == "Strip Light" || part.name == "Level Mark")
                {
                    scenery.Add(part);
                    part.gameObject.SetActive(false);
                }
                else if (part.name == "Left Boundary" || part.name == "Right Boundary")
                    boundaries.Add(part);
                else if (part.name == "Exit Sign")
                    part.gameObject.SetActive(false);
            }
            if (solid[0].Count == 0 || solid[1].Count == 0)
            {
                Debug.LogError("Endless silo needs a solid hold template on each side.", this);
                enabled = false;
                return;
            }
            climber.endless = true;
            RefreshSections();
        }

        private void Update() { RefreshSections(); }

        private void RefreshSections()
        {
            if (climber == null) return;
            int section = Mathf.Max(0, Mathf.FloorToInt(transform.InverseTransformPoint(climber.transform.position).y / SectionHeight));
            if (section == currentSection) return;
            currentSection = section;
            // One section below protects falls; two ahead hide creation above the camera.
            int first = Mathf.Max(0, section - 1);
            int last = section + 2;
            for (int i = first; i <= last; i++)
                if (!sections.ContainsKey(i)) CreateSection(i);
            var retired = new List<int>();
            foreach (var pair in sections)
            {
                if (pair.Key >= first && pair.Key <= last) continue;
                bool held = false;
                foreach (var grip in climber.heldHolds)
                    if (grip != null && grip.transform.IsChildOf(pair.Value.transform)) held = true;
                if (!held) retired.Add(pair.Key);
            }
            foreach (int i in retired)
            {
                sections[i].SetActive(false); // Unregister holds immediately, before Destroy's end-of-frame.
                Destroy(sections[i]);
                sections.Remove(i);
            }
            foreach (var boundary in boundaries)
            {
                var scale = boundary.localScale;
                scale.y = SectionHeight * 5f;
                boundary.localScale = scale;
                var position = boundary.localPosition;
                position.y = section * SectionHeight + SectionHeight;
                boundary.localPosition = position;
            }
        }

        private void CreateSection(int index)
        {
            var root = new GameObject("Silo section " + index);
            root.transform.SetParent(transform, false);
            sections.Add(index, root);
            float bottom = index * SectionHeight;
            foreach (var source in scenery)
            {
                // Floor-level beams would overlap the previous section's top beam.
                if (index > 0 && source.localPosition.y < 3f
                    && source.name != "Wall Panel" && source.name != "Vertical Rib") continue;
                var part = Copy(source, root.transform, source.localPosition + Vector3.up * bottom);
                var label = part.GetComponent<TMP_Text>();
                if (label != null && int.TryParse(label.text, out int metres))
                    label.text = (metres + Mathf.RoundToInt(bottom)).ToString("00");
            }
            var random = new System.Random(unchecked(routeSeed + index * 7919));
            float difficulty = Mathf.Clamp01(index / 5f);
            for (int row = 0; row < Rows; row++)
            {
                int globalRow = index * Rows + row;
                // At most one cracked hold per pair; the other player always has a solid option.
                int fragileSide = globalRow > 2 && random.NextDouble() < 0.30 + difficulty * 0.15
                    ? random.Next(2) : -1;
                for (int side = 0; side < 2; side++)
                {
                    bool starter = globalRow < 2;
                    // Different phases per side remove the old mirrored ladder pattern.
                    float outward = starter ? 1.8f : 1.45f + (float)random.NextDouble() * (0.7f + difficulty * 0.15f);
                    float jitter = starter ? 0 : ((float)random.NextDouble() - 0.5f) * 0.28f;
                    float y = bottom + 3.25f + row * 1.5f + jitter;
                    var choices = side == fragileSide && cracked[side].Count > 0 ? cracked[side] : solid[side];
                    var source = choices[random.Next(choices.Count)];
                    var position = new Vector3(side == 0 ? -outward : outward, y, source.localPosition.z);
                    var part = Copy(source, root.transform, position);
                    part.name = "Hold " + globalRow + (side == 0 ? " Left" : " Right");
                    var hold = part.GetComponent<ClimbingHold>();
                    hold.grabDistance = starter ? 0.65f : Mathf.Lerp(0.60f, 0.54f, difficulty);
                    if (plates[side] != null)
                    {
                        var platePosition = position;
                        platePosition.z = plates[side].localPosition.z;
                        Copy(plates[side], root.transform, platePosition);
                    }
                }
            }
        }

        private static Transform Copy(Transform source, Transform parent, Vector3 position)
        {
            var copy = Instantiate(source.gameObject, parent).transform;
            copy.name = source.name;
            copy.localPosition = position;
            copy.gameObject.SetActive(true);
            return copy;
        }
    }
}
