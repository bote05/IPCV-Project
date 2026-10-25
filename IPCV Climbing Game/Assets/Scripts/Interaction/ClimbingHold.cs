using System.Collections.Generic;
using UnityEngine;

namespace IPCV.Game
{
    public sealed class ClimbingHold : MonoBehaviour
    {
        public static readonly List<ClimbingHold> All = new List<ClimbingHold>();
        public float grabDistance = 0.65f;
        public bool breaksWhenUsed;
        public float breakAfter = 4f;
        public float warningTime = 1.5f;
        public float respawnAfter = 8f;
        public Renderer ring;
        public Transform grip;
        [Tooltip("Separate pieces of a cracked hold. They rattle and fly apart individually.")]
        public Transform[] shards = new Transform[0];
        [Tooltip("Dark inside of a cracked hold, hidden once it shatters.")]
        public GameObject core;
        public Color idleColor = new Color(0.42f, 0.46f, 0.43f);
        public float idleGlow = 0.08f;
        public bool Broken { get; private set; }
        public bool Warning { get; private set; }
        public bool Hovered { get; private set; }
        public bool Held { get; private set; }
        private ClimberBody climber;
        private MaterialPropertyBlock color;
        private MaterialPropertyBlock shardColor;
        private AudioSource sound;
        private AudioClip warningSound;
        private AudioClip breakSound;
        private float lifetime;
        private float brokenTime;
        private float nextBeep;
        private float flashUntil;
        private float phase;
        private bool started;
        private Vector3 gripStart;
        private Quaternion gripRotation;
        private Vector3[] shardStart;
        private Quaternion[] shardRotation;
        private Renderer[] shardRenderers;

        private void OnEnable()
        {
            All.Add(this);
        }

        private void OnDisable()
        {
            All.Remove(this);
        }

        private void Start()
        {
            climber = FindAnyObjectByType<ClimberBody>();
            color = new MaterialPropertyBlock();
            shardColor = new MaterialPropertyBlock();

            // Each hold gets its own timing, so they do not all creak at the same moment.
            phase = Mathf.Repeat(transform.position.x * 3.7f + transform.position.y * 1.3f, 10f);

            if (grip != null)
            {
                gripStart = grip.localPosition;
                gripRotation = grip.localRotation;
            }

            // Remember where every shard starts, so it can be put back later.
            shardStart = new Vector3[shards.Length];
            shardRotation = new Quaternion[shards.Length];
            shardRenderers = new Renderer[shards.Length];
            for (int i = 0; i < shards.Length; i++)
            {
                shardStart[i] = shards[i].localPosition;
                shardRotation[i] = shards[i].localRotation;
                shardRenderers[i] = shards[i].GetComponent<Renderer>();
            }

            if (breaksWhenUsed)
            {
                sound = gameObject.AddComponent<AudioSource>();
                sound.playOnAwake = false;
                sound.volume = 0.3f;
                warningSound = MakeSound(false);
                breakSound = MakeSound(true);
            }
        }

        private void LateUpdate()
        {
            CheckHands();

            if (Broken)
            {
                AnimateBreaking();
                return;
            }

            if (breaksWhenUsed)
            {
                bool justBroke = UpdateCountdown();
                if (justBroke)
                {
                    return;
                }
            }

            if (ring == null)
            {
                return;
            }
            UpdateColors();
        }

        // Works out if a hand is on this hold (Held) or could grab it right now (Hovered).
        private void CheckHands()
        {
            Held = false;
            Hovered = false;
            if (climber == null || Broken)
            {
                return;
            }

            for (int i = 0; i < 4; i++)
            {
                if (climber.heldHolds[i] == this)
                {
                    Held = true;
                }

                // Only free hands that are allowed to grab can hover.
                if (climber.heldHolds[i] != null || !climber.input.canGrab[i])
                {
                    continue;
                }

                Vector3 reach = climber.shoulders[i].position + (Vector3)climber.input.reachOffsets[i];
                if (FindNear(reach) == this)
                {
                    Hovered = true;
                }
            }
        }

        private void AnimateBreaking()
        {
            brokenTime += Time.deltaTime;
            float t = brokenTime;

            if (shards.Length > 0)
            {
                for (int i = 0; i < shards.Length; i++)
                {
                    // Each piece flies outward from the centre, then gravity takes it.
                    Vector3 outward = Vector3.up;
                    if (shardStart[i].sqrMagnitude > 0)
                    {
                        outward = shardStart[i].normalized;
                    }

                    float spin = 1;
                    if (i % 2 != 0)
                    {
                        spin = -1;
                    }

                    float x = outward.x * 1.4f * t;
                    float y = outward.y * 1.1f * t + 0.6f * t - 6f * t * t;
                    float z = -0.9f * t;
                    shards[i].localPosition = shardStart[i] + new Vector3(x, y, z);

                    Quaternion tumble = Quaternion.Euler(t * 260 * outward.y, -t * 210 * outward.x, t * 320 * spin);
                    shards[i].localRotation = shardRotation[i] * tumble;
                }
            }
            else if (grip != null)
            {
                // A hold without shards falls down as one piece.
                grip.localPosition = gripStart + new Vector3(0.2f * t, -4.9f * t * t, -t);
                grip.localRotation = gripRotation * Quaternion.Euler(t * 150, t * 80, 0);
            }

            if (grip != null)
            {
                grip.gameObject.SetActive(t < 1.2f);
            }

            if (brokenTime >= respawnAfter)
            {
                ResetHold();
            }
        }

        // Runs the cracked-hold countdown. Returns true when the hold snapped this frame.
        private bool UpdateCountdown()
        {
            // The countdown starts on the first grab and does not stop when the hand lets go.
            if (Held)
            {
                started = true;
            }
            if (started)
            {
                lifetime += Time.deltaTime;
            }

            Warning = started && lifetime >= breakAfter - warningTime;

            if (lifetime >= breakAfter)
            {
                Break();
                return true;
            }

            // Beep and flash, faster and faster as the end gets closer.
            if (Warning && lifetime >= nextBeep)
            {
                sound.PlayOneShot(warningSound);
                flashUntil = lifetime + 0.075f;
                float remaining = Mathf.Clamp01((breakAfter - lifetime) / warningTime);
                nextBeep = lifetime + Mathf.Lerp(0.1f, 0.35f, remaining);
            }

            Rattle();
            return false;
        }

        private void UpdateColors()
        {
            // Green when held, yellow when a hand can reach it, otherwise the idle colour.
            Color tint = idleColor;
            float glow = idleGlow;
            if (Held)
            {
                tint = new Color(0.2f, 1f, 0.32f);
                glow = 2f;
            }
            else if (Hovered)
            {
                tint = new Color(1f, 0.9f, 0.48f);
                glow = 2.8f;
            }

            // During the warning the ring flashes bright red, with dark red in between.
            bool flash = Warning && lifetime < flashUntil;
            if (Warning)
            {
                if (flash)
                {
                    tint = new Color(1, 0.035f, 0.015f);
                    glow = 3f;
                }
                else
                {
                    tint = new Color(0.25f, 0.015f, 0.005f);
                    glow = 0.1f;
                }
            }

            color.SetColor("_BaseColor", tint);
            color.SetColor("_EmissionColor", tint * glow);
            ring.SetPropertyBlock(color);

            if (shardRenderers.Length == 0)
            {
                return;
            }

            // The rock itself glows hotter as the countdown runs out.
            float heat = 0;
            if (started)
            {
                heat = Mathf.Clamp01(lifetime / breakAfter);
            }

            Color emission;
            if (flash)
            {
                emission = new Color(1f, 0.08f, 0.02f) * 1.6f;
            }
            else
            {
                emission = new Color(0.9f, 0.18f, 0.04f) * heat * heat * 0.8f;
            }

            shardColor.SetColor("_EmissionColor", emission);
            foreach (Renderer piece in shardRenderers)
            {
                if (piece != null)
                {
                    piece.SetPropertyBlock(shardColor);
                }
            }
        }

        // Cracked holds creak every few seconds. Once grabbed they shake harder until they snap.
        private void Rattle()
        {
            if (grip == null)
            {
                return;
            }

            float angle = 0;
            if (started)
            {
                // Fast shaking that gets stronger as the countdown runs out.
                float progress = Mathf.Clamp01(lifetime / breakAfter);
                float strength = Mathf.Lerp(1.2f, 6f, progress * progress);
                angle = Mathf.Sin(Time.time * 47f) * strength;
            }
            else
            {
                // A short wobble every 2.8 seconds that fades out.
                float cycle = Mathf.Repeat(Time.time + phase, 2.8f);
                if (cycle < 0.45f)
                {
                    float wobble = cycle / 0.45f;
                    angle = Mathf.Sin(wobble * Mathf.PI * 3) * 3.5f * (1 - wobble);
                }
            }

            grip.localRotation = gripRotation * Quaternion.Euler(0, 0, angle);
            grip.localPosition = gripStart + new Vector3(0, -Mathf.Abs(angle) * 0.002f, 0);

            // The shards move apart a little while shaking.
            for (int i = 0; i < shards.Length; i++)
            {
                float push = Mathf.Abs(angle) * 0.0025f * (1 + i % 2);
                shards[i].localPosition = shardStart[i] + shardStart[i].normalized * push;
            }
        }

        private void Break()
        {
            Broken = true;
            Warning = false;
            Held = false;
            Hovered = false;
            brokenTime = 0;

            if (ring != null)
            {
                ring.enabled = false;
            }
            if (core != null)
            {
                core.SetActive(false);
            }
            if (grip != null)
            {
                grip.localPosition = gripStart;
                grip.localRotation = gripRotation;
            }
            if (sound != null)
            {
                sound.PlayOneShot(breakSound);
            }

            // Release every hand on this hold, including two hands belonging to one player.
            if (climber == null)
            {
                return;
            }
            for (int i = 0; i < 4; i++)
            {
                if (climber.heldHolds[i] == this)
                {
                    climber.heldHolds[i] = null;
                }
            }
        }

        public void ResetHold()
        {
            Broken = false;
            Warning = false;
            Held = false;
            Hovered = false;
            started = false;
            lifetime = 0;
            brokenTime = 0;
            nextBeep = 0;
            flashUntil = 0;

            if (ring != null)
            {
                ring.enabled = true;
            }
            if (core != null)
            {
                core.SetActive(true);
            }
            if (grip != null)
            {
                grip.gameObject.SetActive(true);
                grip.localPosition = gripStart;
                grip.localRotation = gripRotation;
            }

            // Put every shard back where it started.
            if (shardStart != null)
            {
                for (int i = 0; i < shards.Length; i++)
                {
                    shards[i].localPosition = shardStart[i];
                    shards[i].localRotation = shardRotation[i];
                }
            }

            // Stop the shards glowing.
            if (shardRenderers != null && shardColor != null)
            {
                shardColor.SetColor("_EmissionColor", Color.black);
                foreach (Renderer piece in shardRenderers)
                {
                    if (piece != null)
                    {
                        piece.SetPropertyBlock(shardColor);
                    }
                }
            }
        }

        // Returns the closest hold that is not broken and close enough to grab, or null.
        public static ClimbingHold FindNear(Vector3 position)
        {
            ClimbingHold closest = null;
            float closestDistance = float.PositiveInfinity;

            foreach (ClimbingHold hold in All)
            {
                if (hold.Broken)
                {
                    continue;
                }

                float distance = Vector2.Distance(position, hold.transform.position);
                if (distance <= hold.grabDistance && distance < closestDistance)
                {
                    closest = hold;
                    closestDistance = distance;
                }
            }
            return closest;
        }

        // Makes the warning beep or the snap sound in code, so no audio files are needed.
        private static AudioClip MakeSound(bool breaking)
        {
            const int rate = 24000;
            int length = 1800;
            float fadeSpeed = 50;
            string clipName = "Hold warning";
            if (breaking)
            {
                length = 7200;
                fadeSpeed = 20;
                clipName = "Hold snap";
            }

            float[] samples = new float[length];
            System.Random random = new System.Random(17);
            for (int i = 0; i < samples.Length; i++)
            {
                float t = (float)i / rate;

                // Fade in over the first 50 samples, then die away.
                float fadeIn = Mathf.Min(1, i / 50f);
                float fadeOut = Mathf.Exp(-t * fadeSpeed);
                float envelope = fadeIn * fadeOut;

                float value;
                if (breaking)
                {
                    // A crack: random noise mixed with a low tone.
                    float noise = (float)(random.NextDouble() * 2 - 1);
                    value = noise * 0.65f + Mathf.Sin(t * 1100) * 0.35f;
                }
                else
                {
                    // A beep: an 880 Hz tone.
                    value = Mathf.Sin(t * 2 * Mathf.PI * 880) * 0.5f;
                }
                samples[i] = envelope * value;
            }

            AudioClip clip = AudioClip.Create(clipName, samples.Length, 1, rate, false);
            clip.SetData(samples, 0);
            return clip;
        }

        private void OnDestroy()
        {
            if (warningSound != null)
            {
                Destroy(warningSound);
            }
            if (breakSound != null)
            {
                Destroy(breakSound);
            }
        }
    }
}
