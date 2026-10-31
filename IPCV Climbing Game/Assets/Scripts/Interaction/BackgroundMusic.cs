using UnityEngine;

namespace IPCV.Game
{
    // "Sprite Sprint" from Super 16-Bit Sounds by Dylann Taylor (CC0), quiet so the game sounds stay clear.
    public static class BackgroundMusic
    {
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        private static void Play()
        {
            AudioSource music = new GameObject("Background Music").AddComponent<AudioSource>();
            Object.DontDestroyOnLoad(music.gameObject);
            music.clip = Resources.Load<AudioClip>("Music/SpriteSprint_Loop");
            music.loop = true;
            music.volume = 0.08f;
            music.Play();
        }
    }
}
