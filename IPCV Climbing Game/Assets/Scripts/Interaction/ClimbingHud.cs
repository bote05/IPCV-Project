using TMPro;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.UI;

namespace IPCV.Game
{
    public sealed class ClimbingHud : MonoBehaviour
    {
        public ClimberBody climber;
        public ClimberInput input;
        public TMP_Text heightText;
        public TMP_Text statusText;
        public TMP_Text controlsText;
        public TMP_Text secondPlayerText;
        public Image[] grips = new Image[4];
        public Color firstColor = new Color(0.95f, 0.72f, 0.3f);
        public Color secondColor = new Color(0.94f, 0.43f, 0.3f);
        private bool showHelp = true;
        private bool climbed;

        private void Update()
        {
            // Hide the help text once the players have climbed one metre.
            if (!climbed && climber.BestHeight > 1f)
            {
                climbed = true;
                showHelp = false;
            }

            if (Keyboard.current != null && Keyboard.current.hKey.wasPressedThisFrame)
            {
                showHelp = !showHelp;
            }

            heightText.text = $"{climber.BestHeight:00.0}<size=24> m</size>";

            if (climber.Finished)
            {
                statusText.text = "TOP REACHED!   R / AGAIN";
            }
            else
            {
                string switchTo = "CAMERA";
                if (input.useTracking)
                {
                    switchTo = "KEYS";
                }
                statusText.text = "H / HELP     T / " + switchTo + "     R / RETRY";
            }

            controlsText.text = PlayerText(0);
            secondPlayerText.text = PlayerText(1);

            // Grip icons: full colour when that hand is holding on, faded otherwise.
            for (int i = 0; i < grips.Length; i++)
            {
                Color color = secondColor;
                if (i < 2)
                {
                    color = firstColor;
                }

                if (climber.heldHolds[i] != null)
                {
                    color.a = 1;
                }
                else
                {
                    color.a = 0.25f;
                }
                grips[i].color = color;
            }
        }

        private string PlayerText(int player)
        {
            string label = "02";
            if (player == 0)
            {
                label = "01";
            }

            if (input.useTracking && !input.playerSeen[player])
            {
                return label + "   STEP INTO VIEW";
            }
            if (!showHelp)
            {
                return label;
            }
            if (input.useTracking)
            {
                return label + "   REACH UP / PULL DOWN\n       LOWER HAND TO LET GO";
            }
            if (player == 0)
            {
                return "01   Q E / GRAB\n       W S A D";
            }
            return "02   U O / GRAB\n       ARROWS";
        }
    }
}
