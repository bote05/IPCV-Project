# IPCV Climbing Game

2 player climbing game controlled with the webcam. The two climbers are joined at the waist, so they only get up the wall together. Python does the tracking (OpenCV, MediaPipe) and sends it to Unity over UDP. Unity runs the game.

## Task Division
| Task | Name | What | Folders |
|---|---|---|---|
| 1 Face | Jona Patig | face detection, smoothing and face crops that go on the climbers' heads | `Python/Face` |
| 2 Pose | Alexandra Lupan | MediaPipe pose, filtering, motion signals, pose evaluation tools | `Python/Pose` |
| 3 Identity | Noah Verheijen | player 1 / player 2 identities, positions in metres | `Python/Identity` |
| 4 Interaction | Badr Boubric | the Unity game: climbers, holds, hand input, HUD | `Assets/Scripts/Interaction`, `Assets/Scenes`, `Assets/Art`, `Assets/Materials` |
| 5 Integration (lead) | Daniel Botezatu | combined pipeline, UDP bridge, metrics, testing | `Python/main.py`, `Python/tracking.py`, `Python/bridge`, `Assets/Scripts/Bridge` |

**Only edit your own folders.** If you need something changed in someone else's code, just ask them.

## Setup

Use Python 3.12 and Unity 6000.4.8f1 for the tested setup. From the repository root:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r Python/requirements.txt
cd Python
python -m Pose.tools.download_models   # Models/pose_landmarker_full.task
python -m Face.download_model          # Models/face_detection_yunet_2023mar.onnx, face_recognition_sface_2021dec.onnx
```

## Build the game (once)

Build the Unity game into `Build/ClimbingGame.exe` (the folder is not in git). Either in Unity: File > Build Profiles > Windows > Build, choosing `Build/ClimbingGame.exe` in the repository root; or from the repository root:

```powershell
& "<Unity 6000.4.8f1>\Editor\Unity.exe" -batchmode -quit -projectPath "IPCV Climbing Game" -buildWindows64Player "Build\ClimbingGame.exe"
```

## Run

1. Double-click `play.bat` (or `python main.py --play` from `Python/`). It starts the game in a window and the webcam tracking together. Closing the game window, or pressing `Q` in the webcam window, stops both.
2. The mirrored webcam window shows each player's skeleton, number and face box. Place it next to the game window.
3. Both players stand side by side about 2 m from the camera and face it until `P1` and `P2` appear. Player 1 is the one on the left of the screen and controls the left climber.
4. Both raise a hand to start. Keep both shoulders and wrists in view for the ready check.
5. Optional, for positions in metres: both stand upright at 2 m with shoulders and hips visible, then press `C` in the webcam window (`--calibration-distance` changes the 2 m). A rejected calibration leaves the previous one intact; the terminal explains why.

For development, run the game in the Unity editor instead: open `Assets/Scenes/ClimbingRoom.unity`, press Play, then run `python main.py --webcam` from `Python/`. Without a webcam, `python main.py --demo` sends two fake players.

## How to play

- Reach up with a hand to grab the hold nearest to it, pull down to lift the climbers, lower the hand below the chest to let go.
- Your left hand moves the arm on the left of the screen, like a mirror.
- Cracked holds break a few seconds after they are grabbed.
- At the top, the body stays in place. Both players lower their hands and then raise one to climb again. Missing tracking does not count as lowering hands.

| Where | Key | What |
|---|---|---|
| Webcam window | `C` | calibrate positions at the calibration distance |
| Webcam window | `N` | new players: forget faces and identities |
| Webcam window | `Q` | quit Python |
| Unity | `T` | switch between camera and keyboard control |
| Unity | `R` | restart |
| Unity | `H` | help text |
| Unity | `F1` | bridge panel: tracking, latency, metrics CSV |

Keyboard mode: player 1 grabs with `Q`/`E` and moves with `W A S D`, player 2 grabs with `U`/`O` and moves with the arrow keys.

## Tests

From `Python/`:

```powershell
python -m pytest Pose/tests bridge/tests
```

Unity play mode tests for the bridge are in `Assets/Scripts/Bridge/Tests` (Window > General > Test Runner).

## Folder Structure
```
IPCV-Project/
├── README.md
├── play.bat                     # starts the built game and the tracking together
├── Build/                       # built game (ClimbingGame.exe), not in git
├── Models/                      # downloaded model files, not in git
├── Python/
│   ├── main.py                  # entry point: webcam -> tracking -> UDP, or --demo
│   ├── tracking.py              # runs Face, Pose and Identity on every frame
│   ├── requirements.txt
│   ├── Face/                    # Task 1
│   ├── Pose/                    # Task 2, with tools/ (demo, annotate) and evaluation/
│   ├── Identity/                # Task 3: identity.py, position.py
│   └── bridge/                  # Task 5: UDP protocol and sender, tests
└── IPCV Climbing Game/          # Unity project
    └── Assets/
        ├── Scenes/ClimbingRoom.unity
        ├── Scripts/
        │   ├── Interaction/     # Task 4: ClimberInput, ClimberBody, ClimbingHold, ClimberFace, HUD, camera
        │   └── Bridge/          # Task 5: UDP receivers, metrics CSV, debug panel, tests
        ├── Art/                 # climber models (Quaternius, CC0)
        ├── Materials/, Fonts/, Settings/
        └── TextMesh Pro/        # TMP essentials
```
