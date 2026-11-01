# IPCV Climbing Game

2 player climbing game controlled with the webcam. The two climbers are joined at the waist, so they only get up the wall together. Python does the tracking (OpenCV, MediaPipe) and sends it to Unity over UDP. Unity runs the game.

## Quick start (Windows)

The game build and required models are included. You do not need Unity to play.

1. Install Python 3.12 and extract the ZIP if you downloaded the project that way.
2. Open PowerShell in the project folder, next to `play.bat`, and run:

   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r Python/requirements.txt
   ```

3. Double-click `play.bat`.
4. Both players stand about 2 m from the camera, wait for `P1` and `P2`, then each raise a hand to start.

The default is camera 0. To choose another webcam, run `.\play.bat --camera 1` in PowerShell; try another index if needed. Close the game or press `Q` in the webcam window to stop.

## Task Division
| Task | Name | What | Folders |
|---|---|---|---|
| 1 Face | Jona Patig | face detection, smoothing and face crops that go on the climbers' heads | `Python/Face` |
| 2 Pose | Alexandra Lupan | MediaPipe pose, filtering, motion signals, pose evaluation tools | `Python/Pose` |
| 3 Identity | Noah Verheijen | player 1 / player 2 identities, positions in metres | `Python/Identity` |
| 4 Interaction | Badr Boubric | the Unity game: climbers, holds, hand input, HUD | `Assets/Scripts/Interaction`, `Assets/Scenes`, `Assets/Art`, `Assets/Materials` |
| 5 Integration (lead) | Daniel Botezatu | combined pipeline, UDP bridge, metrics, testing | `Python/main.py`, `Python/tracking.py`, `Python/bridge`, `Assets/Scripts/Bridge` |

**Only edit your own folders.** If you need something changed in someone else's code, just ask them.

## Restore models (optional)

The three runtime models are included in `Models/`. If any are missing, restore them from the project root with:

```powershell
cd Python
..\.venv\Scripts\python.exe -m Pose.tools.download_models
..\.venv\Scripts\python.exe -m Face.download_model
cd ..
```

Model sources and licences are listed in [Models/README.md](Models/README.md).

## Build the game (optional)

The included `Build/ClimbingGame.exe` was built with Unity 6000.4.8f1. Keep the entire `Build/` folder together; the executable needs its data folder and libraries. To rebuild after changing the Unity project, use File > Build Profiles > Windows > Build, choosing `Build/ClimbingGame.exe` in the repository root; or from the repository root:

```powershell
& "<Unity 6000.4.8f1>\Editor\Unity.exe" -batchmode -quit -projectPath "IPCV Climbing Game" -buildWindows64Player "Build\ClimbingGame.exe"
```

## Run

1. Double-click `play.bat`. It starts the game in a window and the webcam tracking together. Closing the game window, or pressing `Q` in the webcam window, stops both.
2. The mirrored webcam window shows each player's skeleton, number and face box. Place it next to the game window.
3. Both players stand side by side about 2 m from the camera and face it until `P1` and `P2` appear. Player 1 is the one on the left of the screen and controls the left climber.
4. Distance from the camera is estimated from torso size, so no calibration is needed to start. For more accurate positions, press `C` in the webcam window and both stand upright at 2 m before the 5 second countdown ends (`--calibration-distance` changes the 2 m). A rejected calibration leaves the previous one intact; the terminal explains why.
5. Both raise a hand to start. Keep both shoulders and wrists in view. Before starting or restarting, the game checks each player's estimated camera distance: 1.5-3 m by default, adjustable on `ClimberInput`. Follow the player's "step closer/back" message. This check does not interrupt an active climb.

When the tracking stops, the terminal prints how long each part of the webcam loop took. `play.bat --camera-thread` reads the webcam in a background thread. In our test that gave 19.5 instead of 17.5 tracked frames per second, but each frame reached the game about 20 ms later (64.5 instead of 44.8 ms median), so it is off by default.

The face photos follow each player's head tilt. Eye landmarks provide roll; yaw and pitch are not estimated. Photos use the upright, mirrored SFace crop so the tilt is applied only once in Unity.

For development, run the game in the Unity editor instead: open `Assets/Scenes/ClimbingRoom.unity`, press Play, then run `..\.venv\Scripts\python.exe main.py --webcam` from `Python/`. Without a webcam, `..\.venv\Scripts\python.exe main.py --demo` sends two fake players.

## How to play

- Raise an arm straight up until the hand touches a hold to grab it. Pull it down to your shoulder to lift the climbers, then drop it to your stomach to let go. Moving a raised hand sideways moves the climber's hand.
- Your left hand moves the arm on the left of the screen, like a mirror.
- Grabbing with your other hand transfers your grip to that hand.
- If the camera briefly loses a player, for example when a hand covers the face, their climber keeps holding for 2 seconds.
- Cracked holds break a few seconds after they are grabbed.
- The silo is endless: new sections appear ahead as you climb, with independently randomized holds for each player. The first two rows are forgiving; later rows have wider spacing, smaller grab tolerance and a mix of solid and cracked holds. Each pair always includes a solid hold.
- Distant sections unload to keep the active level bounded. Falling back recreates the same route; a new Play session chooses a new route. Set `SiloExtender.seed` to a nonzero value for a repeatable layout. Press `R` to retry from the bottom; both players must be ready again in camera mode.

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
..\.venv\Scripts\python.exe -m pytest Pose/tests bridge/tests
```

Unity play mode tests for the bridge are in `Assets/Scripts/Bridge/Tests` (Window > General > Test Runner).

## Folder Structure
```
IPCV-Project/
├── README.md
├── play.bat                     # starts the built game and the tracking together
├── Build/                       # Windows game and its runtime files
├── Models/                      # three required models and their licences
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
        ├── Resources/Music/     # background music: "Sprite Sprint" from Super 16-Bit Sounds (Dylann Taylor, CC0)
        ├── Materials/, Fonts/, Settings/
        └── TextMesh Pro/        # TMP essentials
```
