# IPCV Climbing Game

2 player climbing game controlled with the webcam. Python does the tracking (MediaPipe) and sends data to Unity over UDP. Unity does the game.

## Task Division
| Task | Name | What | Folders |
|---|---|---|---|
| 1 Face | Jona Patig | 256x256 face crop (python), put it on the unity model (C#) | `Python/Face`, `Assets/Scripts/Face` |
| 2 Pose | Alexandra Lupan | MediaPipe pose + smoothing | `Python/Pose`, `Assets/Scripts/Pose` |
| 3 Identity | Noah Verheijen | detect and tell players apart | `Python/Identity`, `Assets/Scripts/Identity` |
| 4 Interaction | Badr Boubric | makes the game in Unity, player physics, climbing holds | `Assets/Scripts/Interaction`, `Assets/Scenes`, `Assets/Prefabs`, `Assets/Materials`, `Assets/Audio` |
| 5 Integration (lead) | Daniel Botezatu | UDP bridge between python and unity | `Python/bridge`, `Python/main.py`, `Assets/Scripts/Bridge` |

**Only edit your own folders.** If you need something changed in someone else's code, just ask them.

## Folder Structure
```
IPCV-Project/
├── Python/                  # tracking side
│   ├── main.py              # entry point (runs everything, sends UDP)
│   ├── Face/                # face crop
│   ├── Pose/                # MediaPipe pose + smoothing
│   ├── Identity/            # tell players apart
│   └── bridge/              # UDP sender
├── Models/                  # model files (e.g. MediaPipe .task files)
└── IPCV Climbing Game/      # Unity project
    └── Assets/
        ├── Scenes/
        ├── Scripts/
        │   ├── Face/        # face on the model
        │   ├── Pose/
        │   ├── Identity/
        │   ├── Interaction/ # player physics, climbing holds
        │   └── Bridge/      # UDP receiver
        ├── Prefabs/
        ├── Materials/
        ├── Models/        
        └── Audio/
```
