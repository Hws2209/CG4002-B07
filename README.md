# README — CG4002 Gesture-Based “Simon Says” Game

## User Interaction and Game Flow
1. UI randomly selects an action and plays the corresponding audio.
2. If “Simon says” is included, the action must be performed; otherwise, stay idle.
3. Wristbands (consisting of IMU and ESP32) detect movement and AI (running on an Ultra96) classifies the action within a 2-second limit.
4. Correct actions trigger a beep; player must return to idle immediately.
5. Each correct action increases the score; final score shown at game over.
6. Objectives:
   - **One-player:** Achieve highest score.
   - **Cooperative:** Work together to maximise score.
   - **Versus:** Last player standing wins.

## List of Actions

### One-Player and Versus Mode
1. Idle  
2. Wave left hand  
3. Wave right hand  
4. Wave both hands  
5. Left back arm circle  
6. Right back arm circle  
7. Both back arm circle  
8. Left front arm circle  
9. Right front arm circle  
10. Both front arm circle  
11. Star jump  

### Cooperative Mode
1. Idle  
2. Shake left hand  
3. Shake right hand  
4. Shake both hands  
5. Left high-five  
6. Right high-five  
7. Both high-five  

---

# Instructions for Running the Project

## 1. Laptop Setup
OS: Ubuntu (Linux) 

**Run:**  
```
python3 Comms/Laptop.py
```

### Notes
- `Laptop.py` contains:
  ```python
  sys.path.append(...)
  ```
  Update this argument to point to the folder containing:
  - `Interface/cli.py`
  - `Interface/audio/`
  
  Alternatively, just run Laptop.py while in CG4002/Comms directory.

### Requirements
- `Interface/cli.py`
- `Interface/audio/`


## 2. Ultra96 Setup
**Run:**  
```
./run_model.sh
```

This script is stored under `AI/Ultra96/run_model.sh` in the GitHub repository. It executes `run_model.py` and loads the FPGA hardware design.

### Requirements
- `AI/Ultra96/run_model.py`
- `AI/Ultra96/design_1.bit`
- `AI/Ultra96/design_1.hwh`
- `AI/Ultra96/design_2.bit`
- `AI/Ultra96/design_2.hwh`


## 3. ESP32 FireBeetle Boards Setup
**Upload to FireBeetle:**  
```
Comms/FireBeetle/FireBeetle.ino
```

### Notes
Modify this line to give each board a unique ID:
```cpp
#define DEVICE_ID 1
```

Device ID
- Player 1 Left  Hand : 1
- Player 1 Right Hand : 2
- Player 2 Left  Hand : 3
- Player 2 Right Hand : 4

### Requirements
- `Comms/FireBeetle/MPU.h`
- `Comms/FireBeetle/Display.h`


## 4. Start Up Sequence
1. Run Laptop.py 
2. Enter game mode in Laptop.py
3. Run ./run_model.sh on the Ultra96
4. Start all the FireBeetles 
