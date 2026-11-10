# BRI Inference Pipeline

Model-agnostic real-time continuous sEMG finger kinematics inference pipeline and MuJoCo Shadow Hand simulator.

The `main` branch provides the model-agnostic streaming, preprocessing, recording, visualization, and Shadow Hand simulation infrastructure. Specific model architectures live on dedicated branches (e.g., `intrasubject-gru-decoder`).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # or source .venv/Scripts/activate on Windows Git Bash
pip install -e ".[dev]"
```

## Adding Your Own Model

To add a new model architecture:

1. Create a new feature branch from `main`:
   ```bash
   git checkout -b my-custom-model
   ```
2. Implement `Model(BaseEMGModel)` in `src/model/model.py`:
   ```python
   import torch
   from src.model.base import BaseEMGModel


   class Model(BaseEMGModel):
       def __init__(self, channels: int = 8, targets: int = 15, window_size: int = 240, **kwargs):
           super().__init__(channels=channels, targets=targets, window_size=window_size)
           # Define your layers here

       def forward(self, x: torch.Tensor) -> torch.Tensor:
           # Map filtered sEMG window (batch, channels, window_size) -> (batch, 15) joint angles in [0, 180] deg
           ...
   ```
   *(For stateful, recurrent, or non-PyTorch models, you can also override `predict_window(self, window: np.ndarray) -> np.ndarray` and `reset(self) -> None`.)*
3. Place your trained checkpoint at `models/best_model.pt` (or pass `--weights path/to/weights.pt`).

## Structure

```
run.sh                       launch MuJoCo Shadow Hand viewer + real-time inference pipeline
view_emg.sh                  launch live 8-channel EMG & 15-DOF joint angle monitor
scan_myo.sh                  scan for nearby Myo armbands over BLE
record_and_train.sh          launch the Myo + MediaPipe Recording Studio & Dataset Manager
shadow_hand_specs.json       physical joint and actuator limits for the Shadow Hand
scripts/
  run_pipeline.py            CLI entry point for real-time streaming inference
  view_emg.py                real-time oscilloscope and angle monitor GUI
  scan_myo.py                BLE scanner utility
  record_intrasubject_myo.py interactive dataset recording studio & session manager
src/
  params.yaml                preprocessing, windowing, kinematics, and UDP parameters
  preprocess.py              Preprocess: causal notch + bandpass filter
  model/
    base.py                  BaseEMGModel: model-agnostic interface contract
    model.py                 Model: template sEMG-to-kinematics decoder (replace on feature branches)
  kinematics/
    mapping.py               closed-form mapping from 15 finger joint angles to 20 Shadow Hand actuators
    udp_sender.py            UDP client transmitting 20 float32 controls (`20f`) on port 5005
  sim/
    config.py                Shadow Hand model configuration
    sim.py                   MuJoCo + mjviser web visualizer (`python -m src.sim`)
    save_joint_ranges.py     utility to export actuator ranges
  stream/
    ring_buffer.py           thread-safe circular buffer for multi-channel EMG
    ble_myo.py               raw 200 Hz BLE client for the Thalmic Myo armband
    mock_myo.py              200 Hz replay / synthetic streamer for offline testing
    worker.py                StreamingInferenceWorker: 10 Hz model-agnostic inference loop
  visualization/
    emg_window.py            Tkinter + Matplotlib live diagnostic GUI
tests/                       unit and integration test suite
```

## Usage

### 1. Run End-to-End Inference + Shadow Hand Simulator
```bash
./run.sh --mode auto
```
Starts the MuJoCo Shadow Hand web viewer (`http://localhost:8080` via `python -m src.sim`) and streams real-time predictions over UDP (`127.0.0.1:5005`). For offline replay without hardware:
```bash
./run.sh --mode mock --duration 10 --log-latency
```

### 2. Live EMG & Kinematics Monitor
```bash
./view_emg.sh --mode auto
```

### 3. Record Synchronized Myo + Webcam Datasets
```bash
./record_and_train.sh
```

### 4. Run Tests
```bash
pytest
```
