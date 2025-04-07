# Soft Trunk Robot Dynamics and Fault Detection

This repository contains the code for the paper *"Motion Dynamics Modeling and Fault Detection of a Soft Trunk Robot"* by Emadodin Jandaghi, Xiaotian Chen, and Chengzhi Yuan. The project implements a machine learning approach using deterministic learning and radial basis function neural networks (RBFNN) to model the dynamics of a soft trunk robot and detect faults.
https://ieeexplore.ieee.org/abstract/document/10196206
## Repository Structure
- `src/`: Python scripts for neural network training, fault detection, and motor control.
  - `dynamic_estimation_fault_detection_nn.py`: Neural network-based fault detection with error analysis.
  - `deterministic_learning_rbfnn.py`: RBFNN training using deterministic learning.
  - `motor_trajectory_control_data_collection.py`: Stepper motor control and data collection.
- `data/`: Placeholder for sample `u1all_Fault3_motor4.npy` data files.
- `docs/`: Research paper in PDF format.

## Arduino Setup
The `motor_trajectory_control_data_collection.py` script requires an Arduino running Firmata to control stepper motors. Follow these steps:
1. Upload `arduino/StandardFirmata.ino` to your Arduino.
2. Connect the Arduino to your computer (default: COM7).
3. Install `pyfirmata` (`pip install pyfirmata`) on your host machine.
4. Run `motor_control_data_collection.py` to collect motion data.

## Requirements
Install dependencies with:
```bash
pip install -r requirements.txt
