# 🛰️ 3-DOF Autonomous Robotic Arm with YOLOv8 & Gemini Cognitive Planning

An advanced, edge-calibrated 3-DOF (Degree of Freedom) Tabletop Robotic Arm Simulator, visual tracker, and real-time path planner. This system fuses high-fidelity **YOLOv8 Computer Vision object tracking** with **Gemini Cognitive Decision Models** to execute complex pick-and-place trajectories, dynamic spatial orientation, and active workspace hazard prevention.

---

## 🏗️ System Architecture

The application implements a decoupled, modern full-stack web and simulation ecosystem:

```
                    ┌─────────────────────────┐
                    │    React + Vite UI      │
                    │  (Control Dashboard)    │
                    └────────────┬────────────┘
                                 │ HTTP / JSON
                                 ▼
                    ┌─────────────────────────┐
                    │   Vite + Express API    │ <─── SQLite Persistent DB
                    │      (server.ts)       │      (State & History Logs)
                    └────────────┬────────────┘
                                 │ Internals
                                 ▼
                    ┌─────────────────────────┐
                    │  Python Vision / VLA    │
                    │     (yolo_detector)     │
                    └────────────┬────────────┘
                                 │
         ┌───────────────────────┴───────────────────────┐
         ▼                                               ▼
┌─────────────────┐                             ┌─────────────────┐
│     YOLOv8      │                             │   Gemini AI     │
│ Neural Pipeline │                             │ Cognitive Core  │
│ (Objects/Class) │                             │ (Trajectory VLA)│
└─────────────────┘                             └─────────────────┘
```

1. **Frontend Interface (React + Vite)**: A responsive workspace control application featuring interactive 3-DOF kinematics visualizers, execution timelines, telemetry indicators, safety hazard alerts, and tabular telemetry logs.
2. **Backend Server (Express)**: Manages local data caching, persistent execution telemetry in SQLite, and bridges communications between client requests, vision pipelines, and planning cores.
3. **Computer Vision Unit (OpenCV + YOLOv8)**: Translates live camera streams and pixel coordinates `(u, v)` into metric physical world coordinates `(X, Y, Z)` in millimeters, tracking multiple targets and identifying human/worker safety violations.
4. **Cognitive Logic Core (Gemini SDK)**: Uses advanced structured LLM prompts to analyze command-line request strings, evaluate detected desktop targets, and generate step-by-step physical joints trajectories (Yaw, Shoulder, Elbow).

---

## ✨ Features

### 🌌 1. Deep Learning Vision Pipeline (YOLOv8)
* **Pre-trained Object Classification**: Real-time object recognition (detects standard workspace cubes, spheres, and containers alongside COCO classes such as `person`, `bottle`, and `cup`).
* **Geometric Spatial Mapping**: Integrates automatic pixel-to-millimeter focal matrices to convert frame coordinate centroids instantly.
* **Persistent Path Tracking**: Tracks unique spatial identities across frame sweeps to preserve historical trajectory timelines.

### 🛡️ 2. Active Workspace Safety Lock (Hazard Prevention)
* **Human Intrusion Detection**: Detects human arms, bodies, or hands entering the workspace boundaries via YOLOv8 Class `0` (Person).
* **Automatic Hardware Cut-off**: Triggers a system-wide safety shutdown state immediately upon intrusion, displaying styled visual hazard warnings and disabling joints actuation until the workspace is cleared.

### 🧠 3. Self-Healing Cognitive Fallback
* **Transient Overload Protection**: Automatically detects Gemini API rate limits (`429`), timeouts, or temporary cloud server limits (`503 Service Unavailable / High Demand`).
* **Silent Edge Cooldowns**: Seamlessly locks down of-cloud requests for 5 minutes and routes operations into high-accuracy local mechanical simulators (`fallback_simulation`), preventing workspace lockups.

### 📐 4. Dual Kinematics Playgrounds
* **Analytical Inverse Kinematics**: Solves trigonometry calculations to translate physical millimeter targets `(x, y, z)` into real-time joint positions ($\theta_1$ Base Yaw, $\theta_2$ Shoulder, $\theta_3$ Elbow) instantly.
* **Forward Joint Simulator**: Animates direct slider movements so operators can test maximum reach ranges, collisions, and workspace coordinate envelopes safely.

---

## 🛠️ Configuration & Secrets

All configuration details are read securely from standard environment configurations. Refer to `.env.example` at the repository root to check syntax.

| Variable Name | Required | Description |
| :--- | :---: | :--- |
| `GEMINI_API_KEY` | Yes (Optional Fallback) | Developer Google Generative AI credentials. If missing, the system utilizes local offline geometric paths automatically. |
| `APP_URL` | No | Auto-configured at runtime to route internal callback telemetry. |

---

## 🚀 Speed-Run Developer Guide

### 🧱 1. Backend & Web Client (Vite + React + Node)
Install relevant node packages, configure environment hooks, and spin up development environments in the sandbox:

```bash
# Install core dependencies
npm install

# Initialize development runtime proxying front-end assets through port 3000
npm run dev

# Compile release distributions and bundle lightweight Node.js Server CJS
npm run build

# Boot compiled distributions inside container environment
npm run start
```

### 🐍 2. Python Vision Engine (OpenCV & YOLO Requirements)
To initialize the deep learning segmenter locally outside the container, populate your workspace python modules:

```bash
# Pull dependencies
pip install -r requirements.txt

# Manually test YOLO tracking pipeline
python vision/yolo_detector.py
```

---

## 🔌 API Endpoints Reference

### 1. Planning Request Proxy
* **Route**: `POST /api/plan`
* **Payload**:
```json
{
  "command": "Pick up the blue container cup and stack it over the red block"
}
```
* **Response**:
```json
{
  "command": "Pick up the blue container cup...",
  "targetReceptor": "red block",
  "detectedObjects": [
    { "name": "cup", "tracking_id": 102, "cx": 360, "cy": 108, "x": 50.0, "y": 165.0, "z": 20.0 }
  ],
  "planSteps": [
    { "step": 1, "action": "MOVE", "jointAngles": { "theta_1": 73.1, "theta_2": 85.0, "theta_3": 92.5 }, "description": "Position hand over cup." }
  ],
  "mode": "cognitive_ai"
}
```

### 2. Execution Logs Telemetry
* **Routes**:
  * `GET /api/logs` - Fetch all saved robotic operations, path calculations, and safety alert histories.
  * `POST /api/logs` - Append active control metrics to the persistent SQLite database.

---

## 📜 License

This project is licensed under the terms of the [MIT License](LICENSE).
