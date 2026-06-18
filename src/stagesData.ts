// Pre-loaded academic stage information and Python/SQL snippets for student review
export interface FolderItem {
  name: string;
  role: string;
  details: string;
}

export interface FolderSection {
  title: string;
  badge: string;
  color: string;
  items: FolderItem[];
}

export const SYSTEM_ARCHITECTURE_ASCII = `+---------------------------------------------------------------------------------------------------------+
|                                  STAGE 6: Streamlit Operator Dashboard Panel                            |
|             Interprets user prose & tracks visual webcam feeds; connects over HTTP on Port 8000           |
+----------------------------------------------------+----------------------------------------------------+
                                                     | (REST POST JSON Payload to Port 8000)
                                                     v
+---------------------------------------------------------------------------------------------------------+
|                                    STAGE 4: Cognitive VLA Planner Module (Gemini)                       |
|   * Evaluates messy text instructions against tabletop world object coordinates fetched from SQLite    |
|   * Outputs sequence coordinates lists following strict, robust JSON constraint response schemas       |
+----------------------------------------------------+----------------------------------------------------+
                                                     | (Trajectory Formulation Arrays)
                                                     v
+----------------------------------------------------+----------------------------------------------------+
|                                    STAGE 3 & 8: Perception (OpenCV & YOLO Detectors)                    |
|   * Monitors Workspace webcam feeds, runs color-space Contours arrays calibration metrics & YOLOv8      |
|   * Transforms 2D plane pixel landmarks (u, v) into physical millimetric workspace coordinates         |
+----------------------------------------------------+----------------------------------------------------+
                                                     | (Writes coordinate telemetry updates)
                                                     v
+----------------------------------------------------+----------------------------------------------------+
|                                    STAGE 7: Relational SQLite Persistent Memory                         |
|   * Lightweight DB storing workspace coordinates, active operator histories, and controller state logs|
+----------------------------------------------------+----------------------------------------------------+
                                                     | (Feeds coordinate targets to local Inverse Kinematics)
                                                     v
+----------------------------------------------------+----------------------------------------------------+
|                                    STAGE 5 & 9: Robot Kinematics Simulator & ROS 2 Drivers              |
|   * Solving Forward & Inverse Kinematics equations to map targets onto joint velocities (Th1,Th2,Th3)  |
|   * Publish trajectories to motor control micro-controllers over ROS 2 Topic publishers                |
+---------------------------------------------------------------------------------------------------------+`;

export const FOLDER_STRUCTURES: FolderSection[] = [
  {
    title: "backend/",
    badge: "FastAPI Core microservice API",
    color: "text-blue-400 border-blue-500/10 bg-blue-500/5",
    items: [
      {
        name: "main.py",
        role: "ASGI application server entry point",
        details: "Configures routers, lifespan handlers, CORS middleware, global error handling, and hooks for continuous database connection loops."
      },
      {
        name: "config.py",
        role: "Environmental settings schema loader",
        details: "Manages runtime settings via Pydantic BaseSettings, loading API secrets securely and keeping configuration states isolated."
      },
      {
        name: "dependencies.py",
        role: "API dependency injection layer",
        details: "Defines reusable core resource dependencies (such as active SQLite database session providers, validation schemas, and system status checkers)."
      }
    ]
  },
  {
    title: "vision/",
    badge: "Computer Vision & YOLO Tracking",
    color: "text-emerald-400 border-emerald-500/10 bg-emerald-500/5",
    items: [
      {
        name: "detector.py",
        role: "OpenCV pipeline for color-space filtering",
        details: "Applies HSV masking, contours detection, morphological denoising, and bounding-box spatial coordinates approximation."
      },
      {
        name: "calibration.py",
        role: "Camera mapping camera matrix transform",
        details: "Applies physical camera factors to transform pixel coordinates (u, v) into physical 3D ground world vectors (x, y, z) in millimeters."
      },
      {
        name: "yolo_onnx.py",
        role: "ONNX YOLO tracker neural network inference",
        details: "Runs YOLO deep learning model parameters. Predicts tabletop shapes under unstructured configurations."
      }
    ]
  },
  {
    title: "robotics_and_planner/",
    badge: "Analytical Kinematics & AI Brain",
    color: "text-indigo-400 border-indigo-500/10 bg-indigo-500/5",
    items: [
      {
        name: "kinematics.py",
        role: "3-DoF Analytical Inverse Kinematics Solver",
        details: "Solves analytical inverse mathematics, mapping millimeter coordinates directly into 3-DoF Joint angle parameters (theta1, theta2, theta3 in degrees)."
      },
      {
        name: "vla_client.py",
        role: "VLA cognitive planning gateway client",
        details: "VLA Model core coordinator using Google GenAI SDK to parse fuzzy prose into structured coordinate matrices."
      }
    ]
  },
  {
    title: "sqlite_memory/",
    badge: "SQLite Database State Manager",
    color: "text-amber-400 border-amber-500/10 bg-amber-500/5",
    items: [
      {
        name: "history_db.py",
        role: "State persistence connection connector",
        details: "Builds schema layouts with performance indexes tracking active coordinate joint positions and historical command registries."
      }
    ]
  }
];

export const OPENCV_CODE_FRAGMENT = `def process_frame(frame):
    # Convert RGB to Hue-Saturation-Value to classify targets
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Establish high-contrast masking criteria
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv_img, lower_red, upper_red)
    
    # Denoise using morphological transformations
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Retrieve centroid values using contour indexes
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected = []
    for c in contours:
        if cv2.contourArea(c) > 300:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                detected.append({"centroid_pixels": (cx, cy)})
    return detected`;

export const PLANNER_INPUT_PAYLOAD = `{
  "command": "Move orange pyramid to safety inside blue box",
  "workspace_state": [
    { "name": "orange pyramid", "x": 40.0, "y": 190.0, "z": 15.0 },
    { "name": "blue box", "x": -100.0, "y": 180.0, "z": 20.0 }
  ]
}`;

export const PLANNER_RESPONSE_PLAN = `{
  "understoodGoal": "Relocate orange pyramid to blue box container",
  "steps": [
    { "action": "move_to", "x": 40.0, "y": 190.0, "z": 100.0 },
    { "action": "grasp", "x": 40.0, "y": 190.0, "z": 15.0 },
    { "action": "transport", "x": -100.0, "y": 180.0, "z": 100.0 },
    { "action": "release", "x": -100.0, "y": 180.0, "z": 30.0 }
  ]
}`;

export const STREAMLIT_APP_CODE = `import streamlit as st
import requests

st.set_page_config(layout="wide")
st.title("VLA Robotic Arm Operator Console")

input_prose = st.text_input("Instruct pick-and-place targets:")
if st.button("Generate Trajectory"):
    response = requests.post(
        "http://localhost:3000/api/plan", 
        json={"command": input_prose}
    )
    st.success("Target path synthesized successfully!")
    st.json(response.json()["planSteps"])`;

export const SQL_SCHEMA_STATEMENTS = `-- Table 1: Real-time monitored physical workspace coordinates
CREATE TABLE workspace_objects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_name TEXT UNIQUE NOT NULL,    -- e.g. "red cube", "blue box"
    shape TEXT NOT NULL,               -- "cube", "sphere", "pyramid", "box"
    color_hex TEXT NOT NULL,           -- Hex color representation
    loc_x REAL NOT NULL,               -- X Coordinate (mm relative to Base)
    loc_y REAL NOT NULL,               -- Y Coordinate (mm)
    loc_z REAL NOT NULL,               -- Height Z (mm)
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table 2: Historic task execution statistics logs
CREATE TABLE task_history (
    task_id TEXT PRIMARY KEY,          -- Unique task identifier
    raw_command TEXT NOT NULL,         -- Human operator original string
    interpreted_goal TEXT NOT NULL,    -- Synthesized command translation
    target_object TEXT NOT NULL,       -- Primary target
    destination TEXT NOT NULL,         -- Placed receptor
    execution_status TEXT CHECK(execution_status IN ('SUCCESS', 'FAILED')),
    execution_time_sec REAL,           -- Execution completion duration
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(target_object) REFERENCES workspace_objects(item_name)
);`;

export const YOLO_CODE_SNIPPET = `from ultralytics import YOLO
import cv2

# Initialize pretrained YOLOv8 model for tabletop block items
model = YOLO("yolov8n-tabletop.pt")

def predict_centroids(img):
    results = model(img)
    objects = []
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Extract bounding polygon boundaries (pixels)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0].item())
            class_name = model.names[cls_id]
            
            # Solve mathematical center of bounding block
            cx_pix = int((x1 + x2) / 2)
            cy_pix = int((y1 + y2) / 2)
            
            objects.append({
                "label": class_name,
                "cx_pix": cx_pix,
                "cy_pix": cy_pix,
                "confidence": float(box.conf[0].item())
            })
            
    return objects`;

export const ROS2_NODE_CODE = `import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import std_msgs.msg

class ArmTrajectoryPublisher(Node):
    def __init__(self):
        super().__init__('arm_trajectory_publisher_node')
        # Setup publisher to stream coordinates onto real motor ROS nodes
        self.publisher_ = self.create_publisher(
            JointTrajectory, 
            '/vla_arm_controller/joint_trajectory', 
            10
        )
        self.get_logger().info('Arm trajectory publisher is initialized and ready.')

    def publish_joint_point(self, base_rad, shoulder_rad, elbow_rad):
        msg = JointTrajectory()
        msg.header = std_msgs.msg.Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = ['joint_base', 'joint_shoulder', 'joint_elbow']
        
        point = JointTrajectoryPoint()
        point.positions = [base_rad, shoulder_rad, elbow_rad]
        point.time_from_start.sec = 2 # Transitions execution time limits
        
        msg.points.append(point)
        self.publisher_.publish(msg)
        self.get_logger().info(f'Published trajectory target: [{base_rad:.3f}, {shoulder_rad:.3f}]')`;
