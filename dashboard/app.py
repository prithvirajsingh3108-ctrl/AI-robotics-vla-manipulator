import streamlit as st
import requests
import json
import numpy as np
import cv2
import pandas as pd
import time
import os

# Set relative system page configuration
st.set_page_config(
    page_title="Operator AI Brain Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Render Custom Styling Blocks
st.markdown("""
<style>
    .reportview-container {
        background: #0b0f19;
    }
    .main .block-container {
        padding-top: 2rem;
    }
    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# Helper references to local modules
from vision.yolo_detector import YoloTabletopTracker
from robotics.kinematics import RoboticArmKinematics
from memory.db_session import get_db_connection

# Initialize local microservice tools
@st.cache_resource
def load_robotics_core():
    return YoloTabletopTracker(), RoboticArmKinematics()

tracker, kinematics = load_robotics_core()

# Sidebar: Core Connection State Diagnostics
st.sidebar.title("🤖 System Core Nodes")
st.sidebar.markdown("---")

api_url = st.sidebar.text_input("FastAPI Endpoint URL:", "http://localhost:8000")

# Check backend health
backend_status = "🔴 Off-line"
try:
    res = requests.get(f"{api_url}/api/v1/health", timeout=1.0)
    if res.status_code == 200:
        backend_status = "🟢 Operational (REST Core)"
except Exception:
    backend_status = "🟡 Running Offline Mode (Direct Client)"

st.sidebar.info(f"Connection Status: {backend_status}")

# Sidebar Project Metadata Details for reviewers
st.sidebar.markdown("""
### 📋 Student Evaluation Spec
*   **B.Tech Capstone Project**
*   **Topic:** Cognitive VLA Robotic Arm
*   **University:** Department of CSE
*   **Advisor:** Senior Robotics Faculty Lead
""")

# Primary Layout Header Banner
st.title("🤖 AI Robotics Brain Operator Console")
st.write("An interactive B.Tech portfolio dashboard controlling visual perception and joint kinematics pipelines.")

tab_dashboard, tab_vision, tab_k_sim, tab_db_history = st.tabs([
    "🎯 Command & Simulator Main",
    "👁️ OpenCV Object Tracking Feed",
    "📐 Inverse Kinematics Playground",
    "🗄️ SQLite Database Logs Workspace"
])

# ==================== TAB 1: COGNITIVE COMMAND OPERATOR & SIMULATOR ====================
with tab_dashboard:
    st.header("VLA Logic Trajectory Generator")
    col_input, col_sim = st.columns([1, 1])
    
    with col_input:
        st.subheader("Natural Language Operator Entry")
        st.write("Submit complex instructions. The cognitive parser extracts item matrices and designs joint targets.")
        
        command = st.text_input(
            "Operator Text Command:",
            "Pick up the red cube and place it inside the yellow container."
        )
        
        col_btn1, col_btn2 = st.columns([1, 1])
        submit_btn = col_btn1.button("Submit Command (Plan Trajectory)", type="primary")
        seed_btn = col_btn2.button("Re-Seeding Workspace Objects")
        
        if seed_btn:
            try:
                from memory.db_session import init_db
                init_db()
                st.success("SQLite table coordinates seeded successfully!")
            except Exception as e:
                st.error(f"Seeding failure: {str(e)}")

        # Fetch current objects coordinates lists
        world_objects = []
        try:
            conn = get_db_connection()
            df_objs = pd.read_sql_query("SELECT item_name as Name, shape as Shape, loc_x as X, loc_y as Y, loc_z as Z FROM workspace_objects", conn)
            conn.close()
            st.markdown("##### Current Monitored physical items coordinates:")
            st.dataframe(df_objs, hide_index=True)
            
            # Map into list parameters dictionary format for planning backend API payloads
            for _, row in df_objs.iterrows():
                world_objects.append({"name": row["Name"], "shape": row["Shape"], "x": float(row["X"]), "y": float(row["Y"]), "z": float(row["Z"])})
        except Exception:
            # Fallback mock logs
            fallback_objs = [
                {"name": "red cube", "shape": "cube", "x": 80.0, "y": 150.0, "z": 10.0},
                {"name": "blue box", "shape": "box", "x": -100.0, "y": 180.0, "z": 20.0},
                {"name": "green sphere", "shape": "sphere", "x": 120.0, "y": 120.0, "z": 10.0}
            ]
            st.warning("Could not read from SQLite DB. Using fallback local cache:")
            st.write(fallback_objs)
            world_objects = fallback_objs
            
        if submit_btn:
            st.subheader("Trajectory Sequencing Output")
            
            # Formulate coordinates pipeline payload
            payload = {"command": command}
            plan_response = None
            
            with st.spinner("Formulating kinematics step trajectories..."):
                if backend_status.startswith("🟢"):
                    # Direct REST Request API core
                    try:
                        res = requests.post(f"{api_url}/api/v1/brain/plan", json=payload)
                        if res.status_code == 200:
                            plan_response = res.json()
                    except Exception as e:
                        st.error(f"REST Core API call failed: {str(e)}. Proceeding offline.")
                        
                if plan_response is None:
                    # Proceed with Local offline cognitive controller simulation
                    from planner.vla_client import VisionLanguageActionPlanner
                    local_planner = VisionLanguageActionPlanner()
                    plan_response = local_planner.formulate_vla_sequence(command, world_objects)
                    
                    # Attach physical kinematics angle degrees configurations offline
                    for step in plan_response["planSteps"]:
                        try:
                            angles = kinematics.solve_inverse_kinematics(step["targetX"], step["targetY"], step["targetZ"])
                            step["angles"] = angles
                            step["vacuum_state"] = 1 if step["action"] in ["grasp", "lift", "transport"] else 0
                        except Exception:
                            step["angles"] = {"theta1_base_yaw": 0.0, "theta2_shoulder_pitch": 0.0, "theta3_elbow_flexion": 0.0}
                            step["vacuum_state"] = 0
            
            st.markdown(f"**VLA Interpreted Task Goal:** *\"{plan_response['understoodCommand']}\"*")
            st.markdown(f"**Identified Target Object:** `{plan_response['targetObject']}` ➔ **Destination Receptor:** `{plan_response['targetReceptor']}`")
            
            # Store plan in session state to animate on simulator columns
            st.session_state["active_plan_seq"] = plan_response["planSteps"]
            st.session_state["active_step_idx"] = 0
            
            # Draw pretty expandable steps lists
            for step in plan_response["planSteps"]:
                with st.expander(f"Step {step['stepNum']}: {step['action'].upper()} - {step['description']}"):
                    st.json(step)

    with col_sim:
        st.subheader("Interactive Tabletop Manipulator Simulator")
        st.write("Renders trajectories of link joint positions. Trigger a plan on left column to see animation.")
        
        # Check plan session states parameters
        if "active_plan_seq" in st.session_state:
            active_plan = st.session_state["active_plan_seq"]
            
            # Render step navigation controller buttons
            col_nav1, col_nav2, col_nav3 = st.columns([1,1,2])
            
            if col_nav1.button("◀ Prev Step"):
                st.session_state["active_step_idx"] = max(0, st.session_state["active_step_idx"] - 1)
            if col_nav2.button("Next Step ▶"):
                st.session_state["active_step_idx"] = min(len(active_plan) - 1, st.session_state["active_step_idx"] + 1)
                
            curr_idx = st.session_state["active_step_idx"]
            curr_step = active_plan[curr_idx]
            
            st.info(f"**Animating Step {curr_idx + 1} of {len(active_plan)}:** {curr_step['action'].upper()} - targetting coordinates *({curr_step['targetX']:.1f}, {curr_step['targetY']:.1f}, {curr_step['targetZ']:.1f})*")
            
            # Display real-time solved target angle degrees calculations
            angles = curr_step["angles"]
            if "theta1_base_yaw" in angles:
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                col_m1.metric("Θ1 Base Yaw", f"{angles['theta1_base_yaw']}°")
                col_m2.metric("Θ2 Shoulder", f"{angles['theta2_shoulder_pitch']}°")
                col_m3.metric("Θ3 Elbow", f"{angles.get('theta3_elbow_flexion', 0.0)}°")
                col_m4.metric("Gripper vacuum", "⚡ ENGAGED" if curr_step.get("vacuum_state", 0) == 1 else "⚪ OFF")
            
            # Setup dynamic synthetic plotting coordinates trajectory charts
            # Generating path coordinates projection points
            path_x, path_y, path_z = [], [], []
            for s in active_plan:
                path_x.append(s["targetX"])
                path_y.append(s["targetY"])
                path_z.append(s["targetZ"])
                
            # Render a 3D scattering plot chart using simple line pandas parameters or native streamlit indicators
            df_path = pd.DataFrame({
                "Joint Stage Steps": [f"Step {s['stepNum']}" for s in active_plan],
                "Target X coordinate": path_x,
                "Target Y coordinate": path_y,
                "Target Z coordinate": path_z
            })
            st.line_chart(df_path.set_index("Joint Stage Steps"))
        else:
            st.warning("No active plan generated. Please input command on left panel and execute to trace trajectory path.")
            
            # Show standard idle home pose metrics
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Θ1 Base Yaw (theta_1)", "0.00° (Idle)")
            col_m2.metric("Θ2 Shoulder (theta_2)", "120.00° (Idle)")
            col_m3.metric("Θ3 Elbow (theta_3)", "100.00° (Idle)")

# ==================== TAB 2: OPENCV COMPUTER VISION TRACKER ====================
with tab_vision:
    st.header("YOLOv8 Deep Learning & Desktop Contour Tracker")
    st.write("Synthesizes neural network detections with real-time geometric pixel conversions.")
    
    # YOLO Engine Status Indicator Panel
    has_yolo_engine = tracker.yolo_detector.has_yolo
    error_message = tracker.yolo_detector.error_msg
    
    col_status_left, col_status_right = st.columns([3, 1])
    with col_status_left:
        if has_yolo_engine:
            st.success("🛰️ **YOLOv8 DEEP LEARNING ACTIVE**: Real-time neural network tensor flows fully running on backend.")
        else:
            st.warning("⚠️ **YOLOv8 FALLBACK ACTIVE**: Utilizing cv2 high-accuracy color HSV segmenters. YOLO loading failed or disabled.")
            if error_message:
                st.error(f"📁 **Detailed Deep Learning Stacktrace error:** `{error_message}`")
    with col_status_right:
        st.info("🎯 **Classes**: person, bottle, cup")

    col_v1, col_v2 = st.columns([2, 1])
    
    with col_v1:
        # Loop frame updates with dynamic noise simulation
        st.markdown("##### Real-time camera stream viz:")
        
        # Pull processed template image from tracker
        frame, detected_data = tracker.process_mock_frame()
        
        # Convert BGR format to RGB format for streamlit image rendering bounds
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, use_container_width=True)
        
        st.caption("Coordinate crosshairs indicate (320, 240) camera focal center. Custom bounding boxes represent YOLO class label envelopes.")

    with col_v2:
        # Inject interactive controls to configure YOLO class objects dynamically
        import json
        import os
        path = ".safety_control.json"
        
        # Initialize defaults in session_state if not present
        if "safety_intruder" not in st.session_state:
            st.session_state.safety_intruder = False
            st.session_state.safety_bottle = True
            st.session_state.safety_cup = True
            
            if os.path.exists(path):
                try:
                    with open(path, "r") as f:
                        data = json.load(f)
                        st.session_state.safety_intruder = data.get("intruder_enabled", False)
                        st.session_state.safety_bottle = data.get("bottle_enabled", True)
                        st.session_state.safety_cup = data.get("cup_enabled", True)
                except Exception:
                    pass

        st.markdown("##### YOLOv8 Sim Target Controller:")
        s_intruder = st.checkbox("🚨 Simulate Human Intruder (Person)", value=st.session_state.safety_intruder)
        s_bottle = st.checkbox("🍼 Simulate Bottle COCO Target", value=st.session_state.safety_bottle)
        s_cup = st.checkbox("☕ Simulate Coffee Cup COCO Target", value=st.session_state.safety_cup)
        
        # Sync to file if changed
        if (s_intruder != st.session_state.safety_intruder or 
            s_bottle != st.session_state.safety_bottle or 
            s_cup != st.session_state.safety_cup):
            
            st.session_state.safety_intruder = s_intruder
            st.session_state.safety_bottle = s_bottle
            st.session_state.safety_cup = s_cup
            
            try:
                with open(path, "w") as f:
                    json.dump({
                        "intruder_enabled": s_intruder,
                        "bottle_enabled": s_bottle,
                        "cup_enabled": s_cup
                    }, f)
                st.rerun()
            except Exception as e:
                st.error(f"Failed to write safety specs: {e}")

        st.markdown("##### Real-time camera calibration metadata:")
        st.caption("Converts coordinate vectors to physical ground workspace coordinates.")
        
        # Display centroid metrics
        for d in detected_data:
            name_lbl = d['name'].replace("_", " ").upper()
            t_id = d.get('tracking_id', 'N/A')
            conf = d.get('confidence', 1.0)
            
            is_yolo_coco = d['name'] in ["person", "bottle", "cup"]
            badge_class = "🟢 Standard Target" if not is_yolo_coco else "🔵 YOLO COCO Class"
            if d['name'] == "person":
                badge_class = "🔴 HAZARD INTRUSION"

            with st.container(border=True):
                st.markdown(f"**{name_lbl}** `[Track-ID: {t_id}]` (_{badge_class}_)")
                st.progress(float(conf), text=f"Inference Confidence: {conf * 100:.1f}%")
                st.write(f"Centroid Coords on Frame: `(u: {d['cx']}, v: {d['cy']})` px")
                st.write(f"Derived Millimeter vector: **X: {d['x']:.1f}mm, Y: {d['y']:.1f}mm, Z: {d['z']:.1f}mm**")
                
                # Check kinematics reaching parameters
                try:
                    ang = kinematics.solve_inverse_kinematics(d["x"], d["y"], d["z"])
                    st.success(f"Target is within reach: Θ1 Yaw: {ang['theta1_base_yaw']:.1f}°")
                except Exception as e:
                    st.error(f"Target state is unreachable: {str(e)}")

        st.button("🔄 Capture & Process Camera Feed Step")

# ==================== TAB 3: INVERSE KINEMATICS PLAYGROUND ====================
with tab_k_sim:
    st.header("3-DoF Geometric Kinematics Solver")
    st.write("Drag controllers to modify end-effector coordinates. Calculates joint servos values in degrees instantly.")

    col_s1, col_s2 = st.columns([1,1])
    
    with col_s1:
        st.markdown("##### Cartesian Input sliders:")
        
        target_x = st.slider("Destination X (Lateral Range -150 to +150 mm)", -150.0, 150.0, 80.0, step=1.0)
        target_y = st.slider("Destination Y (Forward reach 100 to 220 mm)", 100.0, 220.0, 150.0, step=1.0)
        target_z = st.slider("Destination Z (Vertical clearance 0 to 180 mm)", 0.0, 180.0, 10.0, step=1.0)
        
        st.markdown("##### Robotic Manipulator Links configurations:")
        st.write(f"Link 1 (Base Height): `{kinematics.L1}` mm")
        st.write(f"Link 2 (Upper arm span): `{kinematics.L2}` mm")
        st.write(f"Link 3 (Forearm span): `{kinematics.L3}` mm")

    with col_s2:
        st.markdown("##### Calculated Rotational Joints Parameters:")
        
        try:
            angles = kinematics.solve_inverse_kinematics(target_x, target_y, target_z)
            
            st.success("Mathematical constraints solved successfully!")
            
            col_r1, col_r2, col_r3 = st.columns(3)
            col_r1.metric("Theta 1 Base Angle (Yaw)", f"{angles['theta1_base_yaw']}°")
            col_r2.metric("Theta 2 Shoulder Angle (Pitch)", f"{angles['theta2_shoulder_pitch']}°")
            col_r3.metric("Theta 3 Elbow Angle (Flexion)", f"{angles['theta3_elbow_flexion']}°")
            
            # Run forward kinematics validation to verify link geometries align perfectly
            fk_coords = kinematics.solve_forward_kinematics(
                angles["theta1_base_yaw"],
                angles["theta2_shoulder_pitch"],
                angles["theta3_elbow_flexion"]
            )
            
            st.info(f"**Forward Kinematics Validation Checks:** Solved End-Effector tip is positioned at X: {fk_coords['x']}mm, Y: {fk_coords['y']}mm, Z: {fk_coords['z']}mm")
            
        except ValueError as e:
            st.error(f"Kinematics error: {str(e)}")
            st.info("💡 Position value exceeds robot spatial reachable workspace bounds. Please decrease forward Y or side X distance ranges.")

# ==================== TAB 4: SQLITE DATABASE HISTORIES ====================
with tab_db_history:
    st.header("Offline Data Storage Workspace (SQLite)")
    st.write("Displays execution memory tables. Useful for review evaluations and offline optimization training models.")
    
    try:
        conn = get_db_connection()
        
        st.subheader("1. Active Database Workspace Objects coordinates map:")
        df_obj_table = pd.read_sql_query("SELECT * FROM workspace_objects", conn)
        st.dataframe(df_obj_table, use_container_width=True, hide_index=True)
        
        st.subheader("2. Historical Task execution registry (VLA Planner logs):")
        df_task_table = pd.read_sql_query("SELECT * FROM task_history ORDER BY recorded_at DESC", conn)
        st.dataframe(df_task_table, use_container_width=True, hide_index=True)
        
        st.subheader("3. Trajectory Motor angles steps database segment:")
        df_traj_table = pd.read_sql_query("SELECT * FROM trajectory_telemetry ORDER BY id DESC LIMIT 20", conn)
        st.dataframe(df_traj_table, use_container_width=True, hide_index=True)
        
        conn.close()
    except Exception as e:
        st.error(f"Database extraction failed: {str(e)}")
        st.info("Ensure the FastAPI backend server has been initialized or run the Database Seed process inside Tab 1.")
