try:
    import srv_main_module
except Exception:
    pass

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
import sqlite3

# Import microservice components
from memory.db_session import init_db, get_db_connection
from robotics.kinematics import RoboticArmKinematics
from planner.vla_client import VisionLanguageActionPlanner
from vision.yolo_detector import YoloTabletopTracker
from planner.reasoning_engine import AutonomousReasoningEngine

app = FastAPI(
    title="AI Robotics Brain VLA API",
    description="REST backend orchestrating OpenCV detection vectors, SQLite memory, and Gemini planning.",
    version="1.0"
)

# Configure CORS so Streamlit interface can communicate seamlessly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup Lifespan Initialization
@app.on_event("startup")
def startup_event():
    init_db()

# Initiate Kinematics components
kinematics = RoboticArmKinematics()
planner = VisionLanguageActionPlanner()
tracker = YoloTabletopTracker()
reasoning_engine = AutonomousReasoningEngine()

class CommandInput(BaseModel):
    command: str

@app.get("/api/v1/health")
def health_check():
    """Confirms operational diagnostics status of microservices."""
    return {
        "status": "healthy",
        "database": "sqlite3_active",
        "kinematics": "3dof_online",
        "reasoning_engine": "autonomous_loop_online",
        "planner_connected": planner.api_key is not None
    }

@app.get("/api/v1/vision/objects")
def get_world_model_coordinates():
    """Gives active tabletop items locations fetched from SQLite context."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT item_name, shape, color_hex, loc_x, loc_y, loc_z, updated_at FROM workspace_objects")
    rows = cursor.fetchall()
    conn.close()
    
    return {
        "detected_objects": [
            {
                "name": r[0],
                "shape": r[1],
                "color_hex": r[2],
                "x": r[3],
                "y": r[4],
                "z": r[5],
                "updated_at": r[6]
            }
            for r in rows
        ]
    }

@app.post("/api/v1/brain/plan")
def submit_trajectory_plan(payload: CommandInput):
    """
    Accepts natural language command, aggregates database item models,
    and returns simulated or LLM-formulated 3-DOF trajectory sequences.
    """
    # 1. Fetch current world model coordinates from memory database
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT item_name, shape, loc_x, loc_y, loc_z FROM workspace_objects")
    rows = cursor.fetchall()
    conn.close()
    
    world_objects = [
        {"name": r[0], "shape": r[1], "x": r[2], "y": r[3], "z": r[4]}
        for r in rows
    ]

    # 2. Formulate 7-step trajectory plan coordinates index
    plan = planner.formulate_vla_sequence(payload.command, world_objects)
    
    # 3. Log task execution to SQLite database
    task_id = f"TASK-{str(uuid.uuid4())[:8].upper()}"
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
        INSERT INTO task_history (task_id, raw_command, interpreted_goal, target_object, destination, execution_status)
        VALUES (?, ?, ?, ?, ?, 'SUCCESS')
        """, (
            task_id,
            payload.command,
            plan["understoodCommand"],
            plan["targetObject"],
            plan["targetReceptor"],
        ))
        
        # 4. Generate trajectory-joint angle telemetry and store step sequences
        for i, step in enumerate(plan["planSteps"]):
            try:
                # Solve Inverse Kinematics for joint degree configurations (Yaw, Pitch, Flexion)
                angles = kinematics.solve_inverse_kinematics(step["targetX"], step["targetY"], step["targetZ"])
                
                # Determine standard vacuum state depending on intermediate steps
                vacuum_state = 1 if step["action"] in ["grasp", "lift", "transport"] else 0
                
                cursor.execute("""
                INSERT INTO trajectory_telemetry (task_id, sequence_step, joint_1_deg, joint_2_deg, joint_3_deg, vacuum_state)
                VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    task_id,
                    step["stepNum"],
                    angles["theta1_base_yaw"],
                    angles["theta2_shoulder_pitch"],
                    angles["theta3_elbow_flexion"],
                    vacuum_state
                ))
                
                # Attach Joint Angle Degrees directly onto response JSON steps object
                step["angles"] = angles
                step["vacuum_state"] = vacuum_state
                
            except Exception as e:
                # Attach out-of-reach flag with standard fallback configurations if kinematics overflows
                step["angles"] = {"theta1_base_yaw": 0, "theta2_shoulder_pitch": 0, "theta3_elbow_flexion": 0, "error": str(e)}
                step["vacuum_state"] = 0
                
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Database Logging Failed: {str(e)}")
    finally:
        conn.close()

    plan["task_id"] = task_id
    return plan

@app.get("/api/v1/telemetry/history")
def get_historical_tasks():
    """Retrieves list of previous tasks and joint telemetries run."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT task_id, raw_command, interpreted_goal, target_object, destination, execution_status, recorded_at
    FROM task_history ORDER BY recorded_at DESC
    """)
    tasks = cursor.fetchall()
    conn.close()
    
    return [
        {
            "task_id": r[0],
            "command": r[1],
            "goal": r[2],
            "target": r[3],
            "destination": r[4],
            "status": r[5],
            "timestamp": r[6]
        }
        for r in tasks
    ]

@app.post("/api/v1/brain/autonomous_reasoning")
def execute_autonomous_reasoning(payload: CommandInput):
    """
    Triggers the complete, closed-loop Autonomous Reasoning Engine:
    Perceive -> Parse -> Dynamic State-Sync -> Trajectory Formulate -> Kinematics Guard -> Memory Record
    """
    try:
        result = reasoning_engine.run_pipeline(payload.command)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Autonomous loop runtime failure: {str(e)}")

